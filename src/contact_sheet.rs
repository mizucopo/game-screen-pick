use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result, ensure};
use fontdue::{Font, FontSettings};
use image::codecs::jpeg::JpegEncoder;
use image::imageops::{FilterType, resize};
use image::{ExtendedColorType, Rgb, RgbImage};

use crate::quality::{MAX_IMAGE_BYTES, MAX_SELECTED_JPEG_BYTES, read_image_with_limit};

const PAGE_SIZE: usize = 12;
const CANDIDATE_THUMBNAIL_WIDTH: u32 = 320;
const CANDIDATE_COLUMNS: u32 = 3;
const MAX_ENTRIES: usize = 999;
pub(crate) const MAX_SHEET_PIXELS: u64 = 200_000_000;
const BORDER: u32 = 12;
const GAP: u32 = 12;
const BACKGROUND: Rgb<u8> = Rgb([235, 240, 245]);
const CARD: Rgb<u8> = Rgb([255, 255, 255]);
const LETTERBOX: Rgb<u8> = Rgb([31, 39, 49]);
const TEXT: Rgb<u8> = Rgb([25, 36, 51]);
const MUTED: Rgb<u8> = Rgb([74, 88, 105]);

/// The image and actual video timestamp for one before/center/after frame.
#[derive(Clone, Debug)]
pub struct ContextImage {
    pub image: PathBuf,
    pub actual_seconds: f64,
}

/// All labels are supplied from the same candidate/report data as the image.
#[derive(Clone, Debug)]
pub struct SheetEntry {
    pub id: String,
    pub rank: Option<usize>,
    pub source: String,
    pub requested_seconds: f64,
    pub actual_seconds: f64,
    pub scene: String,
    pub image: PathBuf,
    /// Ordered before, center, after. The center timestamp must match the image.
    pub context: Option<[ContextImage; 3]>,
}

/// Render bounded candidate pages into a caller-owned private scratch directory.
/// Context frames, when present, are shown in before/center/after order.
pub fn render_candidate_sheets(entries: &[SheetEntry], scratch: &Path) -> Result<Vec<PathBuf>> {
    validate_entries(entries, false)?;
    let font = font()?;
    prepare_directory(scratch)?;
    let mut paths = Vec::new();
    for (page, entries) in entries.chunks(PAGE_SIZE).enumerate() {
        let path = scratch.join(format!("candidate-contact-sheet-{:02}.jpg", page + 1));
        render(
            entries,
            &path,
            font,
            CANDIDATE_THUMBNAIL_WIDTH,
            CANDIDATE_COLUMNS,
            true,
        )?;
        paths.push(path);
    }
    Ok(paths)
}

/// Render every selected image once, in contiguous rank order, into one JPEG.
/// Candidate review pages carry the context frames; this overview carries the
/// full set of selected main images. The RGB canvas is bounded at 200 MP, and
/// source images are decoded one at a time. Long source labels wrap in full.
pub fn render_selected_sheet(entries: &[SheetEntry], scratch: &Path) -> Result<PathBuf> {
    validate_entries(entries, true)?;
    let font = font()?;
    prepare_directory(scratch)?;
    let mut ordered = entries.to_vec();
    ordered.sort_by_key(|entry| entry.rank);
    let (width, columns) = if entries.len() <= PAGE_SIZE {
        (320, 3)
    } else {
        (160, 6)
    };
    let path = scratch.join("selected-contact-sheet.jpg");
    render(&ordered, &path, font, width, columns, false)?;
    Ok(path)
}

/// Pure lower bound for both required assessment phases with one candidate.
/// Secondary review requires the context strip. Font metrics and wrapped labels
/// add height, so exclude their unknown size rather than assume a maximum page.
pub(crate) fn minimum_candidate_sheet_geometry() -> (u32, u32) {
    let width = canvas_width(cell_width(CANDIDATE_THUMBNAIL_WIDTH), 1);
    let main_height = thumbnail_height(CANDIDATE_THUMBNAIL_WIDTH);
    let primary = card_height(main_height, 0, 0);
    let secondary = card_height(
        main_height,
        0,
        context_height(
            thumbnail_height(context_thumbnail_width(CANDIDATE_THUMBNAIL_WIDTH)),
            0,
        ),
    );
    let height = canvas_height(&[primary.max(secondary)]);
    (width, height)
}

/// Reject only limits that cannot hold even the geometric minimum of one card.
/// Actual rendered pages still require their usual image geometry validation.
pub(crate) fn validate_candidate_sheet_limits(edge: u32, pixels: u64) -> Result<()> {
    let (width, height) = minimum_candidate_sheet_geometry();
    ensure!(
        edge >= width.max(height) && pixels >= u64::from(width) * u64::from(height),
        "configured vision limits cannot fit even one candidate contact sheet"
    );
    Ok(())
}

fn cell_width(thumbnail_width: u32) -> u32 {
    thumbnail_width + 2 * BORDER
}
fn thumbnail_height(thumbnail_width: u32) -> u32 {
    thumbnail_width * 9 / 16
}
fn context_thumbnail_width(thumbnail_width: u32) -> u32 {
    (thumbnail_width - 2 * GAP) / 3
}
fn context_height(thumbnail_height: u32, labels_height: u32) -> u32 {
    thumbnail_height + labels_height + GAP
}
fn card_height(thumbnail_height: u32, labels_height: u32, context_height: u32) -> u32 {
    thumbnail_height + labels_height + context_height + GAP + 2 * BORDER
}
fn canvas_width(cell_width: u32, columns: u32) -> u32 {
    columns * cell_width + (columns + 1) * GAP
}
fn canvas_height(row_heights: &[u32]) -> u32 {
    row_heights.iter().sum::<u32>() + (row_heights.len() as u32 + 1) * GAP
}

fn prepare_directory(path: &Path) -> Result<()> {
    fs::create_dir_all(path).with_context(|| {
        format!(
            "cannot create contact sheet scratch directory: {}",
            path.display()
        )
    })
}

fn font() -> Result<&'static Font> {
    static FONT: OnceLock<std::result::Result<Font, String>> = OnceLock::new();
    FONT.get_or_init(|| {
        Font::from_bytes(
            // Static Regular weight, licensed under assets/OFL.txt.
            // Source: https://github.com/notofonts/noto-cjk/tree/main/Sans/SubsetOTF/JP
            include_bytes!("../assets/NotoSansJP-Regular.otf").as_slice(),
            FontSettings::default(),
        )
        .map_err(str::to_owned)
    })
    .as_ref()
    .map_err(|error| anyhow::anyhow!("cannot load bundled Japanese contact sheet font: {error}"))
}

fn validate_entries(entries: &[SheetEntry], selected: bool) -> Result<()> {
    ensure!(
        !entries.is_empty() && entries.len() <= MAX_ENTRIES,
        "contact sheet requires 1..={MAX_ENTRIES} entries"
    );
    let mut ids = HashSet::new();
    let mut ranks = HashSet::new();
    for entry in entries {
        ensure!(
            !entry.id.trim().is_empty() && entry.id.len() <= 32 && ids.insert(&entry.id),
            "contact sheet candidate IDs must be nonempty, unique, and at most 32 bytes"
        );
        ensure!(
            entry.source.encode_utf16().take(256).count() <= 255 && !entry.source.trim().is_empty(),
            "contact sheet input source must be nonempty and at most 255 UTF-16 units"
        );
        ensure!(
            entry.scene.len() <= 512,
            "contact sheet scene label exceeds 512 bytes"
        );
        for text in [&entry.id, &entry.source, &entry.scene] {
            validate_label(text)?;
        }
        for seconds in [entry.requested_seconds, entry.actual_seconds] {
            ensure!(
                seconds.is_finite() && seconds >= 0.0,
                "contact sheet timestamps must be finite and nonnegative"
            );
        }
        if selected {
            let rank = entry
                .rank
                .context("selected contact sheet entry has no rank")?;
            ensure!(
                (1..=entries.len()).contains(&rank) && ranks.insert(rank),
                "selected contact sheet ranks must be unique and contiguous from 1"
            );
        }
        if let Some(context) = &entry.context {
            ensure!(
                context
                    .iter()
                    .all(|frame| frame.actual_seconds.is_finite() && frame.actual_seconds >= 0.0)
                    && context[0].actual_seconds <= context[1].actual_seconds
                    && context[1].actual_seconds <= context[2].actual_seconds
                    && (context[1].actual_seconds - entry.actual_seconds).abs() <= 0.000_001,
                "contact sheet context must be ordered before/center/after and match the center timestamp"
            );
        }
    }
    Ok(())
}

fn validate_label(text: &str) -> Result<()> {
    ensure!(
        !text.chars().any(char::is_control),
        "contact sheet labels must not contain control characters"
    );
    Ok(())
}

fn display_text(font: &Font, text: &str) -> String {
    let mut display = String::with_capacity(text.len());
    for character in text.chars() {
        if character == '\\' {
            display.push_str("\\\\");
        } else if font.lookup_glyph_index(character) == 0 {
            let codepoint = u32::from(character);
            display.push_str(&if codepoint <= 0xffff {
                format!("\\u{codepoint:04X}")
            } else {
                format!("\\U{codepoint:08X}")
            });
        } else {
            display.push(character);
        }
    }
    display
}

#[derive(Clone, Copy)]
struct Layout {
    thumbnail_width: u32,
    thumbnail_height: u32,
    font_size: f32,
    line_height: u32,
    cell_width: u32,
}

impl Layout {
    fn new(width: u32, font: &Font) -> Self {
        let font_size = if width >= 320 { 16.0 } else { 14.0 };
        Self {
            thumbnail_width: width,
            thumbnail_height: thumbnail_height(width),
            font_size,
            line_height: 0,
            cell_width: cell_width(width),
        }
        .with_font_size(font, font_size)
    }

    fn with_font_size(mut self, font: &Font, font_size: f32) -> Self {
        self.font_size = font_size;
        self.line_height = font
            .horizontal_line_metrics(font_size)
            .expect("bundled font has horizontal metrics")
            .new_line_size
            .ceil() as u32;
        self
    }
}

struct Labels {
    heading: Vec<String>,
    source: Vec<String>,
    requested: Vec<String>,
    actual: Vec<String>,
    scene: Vec<String>,
    context: Option<[Vec<String>; 3]>,
}

impl Labels {
    fn new(entry: &SheetEntry, font: &Font, layout: Layout, context: bool) -> Self {
        let wrap = |text: &str| {
            wrapped_lines(
                font,
                text,
                layout.font_size,
                layout.thumbnail_width,
                usize::MAX,
            )
        };
        let heading = entry.rank.map_or_else(
            || format!("候補 {}", entry.id),
            |rank| format!("順位 {rank}  候補 {}", entry.id),
        );
        let context = if context {
            entry.context.as_ref().map(|frames| {
                std::array::from_fn(|index| {
                    wrapped_lines(
                        font,
                        &format!(
                            "{} {}",
                            ["前", "中心", "後"][index],
                            format_seconds(frames[index].actual_seconds)
                        ),
                        14.0,
                        context_thumbnail_width(layout.thumbnail_width),
                        usize::MAX,
                    )
                })
            })
        } else {
            None
        };
        Self {
            heading: wrap(&heading),
            source: wrap(&format!("入力: {}", entry.source)),
            requested: wrap(&format!(
                "要求: {}",
                format_seconds(entry.requested_seconds)
            )),
            actual: wrap(&format!("実時刻: {}", format_seconds(entry.actual_seconds))),
            scene: wrapped_lines(
                font,
                &format!("場面: {}", entry.scene),
                layout.font_size,
                layout.thumbnail_width,
                1,
            ),
            context,
        }
    }

    fn lower_lines(&self) -> u32 {
        (self.source.len().max(2) + self.requested.len() + self.actual.len() + self.scene.len())
            as u32
    }

    fn height(&self, layout: Layout, context_layout: Layout) -> u32 {
        let context_height = self.context.as_ref().map_or(0, |labels| {
            context_height(
                context_layout.thumbnail_height,
                labels.iter().map(Vec::len).max().unwrap_or(1) as u32 * context_layout.line_height,
            )
        });
        card_height(
            layout.thumbnail_height,
            (self.heading.len() as u32 + self.lower_lines()) * layout.line_height,
            context_height,
        )
    }
}

struct Grid {
    columns: u32,
    width: u32,
    height: u32,
    row_heights: Vec<u32>,
}

struct SheetLayout {
    main: Layout,
    context: Layout,
    labels: Vec<Labels>,
    grid: Grid,
}

fn sheet_layout(
    entries: &[SheetEntry],
    font: &Font,
    width: u32,
    columns: u32,
    show_context: bool,
) -> Result<SheetLayout> {
    let main = Layout::new(width, font);
    let context = Layout::new(context_thumbnail_width(width), font);
    let prepare = |main: Layout| {
        let labels = entries
            .iter()
            .map(|entry| Labels::new(entry, font, main, show_context))
            .collect::<Vec<_>>();
        let grid = grid(&labels, main, context, columns)?;
        Ok(SheetLayout {
            main,
            context,
            labels,
            grid,
        })
    };
    // Escaping a maximum-length source filename can exceed the canvas limit.
    // Retry only the dense final overview, using the same layout measurements.
    let mut result = prepare(main);
    if width == 160 && !show_context {
        for font_size in [12.0, 10.0, 8.0] {
            if result.is_ok() {
                break;
            }
            result = prepare(main.with_font_size(font, font_size));
        }
    }
    result
}

fn grid(labels: &[Labels], layout: Layout, context_layout: Layout, columns: u32) -> Result<Grid> {
    let mut columns = columns.min(labels.len() as u32);
    loop {
        let row_heights = labels
            .chunks(columns as usize)
            .map(|row| {
                row.iter()
                    .map(|labels| labels.height(layout, context_layout))
                    .max()
                    .unwrap_or(0)
            })
            .collect::<Vec<_>>();
        let width = canvas_width(layout.cell_width, columns);
        let height = canvas_height(&row_heights);
        if height <= 65_535 {
            ensure!(
                width <= 65_535 && u64::from(width) * u64::from(height) <= MAX_SHEET_PIXELS,
                "contact sheet exceeds the 200-megapixel JPEG canvas limit"
            );
            return Ok(Grid {
                columns,
                width,
                height,
                row_heights,
            });
        }
        columns += 1;
        ensure!(
            columns <= labels.len() as u32,
            "contact sheet labels exceed JPEG dimensions"
        );
    }
}

fn render(
    entries: &[SheetEntry],
    path: &Path,
    font: &Font,
    width: u32,
    columns: u32,
    show_context: bool,
) -> Result<()> {
    let SheetLayout {
        main: layout,
        context: context_layout,
        labels,
        grid,
    } = sheet_layout(entries, font, width, columns, show_context)?;
    let mut canvas = RgbImage::from_pixel(grid.width, grid.height, BACKGROUND);
    let mut row_y = GAP;
    for (index, (entry, labels)) in entries.iter().zip(&labels).enumerate() {
        let row = index as u32 / grid.columns;
        if index > 0 && (index as u32).is_multiple_of(grid.columns) {
            row_y += grid.row_heights[row as usize - 1] + GAP;
        }
        let card_x = GAP + (index as u32 % grid.columns) * (layout.cell_width + GAP);
        fill_rect(
            &mut canvas,
            card_x,
            row_y,
            layout.cell_width,
            grid.row_heights[row as usize],
            CARD,
        );
        let x = card_x + BORDER;
        let y = row_y + BORDER;
        draw_wrapped(&mut canvas, font, &labels.heading, x, y, layout, TEXT);
        let image_y = y + labels.heading.len() as u32 * layout.line_height;
        draw_image(
            &mut canvas,
            &entry.image,
            x,
            image_y,
            layout.thumbnail_width,
            layout.thumbnail_height,
            if show_context {
                MAX_IMAGE_BYTES
            } else {
                MAX_SELECTED_JPEG_BYTES
            },
        )?;
        let mut labels_y = image_y + layout.thumbnail_height + GAP;
        for (lines, reserve, color) in [
            (&labels.source, labels.source.len().max(2), TEXT),
            (&labels.requested, labels.requested.len(), MUTED),
            (&labels.actual, labels.actual.len(), MUTED),
            (&labels.scene, labels.scene.len(), MUTED),
        ] {
            draw_wrapped(&mut canvas, font, lines, x, labels_y, layout, color);
            labels_y += reserve as u32 * layout.line_height;
        }
        if let (Some(frames), Some(labels)) = (&entry.context, &labels.context) {
            for (index, frame) in frames.iter().enumerate() {
                let x = x + index as u32 * (context_layout.thumbnail_width + GAP);
                draw_image(
                    &mut canvas,
                    &frame.image,
                    x,
                    labels_y,
                    context_layout.thumbnail_width,
                    context_layout.thumbnail_height,
                    MAX_IMAGE_BYTES,
                )?;
                draw_wrapped(
                    &mut canvas,
                    font,
                    &labels[index],
                    x,
                    labels_y + context_layout.thumbnail_height,
                    context_layout,
                    MUTED,
                );
            }
        }
    }
    let file = File::create(path)
        .with_context(|| format!("cannot create contact sheet: {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    JpegEncoder::new_with_quality(&mut writer, 90)
        .encode(
            canvas.as_raw(),
            canvas.width(),
            canvas.height(),
            ExtendedColorType::Rgb8,
        )
        .with_context(|| format!("cannot encode contact sheet: {}", path.display()))?;
    writer
        .flush()
        .with_context(|| format!("cannot flush contact sheet: {}", path.display()))
}

fn format_seconds(seconds: f64) -> String {
    format!("{seconds:.3}秒")
}

fn fill_rect(canvas: &mut RgbImage, x: u32, y: u32, width: u32, height: u32, color: Rgb<u8>) {
    for y in y..y + height {
        for x in x..x + width {
            canvas.put_pixel(x, y, color);
        }
    }
}

fn draw_image(
    canvas: &mut RgbImage,
    path: &Path,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    max_bytes: u64,
) -> Result<()> {
    let source = read_image_with_limit(path, max_bytes)
        .with_context(|| format!("cannot read contact sheet image: {}", path.display()))?;
    let ratio = (f64::from(width) / f64::from(source.width()))
        .min(f64::from(height) / f64::from(source.height()));
    let target_width = (f64::from(source.width()) * ratio).round().max(1.0) as u32;
    let target_height = (f64::from(source.height()) * ratio).round().max(1.0) as u32;
    let thumbnail = resize(&source, target_width, target_height, FilterType::Triangle);
    fill_rect(canvas, x, y, width, height, LETTERBOX);
    let x = x + (width - target_width) / 2;
    let y = y + (height - target_height) / 2;
    for (px, py, pixel) in thumbnail.enumerate_pixels() {
        canvas.put_pixel(x + px, y + py, *pixel);
    }
    Ok(())
}

fn wrapped_lines(font: &Font, text: &str, size: f32, width: u32, limit: usize) -> Vec<String> {
    let text = display_text(font, text);
    let mut lines = Vec::new();
    let mut line = String::new();
    let mut used_width = 0.0;
    for character in text.chars() {
        let advance = font.metrics(character, size).advance_width;
        if !line.is_empty() && used_width + advance > width as f32 {
            lines.push(std::mem::take(&mut line));
            used_width = 0.0;
            if lines.len() == limit {
                let last = lines.last_mut().expect("line was just pushed");
                let ellipsis = font.metrics('…', size).advance_width;
                while !last.is_empty() && text_width(font, last, size) + ellipsis > width as f32 {
                    last.pop();
                }
                last.push('…');
                return lines;
            }
        }
        line.push(character);
        used_width += advance;
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

fn text_width(font: &Font, text: &str, size: f32) -> f32 {
    text.chars()
        .map(|character| font.metrics(character, size).advance_width)
        .sum()
}

fn draw_wrapped(
    canvas: &mut RgbImage,
    font: &Font,
    lines: &[String],
    x: u32,
    y: u32,
    layout: Layout,
    color: Rgb<u8>,
) {
    let ascent = font
        .horizontal_line_metrics(layout.font_size)
        .expect("bundled font has horizontal metrics")
        .ascent
        .ceil() as i32;
    for (line, text) in lines.iter().enumerate() {
        let mut cursor = x as f32;
        let baseline = y as i32 + line as i32 * layout.line_height as i32 + ascent;
        for character in text.chars() {
            let (metrics, bitmap) = font.rasterize(character, layout.font_size);
            let left = cursor.round() as i32 + metrics.xmin;
            let top = baseline - metrics.ymin - metrics.height as i32;
            for py in 0..metrics.height {
                for px in 0..metrics.width {
                    let target_x = left + px as i32;
                    let target_y = top + py as i32;
                    if target_x < x as i32
                        || target_x >= (x + layout.thumbnail_width) as i32
                        || target_y < y as i32
                        || target_y >= (y + lines.len() as u32 * layout.line_height) as i32
                    {
                        continue;
                    }
                    let alpha = u32::from(bitmap[py * metrics.width + px]);
                    if alpha != 0 {
                        let pixel = canvas.get_pixel_mut(target_x as u32, target_y as u32);
                        for channel in 0..3 {
                            pixel[channel] = ((u32::from(color[channel]) * alpha
                                + u32::from(pixel[channel]) * (255 - alpha))
                                / 255) as u8;
                        }
                    }
                }
            }
            cursor += metrics.advance_width;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_geometry_lower_bound_matches_one_card_layout_width() {
        let mut entry = SheetEntry {
            id: "c000001".to_owned(),
            rank: None,
            source: "input.mp4".to_owned(),
            requested_seconds: 0.0,
            actual_seconds: 0.0,
            scene: String::new(),
            image: PathBuf::new(),
            context: None,
        };
        let prepared = sheet_layout(
            &[entry.clone()],
            font().unwrap(),
            CANDIDATE_THUMBNAIL_WIDTH,
            CANDIDATE_COLUMNS,
            true,
        )
        .unwrap();
        let (width, height) = minimum_candidate_sheet_geometry();
        assert_eq!(width, prepared.grid.width);
        // The bundled font's primary labels add height beyond the pure bound.
        assert!(height <= prepared.grid.height);
        assert!(
            u64::from(width) * u64::from(height)
                <= u64::from(prepared.grid.width) * u64::from(prepared.grid.height)
        );
        entry.context = Some(std::array::from_fn(|_| ContextImage {
            image: PathBuf::new(),
            actual_seconds: 0.0,
        }));
        let secondary = sheet_layout(
            &[entry],
            font().unwrap(),
            CANDIDATE_THUMBNAIL_WIDTH,
            CANDIDATE_COLUMNS,
            true,
        )
        .unwrap();
        assert_eq!(width, secondary.grid.width);
        assert!(height <= secondary.grid.height);
        assert!(secondary.grid.height > prepared.grid.height);
    }

    #[test]
    fn candidate_sheet_limits_reject_only_below_the_pure_geometry_boundary() {
        let (width, height) = minimum_candidate_sheet_geometry();
        let edge = width.max(height);
        let pixels = u64::from(width) * u64::from(height);
        assert!(validate_candidate_sheet_limits(edge - 1, pixels).is_err());
        assert!(validate_candidate_sheet_limits(edge, pixels - 1).is_err());
        validate_candidate_sheet_limits(edge, pixels).unwrap();
        assert!(validate_candidate_sheet_limits(200, u64::MAX).is_err());
    }

    #[test]
    fn japanese_glyphs_and_wrapping_fit_the_label_box() {
        let font = font().unwrap();
        let text = "探索・戦闘・会話の画像と前後の時刻。日本語録画の長い入力ファイル名.mp4";
        validate_label(text).unwrap();
        let lines = wrapped_lines(font, text, 14.0, 100, usize::MAX);
        assert!(lines.len() > 2);
        assert_eq!(lines.concat(), text);
        assert!(lines.iter().all(|line| {
            line.chars()
                .map(|ch| font.metrics(ch, 14.0).advance_width)
                .sum::<f32>()
                <= 100.0
        }));
        assert!(font.rasterize('戦', 16.0).1.iter().any(|alpha| *alpha > 0));
    }

    #[test]
    fn unicode_fallbacks_and_literal_backslashes_remain_distinct_after_wrapping() {
        let font = font().unwrap();
        for (raw, expected) in [
            ("日本語🎮", r"日本語\U0001F3AE"),
            (r"日本語\U0001F3AE", r"日本語\\U0001F3AE"),
            ("न", r"\u0928"),
            (r"\u0928", r"\\u0928"),
            ("\\🎮", r"\\\U0001F3AE"),
            (r"\\", r"\\\\"),
        ] {
            let lines = wrapped_lines(font, raw, 14.0, 40, usize::MAX);
            assert_eq!(lines.concat(), expected);
            assert!(
                lines
                    .iter()
                    .all(|line| text_width(font, line, 14.0) <= 40.0)
            );
        }
    }

    #[test]
    fn largest_selected_sheet_and_long_sources_have_bounded_jpeg_dimensions() {
        let font = font().unwrap();
        let entry = SheetEntry {
            id: "c000001".to_owned(),
            rank: Some(999),
            source: "W".repeat(255),
            requested_seconds: 9999.125,
            actual_seconds: 9999.25,
            scene: "探索".to_owned(),
            image: PathBuf::new(),
            context: None,
        };
        let entries = vec![entry.clone(); MAX_ENTRIES];
        let prepared = sheet_layout(&entries, font, 160, 6, false).unwrap();
        assert_eq!(prepared.main.font_size, 14.0);
        let labels = &prepared.labels;
        let dimensions = prepared.grid;
        assert!(dimensions.width <= 65_535 && dimensions.height <= 65_535);
        assert!(u64::from(dimensions.width) * u64::from(dimensions.height) <= MAX_SHEET_PIXELS);
        assert_eq!(labels[0].source.concat(), format!("入力: {}", entry.source));
    }

    #[test]
    fn largest_selected_sheet_fully_wraps_the_widest_unsupported_source_and_maximum_id() {
        let font = font().unwrap();
        let scalar = (0xa0..=0x7ff)
            .filter_map(char::from_u32)
            .filter(|scalar| font.lookup_glyph_index(*scalar) == 0)
            .max_by(|left, right| {
                text_width(font, &display_text(font, &left.to_string()), 14.0).total_cmp(
                    &text_width(font, &display_text(font, &right.to_string()), 14.0),
                )
            })
            .unwrap();
        let entry = SheetEntry {
            id: scalar.to_string().repeat(16),
            rank: Some(999),
            source: format!("W{}.mp4", scalar.to_string().repeat(125)),
            requested_seconds: 9999.125,
            actual_seconds: 9999.25,
            scene: scalar.to_string().repeat(256),
            image: PathBuf::new(),
            context: None,
        };
        assert_eq!(entry.id.len(), 32);
        assert_eq!(entry.source.len(), 255);
        assert_eq!(entry.scene.len(), 512);
        let entries = vec![entry.clone(); MAX_ENTRIES];
        let prepared = sheet_layout(&entries, font, 160, 6, false).unwrap();
        assert_eq!(prepared.main.font_size, 12.0);
        let labels = &prepared.labels;
        assert_eq!(
            labels[0].source.concat(),
            format!("入力: {}", display_text(font, &entry.source))
        );
        assert_eq!(labels[0].scene.len(), 1);
        let dimensions = prepared.grid;
        assert!(dimensions.width <= 65_535 && dimensions.height <= 65_535);
        assert!(u64::from(dimensions.width) * u64::from(dimensions.height) <= MAX_SHEET_PIXELS);
    }

    #[test]
    fn utf16_sized_sources_fit_the_largest_selected_sheet_without_truncation() {
        let font = font().unwrap();
        let widest_missing = |range: std::ops::RangeInclusive<u32>| {
            range
                .filter_map(char::from_u32)
                .filter(|scalar| !scalar.is_control() && font.lookup_glyph_index(*scalar) == 0)
                .max_by(|left, right| {
                    text_width(font, &display_text(font, &left.to_string()), 14.0).total_cmp(
                        &text_width(font, &display_text(font, &right.to_string()), 14.0),
                    )
                })
                .unwrap()
        };
        let id_scalar = widest_missing(0xa0..=0x7ff);
        let source_scalar = widest_missing(0xa0..=0xffff);
        for (source, font_size) in [
            (format!("{}.mp4", "録".repeat(251)), 14.0),
            ("\\".repeat(255), 14.0),
            (format!("{}.webm", "🎮".repeat(125)), 10.0),
            (
                format!("{}.mp4", source_scalar.to_string().repeat(251)),
                8.0,
            ),
        ] {
            let entry = SheetEntry {
                id: id_scalar.to_string().repeat(16),
                rank: Some(999),
                source,
                requested_seconds: 9999.125,
                actual_seconds: 9999.25,
                scene: id_scalar.to_string().repeat(256),
                image: PathBuf::new(),
                context: None,
            };
            assert!(entry.source.encode_utf16().count() <= 255);
            let prepared =
                sheet_layout(&vec![entry.clone(); MAX_ENTRIES], font, 160, 6, false).unwrap();
            assert_eq!(prepared.main.font_size, font_size);
            assert_eq!(
                prepared.labels[0].source.concat(),
                format!("入力: {}", display_text(font, &entry.source))
            );
            assert!(prepared.grid.width <= 65_535 && prepared.grid.height <= 65_535);
            assert!(
                u64::from(prepared.grid.width) * u64::from(prepared.grid.height)
                    <= MAX_SHEET_PIXELS
            );
        }
    }
}
