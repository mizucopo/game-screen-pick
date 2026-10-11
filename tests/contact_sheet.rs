use std::path::{Path, PathBuf};

use game_screen_pick::contact_sheet::{
    ContextImage, SheetEntry, render_candidate_sheets, render_selected_sheet,
};
use image::{Rgb, RgbImage};
use tempfile::TempDir;

fn solid(directory: &Path, name: &str, color: [u8; 3]) -> PathBuf {
    let path = directory.join(name);
    RgbImage::from_pixel(320, 180, Rgb(color))
        .save(&path)
        .unwrap();
    path
}

fn entry(id: usize, image: PathBuf) -> SheetEntry {
    SheetEntry {
        id: format!("c{id:04}"),
        rank: None,
        source: "日本語の録画.mp4".to_owned(),
        requested_seconds: 12.0,
        actual_seconds: 12.125,
        scene: "探索・戦闘・会話".to_owned(),
        image,
        context: None,
    }
}

fn load(path: &Path) -> RgbImage {
    image::open(path).unwrap().into_rgb8()
}

fn close_color(image: &RgbImage, x: u32, y: u32, expected: [u8; 3]) {
    let actual = image.get_pixel(x, y).0;
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(&a, b)| a.abs_diff(b) < 12),
        "pixel ({x}, {y}): {actual:?} != {expected:?}"
    );
}

fn region(image: &RgbImage, x: u32, y: u32, width: u32, height: u32) -> Vec<[u8; 3]> {
    (y..y + height)
        .flat_map(|y| (x..x + width).map(move |x| image.get_pixel(x, y).0))
        .collect()
}

#[test]
fn bounded_candidate_pages_preserve_order_aspect_and_orientation() {
    let temp = TempDir::new().unwrap();
    let portrait = temp.path().join("portrait.png");
    let mut image = RgbImage::from_pixel(90, 160, Rgb([200, 30, 50]));
    for y in 80..160 {
        for x in 0..90 {
            image.put_pixel(x, y, Rgb([20, 190, 60]));
        }
    }
    image.save(&portrait).unwrap();
    let mut entries = vec![entry(1, portrait)];
    for id in 2..=13 {
        entries.push(entry(
            id,
            solid(temp.path(), &format!("{id}.png"), [30, id as u8 * 10, 220]),
        ));
    }
    let sheets = render_candidate_sheets(&entries, &temp.path().join("sheets")).unwrap();
    assert_eq!(sheets.len(), 2);
    let first = load(&sheets[0]);
    let second = load(&sheets[1]);
    assert_eq!(first.dimensions(), (1080, 1500));
    assert_eq!(second.dimensions(), (368, 384));
    // The portrait occupies 101 x 180 pixels, with padding on both sides.
    close_color(&first, 160, 90, [200, 30, 50]);
    close_color(&first, 160, 190, [20, 190, 60]);
    close_color(&first, 60, 138, [31, 39, 49]);
    close_color(&first, 310, 138, [31, 39, 49]);
    close_color(&first, 140, 90, [200, 30, 50]);
    close_color(&first, 225, 90, [200, 30, 50]);
    close_color(&first, 130, 90, [31, 39, 49]);
    close_color(&first, 237, 90, [31, 39, 49]);
    close_color(&first, 540, 138, [30, 20, 220]);
    close_color(&first, 184, 1254, [30, 100, 220]);
    close_color(&second, 184, 138, [30, 130, 220]);
}

#[test]
fn context_triptych_and_final_ranks_follow_the_matching_images() {
    let temp = TempDir::new().unwrap();
    let before = solid(temp.path(), "before.png", [210, 40, 30]);
    let center = solid(temp.path(), "center.png", [20, 190, 40]);
    let after = solid(temp.path(), "after.png", [20, 40, 220]);
    let mut candidate = entry(1, center.clone());
    candidate.context = Some([
        ContextImage {
            image: before.clone(),
            actual_seconds: 11.125,
        },
        ContextImage {
            image: center.clone(),
            actual_seconds: 12.125,
        },
        ContextImage {
            image: after.clone(),
            actual_seconds: 13.125,
        },
    ]);
    let sheets =
        render_candidate_sheets(&[candidate.clone()], &temp.path().join("context")).unwrap();
    let context = load(&sheets[0]);
    assert_eq!(context.dimensions(), (368, 472));
    close_color(&context, 184, 138, [20, 190, 40]);
    close_color(&context, 73, 387, [210, 40, 30]);
    close_color(&context, 183, 387, [20, 190, 40]);
    close_color(&context, 293, 387, [20, 40, 220]);

    let mut rank3 = entry(3, before);
    rank3.rank = Some(3);
    let mut rank1 = entry(1, center);
    rank1.rank = Some(1);
    let mut rank2 = entry(2, after);
    rank2.rank = Some(2);
    let selected =
        render_selected_sheet(&[rank3, rank1, rank2], &temp.path().join("selected")).unwrap();
    let selected = load(&selected);
    assert_eq!(selected.dimensions(), (1080, 384));
    close_color(&selected, 184, 138, [20, 190, 40]);
    close_color(&selected, 540, 138, [20, 40, 220]);
    close_color(&selected, 896, 138, [210, 40, 30]);
}

#[test]
fn labels_render_japanese_and_distinguish_requested_actual_ids_sources_and_rank() {
    let temp = TempDir::new().unwrap();
    let mut original = entry(1, solid(temp.path(), "frame.png", [40, 100, 180]));
    let original_path =
        render_candidate_sheets(&[original.clone()], &temp.path().join("original")).unwrap();
    let original_image = load(&original_path[0]);
    let source = region(&original_image, 24, 240, 320, 48);
    assert!(source.iter().filter(|pixel| pixel[0] < 140).count() > 100);
    original.id = "c0002".to_owned();
    original.source = "別の日本語録画.mp4".to_owned();
    original.requested_seconds = 18.5;
    original.actual_seconds = 18.625;
    original.rank = Some(1);
    let changed_path = render_candidate_sheets(&[original], &temp.path().join("changed")).unwrap();
    let changed = load(&changed_path[0]);
    for (y, height) in [(24, 24), (240, 48), (288, 24), (312, 24)] {
        assert_ne!(
            region(&original_image, 24, y, 320, height),
            region(&changed, 24, y, 320, height)
        );
    }
    assert_eq!(
        region(&original_image, 50, 80, 200, 120),
        region(&changed, 50, 80, 200, 120)
    );
}

#[test]
fn long_source_wraps_without_overlapping_or_discarding_the_last_glyphs() {
    let temp = TempDir::new().unwrap();
    let image = solid(temp.path(), "frame.png", [40, 100, 180]);
    let mut short = entry(1, image.clone());
    short.rank = Some(1);
    let short_path = render_selected_sheet(&[short.clone()], &temp.path().join("short")).unwrap();
    let short_image = load(&short_path);
    let mut long = short;
    long.source = format!("{}末尾.mp4", "W".repeat(245));
    assert_eq!(long.source.len(), 255);
    let long_path = render_selected_sheet(&[long.clone()], &temp.path().join("long")).unwrap();
    let long_image = load(&long_path);
    assert!(long_image.height() > short_image.height());
    assert_eq!(
        region(&short_image, 50, 80, 200, 120),
        region(&long_image, 50, 80, 200, 120)
    );
    long.source = format!("{}別尾.mp4", "W".repeat(245));
    let changed_path = render_selected_sheet(&[long], &temp.path().join("tail")).unwrap();
    let changed = load(&changed_path);
    assert_eq!(long_image.dimensions(), changed.dimensions());
    assert_ne!(
        region(&long_image, 24, 360, 320, long_image.height() - 384),
        region(&changed, 24, 360, 320, changed.height() - 384)
    );
}

#[test]
fn unsupported_unicode_renders_identifiable_labels_without_changing_images_or_metadata() {
    let temp = TempDir::new().unwrap();
    let before = solid(temp.path(), "before.png", [210, 40, 30]);
    let center = solid(temp.path(), "center.png", [20, 190, 40]);
    let after = solid(temp.path(), "after.png", [20, 40, 220]);
    let mut original = entry(1, center.clone());
    original.rank = Some(1);
    original.source = "日本語🎮_नमस्ते.mp4".to_owned();
    original.scene = "探索🎮क".to_owned();
    original.context = Some([
        ContextImage {
            image: before,
            actual_seconds: 11.125,
        },
        ContextImage {
            image: center,
            actual_seconds: 12.125,
        },
        ContextImage {
            image: after,
            actual_seconds: 13.125,
        },
    ]);
    let mut escaped = original.clone();
    escaped.source = "日本語\\U0001F3AE_\\u0928\\u092E\\u0938\\u094D\\u0924\\u0947.mp4".to_owned();
    escaped.scene = "探索\\U0001F3AE\\u0915".to_owned();
    let actual = render_candidate_sheets(
        std::slice::from_ref(&original),
        &temp.path().join("unicode"),
    )
    .unwrap();
    let expected =
        render_candidate_sheets(std::slice::from_ref(&escaped), &temp.path().join("escaped"))
            .unwrap();
    assert_eq!(
        std::fs::read(&actual[0]).unwrap(),
        std::fs::read(&expected[0]).unwrap()
    );
    let actual = render_selected_sheet(
        std::slice::from_ref(&original),
        &temp.path().join("selected-unicode"),
    )
    .unwrap();
    let expected = render_selected_sheet(
        std::slice::from_ref(&escaped),
        &temp.path().join("selected-escaped"),
    )
    .unwrap();
    assert_eq!(
        std::fs::read(actual).unwrap(),
        std::fs::read(expected).unwrap()
    );
    assert_eq!(original.source, "日本語🎮_नमस्ते.mp4");
    assert_eq!(original.scene, "探索🎮क");
}

#[test]
fn expanded_unicode_source_wraps_in_full_and_retains_the_original_scene() {
    let temp = TempDir::new().unwrap();
    let mut original = entry(1, solid(temp.path(), "frame.png", [40, 100, 180]));
    original.rank = Some(1);
    original.source = format!("{}.mp4", "🎮".repeat(62));
    original.scene = "🎮".repeat(128);
    let path =
        render_selected_sheet(std::slice::from_ref(&original), &temp.path().join("long")).unwrap();
    let sheet = load(&path);
    assert!(sheet.height() > 600);
    assert!(u64::from(sheet.width()) * u64::from(sheet.height()) <= 200_000_000);
    original.source = format!("{}🎯.mp4", "🎮".repeat(61));
    let changed_path =
        render_selected_sheet(std::slice::from_ref(&original), &temp.path().join("tail")).unwrap();
    let changed = load(&changed_path);
    assert_eq!(sheet.dimensions(), changed.dimensions());
    assert_eq!(
        region(&sheet, 24, 240, 320, 24),
        region(&changed, 24, 240, 320, 24)
    );
    assert_ne!(
        region(&sheet, 24, sheet.height() - 144, 320, 48),
        region(&changed, 24, changed.height() - 144, 320, 48)
    );
    assert_eq!(original.scene, "🎮".repeat(128));
}

#[test]
fn filesystem_sized_unicode_sources_render_in_full() {
    let temp = TempDir::new().unwrap();
    let mut original = entry(1, solid(temp.path(), "frame.png", [40, 100, 180]));
    original.rank = Some(1);
    for source in [
        format!("{}.mp4", "録".repeat(90)),
        format!("{}.mp4", "録".repeat(251)),
        format!("{}.webm", "🎮".repeat(125)),
    ] {
        assert!(source.len() > 255);
        assert!(source.encode_utf16().count() <= 255);
        if cfg!(target_os = "macos") {
            let named_file = temp.path().join(&source);
            std::fs::write(&named_file, b"filename proof").unwrap();
            assert_eq!(std::fs::read(named_file).unwrap(), b"filename proof");
        }
        original.source = source.clone();
        let path =
            render_selected_sheet(std::slice::from_ref(&original), &temp.path().join("sheet"))
                .unwrap();
        let sheet = load(&path);
        close_color(&sheet, 184, 138, [40, 100, 180]);
        assert!(sheet.height() > 384);
        assert_eq!(original.source, source);
    }
    original.source = format!("{}.mp4", "録".repeat(251));
    let path =
        render_selected_sheet(std::slice::from_ref(&original), &temp.path().join("long")).unwrap();
    let sheet = load(&path);
    original.source = format!("{}画.mp4", "録".repeat(250));
    let changed_path =
        render_selected_sheet(std::slice::from_ref(&original), &temp.path().join("tail")).unwrap();
    let changed = load(&changed_path);
    assert_eq!(sheet.dimensions(), changed.dimensions());
    assert_ne!(
        region(&sheet, 24, sheet.height() - 144, 320, 48),
        region(&changed, 24, changed.height() - 144, 320, 48)
    );
}

#[test]
fn invalid_labels_metadata_corruption_and_context_fail_clearly() {
    let temp = TempDir::new().unwrap();
    let valid = entry(1, solid(temp.path(), "frame.png", [40, 100, 180]));
    let scratch = temp.path().join("sheets");
    let mut invalid = valid.clone();
    invalid.source.push('\0');
    assert!(render_candidate_sheets(&[invalid], &scratch).is_err());
    let mut invalid = valid.clone();
    invalid.id = "c".repeat(33);
    assert!(render_candidate_sheets(&[invalid], &scratch).is_err());
    let mut invalid = valid.clone();
    invalid.source = "f".repeat(256);
    assert!(render_candidate_sheets(&[invalid], &scratch).is_err());
    for source in [
        format!("{}.mp4", "録".repeat(252)),
        format!("{}.mp4", "🎮".repeat(126)),
    ] {
        let mut invalid = valid.clone();
        invalid.source = source;
        assert!(render_candidate_sheets(&[invalid], &scratch).is_err());
    }
    let mut invalid = valid.clone();
    invalid.scene = "s".repeat(513);
    assert!(render_candidate_sheets(&[invalid], &scratch).is_err());
    let mut invalid = valid.clone();
    invalid.id.push('\n');
    assert!(render_candidate_sheets(&[invalid], &scratch).is_err());
    let mut invalid = valid.clone();
    invalid.actual_seconds = f64::NAN;
    assert!(render_candidate_sheets(&[invalid], &scratch).is_err());
    let mut invalid = valid.clone();
    invalid.context = Some(std::array::from_fn(|_| ContextImage {
        image: valid.image.clone(),
        actual_seconds: 11.0,
    }));
    assert!(render_candidate_sheets(&[invalid], &scratch).is_err());
    let mut corrupt = valid.clone();
    corrupt.image = temp.path().join("corrupt.jpg");
    std::fs::write(&corrupt.image, b"not an image").unwrap();
    assert!(render_candidate_sheets(&[corrupt], &scratch).is_err());
    assert!(render_selected_sheet(std::slice::from_ref(&valid), &scratch).is_err());
    let mut duplicate_rank = valid.clone();
    duplicate_rank.rank = Some(1);
    let mut other = entry(2, valid.image.clone());
    other.rank = Some(1);
    assert!(render_selected_sheet(&[duplicate_rank, other], &scratch).is_err());
    assert!(render_candidate_sheets(&[valid.clone(), valid], &scratch).is_err());
}
