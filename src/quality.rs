//! Bounded RGB decoding and conservative image-quality screening.
use std::{fs, io::BufReader, path::Path};

use anyhow::{Context, Result, ensure};
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits, RgbImage, imageops};
use serde::Serialize;

pub const MAX_IMAGE_PIXELS: u64 = 32_000_000;
pub const MAX_IMAGE_EDGE: u32 = 16_384;
pub const MAX_IMAGE_BYTES: u64 = 64 * 1024 * 1024;
pub const NEAR_DUPLICATE_DISTANCE: f64 = 0.035;

#[derive(Debug, Clone, Serialize)]
pub struct Quality {
    pub width: u32,
    pub height: u32,
    pub brightness: f64,
    pub contrast: f64,
    pub sharpness: f64,
    pub dark_fraction: f64,
    pub white_fraction: f64,
    pub score: f64,
    pub rejection: Option<String>,
    #[serde(skip)]
    fingerprint: Vec<u8>,
}

/// Decode one image at a time. Check header dimensions before allocating pixels,
/// including formats whose decoder allocation limit is only best effort.
pub fn read_image(path: &Path) -> Result<RgbImage> {
    let metadata = fs::symlink_metadata(path).context("cannot inspect candidate image")?;
    ensure!(
        metadata.is_file(),
        "candidate image must be an ordinary file"
    );
    ensure!(
        metadata.len() > 0 && metadata.len() <= MAX_IMAGE_BYTES,
        "candidate image exceeds the encoded byte limit or is empty"
    );
    let mut reader = ImageReader::new(BufReader::new(fs::File::open(path)?))
        .with_guessed_format()
        .context("cannot identify candidate image")?;
    ensure!(
        matches!(reader.format(), Some(ImageFormat::Png | ImageFormat::Jpeg)),
        "candidate image must be PNG or JPEG"
    );
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_EDGE);
    limits.max_image_height = Some(MAX_IMAGE_EDGE);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let mut decoder = reader.into_decoder().context("invalid image header")?;
    let (width, height) = decoder.dimensions();
    ensure!(
        width >= 16
            && height >= 16
            && u64::from(width) * u64::from(height) <= MAX_IMAGE_PIXELS
            && width.max(height) <= width.min(height).saturating_mul(32)
            && decoder.total_bytes() <= 192 * 1024 * 1024,
        "candidate image has abnormal dimensions or exceeds the pixel limit"
    );
    let orientation = decoder.orientation().context("invalid image orientation")?;
    let mut image = DynamicImage::from_decoder(decoder).context("corrupt candidate image")?;
    image.apply_orientation(orientation);
    Ok(image.into_rgb8())
}

pub fn analyze(path: &Path) -> Result<Quality> {
    let original = read_image(path)?;
    let (width, height) = original.dimensions();
    let preview = DynamicImage::ImageRgb8(original)
        .thumbnail(512, 512)
        .into_rgb8();
    let luma: Vec<f64> = preview
        .pixels()
        .map(|pixel| {
            (0.2126 * f64::from(pixel[0])
                + 0.7152 * f64::from(pixel[1])
                + 0.0722 * f64::from(pixel[2]))
                / 255.0
        })
        .collect();
    let count = luma.len() as f64;
    let brightness = luma.iter().sum::<f64>() / count;
    let contrast = (luma
        .iter()
        .map(|value| (value - brightness).powi(2))
        .sum::<f64>()
        / count)
        .sqrt();
    let dark_fraction = luma.iter().filter(|&&value| value < 0.06).count() as f64 / count;
    let white_fraction = luma.iter().filter(|&&value| value > 0.94).count() as f64 / count;
    let mut gradient = 0.0;
    let mut edges = 0;
    let stride = preview.width() as usize;
    for y in 0..preview.height() as usize {
        for x in 0..stride {
            let index = y * stride + x;
            if x > 0 {
                gradient += (luma[index] - luma[index - 1]).abs();
                edges += 1;
            }
            if y > 0 {
                gradient += (luma[index] - luma[index - stride]).abs();
                edges += 1;
            }
        }
    }
    let sharpness = gradient / f64::from(edges);
    let rejection = if brightness < 0.025 || dark_fraction > 0.96 {
        Some("dark/black frame".to_owned())
    } else if brightness > 0.975 || white_fraction > 0.96 {
        Some("white/clipped frame".to_owned())
    } else if contrast < 0.025 {
        Some("flat/low-contrast frame".to_owned())
    } else if sharpness < 0.002 {
        Some("indistinct/blurred frame".to_owned())
    } else {
        None
    };
    let score = 100.0
        * (0.45 * (contrast / 0.2).min(1.0)
            + 0.35 * (sharpness / 0.035).min(1.0)
            + 0.2 * (1.0 - (brightness - 0.5).abs() * 2.0));
    let fingerprint = imageops::resize(&preview, 32, 18, imageops::FilterType::Triangle).into_raw();
    Ok(Quality {
        width,
        height,
        brightness,
        contrast,
        sharpness,
        dark_fraction,
        white_fraction,
        score,
        rejection,
        fingerprint,
    })
}

impl Quality {
    /// Mean RGB thumbnail distance; aspect mismatch separates portrait/landscape
    /// content and colour differences distinguish equal-luminance scenes.
    pub fn distance(&self, other: &Self) -> f64 {
        let ratio = f64::from(self.width) / f64::from(self.height);
        let other_ratio = f64::from(other.width) / f64::from(other.height);
        if (ratio / other_ratio - 1.0).abs() > 0.05 {
            return 1.0;
        }
        self.fingerprint
            .iter()
            .zip(&other.fingerprint)
            .map(|(&left, &right)| f64::from(left.abs_diff(right)))
            .sum::<f64>()
            / (self.fingerprint.len() as f64 * 255.0)
    }

    pub fn near_duplicate(&self, other: &Self) -> bool {
        self.distance(other) < NEAR_DUPLICATE_DISTANCE
    }
}
