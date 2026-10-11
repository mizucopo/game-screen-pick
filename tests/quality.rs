use game_screen_pick::quality::{MAX_IMAGE_BYTES, analyze, read_image};
use image::{Rgb, RgbImage};
use std::{fs, path::Path};

fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/images")
        .join(name)
}

#[test]
fn boundary_images_are_screened_and_colour_content_is_kept() {
    for (name, reason) in [
        ("near-black.png", "dark"),
        ("near-white.png", "white"),
        ("low-contrast.png", "contrast"),
    ] {
        let quality = analyze(&fixture(name)).unwrap();
        assert!(quality.rejection.unwrap().contains(reason));
    }
    let grid = analyze(&fixture("rgb-grid.png")).unwrap();
    assert!(grid.rejection.is_none());
    assert_eq!((grid.width, grid.height), (33, 19));
    let directory = tempfile::tempdir().unwrap();
    let image = read_image(&fixture("rgb-grid.png")).unwrap();
    let mut changed = image.clone();
    for pixel in changed.pixels_mut() {
        for value in &mut pixel.0 {
            *value = value.saturating_add(2);
        }
    }
    changed.save(directory.path().join("near.png")).unwrap();
    assert!(grid.near_duplicate(&analyze(&directory.path().join("near.png")).unwrap()));
    image::imageops::flip_horizontal(&image)
        .save(directory.path().join("mirror.png"))
        .unwrap();
    assert!(!grid.near_duplicate(&analyze(&directory.path().join("mirror.png")).unwrap()));
}

#[test]
fn flat_frames_fades_and_blur_do_not_become_valid_fillers() {
    let directory = tempfile::tempdir().unwrap();
    for colour in [0, 8, 127, 248, 255] {
        let path = directory.path().join(format!("{colour}.png"));
        RgbImage::from_pixel(64, 32, Rgb([colour; 3]))
            .save(&path)
            .unwrap();
        assert!(analyze(&path).unwrap().rejection.is_some());
    }
    let path = directory.path().join("fade.png");
    RgbImage::from_fn(64, 32, |x, y| Rgb([((x + y) % 8 + 4) as u8; 3]))
        .save(&path)
        .unwrap();
    assert!(analyze(&path).unwrap().rejection.is_some());
    let path = directory.path().join("blur.png");
    RgbImage::from_fn(512, 512, |x, _| Rgb([(64 + x / 4) as u8; 3]))
        .save(&path)
        .unwrap();
    assert!(
        analyze(&path)
            .unwrap()
            .rejection
            .unwrap()
            .contains("blurred")
    );
}

#[test]
fn corrupt_oversized_and_abnormal_images_are_rejected_before_pixels() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bad.png");
    fs::write(&path, b"not PNG").unwrap();
    assert!(read_image(&path).is_err());
    let bytes = fs::read(fixture("rgb-grid.png")).unwrap();
    fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();
    assert!(read_image(&path).is_err());
    for (width, height) in [(20000, 32), (8192, 8192), (1, 32), (1024, 16)] {
        let encoder = png::Encoder::new(fs::File::create(&path).unwrap(), width, height);
        let _writer = encoder.write_header().unwrap();
        assert!(read_image(&path).is_err(), "{width}x{height}");
    }
    let file = fs::File::create(&path).unwrap();
    file.set_len(MAX_IMAGE_BYTES + 1).unwrap();
    assert!(
        read_image(&path)
            .unwrap_err()
            .to_string()
            .contains("byte limit")
    );
    #[cfg(unix)]
    {
        let link = directory.path().join("link.png");
        std::os::unix::fs::symlink(fixture("rgb-grid.png"), &link).unwrap();
        assert!(read_image(&link).is_err());
    }
}

#[test]
fn oversized_video_frame_is_refused_before_running_a_decoder() {
    use game_screen_pick::media::{FrameTime, MediaTools, Video};
    let video = Video {
        source: "unused.mkv".into(),
        stream_index: 0,
        start_seconds: 0.0,
        duration_seconds: 1.0,
        width: 8192,
        height: 8192,
        frames: vec![FrameTime {
            decode_index: 0,
            seconds: 0.0,
            duration_seconds: 1.0,
        }],
    };
    let tools = MediaTools {
        ffmpeg: "missing-decoder".into(),
        ..MediaTools::default()
    };
    assert!(
        tools
            .extract(&video, 0.0, &std::sync::atomic::AtomicBool::new(false))
            .unwrap_err()
            .to_string()
            .contains("pixel limit")
    );
}
