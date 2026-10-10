//! Input facts only. Product extraction, selection and cache tests belong to their issues.
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const WIDTH: usize = 160;
const HEIGHT: usize = 96;
const FPS: usize = 4;
const SECONDS: usize = 6;
const VIDEOS: [(&str, bool); 2] = [("01-blocks.mkv", false), ("02-mirrored.mkv", true)];
const MULTITRACK_VIDEO: &str = "03-multitrack-rotated.mov";
#[cfg(test)]
const IMAGES: [&str; 4] = [
    "near-black.png",
    "near-white.png",
    "low-contrast.png",
    "rgb-grid.png",
];

fn root() -> PathBuf {
    std::env::var_os("GSP_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("tests/fixtures"))
}

fn rows(file: &str) -> Vec<Vec<String>> {
    fs::read_to_string(root().join(file))
        .expect("input facts")
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| line.split_whitespace().map(String::from).collect())
        .collect()
}

fn video_frames(mirrored: bool) -> Vec<u8> {
    let tiles = rows("video-tiles.tsv");
    assert_eq!(tiles.len(), SECONDS * 6);
    let mut frames = Vec::new();
    for second in 0..SECONDS {
        let mut frame = Vec::with_capacity(WIDTH * HEIGHT * 3);
        for y in 0..HEIGHT {
            let row = &tiles[second * 6 + y / 16];
            assert_eq!(row.len(), 12);
            assert_eq!(row[0].parse::<usize>().unwrap(), second);
            assert_eq!(row[1].parse::<usize>().unwrap(), y / 16);
            for x in 0..WIDTH {
                let tile_x = if mirrored { 9 - x / 16 } else { x / 16 };
                frame.push(row[2 + tile_x].parse::<u8>().unwrap());
            }
        }
        frame.extend(vec![128; WIDTH * HEIGHT * 2]);
        for _ in 0..FPS {
            frames.extend_from_slice(&frame);
        }
    }
    frames
}

#[cfg(test)]
fn video_rgb_frames(mirrored: bool, rotated: bool) -> Vec<u8> {
    let mut pixels = Vec::new();
    let (width, height) = if rotated {
        (HEIGHT, WIDTH)
    } else {
        (WIDTH, HEIGHT)
    };
    for frame in video_frames(mirrored).chunks_exact(WIDTH * HEIGHT * 3) {
        for y in 0..height {
            for x in 0..width {
                // Display rotation +90 is counter-clockwise; neutral chroma makes RGB = luma.
                let source = if rotated {
                    x * WIDTH + WIDTH - 1 - y
                } else {
                    y * WIDTH + x
                };
                pixels.extend([frame[source]; 3]);
            }
        }
    }
    pixels
}

fn image_pixels(row: &[String]) -> Vec<u8> {
    assert_eq!(row.len(), 6);
    let width = row[1].parse::<usize>().unwrap();
    let height = row[2].parse::<usize>().unwrap();
    let mut pixels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            match row[3].as_str() {
                "half" => {
                    let value = row[if x < width / 2 { 4 } else { 5 }]
                        .parse::<u8>()
                        .unwrap();
                    pixels.extend([value; 3]);
                }
                "rgb-grid" => pixels.extend([
                    ((37 * x + 17 * y) % 256) as u8,
                    ((13 * x + 71 * y) % 256) as u8,
                    ((97 * x + 29 * y) % 256) as u8,
                ]),
                recipe => panic!("unknown image recipe: {recipe}"),
            }
        }
    }
    pixels
}

#[cfg(test)]
fn assert_inventory(directory: &str, expected: &[&str]) {
    use std::collections::BTreeSet;
    let actual: BTreeSet<String> = fs::read_dir(root().join(directory))
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            assert!(
                entry.file_type().unwrap().is_file(),
                "regular fixture file required"
            );
            entry.file_name().into_string().unwrap()
        })
        .collect();
    assert_eq!(
        actual,
        expected.iter().map(|name| String::from(*name)).collect()
    );
}

#[cfg(test)]
fn output(tool: &str, args: &[&str], path: &Path, tail: &[&str]) -> Vec<u8> {
    let result = Command::new(tool)
        .args(args)
        .arg(path)
        .args(tail)
        .output()
        .expect("FFmpeg/FFprobe must be installed");
    assert!(
        result.status.success(),
        "{tool}: {}: {}",
        path.display(),
        String::from_utf8_lossy(&result.stderr)
    );
    result.stdout
}

#[cfg(test)]
fn assert_timestamps(path: &Path, stream: &str) {
    let timestamps = output(
        "ffprobe",
        &[
            "-v",
            "error",
            "-select_streams",
            stream,
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ],
        path,
        &[],
    );
    let timestamps: Vec<f64> = String::from_utf8(timestamps)
        .unwrap()
        .lines()
        .map(|line| line.parse().unwrap())
        .collect();
    assert_eq!(timestamps.len(), FPS * SECONDS);
    for (index, timestamp) in timestamps.iter().enumerate() {
        assert_eq!(*timestamp, index as f64 / FPS as f64);
    }
}

#[test]
fn images_match_dimensions_channels_and_orientation() {
    let facts = rows("image-facts.tsv");
    assert_eq!(facts.len(), IMAGES.len(), "image facts inventory");
    let mut names: Vec<&str> = facts.iter().map(|row| row[0].as_str()).collect();
    names.sort_unstable();
    let mut expected = IMAGES;
    expected.sort_unstable();
    assert_eq!(names, expected);
    assert_inventory("images", &IMAGES);
    for row in facts {
        let path = root().join("images").join(&row[0]);
        let dimensions = output(
            "ffprobe",
            &[
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
            ],
            &path,
            &[],
        );
        assert_eq!(
            String::from_utf8(dimensions).unwrap().trim(),
            format!("{}x{}", row[1], row[2])
        );
        let pixels = output(
            "ffmpeg",
            &["-nostdin", "-v", "error", "-i"],
            &path,
            &[
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
        );
        assert!(
            pixels == image_pixels(&row),
            "pixel facts: {}",
            path.display()
        );
    }
}

#[test]
fn videos_match_every_frame_pts_content_and_orientation() {
    assert_inventory("videos", &[VIDEOS[0].0, VIDEOS[1].0, MULTITRACK_VIDEO]);
    for (file, mirrored) in VIDEOS {
        let path = root().join("videos").join(file);
        let metadata = output(
            "ffprobe",
            &[
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,pix_fmt,color_range,r_frame_rate,start_time:format=duration",
                "-of",
                "default=noprint_wrappers=1",
            ],
            &path,
            &[],
        );
        let metadata = String::from_utf8(metadata).unwrap();
        for field in [
            "codec_name=ffv1",
            "width=160",
            "height=96",
            "pix_fmt=yuv444p",
            "color_range=pc",
            "r_frame_rate=4/1",
            "start_time=0.000000",
            "duration=6.000000",
        ] {
            assert!(
                metadata.lines().any(|line| line == field),
                "{file}: missing {field}"
            );
        }
        assert_timestamps(&path, "v:0");
        let frames = output(
            "ffmpeg",
            &["-nostdin", "-v", "error", "-i"],
            &path,
            &[
                "-map", "0:v:0", "-an", "-f", "rawvideo", "-pix_fmt", "yuv444p", "pipe:1",
            ],
        );
        assert!(frames == video_frames(mirrored), "frame facts: {file}");
    }
}

#[test]
fn multiple_tracks_preserve_raw_content_and_selected_display_rotation() {
    let path = root().join("videos").join(MULTITRACK_VIDEO);
    let metadata = output(
        "ffprobe",
        &[
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=index,codec_name,width,height,pix_fmt,color_range,r_frame_rate,start_time:stream_disposition=default,attached_pic:stream_side_data=rotation:format=duration",
            "-of",
            "default",
        ],
        &path,
        &[],
    );
    let metadata = String::from_utf8(metadata).unwrap();
    assert!(metadata.lines().any(|line| line == "duration=6.000000"));
    let streams: Vec<&str> = metadata
        .split("[STREAM]\n")
        .skip(1)
        .map(|stream| stream.split("[/STREAM]").next().unwrap())
        .collect();
    assert_eq!(streams.len(), 2, "two ordinary video tracks");
    for (index, stream) in streams.iter().enumerate() {
        for field in [
            format!("index={index}"),
            "codec_name=png".into(),
            "width=160".into(),
            "height=96".into(),
            "pix_fmt=rgb24".into(),
            "color_range=pc".into(),
            "r_frame_rate=4/1".into(),
            "start_time=0.000000".into(),
            "DISPOSITION:attached_pic=0".into(),
            format!("DISPOSITION:default={index}"),
        ] {
            assert!(stream.lines().any(|line| line == field), "missing {field}");
        }
        let rotations: Vec<&str> = stream
            .lines()
            .filter(|line| line.starts_with("rotation="))
            .collect();
        assert_eq!(
            rotations,
            if index == 0 {
                vec!["rotation=90"]
            } else {
                vec![]
            }
        );
        assert_timestamps(&path, &index.to_string());
        let frames = output(
            "ffmpeg",
            &["-nostdin", "-v", "error", "-noautorotate", "-i"],
            &path,
            &[
                "-map",
                &format!("0:{index}"),
                "-an",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
        );
        assert!(
            frames == video_rgb_frames(index == 1, false),
            "raw track {index}"
        );
    }
    // Select index 0 even though track 1 is marked default. Its display size is 96x160.
    let png = output(
        "ffmpeg",
        &["-nostdin", "-v", "error", "-i"],
        &path,
        &[
            "-map",
            "0:0",
            "-frames:v",
            "1",
            "-c:v",
            "png",
            "-f",
            "image2pipe",
            "pipe:1",
        ],
    );
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
    assert_eq!(
        u32::from_be_bytes(png[16..20].try_into().unwrap()),
        HEIGHT as u32
    );
    assert_eq!(
        u32::from_be_bytes(png[20..24].try_into().unwrap()),
        WIDTH as u32
    );
    let frames = output(
        "ffmpeg",
        &["-nostdin", "-v", "error", "-i"],
        &path,
        &[
            "-map", "0:0", "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ],
    );
    assert!(
        frames == video_rgb_frames(false, true),
        "selected track display orientation"
    );
}

#[test]
fn known_times_distinguish_sources_and_black_interval() {
    let first = video_frames(false);
    let second = video_frames(true);
    let frame_size = WIDTH * HEIGHT * 3;
    let frame = |bytes: &[u8], index| bytes[index * frame_size..(index + 1) * frame_size].to_vec();
    // Same PTS, different input; adjacent scene, different PTS. Black is not a useful candidate.
    assert_ne!(frame(&first, 2), frame(&second, 2));
    assert_ne!(frame(&first, 2), frame(&first, 6));
    assert!(
        frame(&first, 10)[..WIDTH * HEIGHT]
            .iter()
            .all(|value| *value == 0)
    );
}

#[test]
fn fixed_responses_match_the_fixture_contract() {
    let result = Command::new("jq")
        .args(["-se", "--slurpfile", "scenarios"])
        .arg(root().join("scenarios.json"))
        .args(["-f", "tests/fixture_contract.jq"])
        .arg(root().join("responses.json"))
        .output()
        .expect("jq must be installed");
    assert!(result.status.success(), "fixed response contract failed");
}

#[cfg(not(test))]
fn encode(path: &Path, input_args: &[&str], output_args: &[&str], pixels: &[u8]) {
    use std::io::Write;
    use std::process::Stdio;
    let mut child = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-n", "-f", "rawvideo"])
        .args(input_args)
        .args(["-i", "pipe:0"])
        .args(output_args)
        .arg(path)
        .stdin(Stdio::piped())
        .spawn()
        .expect("FFmpeg must be installed");
    child.stdin.take().unwrap().write_all(pixels).unwrap();
    assert!(child.wait().unwrap().success(), "encode {}", path.display());
}

#[cfg(not(test))]
fn main() {
    let mut args = std::env::args_os().skip(1);
    let destination = PathBuf::from(args.next().expect("usage: generate-fixtures NEW_DIRECTORY"));
    assert!(args.next().is_none(), "only one destination allowed");
    fs::create_dir(&destination).expect("destination must not exist");
    fs::create_dir(destination.join("images")).unwrap();
    fs::create_dir(destination.join("videos")).unwrap();
    for file in [
        "image-facts.tsv",
        "video-tiles.tsv",
        "responses.json",
        "scenarios.json",
    ] {
        fs::copy(root().join(file), destination.join(file)).unwrap();
    }
    for row in rows("image-facts.tsv") {
        let size = format!("{}x{}", row[1], row[2]);
        encode(
            &destination.join("images").join(&row[0]),
            &["-pixel_format", "rgb24", "-video_size", &size],
            &["-frames:v", "1", "-c:v", "png", "-threads", "1"],
            &image_pixels(&row),
        );
    }
    for (file, mirrored) in VIDEOS {
        encode(
            &destination.join("videos").join(file),
            &[
                "-pixel_format",
                "yuv444p",
                "-video_size",
                "160x96",
                "-framerate",
                "4",
                "-color_range",
                "pc",
            ],
            &[
                "-an",
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-pix_fmt",
                "yuv444p",
                "-color_range",
                "pc",
                "-threads",
                "1",
                "-fflags",
                "+bitexact",
                "-flags:v",
                "+bitexact",
                "-map_metadata",
                "-1",
            ],
            &video_frames(mirrored),
        );
    }
    let result = Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-display_rotation:v:0",
            "90",
            "-noautorotate",
            "-i",
        ])
        .arg(destination.join("videos").join(VIDEOS[0].0))
        .arg("-i")
        .arg(destination.join("videos").join(VIDEOS[1].0))
        .args([
            "-map",
            "0:v:0",
            "-map",
            "1:v:0",
            "-c:v",
            "png",
            "-pix_fmt",
            "rgb24",
            "-threads",
            "1",
            "-map_metadata",
            "-1",
            "-disposition:v:0",
            "0",
            "-disposition:v:1",
            "default",
        ])
        .arg(destination.join("videos").join(MULTITRACK_VIDEO))
        .status()
        .expect("FFmpeg must support display_rotation");
    assert!(result.success(), "encode {MULTITRACK_VIDEO}");
}
