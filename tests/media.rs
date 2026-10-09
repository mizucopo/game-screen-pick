use game_screen_pick::media::{Interrupted, MediaTools, discover_inputs};
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/videos")
        .join(name)
}

fn pixels(path: &Path) -> (u32, u32, Vec<u8>) {
    let mut decoder = png::Decoder::new(BufReader::new(File::open(path).unwrap()));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().unwrap();
    let mut buffer = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buffer).unwrap();
    assert_eq!(info.color_type, png::ColorType::Rgb);
    buffer.truncate(info.buffer_size());
    (info.width, info.height, buffer)
}

fn expected(second: usize, mirrored: bool, rotated: bool) -> Vec<u8> {
    let rows: Vec<Vec<u8>> = include_str!("fixtures/video-tiles.tsv")
        .lines()
        .filter(|line| !line.starts_with('#') && !line.starts_with("second"))
        .map(|line| {
            line.split('\t')
                .skip(2)
                .map(|number| number.parse().unwrap())
                .collect()
        })
        .collect();
    let (width, height) = if rotated { (96, 160) } else { (160, 96) };
    let mut output = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let (x, y) = if rotated { (159 - y, x) } else { (x, y) };
            let tile_x = if mirrored { 9 - x / 16 } else { x / 16 };
            let luma = rows[second * 6 + y / 16][tile_x];
            output.extend_from_slice(&[luma; 3]);
        }
    }
    output
}

#[test]
fn corpus_extraction_matches_every_source_pts_pixel_and_display_direction() {
    let tools = MediaTools::default();
    let cancelled = AtomicBool::new(false);
    let scenarios: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/scenarios.json")).unwrap();
    assert_eq!(
        scenarios["extraction_cases"][0]["input"],
        "03-multitrack-rotated.mov"
    );
    for (name, mirrored, rotated) in [
        ("01-blocks.mkv", false, false),
        ("02-mirrored.mkv", true, false),
        ("03-multitrack-rotated.mov", false, true),
    ] {
        let video = tools.probe(&fixture(name), &cancelled).unwrap();
        assert_eq!(video.stream_index, 0);
        assert_eq!(
            (video.width, video.height),
            if rotated { (96, 160) } else { (160, 96) }
        );
        assert_eq!(video.start_seconds, 0.0);
        assert_eq!(video.duration_seconds, 6.0);
        assert_eq!(video.frames.len(), 24);
        for index in 0..24 {
            let seconds = f64::from(index) / 4.0;
            assert_eq!(video.frames[index as usize].seconds, seconds);
            let image = tools.extract(&video, seconds, &cancelled).unwrap();
            assert_eq!(image.source, fixture(name).canonicalize().unwrap());
            assert_eq!(image.requested_seconds, seconds);
            assert_eq!(image.actual_seconds, seconds);
            let (width, height, data) = pixels(&image.path());
            assert_eq!((width, height), (video.width, video.height));
            assert_eq!(
                data,
                expected((index / 4) as usize, mirrored, rotated),
                "source {name}, PTS {seconds}"
            );
            let image_path = image.path();
            drop(image);
            assert!(
                !image_path.exists(),
                "owned intermediate must disappear on drop"
            );
        }
    }
}

#[test]
fn boundaries_context_and_same_stream_semantic_chunk() {
    let tools = MediaTools::default();
    let cancelled = AtomicBool::new(false);
    let video = tools
        .probe(&fixture("03-multitrack-rotated.mov"), &cancelled)
        .unwrap();
    for (requested, actual) in [(0.0, 0.0), (0.01, 0.25), (5.9, 5.75), (6.0, 5.75)] {
        let image = tools.extract(&video, requested, &cancelled).unwrap();
        assert_eq!(image.requested_seconds, requested);
        assert_eq!(image.actual_seconds, actual);
        assert_eq!(
            pixels(&image.path()).2,
            expected(actual as usize, false, true)
        );
    }
    for time in [-1.0, 6.01, f64::NAN, f64::INFINITY] {
        assert!(tools.extract(&video, time, &cancelled).is_err());
    }
    let context = tools
        .extract_context(&video, 0.01, 1.0, &cancelled)
        .unwrap();
    assert_eq!(
        context.each_ref().map(|image| image.requested_seconds),
        [0.0, 0.01, 1.01]
    );
    assert_eq!(
        context.each_ref().map(|image| image.actual_seconds),
        [0.0, 0.25, 1.25]
    );
    for image in &context {
        assert_eq!(
            pixels(&image.path()).2,
            expected(image.actual_seconds as usize, false, true)
        );
    }
    let chunk = tools.extract_chunk(&video, 0.9, 2.0, &cancelled).unwrap();
    assert_eq!(chunk.actual_start_seconds, 1.0);
    assert_eq!(chunk.actual_end_seconds, 2.0);
    let decoded = tools.probe(&chunk.path(), &cancelled).unwrap();
    assert_eq!((decoded.width, decoded.height), (96, 160));
    assert_eq!(
        decoded
            .frames
            .iter()
            .map(|frame| frame.seconds)
            .collect::<Vec<_>>(),
        [0.0, 0.25, 0.5, 0.75]
    );
    for frame in &decoded.frames {
        let image = tools.extract(&decoded, frame.seconds, &cancelled).unwrap();
        assert_eq!(pixels(&image.path()).2, expected(1, false, true));
    }
    let chunk_path = chunk.path();
    drop(chunk);
    assert!(!chunk_path.exists());
    assert!(tools.extract_chunk(&video, 1.1, 1.2, &cancelled).is_err());
}

#[test]
fn discovery_is_stable_nonrecursive_and_handles_japanese_space_paths() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("日本語 録画");
    fs::create_dir(&path).unwrap();
    for name in ["z.MOV", "あ mkv.MKV", "a.mp4", "ignored.txt"] {
        fs::write(path.join(name), b"input").unwrap();
    }
    fs::create_dir(path.join("nested")).unwrap();
    fs::write(path.join("nested/inside.mkv"), b"input").unwrap();
    let files = discover_inputs(&path).unwrap();
    assert_eq!(
        files
            .iter()
            .map(|file| file.file_name().unwrap().to_str().unwrap())
            .collect::<Vec<_>>(),
        ["a.mp4", "z.MOV", "あ mkv.MKV"]
    );
    let recording = path.join("記録 空白.webm");
    fs::copy(fixture("01-blocks.mkv"), &recording).unwrap();
    let cancelled = AtomicBool::new(false);
    let tools = MediaTools::default();
    let video = tools.probe(&recording, &cancelled).unwrap();
    let image = tools.extract(&video, 4.0, &cancelled).unwrap();
    assert_eq!(pixels(&image.path()).2, expected(4, false, false));
    let empty = tempfile::tempdir().unwrap();
    assert!(discover_inputs(empty.path()).is_err());
    assert!(discover_inputs(&path.join("missing")).is_err());
    assert!(discover_inputs(&recording).is_err());
}

#[cfg(unix)]
#[test]
fn discovery_rejects_symlinks_special_video_files_and_non_utf8_names() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    use std::os::unix::fs::symlink;
    let directory = tempfile::tempdir().unwrap();
    symlink(fixture("01-blocks.mkv"), directory.path().join("link.mkv")).unwrap();
    assert!(
        discover_inputs(directory.path())
            .unwrap_err()
            .to_string()
            .contains("ordinary file")
    );
    assert!(
        MediaTools::default()
            .probe(&directory.path().join("link.mkv"), &AtomicBool::new(false))
            .is_err()
    );
    let alias = directory.path().join("directory-alias");
    symlink(directory.path(), &alias).unwrap();
    assert!(discover_inputs(&alias).is_err());
    fs::remove_file(directory.path().join("link.mkv")).unwrap();
    fs::create_dir(directory.path().join("directory.mkv")).unwrap();
    assert!(discover_inputs(directory.path()).is_err());
    fs::remove_dir(directory.path().join("directory.mkv")).unwrap();
    let invalid = directory
        .path()
        .join(OsString::from_vec(b"invalid-\xff.mkv".to_vec()));
    // APFS rejects creation of non-UTF-8 filenames itself. Linux permits them,
    // so it additionally exercises the directory-entry validation contract.
    #[cfg(target_os = "linux")]
    {
        fs::write(&invalid, b"input").unwrap();
        assert!(
            discover_inputs(directory.path())
                .unwrap_err()
                .to_string()
                .contains("non-UTF-8")
        );
    }
    assert!(discover_inputs(&invalid).is_err());
    assert!(
        MediaTools::default()
            .probe(&invalid, &AtomicBool::new(false))
            .is_err()
    );
}

fn checked_command(command: &mut Command) {
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn nonzero_offset_and_vfr_use_actual_decoded_pts_at_endpoints() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("offset VFR.mkv");
    // Known source frames 0, 4, 12, 20 get PTS 3, 3.25, 4.5, 5.75.
    // The container's start time is 3, while our time zero is first valid PTS.
    checked_command(Command::new("ffmpeg").args(["-v", "error", "-nostdin", "-i"]).arg(fixture("01-blocks.mkv"))
        .args(["-vf", "select='eq(n,0)+eq(n,4)+eq(n,12)+eq(n,20)',setpts='if(eq(N,0),3,if(eq(N,1),3.25,if(eq(N,2),4.5,5.75)))/TB'", "-fps_mode", "passthrough", "-c:v", "ffv1"]).arg(&source));
    let tools = MediaTools::default();
    let cancelled = AtomicBool::new(false);
    let video = tools.probe(&source, &cancelled).unwrap();
    assert_eq!(video.start_seconds, 3.0);
    assert_eq!(video.duration_seconds, 3.0);
    assert_eq!(
        video
            .frames
            .iter()
            .map(|frame| frame.seconds)
            .collect::<Vec<_>>(),
        [0.0, 0.25, 1.5, 2.75]
    );
    for (requested, actual, second) in [
        (0.0, 0.0, 0),
        (0.01, 0.25, 1),
        (0.3, 1.5, 3),
        (2.75, 2.75, 5),
        (3.0, 2.75, 5),
    ] {
        let image = tools.extract(&video, requested, &cancelled).unwrap();
        assert_eq!(image.actual_seconds, actual);
        assert_eq!(pixels(&image.path()).2, expected(second, false, false));
    }
    for (start, end, indices) in [
        (0.25, 1.0, vec![1]),
        (0.25, 2.0, vec![1, 2]),
        (0.0, 0.2, vec![0]),
        (0.25, 3.0, vec![1, 2, 3]),
    ] {
        let chunk = tools.extract_chunk(&video, start, end, &cancelled).unwrap();
        let chunk_video = tools.probe(&chunk.path(), &cancelled).unwrap();
        assert_eq!(
            chunk_video.duration_seconds,
            chunk.actual_end_seconds - chunk.actual_start_seconds
        );
        assert_eq!(chunk_video.frames.len(), indices.len());
        let output = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_frames",
                "-show_entries",
                "frame=pts_time,duration_time,pkt_duration_time",
                "-of",
                "json",
            ])
            .arg(chunk.path())
            .output()
            .unwrap();
        assert!(output.status.success());
        let raw: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        for (ordinal, index) in indices.into_iter().enumerate() {
            let original = &video.frames[index];
            let decoded = &chunk_video.frames[ordinal];
            assert_eq!(
                decoded.seconds,
                original.seconds - chunk.actual_start_seconds
            );
            assert_eq!(decoded.duration_seconds, original.duration_seconds);
            let raw_duration = raw["frames"][ordinal]["duration_time"]
                .as_str()
                .or_else(|| raw["frames"][ordinal]["pkt_duration_time"].as_str())
                .unwrap()
                .parse::<f64>()
                .unwrap();
            assert_eq!(raw_duration, original.duration_seconds);
            let image = tools
                .extract(&chunk_video, decoded.seconds, &cancelled)
                .unwrap();
            assert_eq!(
                pixels(&image.path()).2,
                expected([0, 1, 3, 5][index], false, false)
            );
        }
    }
}

#[test]
fn chunk_preserves_submillisecond_vfr_pts_and_each_original_display_interval() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("microsecond VFR.mov");
    checked_command(Command::new("ffmpeg").args(["-v", "error", "-nostdin", "-i"]).arg(fixture("01-blocks.mkv"))
        .args(["-vf", "select='eq(n,0)+eq(n,4)+eq(n,12)+eq(n,20)',settb=1/1000000,setpts='if(eq(N,0),3000000,if(eq(N,1),3010001,if(eq(N,2),3020005,3030011)))'", "-fps_mode", "passthrough", "-c:v", "png", "-pix_fmt", "rgb24", "-enc_time_base", "1:1000000", "-video_track_timescale", "1000000", "-bsf:v", "setts=duration=if(eq(N\\,3)\\,7013\\,NEXT_PTS-PTS)"])
        .arg(&source));
    let tools = MediaTools::default();
    let cancelled = AtomicBool::new(false);
    let video = tools.probe(&source, &cancelled).unwrap();
    assert_eq!(video.start_seconds, 3.0);
    let expected_pts = [0.0, 0.010001, 0.020005, 0.030011];
    for (frame, expected) in video.frames.iter().zip(expected_pts) {
        assert!((frame.seconds - expected).abs() < 0.000000001);
    }
    for (index, seconds) in expected_pts.into_iter().enumerate() {
        let image = tools.extract(&video, seconds, &cancelled).unwrap();
        assert!(
            (image.actual_seconds - seconds).abs() < 0.000000001,
            "literal PTS {seconds} extracted {}",
            image.actual_seconds
        );
        assert_eq!(
            pixels(&image.path()).2,
            expected([0, 1, 3, 5][index], false, false)
        );
    }
    let chunk = tools
        .extract_chunk(&video, video.frames[1].seconds, 0.03, &cancelled)
        .unwrap();
    let chunk_video = tools.probe(&chunk.path(), &cancelled).unwrap();
    assert_eq!(chunk_video.frames.len(), 2);
    assert!(
        (chunk_video.duration_seconds - (chunk.actual_end_seconds - chunk.actual_start_seconds))
            .abs()
            < 0.000000001
    );
    for (ordinal, source_index) in [1, 2].into_iter().enumerate() {
        let original = &video.frames[source_index];
        let decoded = &chunk_video.frames[ordinal];
        assert!(
            (decoded.seconds - (original.seconds - chunk.actual_start_seconds)).abs() < 0.000000001
        );
        assert!((decoded.duration_seconds - original.duration_seconds).abs() < 0.000000001);
        let image = tools
            .extract(&chunk_video, decoded.seconds, &cancelled)
            .unwrap();
        assert_eq!(
            pixels(&image.path()).2,
            expected([0, 1, 3, 5][source_index], false, false)
        );
    }
    for (index, frame) in video.frames.iter().enumerate() {
        let end = if index == 0 {
            0.005
        } else {
            frame.seconds + frame.duration_seconds / 2.0
        };
        let chunk = tools
            .extract_chunk(&video, frame.seconds, end, &cancelled)
            .unwrap();
        let decoded = tools.probe(&chunk.path(), &cancelled).unwrap();
        assert_eq!(decoded.frames.len(), 1);
        assert!(
            (decoded.duration_seconds - frame.duration_seconds).abs() < 0.000000001,
            "frame {index}: actual {}, source {}",
            decoded.duration_seconds,
            frame.duration_seconds
        );
        assert!(
            (decoded.duration_seconds - (chunk.actual_end_seconds - chunk.actual_start_seconds))
                .abs()
                < 0.000000001
        );
        let image = tools.extract(&decoded, 0.0, &cancelled).unwrap();
        assert_eq!(
            pixels(&image.path()).2,
            expected([0, 1, 3, 5][index], false, false)
        );
    }
}

#[cfg(unix)]
fn script(directory: &Path, name: &str, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let path = directory.join(name);
    fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
    path
}

#[cfg(unix)]
fn metadata_script(directory: &Path, streams: &str, frames: &str) -> PathBuf {
    script(
        directory,
        "probe",
        &format!(
            "case \"$*\" in\n*show_streams*) cat <<'STREAMS'\n{streams}\nSTREAMS\n;;\n*) cat <<'FRAMES'\n{frames}\nFRAMES\n;;\nesac"
        ),
    )
}

#[cfg(unix)]
#[test]
fn attached_picture_is_excluded_and_invalid_metadata_is_rejected() {
    let directory = tempfile::tempdir().unwrap();
    let stream = r#"{"streams":[{"index":0,"codec_type":"video","width":640,"height":480,"disposition":{"attached_pic":1}},{"index":3,"codec_type":"video","width":160,"height":96,"start_time":"7","duration":"1","disposition":{"attached_pic":0}}]}"#;
    let frames = r#"{"frames":[{"pts_time":"N/A"},{"pts_time":"7","duration_time":"0.25"},{"best_effort_timestamp_time":"7.25"},{"pts_time":"7.5","duration_time":"0.5"}]}"#;
    let mut tools = MediaTools {
        ffprobe: metadata_script(directory.path(), stream, frames),
        ..MediaTools::default()
    };
    let cancelled = AtomicBool::new(false);
    let video = tools.probe(&fixture("01-blocks.mkv"), &cancelled).unwrap();
    assert_eq!(video.stream_index, 3);
    assert_eq!(video.start_seconds, 7.0);
    assert_eq!(video.frames[0].decode_index, 1);
    assert_eq!(video.frames[1].decode_index, 3);
    assert_eq!(video.frames.len(), 2);
    assert!(
        tools
            .extract_chunk(&video, 0.0, 1.0, &cancelled)
            .unwrap_err()
            .to_string()
            .contains("without valid PTS")
    );
    let late_frames = r#"{"frames":[{"pts_time":"7.25"},{"pts_time":"7.75"}]}"#;
    tools.ffprobe = metadata_script(directory.path(), stream, late_frames);
    let late_video = tools.probe(&fixture("01-blocks.mkv"), &cancelled).unwrap();
    assert_eq!(late_video.start_seconds, 7.25);
    assert_eq!(late_video.duration_seconds, 0.75);
    for invalid_frames in [
        "not JSON",
        r#"{"frames":[]}"#,
        r#"{"frames":[{"pts_time":"N/A","best_effort_timestamp_time":"7"}]}"#,
        r#"{"frames":[{"pts_time":"NaN"}]}"#,
        r#"{"frames":[{"pts_time":"7"},{"pts_time":"6"}]}"#,
        r#"{"frames":[{"pts_time":"7"},{"pts_time":"7"}]}"#,
        r#"{"frames":[{"pts_time":"7","duration_time":"-1"}]}"#,
    ] {
        tools.ffprobe = metadata_script(directory.path(), stream, invalid_frames);
        assert!(
            tools.probe(&fixture("01-blocks.mkv"), &cancelled).is_err(),
            "{invalid_frames}"
        );
    }
    for invalid_stream in [
        r#"{"streams":[]}"#,
        r#"{"streams":[{"index":0,"codec_type":"video","width":0,"height":96,"disposition":{"attached_pic":0}}]}"#,
        r#"{"streams":[{"index":0,"codec_type":"video","width":160,"height":96,"duration":"NaN","disposition":{"attached_pic":0}}]}"#,
    ] {
        tools.ffprobe = metadata_script(directory.path(), invalid_stream, frames);
        assert!(tools.probe(&fixture("01-blocks.mkv"), &cancelled).is_err());
    }
}

#[cfg(unix)]
#[test]
fn missing_commands_failure_and_cancellation_clean_owned_files_and_children() {
    let directory = tempfile::tempdir().unwrap();
    let cancelled = AtomicBool::new(false);
    let mut tools = MediaTools {
        ffprobe: directory.path().join("missing ffprobe"),
        ..MediaTools::default()
    };
    assert!(
        tools
            .probe(&fixture("01-blocks.mkv"), &cancelled)
            .unwrap_err()
            .to_string()
            .contains("cannot start")
    );
    tools = MediaTools::default();
    let video = tools.probe(&fixture("01-blocks.mkv"), &cancelled).unwrap();
    let original_source = fs::read(&video.source).unwrap();
    tools.ffmpeg = directory.path().join("missing ffmpeg");
    assert!(tools.extract(&video, 1.0, &cancelled).is_err());
    let recorded_path = directory.path().join("partial-path");
    let pid_path = directory.path().join("child-pid");
    let failing = format!(
        "for path do :; done\nprintf '%s' \"$path\" > '{}'\nprintf partial > \"$path\"\nexit 9",
        recorded_path.display()
    );
    tools.ffmpeg = script(directory.path(), "failing ffmpeg", &failing);
    assert!(
        tools
            .extract(&video, 1.0, &cancelled)
            .unwrap_err()
            .to_string()
            .contains("failed")
    );
    let partial = PathBuf::from(fs::read_to_string(&recorded_path).unwrap());
    assert!(!partial.exists());
    assert!(!partial.parent().unwrap().exists());
    tools.ffmpeg = script(
        directory.path(),
        "invalid PNG ffmpeg",
        &failing.replace("exit 9", "exit 0"),
    );
    assert!(
        tools
            .extract(&video, 1.0, &cancelled)
            .unwrap_err()
            .to_string()
            .contains("truncated PNG")
    );
    let partial = PathBuf::from(fs::read_to_string(&recorded_path).unwrap());
    assert!(!partial.parent().unwrap().exists());
    let hanging = format!(
        "for path do :; done\nprintf '%s' \"$path\" > '{}'\nprintf partial > \"$path\"\nsleep 30 &\nprintf '%s' \"$!\" > '{}'\nwait",
        recorded_path.display(),
        pid_path.display()
    );
    tools.ffmpeg = script(directory.path(), "hanging ffmpeg", &hanging);
    let before = Instant::now();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let deadline = Instant::now() + Duration::from_secs(5);
            while !pid_path.exists() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            cancelled.store(true, Ordering::Relaxed);
        });
        let error = tools.extract(&video, 1.0, &cancelled).unwrap_err();
        assert!(error.downcast_ref::<Interrupted>().is_some());
    });
    assert!(before.elapsed() < Duration::from_secs(6));
    let partial = PathBuf::from(fs::read_to_string(&recorded_path).unwrap());
    assert!(!partial.parent().unwrap().exists());
    let pid: libc::pid_t = fs::read_to_string(&pid_path).unwrap().parse().unwrap();
    let mut running = true;
    for _ in 0..100 {
        // Some Unix init processes reap descendants asynchronously.
        running = unsafe { libc::kill(pid, 0) } == 0;
        if !running {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(!running, "owned child must not remain running");
    assert_eq!(fs::read(&video.source).unwrap(), original_source);
}

#[cfg(unix)]
#[test]
fn probe_output_is_bounded_and_child_diagnostics_are_secret_free() {
    let directory = tempfile::tempdir().unwrap();
    let cancelled = AtomicBool::new(false);
    let mut tools = MediaTools {
        ffprobe: script(directory.path(), "large probe", "head -c 1048577 /dev/zero"),
        ..MediaTools::default()
    };
    assert!(
        format!(
            "{:#}",
            tools
                .probe(&fixture("01-blocks.mkv"), &cancelled)
                .unwrap_err()
        )
        .contains("byte limit")
    );
    tools.ffprobe = script(
        directory.path(),
        "failing probe",
        "printf 'GAME_SCREEN_PICK_API_KEY=secret-example-value' >&2\nexit 1",
    );
    let message = tools
        .probe(&fixture("01-blocks.mkv"), &cancelled)
        .unwrap_err()
        .to_string();
    assert!(message.contains("failed"));
    assert!(!message.contains("secret-example-value"));
    let invalid = directory.path().join("invalid.mkv");
    fs::write(&invalid, b"this is not a video").unwrap();
    assert!(MediaTools::default().probe(&invalid, &cancelled).is_err());
}

#[cfg(unix)]
#[test]
fn exited_command_cannot_leave_descendants_holding_capture_pipes() {
    let directory = tempfile::tempdir().unwrap();
    let tools = MediaTools {
        ffprobe: script(directory.path(), "exited probe", "sleep 30 &\nexit 4"),
        ..MediaTools::default()
    };
    let start = Instant::now();
    assert!(
        tools
            .probe(&fixture("01-blocks.mkv"), &AtomicBool::new(false))
            .is_err()
    );
    assert!(start.elapsed() < Duration::from_secs(3));
}
