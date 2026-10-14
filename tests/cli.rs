use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    thread,
    time::{Duration, Instant},
};

const BINARY: &str = env!("CARGO_BIN_EXE_game-screen-pick");

fn sandbox() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

fn command(directory: &Path, args: &[&str]) -> Output {
    Command::new(BINARY)
        .current_dir(directory)
        .args(args)
        .output()
        .unwrap()
}

fn setup(directory: &Path) {
    fs::write(
        directory.join("config.toml"),
        include_str!("../config.example.toml"),
    )
    .unwrap();
    fs::create_dir(directory.join("録画 空白")).unwrap();
    fs::copy(fixture(), directory.join("録画 空白/動画.mkv")).unwrap();
}

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/videos/03-multitrack-rotated.mov")
}

fn request() -> Vec<&'static str> {
    vec![
        "--config",
        "config.toml",
        "--count",
        "2",
        "--game-context",
        "探索と会話を選ぶ",
        "録画 空白",
        "output",
    ]
}

#[test]
fn help_and_version_need_no_config_input_or_external_commands() {
    let directory = sandbox();
    for args in [
        vec!["--help"],
        vec!["--version"],
        vec!["extract", "--help"],
        vec!["validate", "--help"],
    ] {
        let output = Command::new(BINARY)
            .current_dir(directory.path())
            .env("PATH", "")
            .args(&args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8(output.stdout).unwrap();
        if args == ["--version"] {
            assert_eq!(
                text.trim(),
                format!("game-screen-pick {}", env!("CARGO_PKG_VERSION"))
            );
        } else {
            assert!(text.contains("Usage:"));
        }
    }
}

#[test]
fn parser_errors_escape_dynamic_controls_and_keep_usage_readable() {
    let directory = sandbox();
    for value in [
        "a\nerror: forged",
        "a\rforged",
        "a\tforged",
        "a\u{009b}2Jforged",
    ] {
        for args in [
            vec!["extract", "--at", value, "input.mkv", "output"],
            vec!["--count", value],
            vec![value],
        ] {
            let output = command(directory.path(), &args);
            assert_eq!(output.status.code(), Some(2));
            let diagnostic = String::from_utf8(output.stderr).unwrap();
            assert!(
                !diagnostic.contains(value),
                "raw user value leaked: {diagnostic:?}"
            );
            assert!(!diagnostic.contains("\nerror: forged"));
            assert!(
                !diagnostic
                    .chars()
                    .any(|character| character.is_control() && character != '\n')
            );
            assert!(diagnostic.starts_with("error: "));
        }
    }
    let output = command(directory.path(), &[]);
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("\nUsage:")
    );
}

#[cfg(unix)]
#[test]
fn invocation_name_cannot_inject_controls_into_help_or_usage() {
    use std::os::unix::process::CommandExt;
    let directory = sandbox();
    for name in [
        "renamed\nerror: forged",
        "renamed\rforged",
        "renamed\u{009b}2Jforged",
    ] {
        for args in [vec!["--help"], vec!["--bad"], vec![]] {
            let output = Command::new(BINARY)
                .current_dir(directory.path())
                .arg0(name)
                .args(args)
                .output()
                .unwrap();
            let text = format!(
                "{}{}",
                String::from_utf8(output.stdout).unwrap(),
                String::from_utf8(output.stderr).unwrap()
            );
            assert!(!text.contains(name), "unsafe invocation name: {text:?}");
            assert!(!text.contains("\nerror: forged"));
            assert!(
                !text
                    .chars()
                    .any(|character| character.is_control() && character != '\n')
            );
            assert!(text.contains("game-screen-pick"));
        }
    }
}

#[test]
fn argument_requirements_and_boundaries_are_enforced() {
    let directory = sandbox();
    setup(directory.path());
    for count in ["1", "999"] {
        let mut args = request();
        args[3] = count;
        args.insert(0, "validate");
        assert!(command(directory.path(), &args).status.success());
    }
    for count in ["0", "1000", "-1", "1.5", "abc"] {
        let mut args = request();
        args[3] = count;
        assert_eq!(command(directory.path(), &args).status.code(), Some(2));
    }
    for args in [
        vec![],
        vec![
            "--config",
            "config.toml",
            "--count",
            "2",
            "録画 空白",
            "output",
        ],
        vec![
            "--config",
            "config.toml",
            "--count",
            "2",
            "--game-title",
            "title",
            "--game-context",
            "context",
            "録画 空白",
            "output",
        ],
        vec![
            "--config",
            "config.toml",
            "--count",
            "2",
            "--game-context",
            "  ",
            "録画 空白",
            "output",
        ],
        vec![
            "--config",
            "config.toml",
            "--game-context",
            "context",
            "録画 空白",
            "output",
        ],
    ] {
        assert_eq!(
            command(directory.path(), &args).status.code(),
            Some(2),
            "{args:?}"
        );
    }
}

#[test]
fn validation_is_read_only_and_selection_requires_a_vision_contract() {
    let directory = sandbox();
    setup(directory.path());
    let mut args = request();
    args.insert(0, "validate");
    let output = Command::new(BINARY)
        .current_dir(directory.path())
        .env("PATH", "")
        .args(&args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!directory.path().join("output").exists());
    for method in ["sampled_frames", "semantic_video"] {
        fs::write(
            directory.path().join("config.toml"),
            include_str!("../config.example.toml").replace("sampled_frames", method),
        )
        .unwrap();
        let output = command(directory.path(), &request());
        assert_eq!(output.status.code(), Some(1));
        assert!(String::from_utf8_lossy(&output.stderr).contains("vision settings"));
        assert!(!directory.path().join("output").exists());
    }
    fs::create_dir(directory.path().join("output")).unwrap();
    fs::write(directory.path().join("output/notes.txt"), b"preserve me").unwrap();
    assert_eq!(command(directory.path(), &args).status.code(), Some(2));
    assert_eq!(
        fs::read(directory.path().join("output/notes.txt")).unwrap(),
        b"preserve me"
    );
}

#[test]
fn normal_command_reports_cache_and_output_lock_contention_as_processing_failure() {
    let directory = sandbox();
    setup(directory.path());
    let root = directory.path().canonicalize().unwrap();
    let cache_parent = root.join(".game-screen-pick-cache");
    fs::create_dir(&cache_parent).unwrap();
    let invoke = || {
        Command::new(BINARY)
            .current_dir(&root)
            .env("PATH", "")
            .args(request())
            .output()
            .unwrap()
    };
    let cache = game_screen_pick::cache::Cache::open(&cache_parent).unwrap();
    let output = invoke();
    assert_eq!(
        output.status.code(),
        Some(1),
        "cache lock contention: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("locked by another run"));
    drop(cache);

    let guard = game_screen_pick::output::Output::prepare(
        &root.join("output"),
        &[],
        &std::sync::atomic::AtomicBool::new(false),
    )
    .unwrap();
    let output = invoke();
    assert_eq!(
        output.status.code(),
        Some(1),
        "output lock contention: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("locked by another run"));
    assert!(!root.join("output").exists());
    #[cfg(target_os = "macos")]
    {
        let mut aliases = request();
        aliases[7] = "OUTPUT";
        let output = Command::new(BINARY)
            .current_dir(&root)
            .env("PATH", "")
            .args(aliases)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert!(String::from_utf8_lossy(&output.stderr).contains("locked by another run"));
        assert!(!root.join("OUTPUT").exists());
    }
    drop(guard);

    fs::create_dir(root.join("output")).unwrap();
    fs::write(root.join("output/notes.txt"), b"preserve me").unwrap();
    let invalid = invoke();
    assert_eq!(invalid.status.code(), Some(2));
    assert_eq!(
        fs::read(root.join("output/notes.txt")).unwrap(),
        b"preserve me"
    );
}

#[test]
fn validate_and_normal_reject_control_names_and_hard_link_aliases_without_writes() {
    for hard_link in [false, true] {
        let directory = sandbox();
        setup(directory.path());
        let inputs = directory.path().join("録画 空白");
        let message = if hard_link {
            fs::hard_link(inputs.join("動画.mkv"), inputs.join("alias.mkv")).unwrap();
            "input paths refer to the same file more than once"
        } else {
            fs::copy(fixture(), inputs.join("control\nname.mkv")).unwrap();
            "video source names must not contain controls"
        };
        for diagnostic in [true, false] {
            let mut args = request();
            if diagnostic {
                args.insert(0, "validate");
            }
            let output = Command::new(BINARY)
                .current_dir(directory.path())
                .env("PATH", "")
                .args(args)
                .output()
                .unwrap();
            assert_eq!(output.status.code(), Some(2));
            assert!(String::from_utf8_lossy(&output.stderr).contains(message));
        }
        assert!(!directory.path().join("output").exists());
        assert!(!directory.path().join(".game-screen-pick-cache").exists());
        assert!(!directory.path().join(".gsp-output-locks").exists());
    }
}

#[test]
fn parentless_output_is_an_input_error_in_normal_and_validate_commands() {
    let directory = sandbox();
    setup(directory.path());
    for diagnostic in [false, true] {
        let mut args = request();
        args[7] = "/";
        if diagnostic {
            args.insert(0, "validate");
        }
        let output = Command::new(BINARY)
            .current_dir(directory.path())
            .env("PATH", "")
            .args(args)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains("output requires an existing parent directory"));
        assert!(!error.contains("panicked"));
    }
}

#[test]
fn validate_enforces_the_same_input_count_limit_as_normal_preflight() {
    let directory = sandbox();
    setup(directory.path());
    let inputs = directory.path().join("録画 空白");
    for index in 1..game_screen_pick::selection::MAX_CANDIDATES {
        fs::write(inputs.join(format!("video-{index:04}.mkv")), []).unwrap();
    }
    let invoke = |diagnostic: bool| {
        let mut args = request();
        if diagnostic {
            args.insert(0, "validate");
        }
        Command::new(BINARY)
            .current_dir(directory.path())
            .env("PATH", "")
            .args(args)
            .output()
            .unwrap()
    };
    assert_eq!(invoke(true).status.code(), Some(0));
    fs::write(inputs.join("one-too-many.mkv"), []).unwrap();
    for diagnostic in [true, false] {
        let output = invoke(diagnostic);
        assert_eq!(output.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&output.stderr).contains("input count exceeds 4096"));
    }
    assert!(!directory.path().join("output").exists());
    assert!(!directory.path().join(".game-screen-pick-cache").exists());
}

#[test]
fn normal_command_reports_cache_directory_creation_failures_as_processing() {
    use std::os::unix::fs::PermissionsExt;

    for existing_parent in [false, true] {
        let directory = sandbox();
        setup(directory.path());
        let root = directory.path().canonicalize().unwrap();
        let blocked = root.join("read-only");
        fs::create_dir(&blocked).unwrap();
        let permissions = fs::metadata(&blocked).unwrap().permissions();
        let cache = if existing_parent {
            blocked.clone()
        } else {
            blocked.join("cache")
        };
        let mut args = request();
        args.extend(["--cache-directory", cache.to_str().unwrap()]);
        let invoke = |args: &[&str]| {
            Command::new(BINARY)
                .current_dir(&root)
                .env("PATH", "")
                .args(args)
                .output()
                .unwrap()
        };
        fs::set_permissions(&blocked, fs::Permissions::from_mode(0o500)).unwrap();
        let mut validate = args.clone();
        validate.insert(0, "validate");
        let diagnostic = invoke(&validate);
        let normal = invoke(&args);
        fs::set_permissions(&blocked, permissions).unwrap();
        assert_eq!(diagnostic.status.code(), Some(0));
        let expected = if existing_parent {
            "cannot create Rust cache namespace"
        } else {
            "cannot create cache directory"
        };
        assert!(String::from_utf8_lossy(&normal.stderr).contains(expected));
        assert_eq!(
            normal.status.code(),
            Some(1),
            "{expected}: {}",
            String::from_utf8_lossy(&normal.stderr)
        );
        assert!(!root.join("output").exists());
        assert!(!cache.join(game_screen_pick::cache::NAMESPACE).exists());
    }

    let directory = sandbox();
    setup(directory.path());
    let root = directory.path().canonicalize().unwrap();
    let cache = root.join("cache");
    fs::create_dir(&cache).unwrap();
    let unexpected = cache.join(game_screen_pick::cache::NAMESPACE);
    fs::write(&unexpected, b"preserve unknown file").unwrap();
    let mut args = request();
    args.extend(["--cache-directory", cache.to_str().unwrap()]);
    let invalid = Command::new(BINARY)
        .current_dir(&root)
        .env("PATH", "")
        .args(args)
        .output()
        .unwrap();
    assert_eq!(invalid.status.code(), Some(2));
    assert_eq!(fs::read(unexpected).unwrap(), b"preserve unknown file");
}

#[test]
fn unsafe_paths_and_invalid_config_are_rejected_before_commands() {
    let directory = sandbox();
    setup(directory.path());
    let mut args = request();
    args.insert(0, "validate");
    args[8] = "録画 空白/output";
    assert_eq!(command(directory.path(), &args).status.code(), Some(2));
    args[8] = "../output";
    assert_eq!(command(directory.path(), &args).status.code(), Some(2));
    args[8] = "output";
    fs::write(
        directory.path().join("config.toml"),
        "api_key = SECRET_VALUE",
    )
    .unwrap();
    let output = command(directory.path(), &args);
    assert_eq!(output.status.code(), Some(2));
    assert!(!String::from_utf8_lossy(&output.stderr).contains("SECRET_VALUE"));
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink("録画 空白", directory.path().join("alias")).unwrap();
        args[7] = "alias";
        assert_eq!(command(directory.path(), &args).status.code(), Some(2));
    }
}

#[test]
fn extraction_emits_original_frames_and_timings_without_overwrite() {
    let directory = sandbox();
    setup(directory.path());
    let args = [
        "extract",
        "--at",
        "0.51",
        "--context-seconds",
        "0.25",
        "録画 空白/動画.mkv",
        "inspection",
    ];
    let output = command(directory.path(), &args);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: serde_json::Value = serde_json::from_slice(
        &fs::read(directory.path().join("inspection/extraction.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(metadata["stream_index"], 0);
    assert_eq!(metadata["source"], "動画.mkv");
    for (frame, time) in metadata["frames"]
        .as_array()
        .unwrap()
        .iter()
        .zip([0.5, 0.75, 1.0])
    {
        assert_eq!(frame["actual_seconds"], time);
        assert_eq!(frame["width"], 96);
        assert_eq!(frame["height"], 160);
        let image = directory
            .path()
            .join("inspection")
            .join(frame["file"].as_str().unwrap());
        assert!(fs::metadata(image).unwrap().len() > 8);
    }
    let before = fs::read(directory.path().join("inspection/frame.png")).unwrap();
    assert_eq!(command(directory.path(), &args).status.code(), Some(2));
    assert_eq!(
        fs::read(directory.path().join("inspection/frame.png")).unwrap(),
        before
    );
    for args in [
        vec!["extract", "--at", "0", "missing.mov", "fresh"],
        vec!["extract", "--at", "999", "録画 空白/動画.mkv", "fresh"],
    ] {
        assert_eq!(command(directory.path(), &args).status.code(), Some(2));
        assert!(!directory.path().join("fresh").exists());
    }
}

#[cfg(unix)]
#[test]
fn interruption_reaps_child_and_removes_partial_media() {
    use std::os::unix::fs::PermissionsExt;
    let directory = sandbox();
    setup(directory.path());
    let root = directory.path().canonicalize().unwrap();
    let tools = root.join("tools");
    fs::create_dir(&tools).unwrap();
    let ffmpeg = tools.join("ffmpeg");
    fs::write(&ffmpeg, "#!/bin/sh\nfor output do :; done\nprintf partial > \"$output\"\nprintf '%s\\n' \"$output\" > partial-path\nprintf '%s\\n' \"$$\" > child-pid\nexec /bin/sleep 60\n").unwrap();
    fs::set_permissions(&ffmpeg, fs::Permissions::from_mode(0o755)).unwrap();
    let mut paths = vec![tools];
    paths.extend(std::env::split_paths(&std::env::var_os("PATH").unwrap()));
    let mut child = Command::new(BINARY)
        .current_dir(&root)
        .env("PATH", std::env::join_paths(paths).unwrap())
        .args(["extract", "--at", "0.5", "録画 空白/動画.mkv", "inspection"])
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !root.join("child-pid").exists() {
        assert!(Instant::now() < deadline, "media child did not start");
        thread::sleep(Duration::from_millis(10));
    }
    let media_pid: i32 = fs::read_to_string(root.join("child-pid"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    let partial = fs::read_to_string(root.join("partial-path")).unwrap();
    assert!(Path::new(partial.trim()).exists());
    unsafe {
        assert_eq!(libc::kill(child.id() as i32, libc::SIGINT), 0);
    }
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert_eq!(status.code(), Some(130));
            break;
        }
        assert!(Instant::now() < deadline, "CLI did not handle SIGINT");
        thread::sleep(Duration::from_millis(10));
    }
    assert!(!Path::new(partial.trim()).exists());
    assert!(!root.join("inspection").exists());
    unsafe {
        assert_eq!(libc::kill(media_pid, 0), -1, "owned child survived");
    }
}
