//! Compiled by native integration tests, never part of distribution archives.
use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

fn sleep() -> ! {
    loop {
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn descendant(directory: &Path, name: &str) {
    let child = Command::new(std::env::current_exe().unwrap())
        .arg("descendant")
        .stdin(Stdio::null())
        .spawn()
        .unwrap();
    fs::write(directory.join(name), child.id().to_string()).unwrap();
    // Keep the descendant alive after this Child handle is dropped, as actual
    // wrappers do. It inherits stdout/stderr to exercise pipe cleanup as well.
    drop(child);
}

#[cfg(windows)]
fn ignore_console_events() {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn SetConsoleCtrlHandler(
            handler: Option<unsafe extern "system" fn(u32) -> i32>,
            add: i32,
        ) -> i32;
    }
    unsafe extern "system" fn ignore(event: u32) -> i32 {
        i32::from(event == 0 || event == 1)
    }
    assert_ne!(unsafe { SetConsoleCtrlHandler(None, 0) }, 0);
    assert_ne!(unsafe { SetConsoleCtrlHandler(Some(ignore), 1) }, 0);
}

#[cfg(windows)]
fn console_driver(directory: &Path, arguments: &[String]) {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GenerateConsoleCtrlEvent(event: u32, group: u32) -> i32;
    }
    let mut child = Command::new(&arguments[0])
        .args(&arguments[1..])
        .spawn()
        .unwrap();
    fs::write(directory.join("console-child"), child.id().to_string()).unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    while !directory.join("interrupt-now").exists() {
        if let Some(status) = child.try_wait().unwrap() {
            std::process::exit(status.code().unwrap_or(1));
        }
        assert!(
            std::time::Instant::now() < deadline,
            "interrupt was not requested"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    // This driver owns a separate console. Ctrl-C reaches only its CLI and
    // descendants; this driver's handler ignores it and waits for CLI cleanup.
    assert_ne!(unsafe { GenerateConsoleCtrlEvent(0, 0) }, 0);
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            std::process::exit(status.code().unwrap_or(1));
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("CLI did not exit after Ctrl-C");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn main() {
    #[cfg(windows)]
    ignore_console_events();
    let executable = std::env::current_exe().unwrap();
    let directory = executable.parent().unwrap();
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments.first().is_some_and(|arg| arg == "descendant") {
        sleep();
    }
    let stem = executable.file_stem().unwrap().to_str().unwrap();
    let ffmpeg_behavior = (stem == "ffmpeg")
        .then(|| fs::read_to_string(directory.join("ffmpeg-behavior")).ok())
        .flatten();
    let mode = ffmpeg_behavior.as_deref().unwrap_or(stem);
    match mode {
        "unrelated" => sleep(),
        #[cfg(windows)]
        "console-driver" => console_driver(directory, &arguments),
        "ffmpeg" | "runtime-start-marker" | "runtime-stop-marker" => {
            let marker = match mode {
                "ffmpeg" => "ffmpeg-called",
                "runtime-start-marker" => "runtime-started",
                _ => "runtime-stopped",
            };
            fs::write(marker, b"called").unwrap();
            std::process::exit(1);
        }
        "media-hang" | "media-fail" => {
            let output = arguments.last().unwrap();
            fs::write(output, b"partial").unwrap();
            fs::write(directory.join("partial-path"), output).unwrap();
            if mode == "media-fail" {
                eprintln!("GAME_SCREEN_PICK_API_KEY=fixture-secret-never-print");
                std::process::exit(9);
            }
            descendant(directory, "media-descendant");
            fs::write(
                directory.join("media-leader"),
                std::process::id().to_string(),
            )
            .unwrap();
            sleep();
        }
        "media-exit" => {
            descendant(directory, "media-descendant");
            std::process::exit(4);
        }
        "probe-large" => {
            use std::io::Write;
            std::io::stdout()
                .write_all(&vec![b'x'; 1024 * 1024 + 1])
                .unwrap();
        }
        "probe-metadata" => {
            let file = if arguments.iter().any(|arg| arg == "-show_streams") {
                "streams.json"
            } else {
                "frames.json"
            };
            print!("{}", fs::read_to_string(directory.join(file)).unwrap());
        }
        "runtime-start" | "runtime-exit" | "runtime-exit-alone" => {
            if mode != "runtime-exit-alone" {
                descendant(directory, "runtime-descendant");
            }
            fs::write(
                directory.join("started"),
                std::env::var("GAME_SCREEN_PICK_RUNTIME_SESSION").unwrap(),
            )
            .unwrap();
            fs::write(
                directory.join("runtime-leader"),
                std::process::id().to_string(),
            )
            .unwrap();
            if matches!(mode, "runtime-exit" | "runtime-exit-alone") {
                std::process::exit(7);
            }
            while !directory.join("release-runtime").exists() {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        "runtime-stop" | "runtime-stop-hang" => {
            assert_eq!(
                arguments[1],
                std::env::var("GAME_SCREEN_PICK_RUNTIME_SESSION").unwrap()
            );
            fs::write(directory.join("stopped"), arguments.join("\n")).unwrap();
            if mode == "runtime-stop-hang" {
                descendant(directory, "stop-descendant");
                sleep();
            }
            fs::write(directory.join("release-runtime"), b"stop").unwrap();
        }
        _ => panic!("unknown process fixture mode {mode}"),
    }
}
