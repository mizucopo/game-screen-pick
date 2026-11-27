//! Native executable fixtures: no shell, PowerShell, or global environment edits.
use anyhow::{Context, Result, ensure};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

pub struct Helper {
    directory: tempfile::TempDir,
}

#[allow(dead_code)]
impl Helper {
    pub fn new() -> Self {
        Self {
            directory: tempfile::Builder::new()
                .prefix("process fixtures 日本語 ")
                .tempdir()
                .unwrap(),
        }
    }

    pub fn directory(&self) -> &Path {
        self.directory.path()
    }

    pub fn marker(&self, name: &str) -> PathBuf {
        self.directory().join(name)
    }

    pub fn wait_for_media_start(&self) -> Result<StartedMedia> {
        // The descendant marker is published last. At that point the leader PID
        // and partial file must already be observable, before cancellation.
        eprintln!("startup: waiting for media-descendant PID marker");
        let descendant = poll_marker(&self.marker("media-descendant"))
            .context("timed out waiting for media-descendant PID marker")?
            .parse()
            .context("invalid media-descendant PID marker")?;
        let leader = std::fs::read_to_string(self.marker("media-leader"))
            .context("cannot read media-leader PID marker after descendant became ready")?
            .parse()
            .context("invalid media-leader PID marker")?;
        let partial = PathBuf::from(
            std::fs::read_to_string(self.marker("partial-path"))
                .context("cannot read partial-path marker after descendant became ready")?,
        );
        ensure!(
            partial.is_file(),
            "partial file missing: {}",
            partial.display()
        );
        Ok(StartedMedia {
            leader,
            descendant,
            partial,
        })
    }

    pub fn executable(&self, mode: &str) -> PathBuf {
        static COMPILED: OnceLock<tempfile::TempDir> = OnceLock::new();
        let supplied = std::env::var_os("GAME_SCREEN_PICK_PROCESS_HELPER").map(PathBuf::from);
        let compiled = || {
            COMPILED
                .get_or_init(|| {
                    let directory = tempfile::tempdir().unwrap();
                    let source = directory.path().join("fixture.rs");
                    std::fs::write(&source, include_str!("process_fixture.rs")).unwrap();
                    let output =
                        Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
                            .args(["--edition=2024", "-o"])
                            .arg(directory.path().join("process-fixture.exe"))
                            .arg(source)
                            .output()
                            .expect("Rust compiler is required for native process fixtures");
                    assert!(
                        output.status.success(),
                        "cannot compile process fixture: {}",
                        String::from_utf8_lossy(&output.stderr)
                    );
                    directory
                })
                .path()
                .join("process-fixture.exe")
        };
        let executable = self.marker(&format!("{mode}.exe"));
        std::fs::copy(supplied.unwrap_or_else(compiled), &executable).unwrap();
        executable
    }
}

#[allow(dead_code)]
pub fn wait_for(path: &Path) -> String {
    poll_marker(path)
        .unwrap_or_else(|| panic!("process fixture marker was not created: {}", path.display()))
}

fn poll_marker(path: &Path) -> Option<String> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while std::time::Instant::now() < deadline {
        if let Ok(value) = std::fs::read_to_string(path)
            && !value.is_empty()
        {
            return Some(value);
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    None
}

pub struct StartedMedia {
    leader: u32,
    descendant: u32,
    partial: PathBuf,
}

#[cfg(windows)]
#[allow(dead_code)]
impl StartedMedia {
    pub fn assert_cleaned(&self) {
        eprintln!(
            "cleanup: checking partial file directory and owned processes (leader {}, descendant {})",
            self.leader, self.descendant
        );
        let directory = self
            .partial
            .parent()
            .expect("startup: invalid partial file path");
        assert!(
            !directory
                .try_exists()
                .expect("cleanup: cannot inspect partial file directory"),
            "cleanup: partial file directory remains: {}",
            directory.display()
        );
        assert_gone(self.leader);
        assert_gone(self.descendant);
    }
}

#[cfg(windows)]
#[allow(dead_code)]
pub fn assert_gone(pid: u32) {
    use windows_sys::Win32::Foundation::{
        CloseHandle, ERROR_INVALID_PARAMETER, GetLastError, WAIT_OBJECT_0,
    };
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE, WaitForSingleObject,
    };
    let process = unsafe {
        OpenProcess(
            PROCESS_SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
            0,
            pid,
        )
    };
    if !process.is_null() {
        let result = unsafe { WaitForSingleObject(process, 5000) };
        unsafe { CloseHandle(process) };
        assert_eq!(
            result, WAIT_OBJECT_0,
            "cleanup: owned process {pid} survived"
        );
    } else {
        assert_eq!(
            unsafe { GetLastError() },
            ERROR_INVALID_PARAMETER,
            "cleanup: cannot inspect owned process {pid}"
        );
    }
}
