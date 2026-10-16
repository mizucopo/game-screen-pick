//! Native executable fixtures: no shell, PowerShell, or global environment edits.
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
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while std::time::Instant::now() < deadline {
        if let Ok(value) = std::fs::read_to_string(path)
            && !value.is_empty()
        {
            return value;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    panic!("process fixture marker was not created: {}", path.display());
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
        assert_eq!(result, WAIT_OBJECT_0, "owned process {pid} survived");
    } else {
        assert_eq!(
            unsafe { GetLastError() },
            ERROR_INVALID_PARAMETER,
            "cannot inspect owned process {pid}"
        );
    }
}
