//! Filesystem primitives shared by cache and output ownership.
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};

use crate::media::Interrupted;

/// A processing failure after valid path/ownership checks, including lock
/// contention and recovery I/O. Keep invalid input errors unmarked.
#[derive(Debug)]
pub struct OperationalError(pub &'static str);

impl std::fmt::Display for OperationalError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for OperationalError {}

pub fn checked_path(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut current = PathBuf::new();
    let components = absolute.components().collect::<Vec<_>>();
    for (index, component) in components.iter().enumerate() {
        ensure!(
            !matches!(component, Component::ParentDir),
            "parent traversal is not allowed in paths"
        );
        current.push(component.as_os_str());
        match fs::symlink_metadata(&current) {
            Ok(metadata) => {
                ensure!(
                    !metadata.file_type().is_symlink(),
                    "symlink path components are not allowed"
                );
                ensure!(
                    metadata.is_dir() || metadata.is_file(),
                    "special files are not allowed"
                );
                ensure!(
                    index + 1 == components.len() || metadata.is_dir(),
                    "path parent must be a directory"
                );
            }
            Err(error)
                if error.kind() == std::io::ErrorKind::NotFound
                    && index + 1 == components.len() => {}
            Err(error) => return Err(error).context("cannot inspect path; parent must exist"),
        }
    }
    if current.exists() {
        return current
            .canonicalize()
            .context("cannot resolve existing path");
    }
    let parent = current
        .parent()
        .context("path needs a parent")?
        .canonicalize()?;
    Ok(parent.join(current.file_name().context("path needs a filename")?))
}

pub fn open_regular(path: &Path) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .context("cannot open ordinary file without following links")?;
    ensure!(file.metadata()?.is_file(), "expected an ordinary file");
    Ok(file)
}

pub fn read_regular(path: &Path, max: u64) -> Result<Vec<u8>> {
    let mut file = open_regular(path)?;
    ensure!(file.metadata()?.len() <= max, "file exceeds the byte limit");
    let mut bytes = Vec::new();
    Read::by_ref(&mut file)
        .take(max.saturating_add(1))
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= max, "file exceeds the byte limit");
    Ok(bytes)
}

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
thread_local! {
    static INTERRUPT_DIGEST_AFTER: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) struct DigestInterruption;

#[cfg(test)]
impl Drop for DigestInterruption {
    fn drop(&mut self) {
        INTERRUPT_DIGEST_AFTER.with(|threshold| threshold.set(None));
    }
}

#[cfg(test)]
pub(crate) fn interrupt_digest_after(bytes: u64) -> DigestInterruption {
    INTERRUPT_DIGEST_AFTER.with(|threshold| threshold.set(Some(bytes)));
    DigestInterruption
}

pub fn digest_file_cancelled(path: &Path, cancelled: &AtomicBool) -> Result<(u64, String)> {
    let mut file = open_regular(path)?;
    let mut buffer = [0; 64 * 1024];
    let mut digest = Sha256::new();
    let mut length = 0;
    loop {
        if cancelled.load(Ordering::Relaxed) {
            return Err(Interrupted.into());
        }
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        length += read as u64;
        digest.update(&buffer[..read]);
        #[cfg(test)]
        INTERRUPT_DIGEST_AFTER.with(|threshold| {
            if threshold.get().is_some_and(|bytes| length >= bytes) {
                threshold.set(None);
                cancelled.store(true, Ordering::Relaxed);
            }
        });
    }
    Ok((length, format!("{:x}", digest.finalize())))
}

pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let path = checked_path(path)?;
    if let Ok(metadata) = fs::symlink_metadata(&path) {
        ensure!(
            metadata.is_file(),
            "atomic write target must be an ordinary file"
        );
    }
    let parent = path.parent().context("atomic write needs a parent")?;
    let mut temporary = tempfile::Builder::new()
        .prefix(".gsp-write-")
        .tempfile_in(parent)?;
    temporary.write_all(bytes)?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(&path)
        .map_err(|error| error.error)
        .context("cannot commit atomic file")?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

#[derive(Debug)]
pub struct Lock {
    _file: File,
}

impl Drop for Lock {
    fn drop(&mut self) {
        // A concurrently forked child can retain this open-file description
        // until exec/exit. End the parent's run lock before closing our handle.
        unsafe {
            libc::flock(self._file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

impl Lock {
    pub fn acquire(path: &Path) -> Result<Self> {
        let path = checked_path(path)?;
        match fs::symlink_metadata(&path) {
            Ok(metadata) => ensure!(metadata.is_file(), "lock must be an ordinary file"),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).context(OperationalError("cannot inspect cache/output lock"));
            }
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path)
            .context(OperationalError("cannot open cache/output lock"))?;
        ensure!(
            file.metadata()
                .context(OperationalError("cannot inspect cache/output lock"))?
                .is_file(),
            "lock must be an ordinary file"
        );
        // The descriptor holds the advisory lock; unlinking it would admit a second owner.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            let error = std::io::Error::last_os_error();
            let message = if matches!(error.raw_os_error(), Some(code) if code == libc::EWOULDBLOCK || code == libc::EAGAIN)
            {
                "cache/output is already locked by another run"
            } else {
                "cannot acquire cache/output lock"
            };
            return Err(error).context(OperationalError(message));
        }
        Ok(Self { _file: file })
    }
}
