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

/// Read-only comparison of unresolved siblings on modern macOS APFS. Existing
/// paths and lock representatives still use the filesystem's own lookup. Do
/// not apply APFS's Unicode equivalence to other filesystem formats.
pub(crate) fn native_sibling_names_equal(left: &Path, right: &Path) -> Result<bool> {
    if left == right {
        return Ok(true);
    }
    if left.parent() != right.parent() {
        return Ok(false);
    }
    #[cfg(target_os = "macos")]
    {
        let (Some(left_name), Some(right_name)) = (
            left.file_name().and_then(|name| name.to_str()),
            right.file_name().and_then(|name| name.to_str()),
        ) else {
            // Protected paths keep their native byte names; only UTF-8 names
            // can be Unicode aliases of the validated UTF-8 output leaf.
            return Ok(false);
        };
        let parent = File::open(left.parent().context("path needs a parent")?)?;
        let mut filesystem = std::mem::MaybeUninit::<libc::statfs>::uninit();
        if unsafe { libc::fstatfs(parent.as_raw_fd(), filesystem.as_mut_ptr()) } != 0 {
            return Err(std::io::Error::last_os_error()).context("cannot inspect path filesystem");
        }
        let filesystem = unsafe { filesystem.assume_init() };
        let format = unsafe { std::ffi::CStr::from_ptr(filesystem.f_fstypename.as_ptr()) };
        if format.to_bytes() != b"apfs" {
            return Ok(false);
        }
        let case_sensitive =
            unsafe { libc::fpathconf(parent.as_raw_fd(), libc::_PC_CASE_SENSITIVE) };
        ensure!(
            case_sensitive >= 0,
            "cannot inspect filesystem case sensitivity"
        );
        macos_names::equal(left_name, right_name, case_sensitive != 0)
    }
    #[cfg(not(target_os = "macos"))]
    Ok(false)
}

#[cfg(target_os = "macos")]
mod macos_names {
    use std::ffi::c_void;

    use anyhow::{Result, ensure};

    const UTF8: u32 = 0x0800_0100;
    const CASE_INSENSITIVE: usize = 1;
    const NORMALIZATION_D: isize = 0;

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFStringCreateWithBytes(
            allocator: *const c_void,
            bytes: *const u8,
            length: isize,
            encoding: u32,
            external_representation: u8,
        ) -> *const c_void;
        fn CFStringCreateMutableCopy(
            allocator: *const c_void,
            capacity: isize,
            value: *const c_void,
        ) -> *mut c_void;
        fn CFStringNormalize(value: *mut c_void, form: isize);
        fn CFStringFold(value: *mut c_void, flags: usize, locale: *const c_void);
        fn CFStringCompare(left: *const c_void, right: *const c_void, flags: usize) -> isize;
        fn CFRelease(value: *const c_void);
    }

    struct StringRef(*mut c_void);

    impl StringRef {
        fn normalized(name: &str, case_sensitive: bool) -> Result<Self> {
            // CoreFoundation's fold omits these historical Cyrillic letters
            // from Unicode 9, although APFS treats them as their modern forms.
            // Official Unicode 9 CaseFolding mappings:
            // https://raw.githubusercontent.com/unicode-org/icu/release-58-2/icu4c/source/data/unidata/CaseFolding.txt
            let name = if case_sensitive {
                name.to_owned()
            } else {
                name.chars()
                    .map(|character| match character {
                        '\u{1c80}' => '\u{0432}',
                        '\u{1c81}' => '\u{0434}',
                        '\u{1c82}' => '\u{043e}',
                        '\u{1c83}' => '\u{0441}',
                        '\u{1c84}' | '\u{1c85}' => '\u{0442}',
                        '\u{1c86}' => '\u{044a}',
                        '\u{1c87}' => '\u{0463}',
                        '\u{1c88}' => '\u{a64b}',
                        _ => character,
                    })
                    .collect()
            };
            // kCFStringEncodingUTF8; no locale or filesystem writes are used.
            let value = unsafe {
                CFStringCreateWithBytes(
                    std::ptr::null(),
                    name.as_ptr(),
                    name.len().try_into()?,
                    UTF8,
                    0,
                )
            };
            ensure!(!value.is_null(), "cannot compare APFS path filename");
            let normalized = unsafe { CFStringCreateMutableCopy(std::ptr::null(), 0, value) };
            unsafe { CFRelease(value) };
            ensure!(!normalized.is_null(), "cannot normalize APFS path filename");
            unsafe {
                CFStringNormalize(normalized, NORMALIZATION_D);
                if !case_sensitive {
                    CFStringFold(normalized, CASE_INSENSITIVE, std::ptr::null());
                    CFStringNormalize(normalized, NORMALIZATION_D);
                }
            }
            Ok(Self(normalized))
        }
    }

    impl Drop for StringRef {
        fn drop(&mut self) {
            unsafe { CFRelease(self.0) };
        }
    }

    pub(super) fn equal(left: &str, right: &str, case_sensitive: bool) -> Result<bool> {
        let left = StringRef::normalized(left, case_sensitive)?;
        let right = StringRef::normalized(right, case_sensitive)?;
        // Compare the full folded/decomposed values literally. The combined
        // case/nonliteral comparison omits some full case-fold expansions.
        Ok(unsafe { CFStringCompare(left.0, right.0, 0) } == 0)
    }
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
    digest_hex(&Sha256::digest(bytes))
}

pub(crate) fn digest_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        text.push(char::from(HEX[usize::from(byte >> 4)]));
        text.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    text
}

#[cfg(test)]
mod digest_tests {
    use super::*;

    #[test]
    fn sha256_preserves_known_lowercase_hex_vectors() {
        for (bytes, expected) in [
            (
                Vec::new(),
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                b"abc".to_vec(),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
            (
                (0..=255).collect(),
                "40aff2e9d2d8922e47afd4648e6967497158785fbd1da870e7110266bf944880",
            ),
        ] {
            assert_eq!(sha256(&bytes), expected);
        }
    }

    #[test]
    fn streamed_file_digest_preserves_hex_across_read_chunks() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&vec![0; 65_537]).unwrap();
        let mut bytes_read = 0;
        let (length, digest) =
            digest_file_cancelled(file.path(), &AtomicBool::new(false), |bytes| {
                bytes_read += bytes;
            })
            .unwrap();
        assert_eq!(length, 65_537);
        assert_eq!(bytes_read, length);
        assert_eq!(
            digest,
            "3266304f31be278d06c3bd3eb9aa3e00c59bedec0a890de466568b0b90b0e01f"
        );
    }
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

pub fn digest_file_cancelled(
    path: &Path,
    cancelled: &AtomicBool,
    mut on_read: impl FnMut(u64),
) -> Result<(u64, String)> {
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
        on_read(read as u64);
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
    Ok((length, digest_hex(&digest.finalize())))
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
