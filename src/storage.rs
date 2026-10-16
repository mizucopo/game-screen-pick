//! Filesystem primitives shared by cache and output ownership.
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
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
    #[cfg(windows)]
    windows::validate_path(path)?;
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
        // A Windows drive/UNC prefix is not itself a rooted path. Inspect it
        // only after its following root separator has been added.
        #[cfg(windows)]
        if matches!(component, Component::Prefix(_)) {
            continue;
        }
        match fs::symlink_metadata(&current) {
            Ok(metadata) => {
                ensure!(
                    !is_link(&metadata),
                    "symlink or reparse path components are not allowed"
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

/// Read-only comparison of unresolved siblings on APFS and Windows. Existing
/// paths and lock representatives still use the filesystem's own lookup.
/// APFS Unicode equivalence is never applied to other filesystem formats.
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
    #[cfg(windows)]
    {
        windows::sibling_names_equal(left, right)
    }
    #[cfg(all(not(target_os = "macos"), not(windows)))]
    Ok(false)
}

/// Includes junctions and other Windows reparse points, not only symbolic links.
pub fn is_link(metadata: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        metadata.file_attributes()
            & windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT
            != 0
    }
    #[cfg(not(windows))]
    metadata.file_type().is_symlink()
}

/// Native object identity, independent of the spelling or location of a path.
/// Unix serialization stays compatible with existing ownership journals.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileIdentity {
    pub device: u64,
    pub inode: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub windows_file_id: Option<[u8; 16]>,
}

pub fn file_identity(path: &Path) -> Result<FileIdentity> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let metadata = fs::symlink_metadata(path)?;
        ensure!(!is_link(&metadata), "file identity must not follow links");
        Ok(FileIdentity {
            device: metadata.dev(),
            inode: metadata.ino(),
            windows_file_id: None,
        })
    }
    #[cfg(windows)]
    {
        let path = checked_path(path)?;
        windows::identity(&windows::open_attributes(&path)?)
    }
}

/// POSIX allows fsync of a directory. Windows exposes no equivalent ordinary
/// directory flush: regular files are synced before rename, and the journal
/// recovers ordinary interruption; power-loss durability is not promised.
pub fn sync_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        File::open(path)?.sync_all()?;
    }
    #[cfg(windows)]
    {
        let path = checked_path(path)?;
        ensure!(
            fs::symlink_metadata(path)?.is_dir(),
            "expected an ordinary directory"
        );
    }
    Ok(())
}

/// Move an already-verified transaction directory on the same filesystem.
/// Unix retains rename's empty-directory placeholder semantics. Windows must
/// reject every existing destination: std::fs::rename enables replacement and
/// can otherwise destroy a file created after our ownership checks.
pub(crate) fn rename_directory(source: &Path, destination: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        fs::rename(source, destination)
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Storage::FileSystem::MoveFileExW;

        let wide_path = |path: &Path| -> std::io::Result<Vec<u16>> {
            let mut units = path.as_os_str().encode_wide().collect::<Vec<_>>();
            if units.contains(&0) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "directory move path must not contain NUL",
                ));
            }
            units.push(0);
            Ok(units)
        };
        let source = wide_path(source)?;
        let destination = wide_path(destination)?;
        // No REPLACE_EXISTING or COPY_ALLOWED: the kernel refuses an existing
        // file/directory in this single move, without a separate exists check.
        if unsafe { MoveFileExW(source.as_ptr(), destination.as_ptr(), 0) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
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
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT);
    }
    #[cfg(windows)]
    let native_path = checked_path(path)?;
    #[cfg(windows)]
    let path = native_path.as_path();
    let file = options
        .open(path)
        .context("cannot open ordinary file without following links")?;
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file() && !is_link(&metadata),
        "expected an ordinary file"
    );
    #[cfg(windows)]
    ensure!(
        windows::identity(&file)? == file_identity(path)?,
        "file changed while being opened"
    );
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

#[cfg(all(test, windows))]
mod directory_move_tests {
    use super::*;

    #[test]
    fn transaction_directory_move_refuses_existing_files_and_directories() {
        for kind in ["file", "empty-directory", "populated-directory"] {
            let temporary = tempfile::tempdir().unwrap();
            let root = temporary.path().canonicalize().unwrap();
            let source = root.join("staged");
            let destination = root.join("destination");
            fs::create_dir(&source).unwrap();
            fs::write(source.join("staged.txt"), b"new output").unwrap();
            if kind == "file" {
                fs::write(&destination, vec![0x5a; 65_537]).unwrap();
            } else {
                fs::create_dir(&destination).unwrap();
                if kind == "populated-directory" {
                    fs::write(destination.join("keep.txt"), b"user data").unwrap();
                }
            }
            let original = file_identity(&destination).unwrap();
            assert!(rename_directory(&source, &destination).is_err(), "{kind}");
            assert_eq!(file_identity(&destination).unwrap(), original, "{kind}");
            assert_eq!(fs::read(source.join("staged.txt")).unwrap(), b"new output");
            if kind == "file" {
                assert_eq!(fs::read(&destination).unwrap(), vec![0x5a; 65_537]);
            } else if kind == "populated-directory" {
                assert_eq!(
                    fs::read(destination.join("keep.txt")).unwrap(),
                    b"user data"
                );
            } else {
                assert_eq!(fs::read_dir(&destination).unwrap().count(), 0);
            }
        }
    }

    #[test]
    fn transaction_directory_move_preserves_identity_at_an_absent_destination() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        let source = root.join("staged");
        let destination = root.join("published");
        fs::create_dir(&source).unwrap();
        fs::write(source.join("output.txt"), b"complete output").unwrap();
        let original = file_identity(&source).unwrap();
        rename_directory(&source, &destination).unwrap();
        assert!(!source.exists());
        assert_eq!(file_identity(&destination).unwrap(), original);
        assert_eq!(
            fs::read(destination.join("output.txt")).unwrap(),
            b"complete output"
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
    sync_directory(parent)?;
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
        #[cfg(unix)]
        unsafe {
            libc::flock(self._file.as_raw_fd(), libc::LOCK_UN);
        }
        #[cfg(windows)]
        windows::unlock(&self._file);
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
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            use windows_sys::Win32::Storage::FileSystem::{
                FILE_FLAG_OPEN_REPARSE_POINT, FILE_SHARE_READ, FILE_SHARE_WRITE,
            };
            // Prevent replacing/unlinking the held representative. Other runs
            // can open it to attempt the same exclusive byte-range lock.
            options
                .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
                .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE);
        }
        let file = options
            .open(&path)
            .context(OperationalError("cannot open cache/output lock"))?;
        let metadata = file
            .metadata()
            .context(OperationalError("cannot inspect cache/output lock"))?;
        ensure!(
            metadata.is_file() && !is_link(&metadata),
            "lock must be an ordinary file"
        );
        #[cfg(windows)]
        {
            ensure!(
                windows::identity(&file)? == file_identity(&path)?,
                "lock changed while being opened"
            );
            windows::lock(&file)?;
        }
        // The descriptor holds the advisory lock; unlinking it would admit a second owner.
        #[cfg(unix)]
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

#[cfg(windows)]
pub(crate) mod windows {
    use std::fs::{File, OpenOptions};
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::AsRawHandle;
    use std::path::{Component, Path, Prefix};

    use anyhow::{Context, Result, ensure};
    use windows_sys::Win32::Foundation::{ERROR_INVALID_PARAMETER, ERROR_LOCK_VIOLATION};
    use windows_sys::Win32::Globalization::{CSTR_EQUAL, CompareStringOrdinal};
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_BASIC_INFO, FILE_CASE_SENSITIVE_INFO, FILE_FLAG_BACKUP_SEMANTICS,
        FILE_FLAG_OPEN_REPARSE_POINT, FILE_ID_INFO, FILE_INFO_BY_HANDLE_CLASS,
        FILE_READ_ATTRIBUTES, FileBasicInfo, FileCaseSensitiveInfo, FileIdInfo,
        GetFileInformationByHandleEx, LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY,
        LockFileEx, UnlockFileEx,
    };
    use windows_sys::Win32::System::IO::OVERLAPPED;

    use super::{FileIdentity, OperationalError, is_link};

    pub(super) fn validate_path(path: &Path) -> Result<()> {
        for component in path.components() {
            match component {
                Component::Prefix(prefix) => ensure!(
                    matches!(
                        prefix.kind(),
                        Prefix::Disk(_)
                            | Prefix::VerbatimDisk(_)
                            | Prefix::UNC(_, _)
                            | Prefix::VerbatimUNC(_, _)
                    ) && path.has_root(),
                    "device namespaces and drive-relative paths are not allowed"
                ),
                Component::Normal(name) => {
                    let units = name.encode_wide().collect::<Vec<_>>();
                    ensure!(
                        !units.is_empty()
                            && !units.iter().any(|unit| *unit < 32
                                || matches!(
                                    *unit,
                                    0x22 | 0x2a | 0x2f | 0x3a | 0x3c | 0x3e | 0x3f | 0x5c | 0x7c
                                ))
                            && !matches!(units.last(), Some(0x20 | 0x2e)),
                        "Windows path has an alternate stream, reserved character or trailing dot/space"
                    );
                    // Win32 reserves these even when followed by an extension;
                    // the superscript digits are also DOS device aliases.
                    let stem = name
                        .to_string_lossy()
                        .split('.')
                        .next()
                        .unwrap_or_default()
                        .trim_end_matches(' ')
                        .to_uppercase();
                    let numbered_device = stem
                        .strip_prefix("COM")
                        .or_else(|| stem.strip_prefix("LPT"))
                        .is_some_and(|suffix| {
                            matches!(
                                suffix,
                                "1" | "2"
                                    | "3"
                                    | "4"
                                    | "5"
                                    | "6"
                                    | "7"
                                    | "8"
                                    | "9"
                                    | "¹"
                                    | "²"
                                    | "³"
                            )
                        });
                    ensure!(
                        !matches!(
                            stem.as_str(),
                            "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$"
                        ) && !numbered_device,
                        "Windows reserved device names are not allowed"
                    );
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub(super) fn open_attributes(path: &Path) -> Result<File> {
        let file = OpenOptions::new()
            .access_mode(FILE_READ_ATTRIBUTES)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS)
            .open(path)
            .context("cannot open native file identity")?;
        let metadata = file.metadata()?;
        ensure!(
            (metadata.is_file() || metadata.is_dir()) && !is_link(&metadata),
            "file identity requires an ordinary file or directory without reparse points"
        );
        Ok(file)
    }

    fn query<T: Default>(file: &File, class: FILE_INFO_BY_HANDLE_CLASS) -> std::io::Result<T> {
        let mut value = T::default();
        // All callers pair a documented information class with its matching
        // repr(C) windows-sys structure and pass the structure's exact size.
        if unsafe {
            GetFileInformationByHandleEx(
                file.as_raw_handle(),
                class,
                (&mut value as *mut T).cast(),
                std::mem::size_of::<T>() as u32,
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error());
        }
        Ok(value)
    }

    pub(super) fn identity(file: &File) -> Result<FileIdentity> {
        let info: FILE_ID_INFO = query(file, FileIdInfo)
            .context("filesystem must support stable Windows volume and 128-bit file identities")?;
        Ok(FileIdentity {
            device: info.VolumeSerialNumber,
            inode: u64::from_le_bytes(info.FileId.Identifier[..8].try_into().expect("eight bytes")),
            windows_file_id: Some(info.FileId.Identifier),
        })
    }

    pub(crate) fn input_metadata(path: &Path) -> Result<(FileIdentity, u64, i64, i64)> {
        let file = open_attributes(path)?;
        let metadata = file.metadata()?;
        ensure!(
            metadata.is_file() && !is_link(&metadata),
            "input identity requires an ordinary file"
        );
        let info: FILE_BASIC_INFO =
            query(&file, FileBasicInfo).context("cannot inspect Windows input change time")?;
        Ok((
            identity(&file)?,
            metadata.len(),
            info.LastWriteTime,
            info.ChangeTime,
        ))
    }

    pub(super) fn sibling_names_equal(left: &Path, right: &Path) -> Result<bool> {
        let parent = open_attributes(left.parent().context("path needs a parent")?)?;
        let case_sensitive = match query::<FILE_CASE_SENSITIVE_INFO>(&parent, FileCaseSensitiveInfo)
        {
            // FILE_CS_FLAG_CASE_SENSITIVE_DIR from winnt.h.
            Ok(info) => info.Flags & 1 != 0,
            // Filesystems without per-directory case-sensitivity support use
            // a conservative case-insensitive comparison for absent names.
            Err(error) if error.raw_os_error() == Some(ERROR_INVALID_PARAMETER as i32) => false,
            Err(error) => {
                return Err(error).context("cannot inspect Windows directory case sensitivity");
            }
        };
        let left = left
            .file_name()
            .context("path needs a filename")?
            .encode_wide()
            .collect::<Vec<_>>();
        let right = right
            .file_name()
            .context("path needs a filename")?
            .encode_wide()
            .collect::<Vec<_>>();
        let compared = unsafe {
            CompareStringOrdinal(
                left.as_ptr(),
                left.len().try_into()?,
                right.as_ptr(),
                right.len().try_into()?,
                i32::from(!case_sensitive),
            )
        };
        ensure!(compared != 0, "cannot compare native Windows filenames");
        Ok(compared == CSTR_EQUAL)
    }

    pub(super) fn lock(file: &File) -> Result<()> {
        let mut overlapped = OVERLAPPED::default();
        // Offset zero, length one protects the persistent lock representative.
        // FAIL_IMMEDIATELY makes this synchronous even with a conflicting run.
        if unsafe {
            LockFileEx(
                file.as_raw_handle(),
                LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
                0,
                1,
                0,
                &mut overlapped,
            )
        } == 0
        {
            let error = std::io::Error::last_os_error();
            let message = if error.raw_os_error() == Some(ERROR_LOCK_VIOLATION as i32) {
                "cache/output is already locked by another run"
            } else {
                "cannot acquire cache/output lock"
            };
            return Err(error).context(OperationalError(message));
        }
        Ok(())
    }

    pub(super) fn unlock(file: &File) {
        let mut overlapped = OVERLAPPED::default();
        unsafe { UnlockFileEx(file.as_raw_handle(), 0, 1, 0, &mut overlapped) };
    }
}
