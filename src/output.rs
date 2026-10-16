//! Verified Rust output ownership and recoverable, same-directory publication.
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail, ensure};
use image::ImageDecoder;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as _, Sha256};

use crate::media::Interrupted;
use crate::metrics::timed_cell;
use crate::pipeline::Artifacts;
use crate::storage::{self, Lock, OperationalError};

const NAMESPACE: &str = "game-screen-pick-rust-output";
const SCHEMA: u32 = 1;
const MAX_REPORT_BYTES: u64 = 64 * 1024 * 1024;
const LOCK_DIRECTORY: &str = ".gsp-output-locks";

#[derive(Debug, Clone)]
pub struct OwnedOutput {
    pub report: Value,
    pub fingerprint: String,
    /// Only the report and its declared, published images are read for verification.
    pub bytes_read: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Digest {
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    namespace: String,
    schema: u32,
    report_sha256: String,
    files: BTreeMap<String, Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Identity {
    device: u64,
    inode: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
enum State {
    Absent,
    Empty {
        identity: Identity,
    },
    Owned {
        identity: Identity,
        fingerprint: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Phase {
    Prepared,
    Committed,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Journal {
    namespace: String,
    schema: u32,
    target: String,
    stage: String,
    backup: Option<String>,
    placeholder: Option<Identity>,
    old: State,
    new: State,
    phase: Phase,
}

/// Holds the output lock from preflight through publication and verification.
pub struct Output {
    path: PathBuf,
    parent: PathBuf,
    prefix: String,
    journal: PathBuf,
    initial: State,
    bytes_read: Cell<u64>,
    verification_seconds: Cell<f64>,
    _lock: Lock,
}

impl Output {
    /// Read-only validation for the CLI diagnostic; recovery requires prepare.
    pub fn validate(
        path: &Path,
        protected_paths: &[&Path],
        cancelled: &AtomicBool,
    ) -> Result<Option<OwnedOutput>> {
        check_cancelled(cancelled)?;
        let (mut path, parent, mut prefix) = checked_target(path, protected_paths)?;
        let directory = checked_lock_directory(&path, &parent, protected_paths)?;
        let lock_path = directory.join(path.file_name().context("output filename is required")?);
        match fs::symlink_metadata(&lock_path) {
            Ok(metadata) => {
                ensure!(
                    metadata.is_file() && !metadata.file_type().is_symlink(),
                    "output lock must be an ordinary file"
                );
                let representative = storage::checked_path(&lock_path)?;
                (path, prefix) = native_target(&parent, &representative, protected_paths)?;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                validate_native_aliases(&path, &directory, protected_paths)?;
            }
            Err(error) => return Err(error.into()),
        }
        let journal = parent.join(format!("{prefix}.journal.json"));
        match fs::symlink_metadata(journal) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
            Ok(_) => bail!(
                "output publication recovery is pending; run the normal command to recover or choose a new empty output directory"
            ),
        }
        Ok(inspect(&path, false, &Cell::new(0), cancelled)?.1)
    }

    /// Call before probe, decode, search, inference, or runtime operations.
    /// Protected paths normally include the input directory, config, and cache.
    pub fn prepare(path: &Path, protected_paths: &[&Path], cancelled: &AtomicBool) -> Result<Self> {
        check_cancelled(cancelled)?;
        let (path, parent, _) = checked_target(path, protected_paths)?;
        let directory = checked_lock_directory(&path, &parent, protected_paths)?;
        match fs::create_dir(&directory) {
            Ok(()) => sync_directory(&parent)?,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(error).context(OperationalError("cannot create output lock directory"));
            }
        }
        // Recheck after creation so filesystem-native aliases of the reserved
        // container and protected paths cannot become output directories.
        let (path, _, _) = checked_target(&path, protected_paths)?;
        let directory = checked_lock_directory(&path, &parent, protected_paths)?;
        // Exact leaf names preserve the filesystem's case/Unicode equivalence
        // and its maximum filename length. This lock survives target renames.
        let lock_path = directory.join(path.file_name().context("output filename is required")?);
        let lock = Lock::acquire(&lock_path)?;
        let representative = storage::checked_path(&lock_path)?;
        let (path, prefix) = native_target(&parent, &representative, protected_paths)?;
        let mut output = Self {
            journal: parent.join(format!("{prefix}.journal.json")),
            path,
            parent,
            prefix,
            initial: State::Absent,
            bytes_read: Cell::new(0),
            verification_seconds: Cell::new(0.0),
            _lock: lock,
        };
        output.check_path()?;
        output.recover(cancelled)?;
        let (initial, _) = output.state(&output.path, cancelled)?;
        output.initial = initial;
        Ok(output)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Cumulative content reads, including staging, JPEG validation, ownership
    /// digests and recovery. Directory entries and metadata are not content.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read.get()
    }

    /// Cumulative output ownership inspections, including preflight, staged and
    /// published output and recovery checks. Copy/write/delete time is excluded.
    pub fn verification_seconds(&self) -> f64 {
        self.verification_seconds.get()
    }

    fn state(&self, path: &Path, cancelled: &AtomicBool) -> Result<(State, Option<OwnedOutput>)> {
        timed_cell(&self.verification_seconds, || {
            inspect(path, false, &self.bytes_read, cancelled)
        })
    }

    fn cleanup_state(
        &self,
        path: &Path,
        expected: &State,
        cancelled: &AtomicBool,
    ) -> Result<State> {
        timed_cell(&self.verification_seconds, || {
            cleanup_state(path, expected, &self.bytes_read, cancelled)
        })
    }

    fn remove_verified(&self, path: &Path, expected: &State, cancelled: &AtomicBool) -> Result<()> {
        remove_verified(
            path,
            expected,
            &self.bytes_read,
            &self.verification_seconds,
            cancelled,
        )
    }

    /// Does not consult cache and never decodes an image or scans an input video.
    pub fn verify_existing(&self, cancelled: &AtomicBool) -> Result<Option<OwnedOutput>> {
        check_cancelled(cancelled)?;
        self.check_path()?;
        Ok(self.state(&self.path, cancelled)?.1)
    }

    pub fn publish(
        &mut self,
        artifacts: &Artifacts,
        cancelled: &AtomicBool,
    ) -> Result<OwnedOutput> {
        self.publish_checked(artifacts, cancelled, || Ok(()))
    }

    /// Revalidate caller-owned input identities after staging and immediately
    /// before replacement and commit. Rejection restores the previous output.
    pub fn publish_checked(
        &mut self,
        artifacts: &Artifacts,
        cancelled: &AtomicBool,
        before_replace: impl FnMut() -> Result<()>,
    ) -> Result<OwnedOutput> {
        // This path is created and owned by Rust's private TempDir, unlike a
        // user supplied output path. Resolve the system's temporary-directory
        // alias (for example macOS /var) before applying ordinary path checks.
        let source = artifacts.path().canonicalize()?;
        let record = serde_json::from_slice(&serde_json::to_vec(artifacts.report())?)?;
        self.publish_directory_checked(&source, &record, cancelled, before_replace, |_| Ok(()))
    }

    fn check_path(&self) -> Result<()> {
        ensure!(
            storage::checked_path(&self.path)? == self.path,
            "output path changed after preflight"
        );
        Ok(())
    }

    #[cfg(test)]
    fn publish_directory(
        &mut self,
        source: &Path,
        cancelled: &AtomicBool,
        step: impl FnMut(&str) -> Result<()>,
    ) -> Result<OwnedOutput> {
        self.publish_directory_checked(
            source,
            &tests::report_record(source),
            cancelled,
            || Ok(()),
            step,
        )
    }

    fn publish_directory_checked(
        &mut self,
        source: &Path,
        record: &Value,
        cancelled: &AtomicBool,
        mut before_replace: impl FnMut() -> Result<()>,
        mut step: impl FnMut(&str) -> Result<()>,
    ) -> Result<OwnedOutput> {
        check_cancelled(cancelled)?;
        self.check_path()?;
        ensure!(
            self.state(&self.path, cancelled)?.0 == self.initial,
            "output changed after preflight; preserve it and choose a new empty output directory"
        );
        let stage = tempfile::Builder::new()
            .prefix(&format!("{}-stage-", self.prefix))
            .tempdir_in(&self.parent)
            .context("cannot create output staging directory")?;
        step("staging")?;
        stage_artifacts(source, stage.path(), record, cancelled, &self.bytes_read)?;
        let staged = self.state(stage.path(), cancelled)?.0;
        check_cancelled(cancelled)?;
        step("staged")?;
        before_replace()?;
        // Staging may be lengthy. Recheck the old content, not just its names,
        // immediately before recording a transaction and moving it.
        ensure!(
            self.state(&self.path, cancelled)?.0 == self.initial,
            "output changed during staging; preserve it and choose a new empty output directory"
        );
        let backup = if matches!(self.initial, State::Absent) {
            None
        } else {
            Some(
                tempfile::Builder::new()
                    .prefix(&format!("{}-backup-", self.prefix))
                    .tempdir_in(&self.parent)?,
            )
        };
        let mut journal = Journal {
            namespace: NAMESPACE.to_owned(),
            schema: SCHEMA,
            target: filename(&self.path)?,
            stage: filename(stage.path())?,
            backup: backup
                .as_ref()
                .map(|directory| filename(directory.path()))
                .transpose()?,
            placeholder: backup
                .as_ref()
                .map(|directory| identity(directory.path()))
                .transpose()?,
            old: self.initial.clone(),
            new: staged,
            phase: Phase::Prepared,
        };
        ensure!(
            fs::symlink_metadata(&self.journal)
                .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound),
            "output recovery record already exists"
        );
        storage::atomic_write(&self.journal, &serde_json::to_vec(&journal)?)?;
        // The journal now owns these locations. TempDir must not recursively
        // delete them if a later filesystem operation fails.
        let stage = stage.keep();
        let backup = backup.map(tempfile::TempDir::keep);
        let transaction = (|| -> Result<()> {
            step("prepared")?;
            check_cancelled(cancelled)?;
            self.check_path()?;
            ensure!(
                self.state(&self.path, cancelled)?.0 == journal.old,
                "output changed before replacement"
            );
            before_replace()?;
            if let Some(backup) = &backup {
                fs::rename(&self.path, backup).context("cannot preserve previous output")?;
                sync_directory(&self.parent)?;
            }
            step("backed_up")?;
            check_cancelled(cancelled)?;
            before_replace()?;
            fs::rename(&stage, &self.path).context("cannot publish staged output")?;
            sync_directory(&self.parent)?;
            step("published")?;
            ensure!(
                self.state(&self.path, cancelled)?.0 == journal.new,
                "published output failed completion verification"
            );
            check_cancelled(cancelled)?;
            step("verified")?;
            before_replace()?;
            journal.phase = Phase::Committed;
            storage::atomic_write(&self.journal, &serde_json::to_vec(&journal)?)?;
            step("committed")?;
            Ok(())
        })();
        if let Err(error) = transaction {
            // Complete transaction recovery even after cancellation: prepared
            // restores the prior output; committed finishes verified cleanup.
            let rollback = AtomicBool::new(false);
            self.recover(&rollback)
                .context("publication failed; previous output recovery could not complete")?;
            self.initial = self.state(&self.path, &rollback)?.0;
            return Err(error);
        }
        self.recover(cancelled)?;
        self.initial = self.state(&self.path, cancelled)?.0;
        self.verify_existing(cancelled)?
            .context("published output is missing")
    }

    /// Recovery only moves or deletes directories whose exact file set and
    /// content still match this transaction. Unknown data stops recovery.
    fn recover(&self, cancelled: &AtomicBool) -> Result<()> {
        check_cancelled(cancelled)?;
        let bytes = match fs::symlink_metadata(&self.journal) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => {
                return Err(error)
                    .context(OperationalError("cannot inspect output recovery record"));
            }
            Ok(_) => recovery_read(read_regular(&self.journal, 64 * 1024, &self.bytes_read))?,
        };
        let journal: Journal = serde_json::from_slice(&bytes).context(
            "unknown output recovery record; preserve it and choose a new empty output directory",
        )?;
        ensure!(
            journal.namespace == NAMESPACE
                && journal.schema == SCHEMA
                && journal.target == filename(&self.path)?,
            "unknown output recovery record"
        );
        let stage = self.transaction_path(&journal.stage, "stage")?;
        let backup = journal
            .backup
            .as_ref()
            .map(|name| self.transaction_path(name, "backup"))
            .transpose()?;
        ensure!(
            matches!(journal.new, State::Owned { .. }),
            "invalid output recovery completion record"
        );
        ensure!(
            matches!(journal.old, State::Absent) == backup.is_none()
                && backup.is_some() == journal.placeholder.is_some(),
            "invalid output backup record"
        );
        let (target_state, _) = recovery_read(self.state(&self.path, cancelled))?;
        let stage_state = recovery_read(self.cleanup_state(&stage, &journal.new, cancelled))?;
        ensure!(
            matches!(stage_state, State::Absent) || stage_state == journal.new,
            "staged output changed; preserve recovery files"
        );
        let backup_state = backup
            .as_ref()
            .map(|path| {
                if journal.phase == Phase::Committed {
                    recovery_read(self.cleanup_state(path, &journal.old, cancelled))
                } else {
                    recovery_read(self.state(path, cancelled)).map(|value| value.0)
                }
            })
            .transpose()?;
        let placeholder = journal.placeholder.as_ref().map(|identity| State::Empty {
            identity: identity.clone(),
        });
        ensure!(
            backup_state
                .as_ref()
                .is_none_or(|value| matches!(value, State::Absent)
                    || value == &journal.old
                    || Some(value) == placeholder.as_ref()),
            "previous output changed; preserve recovery files"
        );

        check_cancelled(cancelled)?;
        match journal.phase {
            Phase::Committed => {
                ensure!(
                    target_state == journal.new && matches!(stage_state, State::Absent),
                    "committed output is incomplete; preserve recovery files"
                );
                if let Some(backup) = &backup {
                    ensure!(
                        backup_state.as_ref() == Some(&journal.old)
                            || backup_state == Some(State::Absent),
                        "previous output backup changed"
                    );
                    if backup_state != Some(State::Absent) {
                        self.remove_verified(backup, &journal.old, cancelled)?;
                    }
                }
            }
            Phase::Prepared => {
                if target_state == journal.new {
                    ensure!(
                        matches!(stage_state, State::Absent),
                        "ambiguous output recovery state"
                    );
                    if let Some(backup) = &backup {
                        ensure!(
                            backup_state.as_ref() == Some(&journal.old),
                            "previous output backup is missing"
                        );
                        fs::rename(&self.path, &stage).context(OperationalError(
                            "cannot move uncommitted output back to staging",
                        ))?;
                        fs::rename(backup, &self.path)
                            .context(OperationalError("cannot restore previous output"))?;
                    } else {
                        fs::rename(&self.path, &stage).context(OperationalError(
                            "cannot move uncommitted output back to staging",
                        ))?;
                    }
                } else if matches!(target_state, State::Absent) && backup.is_some() {
                    let backup = backup.as_ref().expect("checked backup");
                    ensure!(
                        backup_state.as_ref() == Some(&journal.old),
                        "previous output backup is missing"
                    );
                    fs::rename(backup, &self.path)
                        .context(OperationalError("cannot restore previous output"))?;
                } else {
                    ensure!(
                        target_state == journal.old,
                        "output changed during recovery; preserve all files"
                    );
                    if let Some(backup) = &backup {
                        ensure!(
                            backup_state.as_ref() == placeholder.as_ref()
                                || backup_state == Some(State::Absent),
                            "ambiguous previous output backup"
                        );
                        if backup_state != Some(State::Absent) {
                            self.remove_verified(
                                backup,
                                placeholder.as_ref().expect("checked placeholder"),
                                cancelled,
                            )?;
                        }
                    }
                }
                ensure!(
                    recovery_read(self.state(&self.path, cancelled))?.0 == journal.old,
                    "previous output was not restored"
                );
                if !matches!(
                    recovery_read(self.cleanup_state(&stage, &journal.new, cancelled))?,
                    State::Absent
                ) {
                    self.remove_verified(&stage, &journal.new, cancelled)?;
                }
            }
        }
        // Refuse to unlink a replaced recovery record, even after recovery.
        ensure!(
            recovery_read(read_regular(&self.journal, 64 * 1024, &self.bytes_read))? == bytes,
            "output recovery record changed"
        );
        check_cancelled(cancelled)?;
        fs::remove_file(&self.journal).context(OperationalError(
            "cannot remove completed output recovery record",
        ))?;
        sync_directory(&self.parent)
    }

    fn transaction_path(&self, name: &str, kind: &str) -> Result<PathBuf> {
        ensure!(
            single_filename(name) && name.starts_with(&format!("{}-{kind}-", self.prefix)),
            "recovery path escapes the output parent"
        );
        let path = self.parent.join(name);
        ensure!(
            storage::checked_path(&path)? == path,
            "recovery path changed"
        );
        Ok(path)
    }
}

fn checked_target(path: &Path, protected_paths: &[&Path]) -> Result<(PathBuf, PathBuf, String)> {
    let path = storage::checked_path(path)?;
    ensure!(
        !path
            .components()
            .any(|component| component.as_os_str() == LOCK_DIRECTORY),
        "output path overlaps the reserved lock directory"
    );
    let parent = path
        .parent()
        .context("output requires an existing parent directory")?
        .to_owned();
    ensure!(
        parent.is_dir(),
        "output requires an existing parent directory"
    );
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("output filename must be UTF-8")?;
    ensure!(!name.is_empty(), "output directory name is required");
    for protected in protected_paths {
        let protected = storage::checked_path(protected)?;
        ensure!(
            !path.starts_with(&protected) && !protected.starts_with(&path),
            "input, config, cache and output paths must not overlap"
        );
    }
    let prefix = format!(".gsp-output-{}", storage::sha256(name.as_bytes()));
    Ok((path, parent, prefix))
}

fn checked_lock_directory(
    path: &Path,
    parent: &Path,
    protected_paths: &[&Path],
) -> Result<PathBuf> {
    let directory = storage::checked_path(&parent.join(LOCK_DIRECTORY))?;
    ensure!(
        !path.starts_with(&directory) && !directory.starts_with(path),
        "output path overlaps the reserved lock directory"
    );
    for protected in protected_paths {
        let protected = storage::checked_path(protected)?;
        ensure!(
            !protected.starts_with(&directory) && !directory.starts_with(&protected),
            "protected path overlaps the reserved output lock directory"
        );
    }
    match fs::symlink_metadata(&directory) {
        Ok(metadata) => ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "output lock container must be an ordinary directory"
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(directory)
}

fn validate_native_aliases(path: &Path, directory: &Path, protected_paths: &[&Path]) -> Result<()> {
    ensure!(
        !storage::native_sibling_names_equal(path, directory)?,
        "output path overlaps the reserved lock directory"
    );
    for protected in protected_paths {
        let protected = storage::checked_path(protected)?;
        ensure!(
            !storage::native_sibling_names_equal(path, &protected)?,
            "input, config, cache and output paths must not overlap"
        );
        ensure!(
            !storage::native_sibling_names_equal(&protected, directory)?,
            "protected path overlaps the reserved output lock directory"
        );
    }
    Ok(())
}

fn native_target(
    parent: &Path,
    representative: &Path,
    protected_paths: &[&Path],
) -> Result<(PathBuf, String)> {
    let name = representative
        .file_name()
        .context("output lock filename is required")?;
    let (path, _, _) = checked_target(&parent.join(name), protected_paths)?;
    // Keep the transaction namespace tied to the held lock, even if a user
    // changes only the output directory's case while a journal is pending.
    let prefix = format!(
        ".gsp-output-{}",
        storage::sha256(filename(representative)?.as_bytes())
    );
    let directory = representative
        .parent()
        .context("output lock parent is required")?;
    for protected in protected_paths {
        let protected = storage::checked_path(protected)?;
        if protected.parent() == Some(parent) {
            let lookup = directory.join(
                protected
                    .file_name()
                    .context("protected path filename is required")?,
            );
            match fs::symlink_metadata(&lookup) {
                Ok(_) => ensure!(
                    storage::checked_path(&lookup)? != representative,
                    "input, config, cache and output paths must not overlap"
                ),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
    }
    Ok((path, prefix))
}

fn filename(path: &Path) -> Result<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .context("output path name must be UTF-8")
}

fn single_filename(name: &str) -> bool {
    let mut components = Path::new(name).components();
    matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
}

fn check_cancelled(cancelled: &AtomicBool) -> Result<()> {
    if cancelled.load(Ordering::Relaxed) {
        return Err(Interrupted.into());
    }
    Ok(())
}

fn identity(path: &Path) -> Result<Identity> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let metadata = fs::symlink_metadata(path)?;
        ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "output must be an ordinary directory"
        );
        Ok(Identity {
            device: metadata.dev(),
            inode: metadata.ino(),
        })
    }
    #[cfg(not(unix))]
    bail!("output ownership is currently supported on macOS and Linux")
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .context(OperationalError(
            "cannot open output directory for synchronization",
        ))?
        .sync_all()
        .context(OperationalError("cannot synchronize output directory"))
}

fn recovery_read<T>(result: Result<T>) -> Result<T> {
    result.map_err(|error| {
        if error.downcast_ref::<std::io::Error>().is_some_and(|io| {
            io.kind() != std::io::ErrorKind::NotFound
                && !matches!(io.raw_os_error(), Some(libc::ELOOP | libc::ENOTDIR))
        }) {
            error.context(OperationalError("cannot read output recovery data"))
        } else {
            error
        }
    })
}

fn read_regular(path: &Path, max: u64, bytes_read: &Cell<u64>) -> Result<Vec<u8>> {
    let bytes = storage::read_regular(path, max)?;
    bytes_read.set(bytes_read.get() + bytes.len() as u64);
    Ok(bytes)
}

struct CountedFile<'a> {
    file: File,
    bytes_read: &'a Cell<u64>,
}

impl Read for CountedFile<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let count = self.file.read(buffer)?;
        self.bytes_read.set(self.bytes_read.get() + count as u64);
        Ok(count)
    }
}

impl Seek for CountedFile<'_> {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.file.seek(position)
    }
}

fn expected_files(report: &Value) -> Result<BTreeSet<String>> {
    ensure!(
        matches!(
            report["method"].as_str(),
            Some("sampled_frames" | "semantic_video")
        ),
        "unknown Rust output report method"
    );
    let count = report["selected_count"]
        .as_u64()
        .context("missing selected count")?;
    ensure!(
        (1..=999).contains(&count) && report["requested_count"].as_u64() == Some(count),
        "incomplete Rust output report"
    );
    let selected = report["selected"]
        .as_array()
        .context("missing selected image records")?;
    ensure!(
        selected.len() as u64 == count,
        "selected image count does not match report"
    );
    let mut files = BTreeSet::from(["selected-contact-sheet.jpg".to_owned()]);
    let width = if count < 100 { 2 } else { 3 };
    for (ordinal, frame) in selected.iter().enumerate() {
        let rank = ordinal + 1;
        let name = format!("selected-{rank:0width$}.jpg");
        ensure!(
            frame["rank"].as_u64() == Some(rank as u64) && frame["image"].as_str() == Some(&name),
            "selected image rank or filename is inconsistent"
        );
        files.insert(name);
    }
    Ok(files)
}

fn directory_files(path: &Path) -> Result<BTreeSet<String>> {
    let mut files = BTreeSet::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        ensure!(
            entry.file_type()?.is_file(),
            "output contains a symlink, directory or special file; preserve it and choose a new empty output directory"
        );
        files.insert(
            entry
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("output contains a non-UTF-8 filename"))?,
        );
        ensure!(
            files.len() <= 1001,
            "output has too many files; preserve it and choose a new empty output directory"
        );
    }
    Ok(files)
}

#[cfg(test)]
fn state(path: &Path) -> Result<(State, Option<OwnedOutput>)> {
    inspect(path, false, &Cell::new(0), &AtomicBool::new(false))
}

fn inspect(
    path: &Path,
    allow_missing: bool,
    bytes_read_total: &Cell<u64>,
    cancelled: &AtomicBool,
) -> Result<(State, Option<OwnedOutput>)> {
    check_cancelled(cancelled)?;
    let identity = match fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok((State::Absent, None));
        }
        Err(error) => return Err(error.into()),
        Ok(_) => identity(path)?,
    };
    let files = directory_files(path)?;
    if files.is_empty() {
        return Ok((State::Empty { identity }, None));
    }
    let bytes = read_regular(&path.join("report.json"), MAX_REPORT_BYTES, bytes_read_total).context("output has no valid Rust ownership report; preserve it and choose a new empty output directory")?;
    check_cancelled(cancelled)?;
    let mut report: Value = serde_json::from_slice(&bytes).context(
        "output has an invalid report; preserve it and choose a new empty output directory",
    )?;
    let publication = report.as_object_mut().context("output report must be an object")?.remove("publication").context("output has an unknown ownership report; preserve it and choose a new empty output directory")?;
    let manifest: Manifest = serde_json::from_value(publication.clone())
        .context("unknown Rust output ownership schema")?;
    ensure!(
        manifest.namespace == NAMESPACE && manifest.schema == SCHEMA,
        "unknown Rust output ownership namespace or schema"
    );
    ensure!(
        manifest.report_sha256 == storage::sha256(&serde_json::to_vec(&report)?),
        "output report was edited; preserve it and choose a new empty output directory"
    );
    let mut expected = expected_files(&report)?;
    ensure!(
        manifest.files.keys().cloned().collect::<BTreeSet<_>>() == expected,
        "output ownership image set does not match report"
    );
    expected.insert("report.json".to_owned());
    ensure!(
        if allow_missing {
            files.is_subset(&expected)
        } else {
            files == expected
        },
        "output contains unknown or missing files; preserve it and choose a new empty output directory"
    );
    let mut bytes_read = bytes.len() as u64;
    for (name, wanted) in &manifest.files {
        check_cancelled(cancelled)?;
        if allow_missing && !files.contains(name) {
            continue;
        }
        ensure!(
            storage::open_regular(&path.join(name))?.metadata()?.len() == wanted.bytes,
            "output image {name:?} size changed; preserve it and choose a new empty output directory"
        );
        let (length, digest) =
            storage::digest_file_cancelled(&path.join(name), cancelled, |bytes| {
                bytes_read_total.set(bytes_read_total.get() + bytes);
            })?;
        ensure!(
            length == wanted.bytes && digest == wanted.sha256,
            "output image {name:?} was edited or damaged; preserve it and choose a new empty output directory"
        );
        bytes_read += length;
    }
    check_cancelled(cancelled)?;
    report
        .as_object_mut()
        .expect("checked object")
        .insert("publication".to_owned(), publication);
    let fingerprint = storage::sha256(&bytes);
    Ok((
        State::Owned {
            identity,
            fingerprint: fingerprint.clone(),
        },
        Some(OwnedOutput {
            report,
            fingerprint,
            bytes_read,
        }),
    ))
}

fn stage_artifacts(
    source: &Path,
    target: &Path,
    record: &Value,
    cancelled: &AtomicBool,
    bytes_read: &Cell<u64>,
) -> Result<()> {
    let source = storage::checked_path(source)?;
    let report_bytes = read_regular(&source.join("report.json"), MAX_REPORT_BYTES, bytes_read)?;
    let mut report: Value = serde_json::from_slice(&report_bytes)?;
    ensure!(
        report.get("publication").is_none(),
        "private artifacts already contain publication metadata"
    );
    // Session adds only this top-level metadata after pipeline completion.
    // Bind every selection field to the immutable pipeline record at the same
    // read whose contents will be signed, avoiding a separate preflight read.
    let cache = report
        .as_object_mut()
        .context("private artifact report must be an object")?
        .remove("cache");
    ensure!(
        report == *record,
        "private report does not match the artifact record"
    );
    if let Some(cache) = cache {
        report
            .as_object_mut()
            .expect("checked object")
            .insert("cache".to_owned(), cache);
    }
    let expected = expected_files(&report)?;
    let mut all = expected.clone();
    all.insert("report.json".to_owned());
    ensure!(
        directory_files(&source)? == all,
        "private artifacts have missing or unexpected files"
    );
    let mut files = BTreeMap::new();
    for name in expected {
        check_cancelled(cancelled)?;
        let staged = target.join(&name);
        let selected = name != "selected-contact-sheet.jpg";
        let max_bytes = if selected {
            crate::quality::MAX_SELECTED_JPEG_BYTES
        } else {
            u64::MAX
        };
        let digest = copy_image(
            &source.join(&name),
            &staged,
            cancelled,
            bytes_read,
            max_bytes,
        )?;
        let decoder = artifact_decoder(&staged, selected, bytes_read)?;
        let image = image::DynamicImage::from_decoder(decoder)
            .context("private artifact is not a decodable JPEG")?;
        if let Some(frame) = report["selected"]
            .as_array()
            .expect("validated selected records")
            .iter()
            .find(|frame| frame["image"].as_str() == Some(&name))
        {
            ensure!(
                frame["width"].as_u64() == Some(u64::from(image.width()))
                    && frame["height"].as_u64() == Some(u64::from(image.height())),
                "private image dimensions do not match report"
            );
        }
        check_cancelled(cancelled)?;
        files.insert(name, digest);
    }
    let manifest = Manifest {
        namespace: NAMESPACE.to_owned(),
        schema: SCHEMA,
        report_sha256: storage::sha256(&serde_json::to_vec(&report)?),
        files,
    };
    report
        .as_object_mut()
        .context("private artifact report must be an object")?
        .insert("publication".to_owned(), serde_json::to_value(manifest)?);
    check_cancelled(cancelled)?;
    write_exclusive(
        &target.join("report.json"),
        &serde_json::to_vec_pretty(&report)?,
    )?;
    sync_directory(target)
}

fn artifact_decoder<'a>(
    path: &Path,
    selected: bool,
    bytes_read: &'a Cell<u64>,
) -> Result<impl ImageDecoder + 'a> {
    let file = storage::open_regular(path)?;
    if selected {
        ensure!(
            file.metadata()?.len() <= crate::quality::MAX_SELECTED_JPEG_BYTES,
            "selected JPEG exceeds the encoded byte limit"
        );
    }
    let mut reader = image::ImageReader::with_format(
        BufReader::new(CountedFile { file, bytes_read }),
        image::ImageFormat::Jpeg,
    );
    let max_pixels = if selected {
        crate::quality::MAX_IMAGE_PIXELS
    } else {
        crate::contact_sheet::MAX_SHEET_PIXELS
    };
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(4 * max_pixels);
    if selected {
        limits.max_image_width = Some(crate::quality::MAX_IMAGE_EDGE);
        limits.max_image_height = Some(crate::quality::MAX_IMAGE_EDGE);
    }
    reader.limits(limits.clone());
    let mut decoder = reader
        .into_decoder()
        .context("private artifact has an invalid JPEG header")?;
    let (width, height) = decoder.dimensions();
    ensure!(
        u64::from(width) * u64::from(height) <= max_pixels,
        "private JPEG exceeds the pixel limit"
    );
    limits.reserve(decoder.total_bytes())?;
    decoder.set_limits(limits)?;
    Ok(decoder)
}

fn write_exclusive(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn copy_image(
    source: &Path,
    target: &Path,
    cancelled: &AtomicBool,
    bytes_read: &Cell<u64>,
    max_bytes: u64,
) -> Result<Digest> {
    let mut source = CountedFile {
        file: storage::open_regular(source)?,
        bytes_read,
    };
    ensure!(
        source.file.metadata()?.len() <= max_bytes,
        "selected JPEG exceeds the encoded byte limit"
    );
    let mut target = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(target)?;
    let digest = copy_jpeg_stream(&mut source, &mut target, cancelled, max_bytes)?;
    target.sync_all()?;
    Ok(digest)
}

fn copy_jpeg_stream(
    mut source: impl Read,
    mut target: impl Write,
    cancelled: &AtomicBool,
    max_bytes: u64,
) -> Result<Digest> {
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut bytes = 0u64;
    let mut first = Vec::with_capacity(2);
    let mut last = [0u8; 2];
    loop {
        check_cancelled(cancelled)?;
        // One extra byte detects growth past the cap without publishing a
        // truncated JPEG or copying/hash-reading the remainder of the file.
        let read_limit = max_bytes
            .saturating_sub(bytes)
            .saturating_add(1)
            .min(buffer.len() as u64) as usize;
        let read = source.read(&mut buffer[..read_limit])?;
        if read == 0 {
            break;
        }
        ensure!(
            bytes + read as u64 <= max_bytes,
            "selected JPEG exceeds the encoded byte limit"
        );
        if first.len() < 2 {
            first.extend_from_slice(&buffer[..read.min(2 - first.len())]);
        }
        last = if read >= 2 {
            [buffer[read - 2], buffer[read - 1]]
        } else {
            [last[1], buffer[0]]
        };
        target.write_all(&buffer[..read])?;
        hash.update(&buffer[..read]);
        bytes += read as u64;
    }
    ensure!(
        bytes >= 4 && first == [0xff, 0xd8] && last == [0xff, 0xd9],
        "private artifact is not a completed JPEG"
    );
    Ok(Digest {
        bytes,
        sha256: storage::digest_hex(&hash.finalize()),
    })
}

fn cleanup_state(
    path: &Path,
    expected: &State,
    bytes_read: &Cell<u64>,
    cancelled: &AtomicBool,
) -> Result<State> {
    let actual = inspect(path, true, bytes_read, cancelled)?.0;
    // A cleanup can fail after deleting the last report but before rmdir. The
    // original directory identity still proves ownership of that empty node.
    if let (
        State::Empty { identity: actual },
        State::Owned {
            identity: wanted, ..
        },
    ) = (&actual, expected)
        && actual == wanted
    {
        return Ok(expected.clone());
    }
    Ok(actual)
}

fn remove_verified(
    path: &Path,
    expected: &State,
    bytes_read: &Cell<u64>,
    verification_seconds: &Cell<f64>,
    cancelled: &AtomicBool,
) -> Result<()> {
    ensure!(
        &recovery_read(timed_cell(verification_seconds, || {
            cleanup_state(path, expected, bytes_read, cancelled)
        }))? == expected,
        "refusing to remove changed output recovery data"
    );
    // Never recursively remove an old directory: only its verified flat files.
    let files = recovery_read(directory_files(path))?;
    // Keep the manifest available if any image deletion fails. Recovery can
    // validate the remaining declared subset without trusting an unknown file.
    for name in files.iter().filter(|name| name.as_str() != "report.json") {
        check_cancelled(cancelled)?;
        fs::remove_file(path.join(name))
            .context(OperationalError("cannot remove verified recovery image"))?;
    }
    if files.contains("report.json") {
        check_cancelled(cancelled)?;
        fs::remove_file(path.join("report.json"))
            .context(OperationalError("cannot remove verified recovery report"))?;
    }
    check_cancelled(cancelled)?;
    fs::remove_dir(path).context(OperationalError(
        "cannot remove verified recovery directory",
    ))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};
    use serde_json::json;

    pub(super) fn report_record(source: &Path) -> Value {
        let mut record: Value =
            serde_json::from_slice(&fs::read(source.join("report.json")).unwrap()).unwrap();
        record.as_object_mut().unwrap().remove("cache");
        record
    }

    #[test]
    fn copied_jpeg_digest_preserves_known_lowercase_hex() {
        let jpeg = [0xff, 0xd8, 0xff, 0xd9];
        let mut copied = Vec::new();
        let digest = copy_jpeg_stream(&jpeg[..], &mut copied, &cancelled(), 4).unwrap();
        assert_eq!(copied, jpeg);
        assert_eq!(digest.bytes, 4);
        assert_eq!(
            digest.sha256,
            "32461d5bd1773012acef0ba15636752949bd7c2ce50f9172159d9f56cf0dd9af"
        );
    }

    #[test]
    fn oversized_encoded_selected_jpeg_is_rejected_without_reading_it() {
        use crate::quality::MAX_SELECTED_JPEG_BYTES;
        let (_directory, root) = workspace();
        let source = artifacts(&root, "padded", 15);
        let path = root.join("result");
        let mut output = Output::prepare(&path, &[], &cancelled()).unwrap();
        let original = publish(&mut output, &source);
        let image_path = source.join("selected-01.jpg");
        let mut file = OpenOptions::new().write(true).open(&image_path).unwrap();
        file.set_len(MAX_SELECTED_JPEG_BYTES + 1).unwrap();
        file.seek(SeekFrom::End(-2)).unwrap();
        file.write_all(&[0xff, 0xd9]).unwrap();
        let decoder_reads = Cell::new(0);
        assert!(artifact_decoder(&image_path, true, &decoder_reads).is_err());
        assert_eq!(
            decoder_reads.get(),
            0,
            "reject before the JPEG decoder allocates its encoded buffer"
        );
        let before = output.bytes_read();
        let error = output
            .publish_directory(&source, &cancelled(), |_| Ok(()))
            .unwrap_err();
        assert!(
            error.to_string().contains("encoded byte limit"),
            "{error:#}"
        );
        assert!(output.bytes_read() - before < original.bytes_read + 64 * 1024);
        assert_eq!(
            output
                .verify_existing(&cancelled())
                .unwrap()
                .unwrap()
                .fingerprint,
            original.fingerprint
        );
        assert!(!output.journal.exists());
    }

    #[test]
    fn encoded_copy_accepts_the_limit_and_rejects_growth_without_truncation() {
        let (_directory, root) = workspace();
        let source = artifacts(&root, "near-limit", 15).join("selected-01.jpg");
        let original = fs::read(&source).unwrap();
        let limit = 64 * 1024 + 4;
        for size in [limit - 1, limit, limit + 1] {
            fs::write(&source, &original).unwrap();
            let mut writer = OpenOptions::new().write(true).open(&source).unwrap();
            writer.set_len(size).unwrap();
            writer.seek(SeekFrom::End(-2)).unwrap();
            writer.write_all(&[0xff, 0xd9]).unwrap();
            let target = root.join(format!("copied-{size}.jpg"));
            let reads = Cell::new(0);
            let result = copy_image(&source, &target, &cancelled(), &reads, limit);
            if size <= limit {
                let digest = result.unwrap();
                assert_eq!(digest.bytes, size);
                assert_eq!(reads.get(), size);
                assert_eq!(fs::read(&target).unwrap(), fs::read(&source).unwrap());
                assert!(image::open(target).is_ok());
            } else {
                assert!(
                    result
                        .unwrap_err()
                        .to_string()
                        .contains("encoded byte limit")
                );
                assert_eq!(reads.get(), 0);
                assert!(!target.exists());
            }
        }

        struct GrowingFile<'a> {
            reader: File,
            grow: Option<File>,
            size: u64,
            reads: &'a Cell<u64>,
        }
        impl Read for GrowingFile<'_> {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                let read = self.reader.read(buffer)?;
                self.reads.set(self.reads.get() + read as u64);
                if let Some(writer) = self.grow.take() {
                    writer.set_len(self.size)?;
                }
                Ok(read)
            }
        }
        fs::write(&source, &original).unwrap();
        let limit = original.len() as u64 + 1024;
        let reads = Cell::new(0);
        let reader = GrowingFile {
            reader: File::open(&source).unwrap(),
            grow: Some(OpenOptions::new().write(true).open(&source).unwrap()),
            size: limit + 1024,
            reads: &reads,
        };
        assert!(reader.reader.metadata().unwrap().len() < limit);
        let mut copied = Vec::new();
        let error = copy_jpeg_stream(reader, &mut copied, &cancelled(), limit).unwrap_err();
        assert!(error.to_string().contains("encoded byte limit"));
        assert_eq!(reads.get(), limit + 1);
        assert_eq!(
            copied, original,
            "growth is rejected before an over-limit write"
        );
    }

    #[test]
    fn producer_32mp_jpeg_above_candidate_limit_renders_and_publishes() {
        use crate::contact_sheet::{SheetEntry, render_selected_sheet};
        use crate::quality::{MAX_IMAGE_BYTES, MAX_SELECTED_JPEG_BYTES};
        let (_directory, root) = workspace();
        let source = artifacts(&root, "large-producer", 15);
        let mut seed = 59u64;
        let mut tile = [Rgb([0; 3]); 64];
        for pixel in &mut tile {
            for channel in &mut pixel.0 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                *channel = if ((seed >> 24) as u8) & 1 == 0 {
                    0
                } else {
                    255
                };
            }
        }
        let image = RgbImage::from_fn(8000, 4000, |x, y| tile[((y % 8) * 8 + x % 8) as usize]);
        let path = source.join("selected-01.jpg");
        crate::pipeline::encode_selected_jpeg(&image, &path).unwrap();
        drop(image);
        let bytes = fs::metadata(&path).unwrap().len();
        assert!(
            bytes > MAX_IMAGE_BYTES && bytes <= MAX_SELECTED_JPEG_BYTES,
            "32 MP quality-95 4:4:4 producer JPEG: {bytes} bytes"
        );
        assert!(
            crate::quality::read_image(&path).is_err(),
            "candidate budget stays separate"
        );
        let sheet = render_selected_sheet(
            &[SheetEntry {
                id: "large".to_owned(),
                rank: Some(1),
                source: "入力.mkv".to_owned(),
                requested_seconds: 1.0,
                actual_seconds: 1.25,
                scene: "32 MP".to_owned(),
                image: path,
                context: None,
            }],
            &source,
        )
        .unwrap();
        assert!(image::open(sheet).is_ok());
        let mut report: Value =
            serde_json::from_slice(&fs::read(source.join("report.json")).unwrap()).unwrap();
        report["selected"][0]["width"] = json!(8000);
        report["selected"][0]["height"] = json!(4000);
        fs::write(
            source.join("report.json"),
            serde_json::to_vec(&report).unwrap(),
        )
        .unwrap();
        let mut output = Output::prepare(&root.join("result"), &[], &cancelled()).unwrap();
        let owned = publish(&mut output, &source);
        assert_eq!(
            owned.report["publication"]["files"]["selected-01.jpg"]["bytes"],
            bytes
        );
        assert_eq!(
            output
                .verify_existing(&cancelled())
                .unwrap()
                .unwrap()
                .fingerprint,
            owned.fingerprint
        );
    }

    fn workspace() -> (tempfile::TempDir, PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().canonicalize().unwrap();
        (directory, path)
    }

    fn artifacts(root: &Path, name: &str, color: u8) -> PathBuf {
        let directory = root.join(name);
        fs::create_dir(&directory).unwrap();
        let image = RgbImage::from_fn(64, 40, |x, y| Rgb([color, (x * 3) as u8, (y * 5) as u8]));
        image.save(directory.join("selected-01.jpg")).unwrap();
        image
            .save(directory.join("selected-contact-sheet.jpg"))
            .unwrap();
        fs::write(directory.join("report.json"), serde_json::to_vec(&json!({
            "method": "sampled_frames", "requested_count": 1, "selected_count": 1,
            "sources": ["入力.mkv"], "cache": {"namespace": "game-screen-pick-rust-v1", "key": "fixture"},
            "selected": [{"rank": 1, "image": "selected-01.jpg", "source": "入力.mkv",
                "requested_seconds": 1.0, "actual_seconds": 1.25, "width": 64, "height": 40}]
        })).unwrap()).unwrap();
        directory
    }

    fn cancelled() -> AtomicBool {
        AtomicBool::new(false)
    }

    fn publish(output: &mut Output, source: &Path) -> OwnedOutput {
        output
            .publish_directory(source, &cancelled(), |_| Ok(()))
            .unwrap()
    }

    #[test]
    fn verified_output_does_not_depend_on_cache_and_detects_valid_jpeg_edits() {
        let (_directory, root) = workspace();
        let source = artifacts(&root, "artifacts", 15);
        let path = root.join("result");
        let cache = root.join("cache");
        fs::create_dir(&cache).unwrap();
        let mut output = Output::prepare(&path, &[&cache], &cancelled()).unwrap();
        let owned = publish(&mut output, &source);
        assert_eq!(owned.report["cache"]["key"], "fixture");
        assert!(owned.bytes_read > fs::metadata(path.join("report.json")).unwrap().len());
        drop(output);
        fs::remove_dir(&cache).unwrap();
        let output = Output::prepare(&path, &[&cache], &cancelled()).unwrap();
        assert_eq!(
            output
                .verify_existing(&cancelled())
                .unwrap()
                .unwrap()
                .fingerprint,
            owned.fingerprint
        );
        drop(output);
        let before_report = fs::read(path.join("report.json")).unwrap();
        RgbImage::from_pixel(64, 40, Rgb([7, 8, 9]))
            .save(path.join("selected-01.jpg"))
            .unwrap();
        let edited = fs::read(path.join("selected-01.jpg")).unwrap();
        assert!(image::open(path.join("selected-01.jpg")).is_ok());
        assert!(Output::prepare(&path, &[], &cancelled()).is_err());
        assert_eq!(fs::read(path.join("selected-01.jpg")).unwrap(), edited);
        assert_eq!(fs::read(path.join("report.json")).unwrap(), before_report);
    }

    #[test]
    fn unknown_files_reports_and_rank_mismatch_are_preserved() {
        let (_directory, root) = workspace();
        let unknown = root.join("unknown");
        fs::create_dir(&unknown).unwrap();
        fs::write(unknown.join("holiday.txt"), b"keep me").unwrap();
        assert!(Output::prepare(&unknown, &[], &cancelled()).is_err());
        assert_eq!(fs::read(unknown.join("holiday.txt")).unwrap(), b"keep me");
        let unowned = artifacts(&root, "unowned", 25);
        let before = fs::read(unowned.join("report.json")).unwrap();
        assert!(Output::prepare(&unowned, &[], &cancelled()).is_err());
        assert_eq!(fs::read(unowned.join("report.json")).unwrap(), before);

        let source = artifacts(&root, "artifacts", 15);
        let path = root.join("result");
        let mut output = Output::prepare(&path, &[], &cancelled()).unwrap();
        publish(&mut output, &source);
        fs::write(path.join("notes.txt"), b"my note").unwrap();
        assert!(output.verify_existing(&cancelled()).is_err());
        assert!(
            output
                .publish_directory(&source, &cancelled(), |_| Ok(()))
                .is_err()
        );
        assert_eq!(fs::read(path.join("notes.txt")).unwrap(), b"my note");
        fs::remove_file(path.join("notes.txt")).unwrap();
        let mut report: Value =
            serde_json::from_slice(&fs::read(path.join("report.json")).unwrap()).unwrap();
        report["selected"][0]["rank"] = json!(9);
        fs::write(
            path.join("report.json"),
            serde_json::to_vec(&report).unwrap(),
        )
        .unwrap();
        assert!(output.verify_existing(&cancelled()).is_err());
    }

    #[test]
    fn preflight_locks_reject_overlap_and_serialize_competing_runs() {
        let (_directory, root) = workspace();
        let input = root.join("input");
        fs::create_dir(&input).unwrap();
        let path = root.join("result");
        assert!(Output::prepare(&input.join("output"), &[&input], &cancelled()).is_err());
        assert!(Output::prepare(&root, &[&input], &cancelled()).is_err());
        assert!(Output::prepare(&path, &[&path], &cancelled()).is_err());
        let output = Output::prepare(&path, &[&input], &cancelled()).unwrap();
        assert!(Output::prepare(&path, &[&input], &cancelled()).is_err());
        drop(output);
        assert!(Output::prepare(&path, &[&input], &cancelled()).is_ok());
    }

    #[test]
    fn diagnostic_validation_never_creates_locks_or_recovers_output() {
        let (_directory, root) = workspace();
        let path = root.join("result");
        assert!(
            Output::validate(&path, &[], &cancelled())
                .unwrap()
                .is_none()
        );
        assert_eq!(fs::read_dir(&root).unwrap().count(), 0);
        let first = artifacts(&root, "first", 15);
        let second = artifacts(&root, "second", 85);
        let mut output = Output::prepare(&path, &[], &cancelled()).unwrap();
        let original = publish(&mut output, &first);
        assert_eq!(
            Output::validate(&path, &[], &cancelled())
                .unwrap()
                .unwrap()
                .fingerprint,
            original.fingerprint
        );
        let journal = pending(&output, &second);
        let before = fs::read(&output.journal).unwrap();
        assert!(Output::validate(&path, &[], &cancelled()).is_err());
        assert_eq!(fs::read(&output.journal).unwrap(), before);
        assert!(root.join(journal.stage).exists());
    }

    #[test]
    fn reserved_lock_paths_and_sidecar_collisions_are_rejected_without_deletion() {
        let (_directory, root) = workspace();
        let container = root.join(LOCK_DIRECTORY);
        assert!(Output::prepare(&container, &[], &cancelled()).is_err());
        assert!(!container.exists());
        fs::create_dir(&container).unwrap();
        assert!(Output::prepare(&container.join("nested-output"), &[], &cancelled()).is_err());
        assert_eq!(fs::read_dir(&container).unwrap().count(), 0);
        fs::write(container.join("input.mkv"), b"keep input").unwrap();
        assert!(Output::prepare(&root.join("result"), &[&container], &cancelled()).is_err());
        assert_eq!(
            fs::read(container.join("input.mkv")).unwrap(),
            b"keep input"
        );
        fs::remove_file(container.join("input.mkv")).unwrap();
        fs::remove_dir(&container).unwrap();
        fs::write(&container, b"user-owned file").unwrap();
        assert!(Output::prepare(&root.join("result"), &[], &cancelled()).is_err());
        assert_eq!(fs::read(&container).unwrap(), b"user-owned file");
    }

    #[test]
    fn actual_recovery_read_permission_failure_is_operational_but_invalid_data_is_not() {
        use std::os::unix::fs::PermissionsExt;
        let (_directory, root) = workspace();
        let first = artifacts(&root, "first", 15);
        let second = artifacts(&root, "second", 85);
        let path = root.join("result");
        let mut output = Output::prepare(&path, &[], &cancelled()).unwrap();
        let original = publish(&mut output, &first);
        let journal = pending(&output, &second);
        let stage_report = root.join(journal.stage).join("report.json");
        let permission = fs::metadata(&stage_report).unwrap().permissions();
        fs::set_permissions(&stage_report, fs::Permissions::from_mode(0o0)).unwrap();
        let result = output.recover(&cancelled());
        fs::set_permissions(&stage_report, permission).unwrap();
        let error = result.unwrap_err();
        assert!(
            error.downcast_ref::<OperationalError>().is_some(),
            "{error:#}"
        );
        assert_eq!(
            output
                .verify_existing(&cancelled())
                .unwrap()
                .unwrap()
                .fingerprint,
            original.fingerprint
        );
        fs::write(&output.journal, b"{\"unknown\": true}").unwrap();
        let error = output.recover(&cancelled()).unwrap_err();
        assert!(
            error.downcast_ref::<OperationalError>().is_none(),
            "{error:#}"
        );
        assert!(stage_report.exists());
    }

    #[test]
    fn renamed_pending_output_does_not_hide_its_original_recovery_record() {
        let (_directory, root) = workspace();
        let first = artifacts(&root, "first", 15);
        let second = artifacts(&root, "second", 85);
        let path = root.join("Result");
        let mut output = Output::prepare(&path, &[], &cancelled()).unwrap();
        let original = publish(&mut output, &first);
        let journal = pending(&output, &second);
        let journal_path = output.journal.clone();
        let record = fs::read(&journal_path).unwrap();
        fs::rename(&path, root.join("result")).unwrap();
        drop(output);
        // On a case-insensitive filesystem the target resolves to "result",
        // but its unchanged lock representative must still find the journal.
        // On a case-sensitive filesystem the original target is absent and
        // recovery also refuses to infer ownership of the renamed directory.
        assert!(Output::prepare(&path, &[], &cancelled()).is_err());
        assert_eq!(fs::read(&journal_path).unwrap(), record);
        assert!(root.join(journal.stage).exists());
        assert_eq!(
            state(&root.join("result")).unwrap().1.unwrap().fingerprint,
            original.fingerprint
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_special_files_and_traversal_are_rejected_without_removing_them() {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        use std::os::unix::fs::symlink;
        let (_directory, root) = workspace();
        let target = root.join("real");
        fs::create_dir(&target).unwrap();
        let link = root.join("link");
        symlink(&target, &link).unwrap();
        assert!(Output::prepare(&link, &[], &cancelled()).is_err());
        assert!(Output::prepare(&link.join("result"), &[], &cancelled()).is_err());
        assert!(Output::prepare(&root.join("result"), &[&link], &cancelled()).is_err());
        assert!(Output::prepare(&root.join("real/../result"), &[], &cancelled()).is_err());
        let fifo = root.join("fifo");
        let filename = CString::new(fifo.as_os_str().as_bytes()).unwrap();
        // A FIFO must be inspected without a blocking read or a write.
        assert_eq!(unsafe { libc::mkfifo(filename.as_ptr(), 0o600) }, 0);
        assert!(Output::prepare(&fifo, &[], &cancelled()).is_err());
        assert!(fs::symlink_metadata(&fifo).is_ok());
        symlink(&fifo, target.join("selected-01.jpg")).unwrap();
        assert!(Output::prepare(&target, &[], &cancelled()).is_err());
        assert!(
            fs::symlink_metadata(target.join("selected-01.jpg"))
                .unwrap()
                .file_type()
                .is_symlink()
        );
    }

    #[test]
    fn staged_corrupt_jpegs_preserve_previous_output() {
        let (_directory, root) = workspace();
        let first = artifacts(&root, "first", 15);
        let path = root.join("result");
        let mut output = Output::prepare(&path, &[], &cancelled()).unwrap();
        let original = publish(&mut output, &first);
        for (ordinal, name) in ["selected-01.jpg", "selected-contact-sheet.jpg"]
            .into_iter()
            .enumerate()
        {
            let second = artifacts(&root, &format!("second-{ordinal}"), 85);
            fs::write(second.join(name), [0xff, 0xd8, 0xff, 0xd9]).unwrap();
            assert!(image::open(second.join(name)).is_err());
            assert!(
                output
                    .publish_directory(&second, &cancelled(), |_| Ok(()))
                    .is_err()
            );
            assert_eq!(
                output
                    .verify_existing(&cancelled())
                    .unwrap()
                    .unwrap()
                    .fingerprint,
                original.fingerprint
            );
            assert!(!output.journal.exists());
        }
    }

    #[test]
    fn cancelled_preflight_stops_during_output_digest() {
        assert_cancelled_ownership("prepare");
    }

    #[test]
    fn cancelled_diagnostic_stops_during_output_digest() {
        assert_cancelled_ownership("validate");
    }

    #[test]
    fn cancelled_warm_verification_stops_during_output_digest() {
        assert_cancelled_ownership("verify");
    }

    #[test]
    fn cancelled_pending_recovery_retains_its_journal_and_output() {
        assert_cancelled_ownership("recover");
    }

    fn assert_cancelled_ownership(operation: &str) {
        let (_directory, root) = workspace();
        let source = artifacts(&root, "artifacts", 15);
        let mut encoded = fs::read(source.join("selected-01.jpg")).unwrap();
        encoded.resize(256 * 1024, 0);
        encoded.extend([0xff, 0xd9]);
        fs::write(source.join("selected-01.jpg"), encoded).unwrap();
        assert!(image::open(source.join("selected-01.jpg")).is_ok());
        let path = root.join("result");
        let mut output = Some(Output::prepare(&path, &[], &cancelled()).unwrap());
        let original = publish(output.as_mut().unwrap(), &source);
        let snapshot = || {
            directory_files(&path)
                .unwrap()
                .into_iter()
                .map(|name| {
                    let bytes = fs::read(path.join(&name)).unwrap();
                    (name, bytes)
                })
                .collect::<BTreeMap<_, _>>()
        };
        let before = snapshot();
        let journal = (operation == "recover").then(|| pending(output.as_ref().unwrap(), &source));
        let journal_path = output.as_ref().unwrap().journal.clone();
        let journal_bytes = journal.as_ref().map(|_| fs::read(&journal_path).unwrap());
        let bytes_read_before = output.as_ref().unwrap().bytes_read();
        let partial_bytes = before["report.json"].len() as u64
            + 64 * 1024
            + journal_bytes.as_ref().map_or(0, |bytes| bytes.len() as u64);
        let flag = AtomicBool::new(false);
        let interruption = storage::interrupt_digest_after(64 * 1024);
        let result = match operation {
            "prepare" => {
                drop(output.take());
                Output::prepare(&path, &[], &flag).map(|_| ())
            }
            "validate" => Output::validate(&path, &[], &flag).map(|_| ()),
            "verify" => output.as_ref().unwrap().verify_existing(&flag).map(|_| ()),
            "recover" => output.as_ref().unwrap().recover(&flag),
            _ => unreachable!(),
        };
        drop(interruption);
        assert!(
            flag.load(Ordering::Relaxed),
            "the caller's flag must change during the actual digest"
        );
        let error = result.unwrap_err();
        assert!(error.downcast_ref::<Interrupted>().is_some(), "{error:#}");
        assert_eq!(snapshot(), before);
        if matches!(operation, "verify" | "recover") {
            assert_eq!(
                output.as_ref().unwrap().bytes_read() - bytes_read_before,
                partial_bytes,
                "count the report and the image chunk read before interruption"
            );
        }
        if operation == "verify" {
            let output = output.as_ref().unwrap();
            let retried = output.verify_existing(&cancelled()).unwrap().unwrap();
            assert_eq!(
                output.bytes_read() - bytes_read_before,
                partial_bytes + retried.bytes_read,
                "a retry on the same output keeps the interrupted read cost"
            );
        }
        if let Some(journal) = journal {
            let bytes = fs::read(&journal_path).unwrap();
            assert_eq!(Some(&bytes), journal_bytes.as_ref());
            assert!(serde_json::from_slice::<Journal>(&bytes).is_ok());
            assert!(root.join(journal.stage).join("report.json").is_file());
            let output = output.as_ref().unwrap();
            output.recover(&cancelled()).unwrap();
            // This prepared transaction checks the target twice and the stage
            // three times, then reads the unchanged journal before unlinking.
            let retry_bytes = 5 * original.bytes_read + 2 * bytes.len() as u64;
            assert_eq!(
                output.bytes_read() - bytes_read_before,
                partial_bytes + retry_bytes,
                "recovery on the same output keeps the interrupted read cost"
            );
        }
        drop(output);
        let output = Output::prepare(&path, &[], &cancelled()).unwrap();
        assert_eq!(
            output
                .verify_existing(&cancelled())
                .unwrap()
                .unwrap()
                .fingerprint,
            original.fingerprint
        );
    }

    #[test]
    fn selected_jpeg_headers_are_bounded_before_allocating_pixels() {
        let (_directory, root) = workspace();
        let source = artifacts(&root, "headers", 15);
        let original = fs::read(source.join("selected-01.jpg")).unwrap();
        let marker = original
            .windows(2)
            .position(|bytes| bytes == [0xff, 0xc0])
            .unwrap();
        let path = source.join("selected-01.jpg");
        let output_path = root.join("result");
        let mut output = Output::prepare(&output_path, &[], &cancelled()).unwrap();
        let owned = publish(&mut output, &source);
        let dimensions = |width: u16, height: u16| {
            let mut bytes = original.clone();
            bytes[marker + 5..marker + 7].copy_from_slice(&height.to_be_bytes());
            bytes[marker + 7..marker + 9].copy_from_slice(&width.to_be_bytes());
            fs::write(&path, bytes).unwrap();
        };
        for (width, height) in [(6000, 6000), (16_385, 16), (16, 16_385)] {
            dimensions(width, height);
            assert!(
                artifact_decoder(&path, true, &Cell::new(0)).is_err(),
                "oversized selected header {width}x{height} must fail before pixel decode"
            );
            assert!(
                output
                    .publish_directory(&source, &cancelled(), |_| Ok(()))
                    .is_err()
            );
            assert_eq!(
                output
                    .verify_existing(&cancelled())
                    .unwrap()
                    .unwrap()
                    .fingerprint,
                owned.fingerprint
            );
            assert!(!output.journal.exists());
        }
        for (width, height) in [(8000, 4000), (16_384, 16), (10_000, 20_000)] {
            dimensions(width, height);
            let selected = height <= 16_384;
            let bytes_read = Cell::new(0);
            let decoder = artifact_decoder(&path, selected, &bytes_read).unwrap();
            assert_eq!(decoder.dimensions(), (u32::from(width), u32::from(height)));
            assert!(bytes_read.get() < 1024);
            // Dropping the header decoder never allocates the advertised pixels.
        }
        dimensions(10_001, 20_000);
        assert!(
            artifact_decoder(&path, false, &Cell::new(0)).is_err(),
            "contact sheet header above 200MP must fail before pixel decode"
        );
    }

    #[test]
    fn staged_image_dimensions_must_match_report() {
        let (_directory, root) = workspace();
        let first = artifacts(&root, "first", 15);
        let second = artifacts(&root, "second", 85);
        let path = root.join("result");
        let mut output = Output::prepare(&path, &[], &cancelled()).unwrap();
        let original = publish(&mut output, &first);
        RgbImage::from_pixel(32, 40, Rgb([7, 8, 9]))
            .save(second.join("selected-01.jpg"))
            .unwrap();
        assert!(image::open(second.join("selected-01.jpg")).is_ok());
        assert!(
            output
                .publish_directory(&second, &cancelled(), |_| Ok(()))
                .is_err()
        );
        assert_eq!(
            output
                .verify_existing(&cancelled())
                .unwrap()
                .unwrap()
                .fingerprint,
            original.fingerprint
        );
        assert!(!output.journal.exists());
    }

    #[test]
    fn staging_failures_and_cancellation_restore_the_previous_complete_output() {
        let (_directory, root) = workspace();
        let first = artifacts(&root, "first", 15);
        let second = artifacts(&root, "second", 85);
        for (ordinal, failure) in ["staging", "prepared", "backed_up", "published"]
            .iter()
            .enumerate()
        {
            let path = root.join(format!("result-{ordinal}"));
            let mut output = Output::prepare(&path, &[], &cancelled()).unwrap();
            let original = publish(&mut output, &first);
            let error = output
                .publish_directory(&second, &cancelled(), |step| {
                    ensure!(step != *failure, "injected {step} failure");
                    Ok(())
                })
                .unwrap_err();
            assert!(error.to_string().contains("injected"), "{error:#}");
            assert_eq!(
                output
                    .verify_existing(&cancelled())
                    .unwrap()
                    .unwrap()
                    .fingerprint,
                original.fingerprint
            );
            assert!(!output.journal.exists());
            assert!(fs::read_dir(&root).unwrap().all(|entry| {
                !entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .contains(&format!("{}-stage-", output.prefix))
            }));

            let cancel = cancelled();
            let error = output
                .publish_directory(&second, &cancel, |step| {
                    if step == *failure {
                        cancel.store(true, Ordering::Relaxed);
                    }
                    Ok(())
                })
                .unwrap_err();
            assert!(error.downcast_ref::<Interrupted>().is_some(), "{error:#}");
            assert_eq!(
                output
                    .verify_existing(&cancelled())
                    .unwrap()
                    .unwrap()
                    .fingerprint,
                original.fingerprint
            );
        }
    }

    #[test]
    fn new_and_empty_outputs_are_restored_on_cancel() {
        let (_directory, root) = workspace();
        let source = artifacts(&root, "artifacts", 15);
        for empty in [false, true] {
            let path = root.join(if empty { "empty" } else { "new" });
            if empty {
                fs::create_dir(&path).unwrap();
            }
            let mut output = Output::prepare(&path, &[], &cancelled()).unwrap();
            let before = state(&path).unwrap().0;
            let cancel = cancelled();
            let error = output
                .publish_directory(&source, &cancel, |step| {
                    if step == "published" {
                        cancel.store(true, Ordering::Relaxed);
                    }
                    Ok(())
                })
                .unwrap_err();
            assert!(error.downcast_ref::<Interrupted>().is_some());
            assert_eq!(state(&path).unwrap().0, before);
        }
    }

    #[test]
    fn failed_rename_keeps_old_backup_and_recovers_after_conflict_is_removed() {
        let (_directory, root) = workspace();
        let first = artifacts(&root, "first", 15);
        let second = artifacts(&root, "second", 85);
        let path = root.join("result");
        let mut output = Output::prepare(&path, &[], &cancelled()).unwrap();
        let original = publish(&mut output, &first);
        // A regular file at the destination makes the actual directory rename
        // fail on both target OSes. Recovery must retain this unknown file.
        assert!(
            output
                .publish_directory(&second, &cancelled(), |step| {
                    if step == "backed_up" {
                        fs::write(&path, b"user conflict")?;
                    }
                    Ok(())
                })
                .is_err()
        );
        assert_eq!(fs::read(&path).unwrap(), b"user conflict");
        let journal: Journal = serde_json::from_slice(&fs::read(&output.journal).unwrap()).unwrap();
        let backup = root.join(journal.backup.unwrap());
        assert_eq!(
            state(&backup).unwrap().1.unwrap().fingerprint,
            original.fingerprint
        );
        assert!(root.join(journal.stage).exists());
        drop(output);
        fs::remove_file(&path).unwrap(); // Only the test-created conflict is removed.
        let output = Output::prepare(&path, &[], &cancelled()).unwrap();
        assert_eq!(
            output
                .verify_existing(&cancelled())
                .unwrap()
                .unwrap()
                .fingerprint,
            original.fingerprint
        );
        assert!(!backup.exists());
    }

    #[test]
    fn edits_during_staging_are_kept_instead_of_being_overwritten() {
        let (_directory, root) = workspace();
        let first = artifacts(&root, "first", 15);
        let second = artifacts(&root, "second", 85);
        let path = root.join("result");
        let mut output = Output::prepare(&path, &[], &cancelled()).unwrap();
        publish(&mut output, &first);
        let edited = fs::read(second.join("selected-01.jpg")).unwrap();
        assert!(
            output
                .publish_directory(&second, &cancelled(), |step| {
                    if step == "staging" {
                        fs::write(path.join("selected-01.jpg"), &edited)?;
                    }
                    Ok(())
                })
                .is_err()
        );
        assert_eq!(fs::read(path.join("selected-01.jpg")).unwrap(), edited);
        assert!(!output.journal.exists());
    }

    #[test]
    fn input_changes_during_staging_or_publish_verify_do_not_replace_old_output() {
        use std::cell::Cell;
        let (_directory, root) = workspace();
        let first = artifacts(&root, "first", 15);
        let second = artifacts(&root, "second", 85);
        let input = root.join("input.mkv");
        for boundary in ["staged", "backed_up", "verified"] {
            fs::write(&input, b"original input").unwrap();
            let path = root.join(format!("result-{boundary}"));
            let mut output = Output::prepare(&path, &[&input], &cancelled()).unwrap();
            let original = publish(&mut output, &first);
            let checks = Cell::new(0);
            let error = output
                .publish_directory_checked(
                    &second,
                    &report_record(&second),
                    &cancelled(),
                    || {
                        checks.set(checks.get() + 1);
                        ensure!(
                            fs::read(&input)? == b"original input",
                            "input changed during publication"
                        );
                        Ok(())
                    },
                    |step| {
                        if step == boundary {
                            fs::write(&input, b"changed during staging or verification")?;
                        }
                        Ok(())
                    },
                )
                .unwrap_err();
            assert!(error.to_string().contains("input changed"), "{error:#}");
            assert!(checks.get() >= 1);
            assert_eq!(
                output
                    .verify_existing(&cancelled())
                    .unwrap()
                    .unwrap()
                    .fingerprint,
                original.fingerprint
            );
            assert!(!output.journal.exists());
        }
    }

    #[test]
    fn cumulative_output_bytes_include_staging_decoding_and_all_ownership_passes() {
        use std::cell::Cell;
        let (_directory, root) = workspace();
        let source = artifacts(&root, "artifacts", 15);
        let raw_report = fs::metadata(source.join("report.json")).unwrap().len();
        let images = fs::metadata(source.join("selected-01.jpg")).unwrap().len()
            + fs::metadata(source.join("selected-contact-sheet.jpg"))
                .unwrap()
                .len();
        let path = root.join("result");
        let mut output = Output::prepare(&path, &[], &cancelled()).unwrap();
        assert_eq!(output.bytes_read(), 0);
        let journal_bytes = Cell::new(0);
        let journal_path = output.journal.clone();
        let owned = output
            .publish_directory(&source, &cancelled(), |step| {
                if step == "committed" {
                    journal_bytes.set(fs::metadata(&journal_path)?.len());
                }
                Ok(())
            })
            .unwrap();
        // The staged report/images are verified, then the renamed output,
        // recovery target, saved state and returned output each read that set.
        // Source copy, staged JPEG decode and both journal reads are additional.
        let minimum = raw_report + 2 * images + 5 * owned.bytes_read + 2 * journal_bytes.get();
        assert!(
            output.bytes_read() >= minimum,
            "{} < {minimum}",
            output.bytes_read()
        );
        drop(output);
        let output = Output::prepare(&path, &[], &cancelled()).unwrap();
        assert_eq!(output.bytes_read(), owned.bytes_read);
        let before = output.bytes_read();
        output.verify_existing(&cancelled()).unwrap();
        assert_eq!(output.bytes_read() - before, owned.bytes_read);
    }

    fn pending(output: &Output, source: &Path) -> Journal {
        let stage = tempfile::Builder::new()
            .prefix(&format!("{}-stage-", output.prefix))
            .tempdir_in(&output.parent)
            .unwrap();
        stage_artifacts(
            source,
            stage.path(),
            &report_record(source),
            &cancelled(),
            &Cell::new(0),
        )
        .unwrap();
        let backup = tempfile::Builder::new()
            .prefix(&format!("{}-backup-", output.prefix))
            .tempdir_in(&output.parent)
            .unwrap();
        let journal = Journal {
            namespace: NAMESPACE.to_owned(),
            schema: SCHEMA,
            target: filename(&output.path).unwrap(),
            stage: filename(stage.path()).unwrap(),
            backup: Some(filename(backup.path()).unwrap()),
            placeholder: Some(identity(backup.path()).unwrap()),
            old: output.initial.clone(),
            new: state(stage.path()).unwrap().0,
            phase: Phase::Prepared,
        };
        storage::atomic_write(&output.journal, &serde_json::to_vec(&journal).unwrap()).unwrap();
        let _ = stage.keep();
        let _ = backup.keep();
        journal
    }

    #[test]
    fn reopening_recovers_each_recorded_precommit_rename_boundary() {
        let (_directory, root) = workspace();
        let first = artifacts(&root, "first", 15);
        let second = artifacts(&root, "second", 85);
        for boundary in 0..3 {
            let path = root.join(format!("result-{boundary}"));
            let mut output = Output::prepare(&path, &[], &cancelled()).unwrap();
            let original = publish(&mut output, &first);
            let journal = pending(&output, &second);
            if boundary >= 1 {
                fs::rename(&path, root.join(journal.backup.as_ref().unwrap())).unwrap();
            }
            if boundary >= 2 {
                fs::rename(root.join(&journal.stage), &path).unwrap();
            }
            drop(output);
            let output = Output::prepare(&path, &[], &cancelled()).unwrap();
            assert_eq!(
                output
                    .verify_existing(&cancelled())
                    .unwrap()
                    .unwrap()
                    .fingerprint,
                original.fingerprint
            );
            assert!(!output.journal.exists());
            assert!(!root.join(&journal.stage).exists());
            assert!(!root.join(journal.backup.unwrap()).exists());
        }
    }

    #[test]
    fn committed_cleanup_can_resume_after_partial_file_removal() {
        let (_directory, root) = workspace();
        let first = artifacts(&root, "first", 15);
        let second = artifacts(&root, "second", 85);
        let path = root.join("result");
        let mut output = Output::prepare(&path, &[], &cancelled()).unwrap();
        publish(&mut output, &first);
        let mut journal = pending(&output, &second);
        let backup = root.join(journal.backup.as_ref().unwrap());
        fs::rename(&path, &backup).unwrap();
        fs::rename(root.join(&journal.stage), &path).unwrap();
        journal.phase = Phase::Committed;
        storage::atomic_write(&output.journal, &serde_json::to_vec(&journal).unwrap()).unwrap();
        fs::remove_file(backup.join("selected-01.jpg")).unwrap();
        let wanted = state(&path).unwrap().1.unwrap().fingerprint;
        drop(output);
        let output = Output::prepare(&path, &[], &cancelled()).unwrap();
        assert_eq!(
            output
                .verify_existing(&cancelled())
                .unwrap()
                .unwrap()
                .fingerprint,
            wanted
        );
        assert!(!backup.exists());
        assert!(!output.journal.exists());
    }

    #[test]
    fn edited_recovery_files_are_preserved_and_path_escape_records_are_rejected() {
        let (_directory, root) = workspace();
        let first = artifacts(&root, "first", 15);
        let second = artifacts(&root, "second", 85);
        let path = root.join("result");
        let mut output = Output::prepare(&path, &[], &cancelled()).unwrap();
        let original = publish(&mut output, &first);
        let mut journal = pending(&output, &second);
        let stage = root.join(&journal.stage);
        fs::write(stage.join("user-note.txt"), b"keep").unwrap();
        let record = fs::read(&output.journal).unwrap();
        assert!(output.recover(&cancelled()).is_err());
        assert_eq!(fs::read(stage.join("user-note.txt")).unwrap(), b"keep");
        assert_eq!(fs::read(&output.journal).unwrap(), record);
        assert_eq!(
            output
                .verify_existing(&cancelled())
                .unwrap()
                .unwrap()
                .fingerprint,
            original.fingerprint
        );
        journal.stage = "../outside".to_owned();
        storage::atomic_write(&output.journal, &serde_json::to_vec(&journal).unwrap()).unwrap();
        assert!(output.recover(&cancelled()).is_err());
        assert!(stage.exists());
    }

    #[test]
    fn malformed_private_artifacts_cannot_replace_owned_output() {
        let (_directory, root) = workspace();
        let first = artifacts(&root, "first", 15);
        let second = artifacts(&root, "second", 85);
        let path = root.join("result");
        let mut output = Output::prepare(&path, &[], &cancelled()).unwrap();
        let original = publish(&mut output, &first);
        fs::write(second.join("selected-contact-sheet.jpg"), b"not an image").unwrap();
        assert!(
            output
                .publish_directory(&second, &cancelled(), |_| Ok(()))
                .is_err()
        );
        assert_eq!(
            output
                .verify_existing(&cancelled())
                .unwrap()
                .unwrap()
                .fingerprint,
            original.fingerprint
        );
        assert!(!output.journal.exists());
    }
}
