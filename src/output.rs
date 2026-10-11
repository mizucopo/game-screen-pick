//! Verified Rust output ownership and recoverable, same-directory publication.
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as _, Sha256};

use crate::media::Interrupted;
use crate::pipeline::Artifacts;
use crate::storage::{self, Lock};

const NAMESPACE: &str = "game-screen-pick-rust-output";
const SCHEMA: u32 = 1;
const MAX_REPORT_BYTES: u64 = 64 * 1024 * 1024;

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
    preflight_bytes_read: u64,
    _lock: Lock,
}

impl Output {
    /// Read-only validation for the CLI diagnostic; recovery requires prepare.
    pub fn validate(path: &Path, protected_paths: &[&Path]) -> Result<Option<OwnedOutput>> {
        let (path, parent, prefix) = checked_target(path, protected_paths)?;
        let journal = parent.join(format!("{prefix}.journal.json"));
        match fs::symlink_metadata(journal) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
            Ok(_) => bail!(
                "output publication recovery is pending; run the normal command to recover or choose a new empty output directory"
            ),
        }
        Ok(state(&path)?.1)
    }

    /// Call before probe, decode, search, inference, or runtime operations.
    /// Protected paths normally include the input directory, config, and cache.
    pub fn prepare(path: &Path, protected_paths: &[&Path]) -> Result<Self> {
        let (path, parent, prefix) = checked_target(path, protected_paths)?;
        // The stable sibling lock survives directory replacement; output itself
        // contains only the files declared in report.json.
        let lock = Lock::acquire(&parent.join(format!("{prefix}.lock")))?;
        let mut output = Self {
            journal: parent.join(format!("{prefix}.journal.json")),
            path,
            parent,
            prefix,
            initial: State::Absent,
            preflight_bytes_read: 0,
            _lock: lock,
        };
        output.check_path()?;
        output.recover()?;
        let (initial, owned) = state(&output.path)?;
        output.initial = initial;
        output.preflight_bytes_read = owned.map_or(0, |owned| owned.bytes_read);
        Ok(output)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Content bytes read by normal ownership preflight; recovery is separate.
    pub fn preflight_bytes_read(&self) -> u64 {
        self.preflight_bytes_read
    }

    /// Does not consult cache and never decodes an image or scans an input video.
    pub fn verify_existing(&self) -> Result<Option<OwnedOutput>> {
        self.check_path()?;
        Ok(state(&self.path)?.1)
    }

    pub fn publish(
        &mut self,
        artifacts: &Artifacts,
        cancelled: &AtomicBool,
    ) -> Result<OwnedOutput> {
        // This path is created and owned by Rust's private TempDir, unlike a
        // user supplied output path. Resolve the system's temporary-directory
        // alias (for example macOS /var) before applying ordinary path checks.
        let source = artifacts.path().canonicalize()?;
        self.publish_directory(&source, cancelled, |_| Ok(()))
    }

    fn check_path(&self) -> Result<()> {
        ensure!(
            storage::checked_path(&self.path)? == self.path,
            "output path changed after preflight"
        );
        Ok(())
    }

    fn publish_directory(
        &mut self,
        source: &Path,
        cancelled: &AtomicBool,
        mut step: impl FnMut(&str) -> Result<()>,
    ) -> Result<OwnedOutput> {
        check_cancelled(cancelled)?;
        self.check_path()?;
        ensure!(
            state(&self.path)?.0 == self.initial,
            "output changed after preflight; preserve it and choose a new empty output directory"
        );
        let stage = tempfile::Builder::new()
            .prefix(&format!("{}-stage-", self.prefix))
            .tempdir_in(&self.parent)
            .context("cannot create output staging directory")?;
        step("staging")?;
        stage_artifacts(source, stage.path(), cancelled)?;
        let staged = state(stage.path())?.0;
        check_cancelled(cancelled)?;
        // Staging may be lengthy. Recheck the old content, not just its names,
        // immediately before recording a transaction and moving it.
        ensure!(
            state(&self.path)?.0 == self.initial,
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
                state(&self.path)?.0 == journal.old,
                "output changed before replacement"
            );
            if let Some(backup) = &backup {
                fs::rename(&self.path, backup).context("cannot preserve previous output")?;
                sync_directory(&self.parent)?;
            }
            step("backed_up")?;
            check_cancelled(cancelled)?;
            fs::rename(&stage, &self.path).context("cannot publish staged output")?;
            sync_directory(&self.parent)?;
            step("published")?;
            ensure!(
                state(&self.path)?.0 == journal.new,
                "published output failed completion verification"
            );
            check_cancelled(cancelled)?;
            journal.phase = Phase::Committed;
            storage::atomic_write(&self.journal, &serde_json::to_vec(&journal)?)?;
            step("committed")?;
            Ok(())
        })();
        if let Err(error) = transaction {
            self.recover()
                .context("publication failed; previous output recovery could not complete")?;
            self.initial = state(&self.path)?.0;
            return Err(error);
        }
        self.recover()?;
        self.initial = state(&self.path)?.0;
        self.verify_existing()?
            .context("published output is missing")
    }

    /// Recovery only moves or deletes directories whose exact file set and
    /// content still match this transaction. Unknown data stops recovery.
    fn recover(&self) -> Result<()> {
        let bytes = match fs::symlink_metadata(&self.journal) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.into()),
            Ok(_) => storage::read_regular(&self.journal, 64 * 1024)?,
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
        let (target_state, _) = state(&self.path)?;
        let stage_state = cleanup_state(&stage, &journal.new)?;
        ensure!(
            matches!(stage_state, State::Absent) || stage_state == journal.new,
            "staged output changed; preserve recovery files"
        );
        let backup_state = backup
            .as_ref()
            .map(|path| {
                if journal.phase == Phase::Committed {
                    cleanup_state(path, &journal.old)
                } else {
                    state(path).map(|value| value.0)
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
                        remove_verified(backup, &journal.old)?;
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
                        fs::rename(&self.path, &stage)?;
                        fs::rename(backup, &self.path)?;
                    } else {
                        fs::rename(&self.path, &stage)?;
                    }
                } else if matches!(target_state, State::Absent) && backup.is_some() {
                    let backup = backup.as_ref().expect("checked backup");
                    ensure!(
                        backup_state.as_ref() == Some(&journal.old),
                        "previous output backup is missing"
                    );
                    fs::rename(backup, &self.path)?;
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
                            remove_verified(
                                backup,
                                placeholder.as_ref().expect("checked placeholder"),
                            )?;
                        }
                    }
                }
                ensure!(
                    state(&self.path)?.0 == journal.old,
                    "previous output was not restored"
                );
                if !matches!(cleanup_state(&stage, &journal.new)?, State::Absent) {
                    remove_verified(&stage, &journal.new)?;
                }
            }
        }
        // Refuse to unlink a replaced recovery record, even after recovery.
        ensure!(
            storage::read_regular(&self.journal, 64 * 1024)? == bytes,
            "output recovery record changed"
        );
        fs::remove_file(&self.journal)?;
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
    File::open(path)?
        .sync_all()
        .context("cannot synchronize output directory")
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

fn state(path: &Path) -> Result<(State, Option<OwnedOutput>)> {
    inspect(path, false)
}

fn inspect(path: &Path, allow_missing: bool) -> Result<(State, Option<OwnedOutput>)> {
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
    let bytes = storage::read_regular(&path.join("report.json"), MAX_REPORT_BYTES).context("output has no valid Rust ownership report; preserve it and choose a new empty output directory")?;
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
        if allow_missing && !files.contains(name) {
            continue;
        }
        ensure!(
            storage::open_regular(&path.join(name))?.metadata()?.len() == wanted.bytes,
            "output image {name:?} size changed; preserve it and choose a new empty output directory"
        );
        let (length, digest) = storage::digest_file(&path.join(name))?;
        ensure!(
            length == wanted.bytes && digest == wanted.sha256,
            "output image {name:?} was edited or damaged; preserve it and choose a new empty output directory"
        );
        bytes_read += length;
    }
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

fn stage_artifacts(source: &Path, target: &Path, cancelled: &AtomicBool) -> Result<()> {
    let source = storage::checked_path(source)?;
    let report_bytes = storage::read_regular(&source.join("report.json"), MAX_REPORT_BYTES)?;
    let mut report: Value = serde_json::from_slice(&report_bytes)?;
    ensure!(
        report.get("publication").is_none(),
        "private artifacts already contain publication metadata"
    );
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
        let digest = copy_image(&source.join(&name), &staged, cancelled)?;
        let image = image::ImageReader::with_format(
            BufReader::new(storage::open_regular(&staged)?),
            image::ImageFormat::Jpeg,
        )
        .decode()
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

fn write_exclusive(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn copy_image(source: &Path, target: &Path, cancelled: &AtomicBool) -> Result<Digest> {
    let mut source = storage::open_regular(source)?;
    let mut target = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(target)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut bytes = 0u64;
    let mut first = Vec::with_capacity(2);
    let mut last = [0u8; 2];
    loop {
        check_cancelled(cancelled)?;
        let read = source.read(&mut buffer)?;
        if read == 0 {
            break;
        }
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
    target.sync_all()?;
    Ok(Digest {
        bytes,
        sha256: format!("{:x}", hash.finalize()),
    })
}

fn cleanup_state(path: &Path, expected: &State) -> Result<State> {
    let actual = inspect(path, true)?.0;
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

fn remove_verified(path: &Path, expected: &State) -> Result<()> {
    ensure!(
        &cleanup_state(path, expected)? == expected,
        "refusing to remove changed output recovery data"
    );
    // Never recursively remove an old directory: only its verified flat files.
    let files = directory_files(path)?;
    // Keep the manifest available if any image deletion fails. Recovery can
    // validate the remaining declared subset without trusting an unknown file.
    for name in files.iter().filter(|name| name.as_str() != "report.json") {
        fs::remove_file(path.join(name))?;
    }
    if files.contains("report.json") {
        fs::remove_file(path.join("report.json"))?;
    }
    fs::remove_dir(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};
    use serde_json::json;

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
        let mut output = Output::prepare(&path, &[&cache]).unwrap();
        let owned = publish(&mut output, &source);
        assert_eq!(owned.report["cache"]["key"], "fixture");
        assert!(owned.bytes_read > fs::metadata(path.join("report.json")).unwrap().len());
        drop(output);
        fs::remove_dir(&cache).unwrap();
        let output = Output::prepare(&path, &[&cache]).unwrap();
        assert_eq!(
            output.verify_existing().unwrap().unwrap().fingerprint,
            owned.fingerprint
        );
        drop(output);
        let before_report = fs::read(path.join("report.json")).unwrap();
        RgbImage::from_pixel(64, 40, Rgb([7, 8, 9]))
            .save(path.join("selected-01.jpg"))
            .unwrap();
        let edited = fs::read(path.join("selected-01.jpg")).unwrap();
        assert!(image::open(path.join("selected-01.jpg")).is_ok());
        assert!(Output::prepare(&path, &[]).is_err());
        assert_eq!(fs::read(path.join("selected-01.jpg")).unwrap(), edited);
        assert_eq!(fs::read(path.join("report.json")).unwrap(), before_report);
    }

    #[test]
    fn unknown_files_reports_and_rank_mismatch_are_preserved() {
        let (_directory, root) = workspace();
        let unknown = root.join("unknown");
        fs::create_dir(&unknown).unwrap();
        fs::write(unknown.join("holiday.txt"), b"keep me").unwrap();
        assert!(Output::prepare(&unknown, &[]).is_err());
        assert_eq!(fs::read(unknown.join("holiday.txt")).unwrap(), b"keep me");
        let legacy = artifacts(&root, "legacy", 25);
        let before = fs::read(legacy.join("report.json")).unwrap();
        assert!(Output::prepare(&legacy, &[]).is_err());
        assert_eq!(fs::read(legacy.join("report.json")).unwrap(), before);

        let source = artifacts(&root, "artifacts", 15);
        let path = root.join("result");
        let mut output = Output::prepare(&path, &[]).unwrap();
        publish(&mut output, &source);
        fs::write(path.join("notes.txt"), b"my note").unwrap();
        assert!(output.verify_existing().is_err());
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
        assert!(output.verify_existing().is_err());
    }

    #[test]
    fn preflight_locks_reject_overlap_and_serialize_competing_runs() {
        let (_directory, root) = workspace();
        let input = root.join("input");
        fs::create_dir(&input).unwrap();
        let path = root.join("result");
        assert!(Output::prepare(&input.join("output"), &[&input]).is_err());
        assert!(Output::prepare(&root, &[&input]).is_err());
        assert!(Output::prepare(&path, &[&path]).is_err());
        let output = Output::prepare(&path, &[&input]).unwrap();
        assert!(Output::prepare(&path, &[&input]).is_err());
        drop(output);
        assert!(Output::prepare(&path, &[&input]).is_ok());
    }

    #[test]
    fn diagnostic_validation_never_creates_locks_or_recovers_output() {
        let (_directory, root) = workspace();
        let path = root.join("result");
        assert!(Output::validate(&path, &[]).unwrap().is_none());
        assert_eq!(fs::read_dir(&root).unwrap().count(), 0);
        let first = artifacts(&root, "first", 15);
        let second = artifacts(&root, "second", 85);
        let mut output = Output::prepare(&path, &[]).unwrap();
        let original = publish(&mut output, &first);
        assert_eq!(
            Output::validate(&path, &[]).unwrap().unwrap().fingerprint,
            original.fingerprint
        );
        let journal = pending(&output, &second);
        let before = fs::read(&output.journal).unwrap();
        assert!(Output::validate(&path, &[]).is_err());
        assert_eq!(fs::read(&output.journal).unwrap(), before);
        assert!(root.join(journal.stage).exists());
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
        assert!(Output::prepare(&link, &[]).is_err());
        assert!(Output::prepare(&link.join("result"), &[]).is_err());
        assert!(Output::prepare(&root.join("result"), &[&link]).is_err());
        assert!(Output::prepare(&root.join("real/../result"), &[]).is_err());
        let fifo = root.join("fifo");
        let filename = CString::new(fifo.as_os_str().as_bytes()).unwrap();
        // A FIFO must be inspected without a blocking read or a write.
        assert_eq!(unsafe { libc::mkfifo(filename.as_ptr(), 0o600) }, 0);
        assert!(Output::prepare(&fifo, &[]).is_err());
        assert!(fs::symlink_metadata(&fifo).is_ok());
        symlink(&fifo, target.join("selected-01.jpg")).unwrap();
        assert!(Output::prepare(&target, &[]).is_err());
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
        let mut output = Output::prepare(&path, &[]).unwrap();
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
                output.verify_existing().unwrap().unwrap().fingerprint,
                original.fingerprint
            );
            assert!(!output.journal.exists());
        }
    }

    #[test]
    fn staged_image_dimensions_must_match_report() {
        let (_directory, root) = workspace();
        let first = artifacts(&root, "first", 15);
        let second = artifacts(&root, "second", 85);
        let path = root.join("result");
        let mut output = Output::prepare(&path, &[]).unwrap();
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
            output.verify_existing().unwrap().unwrap().fingerprint,
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
            let mut output = Output::prepare(&path, &[]).unwrap();
            let original = publish(&mut output, &first);
            let error = output
                .publish_directory(&second, &cancelled(), |step| {
                    ensure!(step != *failure, "injected {step} failure");
                    Ok(())
                })
                .unwrap_err();
            assert!(error.to_string().contains("injected"), "{error:#}");
            assert_eq!(
                output.verify_existing().unwrap().unwrap().fingerprint,
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
                output.verify_existing().unwrap().unwrap().fingerprint,
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
            let mut output = Output::prepare(&path, &[]).unwrap();
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
        let mut output = Output::prepare(&path, &[]).unwrap();
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
        let output = Output::prepare(&path, &[]).unwrap();
        assert_eq!(
            output.verify_existing().unwrap().unwrap().fingerprint,
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
        let mut output = Output::prepare(&path, &[]).unwrap();
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

    fn pending(output: &Output, source: &Path) -> Journal {
        let stage = tempfile::Builder::new()
            .prefix(&format!("{}-stage-", output.prefix))
            .tempdir_in(&output.parent)
            .unwrap();
        stage_artifacts(source, stage.path(), &cancelled()).unwrap();
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
            let mut output = Output::prepare(&path, &[]).unwrap();
            let original = publish(&mut output, &first);
            let journal = pending(&output, &second);
            if boundary >= 1 {
                fs::rename(&path, root.join(journal.backup.as_ref().unwrap())).unwrap();
            }
            if boundary >= 2 {
                fs::rename(root.join(&journal.stage), &path).unwrap();
            }
            drop(output);
            let output = Output::prepare(&path, &[]).unwrap();
            assert_eq!(
                output.verify_existing().unwrap().unwrap().fingerprint,
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
        let mut output = Output::prepare(&path, &[]).unwrap();
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
        let output = Output::prepare(&path, &[]).unwrap();
        assert_eq!(
            output.verify_existing().unwrap().unwrap().fingerprint,
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
        let mut output = Output::prepare(&path, &[]).unwrap();
        let original = publish(&mut output, &first);
        let mut journal = pending(&output, &second);
        let stage = root.join(&journal.stage);
        fs::write(stage.join("user-note.txt"), b"keep").unwrap();
        let record = fs::read(&output.journal).unwrap();
        assert!(output.recover().is_err());
        assert_eq!(fs::read(stage.join("user-note.txt")).unwrap(), b"keep");
        assert_eq!(fs::read(&output.journal).unwrap(), record);
        assert_eq!(
            output.verify_existing().unwrap().unwrap().fingerprint,
            original.fingerprint
        );
        journal.stage = "../outside".to_owned();
        storage::atomic_write(&output.journal, &serde_json::to_vec(&journal).unwrap()).unwrap();
        assert!(output.recover().is_err());
        assert!(stage.exists());
    }

    #[test]
    fn malformed_private_artifacts_cannot_replace_owned_output() {
        let (_directory, root) = workspace();
        let first = artifacts(&root, "first", 15);
        let second = artifacts(&root, "second", 85);
        let path = root.join("result");
        let mut output = Output::prepare(&path, &[]).unwrap();
        let original = publish(&mut output, &first);
        fs::write(second.join("selected-contact-sheet.jpg"), b"not an image").unwrap();
        assert!(
            output
                .publish_directory(&second, &cancelled(), |_| Ok(()))
                .is_err()
        );
        assert_eq!(
            output.verify_existing().unwrap().unwrap().fingerprint,
            original.fingerprint
        );
        assert!(!output.journal.exists());
    }
}
