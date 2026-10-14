//! Locked Rust checkpoints; legacy caches are never read or modified.
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::media::Interrupted;
use crate::storage::{self, Lock};

pub const NAMESPACE: &str = "game-screen-pick-rust-v1";
const SCHEMA: u32 = 1;
const MAX_RECORD_BYTES: u64 = 64 * 1024 * 1024;
const MAX_INPUTS: usize = 4096;

/// Paths are deliberately excluded, so moving an input directory preserves hits.
/// Ordinary edits/replacements are detected by these Unix metadata fields. An
/// adversarial writer restoring all metadata, or a filesystem returning stale
/// metadata, is outside this detection guarantee.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputStamp {
    pub device: u64,
    pub inode: u64,
    pub length: u64,
    pub modified_seconds: i64,
    pub modified_nanoseconds: i64,
    pub changed_seconds: i64,
    pub changed_nanoseconds: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputIdentity {
    pub content_digest: String,
    pub metadata: InputStamp,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct CacheStats {
    /// Checkpoint payloads and cold input content; metadata reads have no bytes.
    pub bytes_read: u64,
    pub input_hashes: u64,
    pub hits: u64,
    pub misses: u64,
    pub elapsed_seconds: f64,
    /// Actual input identification (including cold hashes/index writes) and
    /// checkpoint reads/domain validation. Nested lookup time is counted once.
    pub verification_seconds: f64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    schema: u32,
    kind: String,
    key: String,
    payload_digest: String,
    payload: serde_json::Value,
}

pub struct Cache {
    root: PathBuf,
    _lock: Lock,
    stats: CacheStats,
    started: Instant,
}

/// Conditions must contain only non-secret effective inputs, with explicit
/// algorithm/schema versions. The caller decides which dependencies belong here.
pub fn key(conditions: &impl Serialize) -> Result<String> {
    let value = serde_json::to_value(conditions).context("cannot encode cache conditions")?;
    Ok(storage::sha256(&serde_json::to_vec(&value)?))
}

/// Recheck current metadata around external work and before a warm return or
/// publication, without reading video content or the persisted identity index.
pub fn verify_inputs(paths: &[PathBuf], identities: &[InputIdentity]) -> Result<()> {
    ensure!(
        !paths.is_empty() && paths.len() <= MAX_INPUTS && paths.len() == identities.len(),
        "input paths and identities must have matching bounded lengths"
    );
    let mut files = BTreeSet::new();
    for (path, identity) in paths.iter().zip(identities) {
        let path = storage::checked_path(path)?;
        let current = stamp(&path)?;
        ensure!(
            files.insert((current.device, current.inode)),
            "input paths refer to the same file more than once"
        );
        ensure!(
            valid_digest(&identity.content_digest) && current == identity.metadata,
            "input changed during processing"
        );
    }
    Ok(())
}

impl Cache {
    /// The caller preflights cache/input/output overlap and creates the parent.
    /// Acquire this guard before probe, search, inference or runtime operations.
    pub fn open(cache_parent: &Path) -> Result<Self> {
        let started = Instant::now();
        let parent = storage::checked_path(cache_parent)?;
        ensure!(
            fs::symlink_metadata(&parent)?.is_dir(),
            "cache parent must be an ordinary directory"
        );
        let root = storage::checked_path(&parent.join(NAMESPACE))?;
        match fs::create_dir(&root) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(error).context(storage::OperationalError(
                    "cannot create Rust cache namespace",
                ));
            }
        }
        ensure!(
            fs::symlink_metadata(&root)?.is_dir(),
            "Rust cache namespace must be an ordinary directory"
        );
        let lock = Lock::acquire(&root.join(".lock"))?;
        Ok(Self {
            root,
            _lock: lock,
            stats: CacheStats::default(),
            started,
        })
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            elapsed_seconds: self.started.elapsed().as_secs_f64(),
            ..self.stats.clone()
        }
    }

    /// Verify the storage envelope/type only. Domain checkpoints should use
    /// read_validated so rejected references are counted as misses.
    pub fn read<T: DeserializeOwned>(&mut self, kind: &str, key: &str) -> Result<Option<T>> {
        self.read_validated(kind, key, |_| true)
    }

    /// A hit is counted only after the typed payload passes its domain checks.
    /// Missing/corrupt/rejected records count once as a miss. Unsafe filesystem
    /// objects fail; the validator is never called for a bad envelope/type.
    pub fn read_validated<T: DeserializeOwned>(
        &mut self,
        kind: &str,
        key: &str,
        validate: impl FnOnce(&T) -> bool,
    ) -> Result<Option<T>> {
        let started = Instant::now();
        let result = self.read_validated_inner(kind, key, validate);
        self.stats.verification_seconds += started.elapsed().as_secs_f64();
        result
    }

    fn read_validated_inner<T: DeserializeOwned>(
        &mut self,
        kind: &str,
        key: &str,
        validate: impl FnOnce(&T) -> bool,
    ) -> Result<Option<T>> {
        let Some(path) = self.record_path(kind, key, false)? else {
            return Ok(self.miss());
        };
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(self.miss()),
            Err(error) => return Err(error).context("cannot inspect cache checkpoint"),
        };
        ensure!(
            metadata.is_file(),
            "cache checkpoint must be an ordinary file"
        );
        if metadata.len() > MAX_RECORD_BYTES {
            return Ok(self.miss());
        }
        let bytes = storage::read_regular(&path, MAX_RECORD_BYTES)?;
        self.stats.bytes_read += bytes.len() as u64;
        let Some(value) = decode::<T>(&bytes, kind, key) else {
            return Ok(self.miss());
        };
        if !validate(&value) {
            return Ok(self.miss());
        }
        self.stats.hits += 1;
        Ok(Some(value))
    }

    /// One completed batch/chunk at a time. Serialization/write failure leaves
    /// the previously committed record intact; abandoned tempfiles are not hits.
    pub fn write<T: Serialize>(&mut self, kind: &str, key: &str, value: &T) -> Result<()> {
        let payload = serde_json::to_value(value).context("cannot encode cache checkpoint")?;
        let envelope = Envelope {
            schema: SCHEMA,
            kind: kind.to_owned(),
            key: key.to_owned(),
            payload_digest: storage::sha256(&serde_json::to_vec(&payload)?),
            payload,
        };
        let bytes = serde_json::to_vec(&envelope)?;
        ensure!(
            bytes.len() as u64 <= MAX_RECORD_BYTES,
            "cache checkpoint exceeds 64 MiB"
        );
        let path = self
            .record_path(kind, key, true)?
            .context("cannot create cache checkpoint directory")?;
        // Reject existing links/special files even though atomic replacement
        // would replace the directory entry rather than follow it.
        match fs::symlink_metadata(&path) {
            Ok(metadata) => ensure!(
                metadata.is_file(),
                "cache checkpoint must be an ordinary file"
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).context("cannot inspect cache checkpoint"),
        }
        storage::atomic_write(&path, &bytes)
    }

    /// All metadata is preflighted first. Each completed cold content hash is
    /// immediately checkpointed, so interruption retains earlier input hashes.
    /// A normal warm lookup reads metadata and this bounded index only.
    pub fn identify(
        &mut self,
        paths: &[PathBuf],
        cancelled: &AtomicBool,
    ) -> Result<Vec<InputIdentity>> {
        let started = Instant::now();
        let preceding_seconds = self.stats.verification_seconds;
        let result = self.identify_inner(paths, cancelled);
        // identify includes its read_validated call; avoid counting it twice.
        self.stats.verification_seconds = preceding_seconds + started.elapsed().as_secs_f64();
        result
    }

    fn identify_inner(
        &mut self,
        paths: &[PathBuf],
        cancelled: &AtomicBool,
    ) -> Result<Vec<InputIdentity>> {
        check_cancelled(cancelled)?;
        ensure!(
            !paths.is_empty() && paths.len() <= MAX_INPUTS,
            "input identity count must be between 1 and 4096"
        );
        let mut current = BTreeSet::new();
        let mut files = BTreeSet::new();
        let mut stamps = Vec::with_capacity(paths.len());
        for path in paths {
            check_cancelled(cancelled)?;
            let path = storage::checked_path(path)?;
            let stamp = stamp(&path)?;
            ensure!(
                files.insert((stamp.device, stamp.inode)),
                "input paths refer to the same file more than once"
            );
            current.insert(stamp.clone());
            stamps.push((path, stamp));
        }
        let index_key = key(&"input-metadata-v1")?;
        let records = self
            .read_validated::<Vec<InputIdentity>>("inputs", &index_key, |records| {
                valid_index(records)
            })?
            .unwrap_or_default();
        let mut index = records
            .into_iter()
            .map(|identity| (identity.metadata, identity.content_digest))
            .collect::<BTreeMap<_, _>>();
        let mut identities = Vec::with_capacity(stamps.len());
        for (path, metadata) in stamps {
            check_cancelled(cancelled)?;
            let digest = match index.get(&metadata) {
                Some(digest) => digest.clone(),
                None => {
                    let (bytes, digest) =
                        storage::digest_file_cancelled(&path, cancelled, |bytes| {
                            self.stats.bytes_read += bytes;
                        })?;
                    self.stats.input_hashes += 1;
                    ensure!(
                        bytes == metadata.length && stamp(&path)? == metadata,
                        "input changed while its identity was being computed"
                    );
                    check_cancelled(cancelled)?;
                    if index.len() >= MAX_INPUTS {
                        index.retain(|stamp, _| current.contains(stamp));
                    }
                    index.insert(metadata.clone(), digest.clone());
                    let records = index
                        .iter()
                        .map(|(metadata, digest)| InputIdentity {
                            content_digest: digest.clone(),
                            metadata: metadata.clone(),
                        })
                        .collect::<Vec<_>>();
                    self.write("inputs", &index_key, &records)?;
                    digest
                }
            };
            identities.push(InputIdentity {
                content_digest: digest,
                metadata,
            });
        }
        Ok(identities)
    }

    fn miss<T>(&mut self) -> Option<T> {
        self.stats.misses += 1;
        None
    }

    fn record_path(&self, kind: &str, key: &str, create: bool) -> Result<Option<PathBuf>> {
        ensure!(
            !kind.is_empty()
                && kind.len() <= 64
                && kind
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-'),
            "invalid cache checkpoint namespace"
        );
        ensure!(
            valid_digest(key),
            "cache key must be a lowercase SHA-256 digest"
        );
        let directory = storage::checked_path(&self.root.join(kind))?;
        if create {
            match fs::create_dir(&directory) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error).context("cannot create checkpoint directory"),
            }
        }
        match fs::symlink_metadata(&directory) {
            Ok(metadata) => ensure!(
                metadata.is_dir(),
                "checkpoint namespace must be an ordinary directory"
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error).context("cannot inspect checkpoint directory"),
        }
        Ok(Some(storage::checked_path(
            &directory.join(format!("{key}.json")),
        )?))
    }
}

fn decode<T: DeserializeOwned>(bytes: &[u8], kind: &str, key: &str) -> Option<T> {
    let envelope = serde_json::from_slice::<Envelope>(bytes).ok()?;
    if envelope.schema != SCHEMA || envelope.key != key || envelope.kind != kind {
        return None;
    }
    let payload = serde_json::to_vec(&envelope.payload).ok()?;
    if storage::sha256(&payload) != envelope.payload_digest {
        return None;
    }
    serde_json::from_value(envelope.payload).ok()
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_index(records: &[InputIdentity]) -> bool {
    if records.len() > MAX_INPUTS {
        return false;
    }
    let mut stamps = BTreeSet::new();
    records
        .iter()
        .all(|identity| valid_digest(&identity.content_digest) && stamps.insert(&identity.metadata))
}

fn stamp(path: &Path) -> Result<InputStamp> {
    let metadata = fs::symlink_metadata(path).context("cannot inspect input identity")?;
    ensure!(
        metadata.is_file(),
        "input identity requires an ordinary file"
    );
    Ok(InputStamp {
        device: metadata.dev(),
        inode: metadata.ino(),
        length: metadata.len(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
    })
}

fn check_cancelled(cancelled: &AtomicBool) -> Result<()> {
    if cancelled.load(Ordering::Relaxed) {
        return Err(Interrupted.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interrupted_input_hash_counts_partial_bytes_before_same_cache_retry() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let source = root.join("recording.mkv");
        let contents = vec![7; 256 * 1024];
        fs::write(&source, &contents).unwrap();
        let parent = root.join("cache");
        fs::create_dir(&parent).unwrap();
        let mut cache = Cache::open(&parent).unwrap();
        let cancelled = AtomicBool::new(false);
        let interruption = storage::interrupt_digest_after(64 * 1024);
        let error = cache
            .identify(std::slice::from_ref(&source), &cancelled)
            .unwrap_err();
        drop(interruption);
        assert!(error.downcast_ref::<Interrupted>().is_some());
        assert_eq!(cache.stats().bytes_read, 64 * 1024);
        assert_eq!(cache.stats().input_hashes, 0);
        let interrupted_seconds = cache.stats().verification_seconds;
        assert!(interrupted_seconds > 0.0);
        cancelled.store(false, Ordering::Relaxed);
        let identities = cache
            .identify(std::slice::from_ref(&source), &cancelled)
            .unwrap();
        assert_eq!(identities[0].content_digest, storage::sha256(&contents));
        assert_eq!(cache.stats().bytes_read, 64 * 1024 + contents.len() as u64);
        assert_eq!(cache.stats().input_hashes, 1);
        assert!(cache.stats().verification_seconds > interrupted_seconds);
    }

    #[test]
    fn checkpoint_timing_includes_domain_validation_without_idle_time() {
        let directory = tempfile::tempdir().unwrap();
        let parent = directory.path().canonicalize().unwrap();
        let mut cache = Cache::open(&parent).unwrap();
        let key = key(&"timed-checkpoint").unwrap();
        cache.write("test", &key, &42_u64).unwrap();
        let delay = std::time::Duration::from_millis(5);
        std::thread::sleep(delay);
        assert_eq!(cache.stats().verification_seconds, 0.0);
        let hit = cache
            .read_validated::<u64>("test", &key, |value| {
                std::thread::sleep(delay);
                *value == 42
            })
            .unwrap();
        assert_eq!(hit, Some(42));
        assert!(cache.stats().verification_seconds >= delay.as_secs_f64());
    }
}
