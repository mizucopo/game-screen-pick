use std::fs::{self, File};
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use game_screen_pick::cache::{Cache, InputIdentity, NAMESPACE, key, verify_inputs};
use game_screen_pick::selection::{Assessment, Candidate, validate_assessments};
use serde::Serialize;
use serde_json::{Value, json};
use tempfile::TempDir;

fn temporary() -> TempDir {
    // macOS system temp aliases are not user input symlinks.
    TempDir::new_in(std::env::temp_dir().canonicalize().unwrap()).unwrap()
}

fn checkpoint(parent: &Path, kind: &str, cache_key: &str) -> PathBuf {
    parent
        .join(NAMESPACE)
        .join(kind)
        .join(format!("{cache_key}.json"))
}

#[test]
fn dropping_lock_releases_it_while_a_forked_child_retains_its_descriptor() {
    use std::io::{Read, Write};
    use std::os::fd::{AsRawFd, FromRawFd};

    let parent = temporary();
    let path = parent.path().join("run.lock");
    let guard = game_screen_pick::storage::Lock::acquire(&path).unwrap();
    let pipe = || {
        let mut descriptors = [0; 2];
        assert_eq!(unsafe { libc::pipe(descriptors.as_mut_ptr()) }, 0);
        descriptors.map(|descriptor| unsafe { File::from_raw_fd(descriptor) })
    };
    let [mut ready_read, ready_write] = pipe();
    let [release_read, mut release_write] = pipe();
    let ready_read_fd = ready_read.as_raw_fd();
    let ready_write_fd = ready_write.as_raw_fd();
    let release_read_fd = release_read.as_raw_fd();
    let release_write_fd = release_write.as_raw_fd();
    let child = unsafe { libc::fork() };
    if child == 0 {
        // Cargo tests are multithreaded: no allocation, Rust destructors or
        // non-async-signal-safe library calls may run in the forked child.
        unsafe {
            libc::close(ready_read_fd);
            libc::close(release_write_fd);
            let ready = 1_u8;
            if libc::write(ready_write_fd, (&ready as *const u8).cast(), 1) != 1 {
                libc::_exit(91);
            }
            let mut release = 0_u8;
            if libc::read(release_read_fd, (&mut release as *mut u8).cast(), 1) != 1 {
                libc::_exit(92);
            }
            libc::close(ready_write_fd);
            libc::close(release_read_fd);
            libc::_exit(0);
        }
    }
    assert!(child > 0);
    drop(ready_write);
    drop(release_read);
    let mut ready = [0];
    let synchronized = ready_read.read_exact(&mut ready);
    drop(guard);
    // The child cannot exit until this acquisition has been attempted.
    let reacquired = game_screen_pick::storage::Lock::acquire(&path);
    let released = release_write.write_all(&[1]);
    let mut status = 0;
    let reaped = unsafe { libc::waitpid(child, &mut status, 0) };
    // Always release/reap the child before asserting, including the red case.
    synchronized.unwrap();
    released.unwrap();
    assert_eq!(reaped, child);
    assert_eq!(status, 0);
    assert!(
        reacquired.is_ok(),
        "the parent run ended while a child still retained its inherited descriptor: {:?}",
        reacquired.err()
    );
}

#[cfg(target_os = "macos")]
#[test]
fn existing_case_aliases_share_output_lock_and_reject_protected_overlap() {
    use std::os::unix::fs::MetadataExt;

    let parent = temporary();
    let actual = parent.path().join("Result");
    let alias = parent.path().join("result");
    fs::create_dir(&actual).unwrap();
    assert_eq!(
        fs::metadata(&actual).unwrap().ino(),
        fs::metadata(&alias)
            .expect("this native regression requires the default case-insensitive macOS volume")
            .ino()
    );
    let first =
        game_screen_pick::output::Output::prepare(&actual, &[], &AtomicBool::new(false)).unwrap();
    let competing = game_screen_pick::output::Output::prepare(&alias, &[], &AtomicBool::new(false));
    let lock_blocked = competing.is_err();
    drop(competing);
    drop(first);
    let overlap =
        game_screen_pick::output::Output::prepare(&alias, &[&actual], &AtomicBool::new(false));
    assert!(
        lock_blocked && overlap.is_err(),
        "same native directory must share its lock and reject protected overlap: lock_blocked={lock_blocked}, overlap_rejected={}",
        overlap.is_err()
    );
}

#[cfg(target_os = "macos")]
#[test]
fn absent_case_and_unicode_aliases_share_output_lock_and_reject_protected_overlap() {
    for (actual_name, alias_name) in [("Result", "result"), ("Résult", "Re\u{301}sult")] {
        let parent = temporary();
        let actual = parent.path().join(actual_name);
        let alias = parent.path().join(alias_name);
        let first =
            game_screen_pick::output::Output::prepare(&actual, &[], &AtomicBool::new(false))
                .unwrap();
        let competing =
            game_screen_pick::output::Output::prepare(&alias, &[], &AtomicBool::new(false));
        let lock_blocked = competing.is_err();
        assert!(!actual.exists(), "locking must not precreate the output");
        drop(competing);
        drop(first);
        let overlap =
            game_screen_pick::output::Output::prepare(&actual, &[&alias], &AtomicBool::new(false));
        assert!(
            lock_blocked && overlap.is_err(),
            "absent native aliases must share their lock and reject protected overlap: {actual_name:?}/{alias_name:?}, lock_blocked={lock_blocked}, overlap_rejected={}",
            overlap.is_err()
        );
        assert!(!actual.exists(), "rejection must not precreate the output");
    }
}

#[test]
fn native_output_locks_allow_maximum_length_leaf_without_precreation() {
    let parent = temporary();
    let actual = parent.path().join("a".repeat(255));
    let first =
        game_screen_pick::output::Output::prepare(&actual, &[], &AtomicBool::new(false)).unwrap();
    assert!(
        game_screen_pick::output::Output::prepare(&actual, &[], &AtomicBool::new(false)).is_err()
    );
    assert!(!actual.exists());
    drop(first);
    assert!(
        game_screen_pick::output::Output::prepare(&actual, &[], &AtomicBool::new(false)).is_ok()
    );
    assert!(!actual.exists());
}

#[test]
fn cold_warm_identity_survives_directory_move_and_reuses_unchanged_inputs() {
    let workspace = temporary();
    let inputs = workspace.path().join("inputs");
    let cache_parent = workspace.path().join("cache");
    fs::create_dir(&inputs).unwrap();
    fs::create_dir(&cache_parent).unwrap();
    let source = inputs.join("video.mkv");
    fs::write(&source, vec![0x3b; 2 * 1024 * 1024]).unwrap();
    let cancelled = AtomicBool::new(false);
    let mut cache = Cache::open(&cache_parent).unwrap();
    let original = cache.identify(&[source], &cancelled).unwrap();
    assert_eq!(cache.stats().input_hashes, 1);
    assert!(cache.stats().bytes_read >= 2 * 1024 * 1024);
    drop(cache);

    let moved = workspace.path().join("renamed");
    fs::rename(inputs, &moved).unwrap();
    let source = moved.join("video.mkv");
    let mut cache = Cache::open(&cache_parent).unwrap();
    assert_eq!(
        cache
            .identify(std::slice::from_ref(&source), &cancelled)
            .unwrap(),
        original
    );
    assert_eq!(cache.stats().input_hashes, 0);
    assert!(cache.stats().bytes_read < 16 * 1024);

    let second = moved.join("additional.mkv");
    fs::write(&second, b"additional content").unwrap();
    let identities = cache
        .identify(&[source.clone(), second.clone()], &cancelled)
        .unwrap();
    assert_eq!(identities[0], original[0]);
    assert_eq!(cache.stats().input_hashes, 1);
    // Replacement with equal length still changes the inode and content digest.
    let replacement = moved.join("replacement");
    fs::write(&replacement, vec![0x52; 2 * 1024 * 1024]).unwrap();
    fs::rename(replacement, &source).unwrap();
    let changed = cache.identify(&[source, second], &cancelled).unwrap();
    assert_ne!(changed[0].content_digest, original[0].content_digest);
    assert_ne!(changed[0].metadata.inode, original[0].metadata.inode);
    assert_eq!(cache.stats().input_hashes, 2);
}

#[test]
fn metadata_changes_rehash_content_without_a_path_dependency() {
    let workspace = temporary();
    let source = workspace.path().join("video.mkv");
    fs::write(&source, b"before").unwrap();
    let cancelled = AtomicBool::new(false);
    let mut cache = Cache::open(workspace.path()).unwrap();
    let original = cache
        .identify(std::slice::from_ref(&source), &cancelled)
        .unwrap();
    verify_inputs(std::slice::from_ref(&source), &original).unwrap();
    fs::write(&source, b"after!").unwrap();
    let changed_time = std::time::UNIX_EPOCH + std::time::Duration::from_secs(42);
    File::options()
        .write(true)
        .open(&source)
        .unwrap()
        .set_times(fs::FileTimes::new().set_modified(changed_time))
        .unwrap();
    let changed = cache.identify(&[source], &cancelled).unwrap();
    assert_eq!(original[0].metadata.length, changed[0].metadata.length);
    assert_eq!(original[0].metadata.inode, changed[0].metadata.inode);
    assert_ne!(original[0].content_digest, changed[0].content_digest);
    assert_eq!(cache.stats().input_hashes, 2);
    assert!(verify_inputs(&[workspace.path().join("video.mkv")], &original).is_err());
}

#[test]
fn verified_records_reject_corrupt_payload_wrong_key_schema_and_type() {
    let parent = temporary();
    let mut cache = Cache::open(parent.path()).unwrap();
    let first = key(&json!({"model": "fixed", "revision": 1})).unwrap();
    let second = key(&json!({"model": "fixed", "revision": 2})).unwrap();
    let value = vec!["completed", "日本語"];
    cache.write("batch", &first, &value).unwrap();
    assert_eq!(
        cache.read::<Vec<String>>("batch", &first).unwrap(),
        Some(value.iter().map(|text| text.to_string()).collect())
    );
    let path = checkpoint(parent.path(), "batch", &first);
    let original = fs::read(&path).unwrap();
    let mut envelope: Value = serde_json::from_slice(&original).unwrap();
    envelope["payload"][0] = "different-input".into();
    fs::write(&path, serde_json::to_vec(&envelope).unwrap()).unwrap();
    assert!(
        cache
            .read::<Vec<String>>("batch", &first)
            .unwrap()
            .is_none()
    );
    fs::write(&path, &original).unwrap();
    fs::copy(&path, checkpoint(parent.path(), "batch", &second)).unwrap();
    assert!(
        cache
            .read::<Vec<String>>("batch", &second)
            .unwrap()
            .is_none()
    );
    cache.write("other", &first, &value).unwrap();
    fs::write(checkpoint(parent.path(), "other", &first), &original).unwrap();
    assert!(
        cache
            .read::<Vec<String>>("other", &first)
            .unwrap()
            .is_none()
    );
    envelope = serde_json::from_slice(&original).unwrap();
    envelope["schema"] = 2.into();
    fs::write(&path, serde_json::to_vec(&envelope).unwrap()).unwrap();
    assert!(
        cache
            .read::<Vec<String>>("batch", &first)
            .unwrap()
            .is_none()
    );
    fs::write(&path, &original).unwrap();
    assert!(cache.read::<u64>("batch", &first).unwrap().is_none());
    fs::write(&path, b"{\"schema\":").unwrap();
    assert!(
        cache
            .read::<Vec<String>>("batch", &first)
            .unwrap()
            .is_none()
    );
    cache.write("batch", &first, &value).unwrap();
    assert!(
        cache
            .read::<Vec<String>>("batch", &first)
            .unwrap()
            .is_some()
    );
    assert_eq!(cache.stats().hits, 2);
    assert_eq!(cache.stats().misses, 6);
}

#[test]
fn validated_reads_count_rejected_references_once_and_skip_bad_envelope_validators() {
    let parent = temporary();
    let mut cache = Cache::open(parent.path()).unwrap();
    let image = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/images/rgb-grid.png");
    let candidate = Candidate {
        id: "expected-frame".into(),
        source: "video.mkv".into(),
        requested_seconds: 0.0,
        actual_seconds: 0.0,
        duration_seconds: 1.0,
        quality: game_screen_pick::quality::analyze(&image).unwrap(),
        image,
        event: None,
    };
    let cache_key = key(&"typed-reference").unwrap();
    let mut assessment = Assessment {
        id: "different-frame".into(),
        blog_score: 80.0,
        transition: false,
        scene: "通常進行".into(),
        reason: "読みやすい録画場面".into(),
        special: false,
    };
    cache
        .write("assessments", &cache_key, &vec![assessment.clone()])
        .unwrap();
    let validate = |records: &Vec<Assessment>| {
        validate_assessments(std::slice::from_ref(&candidate), records).is_ok()
    };
    assert!(
        cache
            .read_validated::<Vec<Assessment>>("assessments", &cache_key, validate)
            .unwrap()
            .is_none()
    );
    assert_eq!(cache.stats().hits, 0);
    assert_eq!(cache.stats().misses, 1);
    assert!(cache.stats().bytes_read > 0);
    // The public storage-only API remains usable without a domain validator.
    assert!(
        cache
            .read::<Vec<Assessment>>("assessments", &cache_key)
            .unwrap()
            .is_some()
    );
    assessment.id = candidate.id.clone();
    cache
        .write("assessments", &cache_key, &vec![assessment])
        .unwrap();
    assert!(
        cache
            .read_validated::<Vec<Assessment>>("assessments", &cache_key, validate)
            .unwrap()
            .is_some()
    );
    assert_eq!(cache.stats().hits, 2);
    assert_eq!(cache.stats().misses, 1);
    fs::write(checkpoint(parent.path(), "assessments", &cache_key), b"{").unwrap();
    assert!(
        cache
            .read_validated::<Vec<Assessment>>("assessments", &cache_key, |_| {
                panic!("a corrupt envelope must not reach the domain validator")
            })
            .unwrap()
            .is_none()
    );
    cache.write("assessments", &cache_key, &7).unwrap();
    assert!(
        cache
            .read_validated::<Vec<Assessment>>("assessments", &cache_key, |_| {
                panic!("the wrong payload type must not reach the domain validator")
            })
            .unwrap()
            .is_none()
    );
    cache.write("contexts", &cache_key, &"  ").unwrap();
    assert!(
        cache
            .read_validated::<String>("contexts", &cache_key, |context| {
                !context.trim().is_empty()
            })
            .unwrap()
            .is_none()
    );
    assert_eq!(cache.stats().hits, 2);
    assert_eq!(cache.stats().misses, 4);
}

#[test]
fn old_namespaces_and_unknown_files_are_preserved_and_never_imported() {
    let parent = temporary();
    let legacy = parent.path().join("legacy-cache");
    fs::create_dir(&legacy).unwrap();
    fs::write(legacy.join("report.json"), b"old schema").unwrap();
    fs::write(parent.path().join("unknown"), b"keep").unwrap();
    let mut cache = Cache::open(parent.path()).unwrap();
    let cache_key = key(&"legacy-looking-condition").unwrap();
    assert!(cache.read::<String>("batch", &cache_key).unwrap().is_none());
    cache
        .write("batch", &cache_key, &"Rust checkpoint")
        .unwrap();
    assert_eq!(fs::read(legacy.join("report.json")).unwrap(), b"old schema");
    assert_eq!(fs::read(parent.path().join("unknown")).unwrap(), b"keep");
}

#[test]
fn locks_unsafe_paths_aliases_and_special_files_are_rejected() {
    let parent = temporary();
    let mut cache = Cache::open(parent.path()).unwrap();
    assert!(Cache::open(parent.path()).is_err());
    let cache_key = key(&"safe").unwrap();
    assert!(cache.write("../outside", &cache_key, &"unsafe").is_err());
    assert!(cache.write("safe", "../outside", &"unsafe").is_err());
    let outside = temporary();
    symlink(outside.path(), parent.path().join(NAMESPACE).join("linked")).unwrap();
    assert!(cache.read::<String>("linked", &cache_key).is_err());
    assert!(cache.write("linked", &cache_key, &"unsafe").is_err());

    cache.write("batch", &cache_key, &"original").unwrap();
    let record = checkpoint(parent.path(), "batch", &cache_key);
    fs::remove_file(&record).unwrap();
    symlink(outside.path().join("secret"), &record).unwrap();
    assert!(cache.read::<String>("batch", &cache_key).is_err());
    assert!(cache.write("batch", &cache_key, &"unsafe").is_err());
    fs::remove_file(&record).unwrap();
    fs::create_dir(&record).unwrap();
    assert!(cache.read::<String>("batch", &cache_key).is_err());
    assert!(cache.write("batch", &cache_key, &"unsafe").is_err());
    fs::remove_dir(&record).unwrap();
    let fifo = std::ffi::CString::new(record.as_os_str().as_encoded_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0);
    assert!(cache.read::<String>("batch", &cache_key).is_err());
    assert!(cache.write("batch", &cache_key, &"unsafe").is_err());

    let source = outside.path().join("video.mkv");
    let alias = outside.path().join("alias.mkv");
    fs::write(&source, b"video").unwrap();
    fs::hard_link(&source, &alias).unwrap();
    let cancelled = AtomicBool::new(false);
    assert!(
        cache
            .identify(&[source.clone(), alias], &cancelled)
            .is_err()
    );
    let link = outside.path().join("symlink.mkv");
    symlink(source, &link).unwrap();
    assert!(cache.identify(&[link], &cancelled).is_err());
    assert_eq!(cache.stats().input_hashes, 0);
    drop(cache);
    assert!(Cache::open(parent.path()).is_ok());
}

#[test]
fn committed_checkpoint_survives_an_interrupted_batch_and_failed_overwrite() {
    let parent = temporary();
    let first = key(&"batch-1").unwrap();
    let missing = key(&"batch-2").unwrap();
    let mut cache = Cache::open(parent.path()).unwrap();
    cache.write("assessment", &first, &vec![1, 2, 3]).unwrap();
    // Abandoned scratch never masquerades as a completed checkpoint.
    fs::write(
        parent
            .path()
            .join(NAMESPACE)
            .join("assessment")
            .join(".partial"),
        b"uncommitted",
    )
    .unwrap();
    struct FailedPayload;
    impl Serialize for FailedPayload {
        fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom("injected serialization failure"))
        }
    }
    assert!(cache.write("assessment", &first, &FailedPayload).is_err());
    drop(cache);
    let mut resumed = Cache::open(parent.path()).unwrap();
    assert_eq!(
        resumed.read::<Vec<u32>>("assessment", &first).unwrap(),
        Some(vec![1, 2, 3])
    );
    assert!(
        resumed
            .read::<Vec<u32>>("assessment", &missing)
            .unwrap()
            .is_none()
    );
    let source = parent.path().join("input.mkv");
    fs::write(&source, b"input").unwrap();
    let cancelled = AtomicBool::new(true);
    assert!(resumed.identify(&[source], &cancelled).is_err());
    assert_eq!(resumed.stats().input_hashes, 0);
}

#[test]
fn oversized_records_and_invalid_identity_index_are_misses_not_hits() {
    let parent = temporary();
    let mut cache = Cache::open(parent.path()).unwrap();
    let cache_key = key(&"oversized").unwrap();
    cache.write("batch", &cache_key, &"small").unwrap();
    let file = File::options()
        .write(true)
        .open(checkpoint(parent.path(), "batch", &cache_key))
        .unwrap();
    file.set_len(64 * 1024 * 1024 + 1).unwrap();
    assert!(cache.read::<String>("batch", &cache_key).unwrap().is_none());
    assert_eq!(cache.stats().bytes_read, 0);
    let source = parent.path().join("video.mkv");
    fs::write(&source, b"current video").unwrap();
    let cancelled = AtomicBool::new(false);
    let identities = cache
        .identify(std::slice::from_ref(&source), &cancelled)
        .unwrap();
    let mut invalid = identities;
    invalid[0].content_digest = "not-a-digest".into();
    let index_key = key(&"input-metadata-v1").unwrap();
    cache.write("inputs", &index_key, &invalid).unwrap();
    assert!(
        cache
            .read::<Vec<InputIdentity>>("inputs", &index_key)
            .unwrap()
            .is_some()
    );
    let baseline = cache.stats();
    let recovered = cache
        .identify(std::slice::from_ref(&source), &cancelled)
        .unwrap();
    assert_ne!(recovered[0].content_digest, invalid[0].content_digest);
    assert_eq!(cache.stats().input_hashes, 2);
    assert_eq!(cache.stats().hits, baseline.hits);
    assert_eq!(cache.stats().misses, baseline.misses + 1);
    let duplicate = vec![recovered[0].clone(), recovered[0].clone()];
    cache.write("inputs", &index_key, &duplicate).unwrap();
    let baseline = cache.stats();
    assert_eq!(cache.identify(&[source], &cancelled).unwrap(), recovered);
    assert_eq!(cache.stats().hits, baseline.hits);
    assert_eq!(cache.stats().misses, baseline.misses + 1);
    assert_eq!(cache.stats().input_hashes, baseline.input_hashes + 1);
}
