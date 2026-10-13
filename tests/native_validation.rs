use std::fs;
#[cfg(target_os = "macos")]
use std::path::Path;
use std::sync::atomic::AtomicBool;

use game_screen_pick::output::Output;
use tempfile::TempDir;

fn temporary() -> TempDir {
    TempDir::new_in(std::env::temp_dir().canonicalize().unwrap()).unwrap()
}

#[cfg(target_os = "macos")]
fn native_aliases(parent: &Path, name: &str, alias: &str) -> bool {
    use std::os::unix::fs::MetadataExt;

    let probe = TempDir::new_in(parent).unwrap();
    let original = probe.path().join(name);
    fs::create_dir(&original).unwrap();
    fs::metadata(probe.path().join(alias))
        .is_ok_and(|metadata| metadata.ino() == fs::metadata(&original).unwrap().ino())
}

#[cfg(target_os = "macos")]
#[test]
fn absent_native_alias_validation_agrees_with_filesystem_and_prepare_without_writes() {
    let cancelled = AtomicBool::new(false);
    for (name, alias) in [
        ("Result", "result"),
        ("Résult", "Re\u{301}sult"),
        ("straße", "STRASSE"),
        ("ﬀ", "ff"),
        ("İ", "i\u{307}"),
        ("Σ", "ς"),
        ("Ა", "ა"),
        ("K", "K"),
        ("Ω", "Ω"),
        ("豈", "豈"),
        ("가", "가"),
        ("e\u{301}\u{323}", "e\u{323}\u{301}"),
        ("Résult", "Result"),
        ("I", "ı"),
        ("Ａ", "A"),
        ("result-1", "result1"),
        ("a\u{ad}", "a"),
        ("x\u{200d}", "x"),
    ] {
        let parent = temporary();
        let expected_overlap = native_aliases(parent.path(), name, alias);
        let output = parent.path().join(name);
        let protected = parent.path().join(alias);
        let validation = Output::validate(&output, &[&protected], &cancelled);
        assert_eq!(
            fs::read_dir(parent.path()).unwrap().count(),
            0,
            "readonly validation must not create output/cache/lock representatives"
        );
        let preparation = Output::prepare(&output, &[&protected], &cancelled);
        assert_eq!(
            validation.is_err(),
            expected_overlap,
            "absent {name:?}/{alias:?}: readonly validation must match native lookup: {:?}",
            validation.err()
        );
        assert_eq!(
            preparation.is_err(),
            expected_overlap,
            "absent {name:?}/{alias:?}: normal prepare must match native lookup"
        );
        assert!(!output.exists());
        assert!(!protected.exists());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn absent_reserved_directory_aliases_are_rejected_without_writes() {
    let cancelled = AtomicBool::new(false);
    for protected_alias in [false, true] {
        let parent = temporary();
        let alias = parent.path().join(".GSP-OUTPUT-LOCKS");
        let output = if protected_alias {
            parent.path().join("result")
        } else {
            alias.clone()
        };
        let expected_overlap =
            native_aliases(parent.path(), ".gsp-output-locks", ".GSP-OUTPUT-LOCKS");
        let protected = if protected_alias {
            vec![alias.as_path()]
        } else {
            vec![]
        };
        let validation = Output::validate(&output, &protected, &cancelled);
        assert_eq!(fs::read_dir(parent.path()).unwrap().count(), 0);
        assert_eq!(validation.is_err(), expected_overlap);
        let preparation = Output::prepare(&output, &protected, &cancelled);
        assert_eq!(preparation.is_err(), expected_overlap);
        if protected_alias || !expected_overlap {
            assert!(!output.exists());
        }
    }
}

#[test]
fn validation_with_existing_lock_representative_agrees_with_prepare() {
    let cancelled = AtomicBool::new(false);
    for (name, alias) in [
        ("Result", "result"),
        ("Résult", "Re\u{301}sult"),
        ("Result", "Cache"),
    ] {
        let parent = temporary();
        let output = parent.path().join(name);
        let protected = parent.path().join(alias);
        drop(Output::prepare(&output, &[], &cancelled).unwrap());
        let lock_path = parent.path().join(".gsp-output-locks").join(name);
        let before = fs::read(&lock_path).unwrap();
        let modified = fs::metadata(&lock_path).unwrap().modified().unwrap();
        let validation = Output::validate(&output, &[&protected], &cancelled);
        assert_eq!(fs::read(&lock_path).unwrap(), before);
        assert_eq!(
            fs::metadata(&lock_path).unwrap().modified().unwrap(),
            modified
        );
        let preparation = Output::prepare(&output, &[&protected], &cancelled);
        assert_eq!(validation.is_err(), preparation.is_err());
        assert!(!output.exists());
        assert!(!protected.exists());
    }
}

#[test]
fn distinct_absent_output_and_cache_names_remain_valid() {
    let parent = temporary();
    let output = parent.path().join("result");
    let protected = parent.path().join("cache");
    let cancelled = AtomicBool::new(false);
    assert!(
        Output::validate(&output, &[&protected], &cancelled)
            .unwrap()
            .is_none()
    );
    assert_eq!(fs::read_dir(parent.path()).unwrap().count(), 0);
    assert!(Output::prepare(&output, &[&protected], &cancelled).is_ok());
    assert!(!output.exists());
    assert!(!protected.exists());
}
