use std::fs;

use game_screen_pick::{
    config::{Backend, Config, InferenceLevel},
    domain::{GameInput, SelectionMethod},
};

const VALID: &str = r#"
[selection]
method = "sampled_frames"
[ai]
backend = "vllm"
base_url = "http://127.0.0.1:8000/v1"
model = "vision-model"
inference_level = "none"
timeout_seconds = 900
cache_revision = "1"
"#;

fn error_text(source: &str) -> String {
    let error = Config::parse(source).expect_err("invalid configuration must fail");
    format!("{error:#}\n{error:?}")
}

#[test]
fn explicit_configuration_and_example_parse() {
    let config = Config::parse(VALID).unwrap();
    assert_eq!(config.selection.method, SelectionMethod::SampledFrames);
    assert_eq!(config.ai.backend, Backend::Vllm);
    assert_eq!(config.ai.inference_level, InferenceLevel::None);
    assert_eq!(config.ai.timeout_seconds, 900);
    assert_eq!(config.ai.effective_api_key(None), None);
    Config::parse(include_str!("../config.example.toml")).unwrap();

    let other = VALID
        .replace("sampled_frames", "semantic_video")
        .replace("vllm", "strata")
        .replace("\"none\"", "\"high\"");
    let config = Config::parse(&other).unwrap();
    assert_eq!(config.selection.method, SelectionMethod::SemanticVideo);
    assert_eq!(config.ai.backend, Backend::Strata);
    assert_eq!(config.ai.inference_level, InferenceLevel::High);
}

#[test]
fn every_required_field_is_explicit() {
    for assignment in [
        "method = \"sampled_frames\"\n",
        "backend = \"vllm\"\n",
        "base_url = \"http://127.0.0.1:8000/v1\"\n",
        "model = \"vision-model\"\n",
        "inference_level = \"none\"\n",
        "timeout_seconds = 900\n",
        "cache_revision = \"1\"\n",
    ] {
        assert!(
            error_text(&VALID.replace(assignment, "")).contains("missing required"),
            "{assignment} must not have a default"
        );
    }
    assert!(Config::parse("").is_err());
}

#[test]
fn unknown_keys_and_future_settings_are_rejected() {
    for source in [
        format!("extra = 1\n{VALID}"),
        VALID.replace("[ai]", "unexpected = true\n[ai]"),
        format!("{VALID}ollama_api_key = \"SECRET_VALUE\"\n"),
        format!("{VALID}\n[runtime]\nstart_command = [\"echo\"]\n"),
        format!("{VALID}\n[media]\nworkers = 1\n"),
        format!("{VALID}\n[search]\nprovider = \"brave\"\n"),
    ] {
        let error = error_text(&source);
        assert!(error.contains("unknown configuration key"), "{error}");
        assert!(!error.contains("SECRET_VALUE"));
    }
}

#[test]
fn invalid_types_and_toml_do_not_quote_secrets() {
    const SECRET: &str = "DO_NOT_DISCLOSE_THIS_AUTH_VALUE";
    for source in [
        format!("{VALID}api_key = [\"{SECRET}\"]\n"),
        format!("{VALID}api_key = \"{SECRET}\n"),
        format!("{VALID}{SECRET} = 1\n"),
        VALID.replace(
            "timeout_seconds = 900",
            &format!("timeout_seconds = \"{SECRET}\""),
        ),
        VALID.replace(
            "model = \"vision-model\"",
            &format!("model = [\"{SECRET}\"]"),
        ),
        format!("{VALID}api_key = \"{SECRET}\"\napi_key = \"duplicate\"\n"),
    ] {
        let error = error_text(&source);
        assert!(!error.contains(SECRET), "parse diagnostic exposed a secret");
        assert!(error.contains("configuration") || error.contains("TOML"));
    }
}

#[test]
fn unsupported_values_blank_fields_and_timeout_bounds_fail() {
    for (from, to) in [
        ("\"sampled_frames\"", "\"unknown\""),
        ("\"vllm\"", "\"ollama\""),
        ("\"none\"", "\"maximum\""),
        ("model = \"vision-model\"", "model = \"  \""),
        ("cache_revision = \"1\"", "cache_revision = \"\\t\""),
        ("timeout_seconds = 900", "timeout_seconds = 0"),
        ("timeout_seconds = 900", "timeout_seconds = 3601"),
        ("timeout_seconds = 900", "timeout_seconds = -1"),
        ("timeout_seconds = 900", "timeout_seconds = 1.5"),
        ("timeout_seconds = 900", "timeout_seconds = true"),
    ] {
        assert!(Config::parse(&VALID.replace(from, to)).is_err(), "{to}");
    }
    for timeout in [1, 3600] {
        Config::parse(&VALID.replace(
            "timeout_seconds = 900",
            &format!("timeout_seconds = {timeout}"),
        ))
        .unwrap();
    }
    for level in ["none", "low", "medium", "high"] {
        Config::parse(&VALID.replace("\"none\"", &format!("\"{level}\""))).unwrap();
    }
}

#[test]
fn connection_url_is_normalized_without_assuming_a_service_path() {
    for (provided, expected) in [
        ("HTTP://LOCALHOST:80/v1/", "http://localhost/v1"),
        (
            "https://example.test/ai/custom///",
            "https://example.test/ai/custom",
        ),
        ("https://example.test", "https://example.test/"),
        (
            "https://example.test/custom%20service/",
            "https://example.test/custom%20service",
        ),
    ] {
        let config = Config::parse(&VALID.replace("http://127.0.0.1:8000/v1", provided)).unwrap();
        assert_eq!(config.ai.base_url.as_str(), expected);
    }
}

#[test]
fn connection_url_rejects_invalid_or_secret_components() {
    for provided in [
        "relative/v1",
        "http:/example.test/v1",
        "file:///tmp/service",
        "https://",
        "https://DO_NOT_DISCLOSE@example.test/v1",
        "https://user:DO_NOT_DISCLOSE@example.test/v1",
        "https://@example.test/v1",
        "https://example.test/v1?api_key=DO_NOT_DISCLOSE",
        "https://example.test/v1?",
        "https://example.test/v1#DO_NOT_DISCLOSE",
        "https://example.test/v1#",
        "https://example.test/\\nDO_NOT_DISCLOSE",
    ] {
        let source = VALID.replace("http://127.0.0.1:8000/v1", provided);
        let error = error_text(&source);
        assert!(error.contains("ai.base_url"), "{provided}: {error}");
        assert!(!error.contains("DO_NOT_DISCLOSE"));
    }
}

#[test]
fn authentication_precedence_and_redaction() {
    let configured = Config::parse(&format!("{VALID}api_key = \" CONFIG_SECRET \"\n")).unwrap();
    assert_eq!(
        configured.ai.effective_api_key(Some("ENV_SECRET")),
        Some("CONFIG_SECRET")
    );
    assert_eq!(configured.ai.effective_api_key(None), Some("CONFIG_SECRET"));
    let debug = format!("{configured:?}");
    assert!(!debug.contains("CONFIG_SECRET"));
    assert!(debug.contains("[REDACTED]"));

    for suffix in ["", "api_key = \"\"\n", "api_key = \" \\t \"\n"] {
        let config = Config::parse(&format!("{VALID}{suffix}")).unwrap();
        assert_eq!(
            config.ai.effective_api_key(Some(" ENV_SECRET ")),
            Some("ENV_SECRET")
        );
        assert_eq!(config.ai.effective_api_key(Some(" ")), None);
        assert_eq!(config.ai.effective_api_key(None), None);
        assert!(!format!("{config:?}").contains("ENV_SECRET"));
    }
}

#[test]
fn named_environment_fallback_is_resolved_without_process_global_mutation() {
    if std::env::var("GSP_AUTH_TEST_CHILD").as_deref() == Ok("1") {
        let config = Config::parse(VALID).unwrap();
        assert_eq!(config.ai.resolved_api_key().as_deref(), Some("ENV_SECRET"));
        let config = Config::parse(&format!("{VALID}api_key = \"CONFIG_SECRET\"\n")).unwrap();
        assert_eq!(
            config.ai.resolved_api_key().as_deref(),
            Some("CONFIG_SECRET")
        );
        return;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "named_environment_fallback_is_resolved_without_process_global_mutation",
        ])
        .env("GSP_AUTH_TEST_CHILD", "1")
        .env("GAME_SCREEN_PICK_API_KEY", " ENV_SECRET ")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "environment fallback subprocess failed"
    );
    assert!(!String::from_utf8_lossy(&output.stdout).contains("ENV_SECRET"));
}

#[test]
fn loading_a_missing_or_invalid_file_does_not_fall_back() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("設定.toml");
    assert!(Config::load(&path).is_err());
    fs::write(&path, "invalid SECRET_VALUE").unwrap();
    let error = Config::load(&path).unwrap_err();
    assert!(!format!("{error:#}\n{error:?}").contains("SECRET_VALUE"));
    fs::write(&path, [0xff]).unwrap();
    assert!(Config::load(&path).is_err());
    fs::write(&path, VALID).unwrap();
    assert_eq!(Config::load(&path).unwrap().ai.model, "vision-model");
}

#[test]
fn game_input_requires_one_nonblank_source() {
    for (title, context) in [
        (None, None),
        (Some("title"), Some("context")),
        (Some(" "), None),
        (None, Some("\t\n")),
    ] {
        assert!(GameInput::new(title.map(str::to_owned), context.map(str::to_owned)).is_err());
    }
    assert_eq!(
        GameInput::new(Some(" ゲーム名 ".to_owned()), None).unwrap(),
        GameInput::Title("ゲーム名".to_owned())
    );
    assert_eq!(
        GameInput::new(None, Some(" 探索・会話を重視 ".to_owned())).unwrap(),
        GameInput::Context("探索・会話を重視".to_owned())
    );
}
