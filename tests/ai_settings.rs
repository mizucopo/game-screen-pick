use game_screen_pick::ai::OpenAiClient;
use game_screen_pick::cache;
use game_screen_pick::config::Config;
use game_screen_pick::run::ResolvedGameContext;
use serde_json::json;

const BASE: &str = r#"
[selection]
method = "sampled_frames"
[ai]
backend = "vllm"
base_url = "http://127.0.0.1:1/custom"
model = "fixture-vision"
inference_level = "none"
timeout_seconds = 10
cache_revision = "weights-server-v1"
"#;

const PROTOCOL: &str = r#"
[ai.protocol]
reasoning_format = "reasoning_effort"
supported_inference_levels = ["none", "low"]
json_schema = false
max_output_tokens = 2048
max_request_bytes = 1048576
max_response_bytes = 1048576
max_attempts = 3
[ai.protocol.media]
vision_revision = "encoder-projector-v1"
max_images = 12
max_image_pixels = 32000000
max_image_edge = 16384
max_video_bytes = 1048576
max_video_seconds = 10.0
"#;

fn config(source: &str) -> Config {
    Config::parse(source).unwrap()
}

fn key(source: &str) -> String {
    let config = config(source);
    let conditions = config
        .ai
        .cache_conditions(
            "prompt-v1",
            "schema-v1",
            "chunk-v1",
            ResolvedGameContext::direct("fixture context"),
        )
        .unwrap();
    cache::key(&json!({"conditions":conditions,"model":config.ai.model_conditions()})).unwrap()
}

#[test]
fn explicit_protocol_is_required_only_for_the_inference_boundary() {
    let base = config(BASE);
    assert!(OpenAiClient::new(&base.ai).is_err());
    assert!(
        base.ai
            .cache_conditions("p", "s", "m", ResolvedGameContext::direct("c"))
            .is_err()
    );
    let ready = config(&format!("{BASE}{PROTOCOL}"));
    // Port 1 has no server. Construction must still succeed without I/O.
    OpenAiClient::new(&ready.ai).unwrap();
}

#[test]
fn invalid_protocol_contracts_fail_without_disclosing_configuration_values() {
    let source = format!("{BASE}api_key = \"PRIVATE_AUTH\"\n{PROTOCOL}");
    for changed in [
        source.replace("reasoning_effort", "PRIVATE_AUTH"),
        source.replace("[\"none\", \"low\"]", "[\"none\", \"none\"]"),
        source.replace("[\"none\", \"low\"]", "[\"high\"]"),
        source.replace("max_attempts = 3", "max_attempts = 4"),
        source.replace("max_attempts = 3", "max_attempts = 0"),
        source.replace("max_output_tokens = 2048", "max_output_tokens = 0"),
        source.replace("max_response_bytes = 1048576", "max_response_bytes = 0"),
        source.replace(
            "max_request_bytes = 1048576",
            "max_request_bytes = 268435457",
        ),
        source.replace("encoder-projector-v1", " "),
        source.replace("max_images = 12", "max_images = 65"),
        source.replace(
            "max_image_pixels = 32000000",
            "max_image_pixels = 200000001",
        ),
        source.replace("max_image_edge = 16384", "max_image_edge = 0"),
        source.replace("max_video_seconds = 10.0", "max_video_seconds = nan"),
        source.replace("max_video_bytes = 1048576", "max_video_bytes = 0"),
        source.replace("max_video_seconds = 10.0", "max_video_seconds = 0.0"),
        source.replace("max_attempts = 3", "PRIVATE_AUTH = true"),
    ] {
        let error = Config::parse(&changed).unwrap_err();
        assert!(!format!("{error:#}\n{error:?}").contains("PRIVATE_AUTH"));
    }
}

#[test]
fn strata_rejects_a_declared_video_path_but_accepts_image_and_json_contracts() {
    let strata = format!("{BASE}{PROTOCOL}").replace("\"vllm\"", "\"strata\"");
    assert!(
        Config::parse(&strata)
            .unwrap_err()
            .to_string()
            .contains("semantic_video")
    );
    let images = strata
        .replace("max_video_bytes = 1048576", "max_video_bytes = 0")
        .replace("max_video_seconds = 10.0", "max_video_seconds = 0.0")
        .replace("json_schema = false", "json_schema = true");
    OpenAiClient::new(&config(&images).ai).unwrap();
}

#[test]
fn inference_cache_keys_include_all_effective_protocol_and_media_conditions() {
    let source = format!("{BASE}{PROTOCOL}");
    let original = key(&source);
    for (from, to) in [
        ("fixture-vision", "other-model"),
        ("127.0.0.1:1/custom", "localhost:2/custom"),
        ("inference_level = \"none\"", "inference_level = \"low\""),
        ("timeout_seconds = 10", "timeout_seconds = 11"),
        ("weights-server-v1", "weights-server-v2"),
        ("reasoning_effort", "chat_template"),
        ("json_schema = false", "json_schema = true"),
        ("max_output_tokens = 2048", "max_output_tokens = 4096"),
        ("max_request_bytes = 1048576", "max_request_bytes = 2097152"),
        (
            "max_response_bytes = 1048576",
            "max_response_bytes = 2097152",
        ),
        ("max_attempts = 3", "max_attempts = 2"),
        ("encoder-projector-v1", "encoder-projector-v2"),
        ("max_images = 12", "max_images = 6"),
        ("max_image_pixels = 32000000", "max_image_pixels = 16000000"),
        ("max_image_edge = 16384", "max_image_edge = 8192"),
        ("max_video_bytes = 1048576", "max_video_bytes = 2097152"),
        ("max_video_seconds = 10.0", "max_video_seconds = 11.0"),
    ] {
        assert_ne!(
            original,
            key(&source.replace(from, to)),
            "{from} must invalidate inference cache"
        );
    }
    // Backend switch must change the key even for the same declared image contract.
    let image_only = source
        .replace("max_video_bytes = 1048576", "max_video_bytes = 0")
        .replace("max_video_seconds = 10.0", "max_video_seconds = 0.0");
    assert_ne!(
        key(&image_only),
        key(&image_only.replace("\"vllm\"", "\"strata\""))
    );
    assert_eq!(
        original,
        key(&format!("{BASE}api_key = \"PRIVATE_AUTH\"\n{PROTOCOL}"))
    );
    let config = config(&format!("{BASE}api_key = \"PRIVATE_AUTH\"\n{PROTOCOL}"));
    let conditions = config
        .ai
        .cache_conditions("p", "s", "m", ResolvedGameContext::direct("c"))
        .unwrap();
    let metadata = serde_json::to_string(
        &json!({"conditions":conditions,"model":config.ai.model_conditions()}),
    )
    .unwrap();
    assert!(!metadata.contains("PRIVATE_AUTH"));
    assert!(!format!("{:?}", config.ai).contains("PRIVATE_AUTH"));
}
