//! Loopback contracts only; no live Brave plan or model capability is claimed.
use super::*;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

const SECRET: &str = "BRAVE_AUTHENTICATION_MUST_STAY_PRIVATE";

#[derive(Clone)]
struct Received {
    target: String,
    headers: String,
    body: Value,
}

struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    header_delay: Duration,
    body_delay: Duration,
    include_length: bool,
}

impl Reply {
    fn json(value: Value) -> Self {
        Self {
            status: 200,
            headers: vec![("Content-Type".to_owned(), "application/json".to_owned())],
            body: serde_json::to_vec(&value).unwrap(),
            header_delay: Duration::ZERO,
            body_delay: Duration::ZERO,
            include_length: true,
        }
    }

    fn redirect(destination: String) -> Self {
        Self {
            status: 307,
            headers: vec![("Location".to_owned(), destination)],
            ..Self::json(json!({}))
        }
    }
}

struct Mock {
    url: String,
    requests: Arc<Mutex<Vec<Received>>>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl Mock {
    fn new(mut handler: impl FnMut(&Received, usize) -> Reply + Send + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = requests.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let worker = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(15);
            while !stopping.load(Ordering::Relaxed) && Instant::now() < deadline {
                let (mut stream, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(3));
                        continue;
                    }
                    Err(error) => panic!("mock accept: {error}"),
                };
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut bytes = Vec::new();
                let mut chunk = [0; 4096];
                let header_end = loop {
                    let count = stream.read(&mut chunk).unwrap();
                    assert_ne!(count, 0);
                    bytes.extend_from_slice(&chunk[..count]);
                    assert!(bytes.len() <= 16 * 1024);
                    if let Some(end) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
                        break end + 4;
                    }
                };
                let headers = String::from_utf8(bytes[..header_end].to_vec()).unwrap();
                let length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap_or(0);
                assert!(length <= 256 * 1024);
                while bytes.len() - header_end < length {
                    let count = stream.read(&mut chunk).unwrap();
                    assert_ne!(count, 0);
                    bytes.extend_from_slice(&chunk[..count]);
                }
                let first = headers.lines().next().unwrap();
                assert!(first.starts_with("GET ") || first.starts_with("POST "));
                let request = Received {
                    target: first.split_whitespace().nth(1).unwrap().to_owned(),
                    headers,
                    body: if length == 0 {
                        Value::Null
                    } else {
                        serde_json::from_slice(&bytes[header_end..header_end + length]).unwrap()
                    },
                };
                let index = {
                    let mut requests = recorded.lock().unwrap();
                    let index = requests.len();
                    requests.push(request.clone());
                    index
                };
                let reply = handler(&request, index);
                thread::sleep(reply.header_delay);
                let mut head = format!("HTTP/1.1 {} Test\r\nConnection: close\r\n", reply.status);
                if reply.include_length {
                    head.push_str(&format!("Content-Length: {}\r\n", reply.body.len()));
                }
                for (name, value) in reply.headers {
                    head.push_str(&format!("{name}: {value}\r\n"));
                }
                head.push_str("\r\n");
                if stream.write_all(head.as_bytes()).is_ok() {
                    thread::sleep(reply.body_delay);
                    let _ = stream.write_all(&reply.body);
                }
            }
        });
        Self {
            url,
            requests,
            stop,
            worker: Some(worker),
        }
    }

    fn requests(&self) -> Vec<Received> {
        self.requests.lock().unwrap().clone()
    }
}

impl Drop for Mock {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let result = self.worker.take().unwrap().join();
        if !thread::panicking() {
            result.unwrap();
        }
    }
}

fn config(url: &str) -> BraveConfig {
    BraveConfig {
        endpoint: format!("{url}/res/v1/web/search"),
        api_key: Some(SECRET.to_owned()),
        timeout_seconds: 5,
        max_response_bytes: 1024 * 1024,
        result_count: 5,
        country: "US".to_owned(),
        search_lang: "en".to_owned(),
        quota_stop_confirmed: true,
        quota_contract_reference: Some("applicable-free-hard-stop-account-contract-v1".to_owned()),
        context_generation_confirmed: true,
        context_contract_reference: Some("applicable-context-inference-contract-v1".to_owned()),
        generated_context_persistence_allowed: false,
        storage_contract_reference: None,
    }
}

fn result() -> Value {
    json!({"type":"search","web":{"results":[{"title":"Example game official", "url":"https://example.com/game",
        "description":"An exploration game with quests and unlocks.", "extra_snippets":["Representative exploration and encounter screens."]}]}})
}

fn evidence() -> SearchEvidence {
    SearchEvidence {
        queries: vec!["first query".to_owned(), "second query".to_owned()],
        results: vec![SearchResult {
            query_index: 0,
            title: "Example game official".to_owned(),
            url: "https://example.com/game".to_owned(),
            description: "An exploration game with quests and unlocks.".to_owned(),
            extra_snippets: vec![],
        }],
    }
}

fn context() -> Value {
    json!({"game_title":"Example", "identification":"Example from Studio, PC edition", "genre":"Exploration RPG",
        "progression":["Complete quests to unlock regions"],"representative_screens":["Exploration in a newly unlocked region"],
        "selection_priorities":["Readable quest and exploration scenes"],"ambiguous":false,"contradictions":[],
        "source_urls":["https://example.com/game"]})
}

#[test]
fn fixed_queries_parameters_and_transient_evidence() {
    let mock = Mock::new(|_, _| Reply::json(result()));
    let evidence = BraveClient::new(&config(&mock.url))
        .unwrap()
        .search("Example + 日本語")
        .unwrap();
    assert_eq!(evidence.queries.len(), 2);
    assert_eq!(evidence.results.len(), 2);
    assert_eq!(evidence.results[1].query_index, 1);
    let requests = mock.requests();
    assert_eq!(requests.len(), 2);
    for (i, request) in requests.iter().enumerate() {
        let url = Url::parse(&format!("{}{}", mock.url, request.target)).unwrap();
        assert_eq!(url.path(), "/res/v1/web/search");
        let query: std::collections::HashMap<_, _> = url.query_pairs().into_owned().collect();
        assert_eq!(query["q"], evidence.queries[i]);
        for (name, value) in [
            ("count", "5"),
            ("country", "US"),
            ("search_lang", "en"),
            ("extra_snippets", "true"),
            ("text_decorations", "false"),
            ("spellcheck", "false"),
            ("safesearch", "moderate"),
        ] {
            assert_eq!(query[name], value);
        }
        assert!(request.headers.contains(SECRET));
        assert!(!request.target.contains(SECRET));
    }
    assert!(!serde_json::to_string(&evidence).unwrap().contains(SECRET));
}

#[test]
fn strict_configuration_and_contract_are_separate() {
    let absent: BraveConfig = toml::from_str("").unwrap();
    absent.validate().unwrap();
    assert!(!absent.persist_context());
    assert!(absent.validate_search_contract().is_err());
    assert!(BraveClient::new(&absent).is_err());
    assert!(toml::from_str::<BraveConfig>("result_count = '5'").is_err());
    assert!(toml::from_str::<BraveConfig>("provider = 'paid-fallback'").is_err());
    assert!(toml::from_str::<BraveConfig>("quota_stop_confirmed = 'true'").is_err());
}

#[test]
fn no_account_contract_means_no_search_even_with_key() {
    let mock = Mock::new(|_, _| Reply::json(result()));
    for field in ["quota", "generation", "storage"] {
        let mut settings = config(&mock.url);
        match field {
            "quota" => settings.quota_stop_confirmed = false,
            "generation" => settings.context_contract_reference = None,
            "storage" => settings.generated_context_persistence_allowed = true,
            _ => unreachable!(),
        }
        settings.validate().unwrap();
        assert!(BraveClient::new(&settings).is_err());
    }
    assert!(mock.requests().is_empty());
}

#[test]
fn search_identity_excludes_credentials_includes_policy_and_parameters() {
    let mut settings = config("http://127.0.0.1:12345");
    let original = settings.search_conditions().unwrap();
    assert!(!original.contains(SECRET));
    assert!(!format!("{settings:?}").contains(SECRET));
    settings.api_key = Some("OTHER_AUTHENTICATION_MATERIAL".to_owned());
    assert_eq!(original, settings.search_conditions().unwrap());
    settings.result_count = 10;
    assert_ne!(original, settings.search_conditions().unwrap());
    settings.result_count = 5;
    settings.quota_contract_reference = Some("account-contract-v2".to_owned());
    assert_ne!(original, settings.search_conditions().unwrap());
    settings.quota_contract_reference =
        Some("applicable-free-hard-stop-account-contract-v1".to_owned());
    settings.generated_context_persistence_allowed = true;
    settings.storage_contract_reference =
        Some("generated-and-downstream-storage-granted".to_owned());
    assert!(settings.persist_context());
    assert_ne!(original, settings.search_conditions().unwrap());
    settings.storage_contract_reference = Some(settings.api_key.clone().unwrap());
    assert!(settings.search_conditions().is_err());
    assert!(!format!("{settings:?}").contains("OTHER_AUTHENTICATION_MATERIAL"));
}

#[test]
fn only_official_https_and_literal_loopback_mock_endpoints_are_allowed() {
    let mut settings = config("http://127.0.0.1:12345");
    for endpoint in [
        BRAVE_ENDPOINT,
        "http://127.0.0.1:12345/res/v1/web/search",
        "http://[::1]:12345/res/v1/web/search",
    ] {
        settings.endpoint = endpoint.to_owned();
        settings.validate().unwrap();
    }
    for endpoint in [
        "http://api.search.brave.com/res/v1/web/search",
        "https://other.example/res/v1/web/search",
        "http://localhost:12345/res/v1/web/search",
        "http://127.0.0.1:12345/wrong-path",
        "http://key@127.0.0.1:12345/res/v1/web/search",
        "https://api.search.brave.com/res/v1/web/search?key=value",
    ] {
        settings.endpoint = endpoint.to_owned();
        assert!(settings.validate().is_err(), "{endpoint}");
    }
}

#[test]
fn invalid_limits_and_references_fail_structural_validation() {
    let mut settings = config("http://127.0.0.1:12345");
    settings.result_count = 21;
    assert!(settings.validate().is_err());
    settings.result_count = 5;
    settings.country = "us".to_owned();
    assert!(settings.validate().is_err());
    settings.country = "US".to_owned();
    settings.quota_contract_reference = Some(" \nsecret".to_owned());
    assert!(settings.validate().is_err());
}

#[test]
fn authentication_quota_and_http_errors_stop_without_retry_or_fallback() {
    for status in [401, 403, 429, 500] {
        for first_error in [0, 1] {
            let mock = Mock::new(move |_, i| {
                if i == first_error {
                    Reply {
                        status,
                        ..Reply::json(json!({"error":SECRET}))
                    }
                } else {
                    Reply::json(result())
                }
            });
            let error = BraveClient::new(&config(&mock.url))
                .unwrap()
                .search("Example")
                .unwrap_err()
                .to_string();
            assert!(!error.contains(SECRET));
            assert_eq!(mock.requests().len(), first_error + 1);
        }
    }
}

#[test]
fn exhausted_or_invalid_remaining_header_stops_before_the_next_query() {
    for remaining in ["0, 200", "2, 0", "invalid"] {
        let value = remaining.to_owned();
        let mock = Mock::new(move |_, _| Reply {
            headers: vec![
                ("Content-Type".to_owned(), "application/json".to_owned()),
                ("X-RateLimit-Remaining".to_owned(), value.clone()),
            ],
            ..Reply::json(result())
        });
        assert!(
            BraveClient::new(&config(&mock.url))
                .unwrap()
                .search("Example")
                .is_err()
        );
        assert_eq!(mock.requests().len(), 1);
    }
}

#[test]
fn every_repeated_quota_header_must_be_valid_and_unexhausted() {
    for remaining in ["0", "invalid"] {
        let value = remaining.to_owned();
        let mock = Mock::new(move |_, _| Reply {
            headers: vec![
                ("Content-Type".to_owned(), "application/json".to_owned()),
                ("X-RateLimit-Remaining".to_owned(), "2".to_owned()),
                ("X-RateLimit-Remaining".to_owned(), value.clone()),
            ],
            ..Reply::json(result())
        });
        assert!(
            BraveClient::new(&config(&mock.url))
                .unwrap()
                .search("Example")
                .is_err()
        );
        assert_eq!(mock.requests().len(), 1);
    }
}

#[test]
fn exhausted_redirect_response_is_not_followed() {
    let mock = Mock::new(|_, _| {
        let mut reply = Reply::redirect("/redirected".to_owned());
        reply
            .headers
            .push(("X-RateLimit-Remaining".to_owned(), "0".to_owned()));
        reply
    });
    assert!(
        BraveClient::new(&config(&mock.url))
            .unwrap()
            .search("Example")
            .is_err()
    );
    assert_eq!(mock.requests().len(), 1);
}

#[test]
fn generation_uses_shared_client_and_retries_domain_contract_failures() {
    let mock = Mock::new(|request, i| {
        assert_eq!(request.target, "/v1/chat/completions");
        assert_eq!(request.body["model"], "context-contract");
        assert_eq!(request.body["reasoning_effort"], "none");
        assert_eq!(request.body["response_format"]["type"], "json_schema");
        assert!(request.body.to_string().contains("untrusted external data"));
        assert!(
            request
                .body
                .to_string()
                .contains("https://example.com/game")
        );
        assert!(!request.body.to_string().contains(SECRET));
        let mut response = context();
        if i == 0 {
            response["ambiguous"] = json!(true);
        }
        Reply::json(
            json!({"model":"context-contract","choices":[{"index":0,"finish_reason":"stop",
            "message":{"role":"assistant","content":response.to_string(),"refusal":null}}]}),
        )
    });
    let settings = crate::config::Config::parse(&format!(
        r#"
[selection]
method = "sampled_frames"
[ai]
backend = "strata"
base_url = "{}/v1"
model = "context-contract"
inference_level = "none"
timeout_seconds = 5
cache_revision = "mock-context-v1"
[ai.protocol]
reasoning_format = "reasoning_effort"
supported_inference_levels = ["none"]
json_schema = true
max_output_tokens = 1024
max_request_bytes = 1048576
max_response_bytes = 16384
max_attempts = 2
"#,
        mock.url
    ))
    .unwrap()
    .ai;
    let client = OpenAiClient::new(&settings).unwrap();
    let generated = generate_context(&client, "Example", &evidence()).unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&generated).unwrap(),
        context()
    );
    assert_eq!(mock.requests().len(), 2);
}

#[test]
fn same_origin_redirect_preserves_authentication() {
    let mock = Mock::new(|_, i| {
        if i == 0 {
            Reply::redirect("/redirected?q=fixed".to_owned())
        } else {
            Reply::json(result())
        }
    });
    BraveClient::new(&config(&mock.url))
        .unwrap()
        .search("Example")
        .unwrap();
    let requests = mock.requests();
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[1].target, "/redirected?q=fixed");
    assert!(requests.iter().all(|r| r.headers.contains(SECRET)));
}

#[test]
fn cross_origin_redirect_never_receives_token() {
    let destination = Mock::new(|_, _| Reply::json(result()));
    let redirect_url = format!("{}/res/v1/web/search", destination.url);
    let mock = Mock::new(move |_, _| Reply::redirect(redirect_url.clone()));
    let error = BraveClient::new(&config(&mock.url))
        .unwrap()
        .search("Example")
        .unwrap_err()
        .to_string();
    assert!(error.contains("origin"));
    assert_eq!(mock.requests().len(), 1);
    assert!(destination.requests().is_empty());
}

#[test]
fn redirect_loop_has_a_bounded_attempt_count() {
    let mock = Mock::new(|_, _| Reply::redirect("/again".to_owned()));
    assert!(
        BraveClient::new(&config(&mock.url))
            .unwrap()
            .search("Example")
            .is_err()
    );
    assert_eq!(mock.requests().len(), 4);
}

#[test]
fn oversized_body_is_bounded_with_or_without_content_length() {
    for include_length in [true, false] {
        let mock = Mock::new(move |_, _| Reply {
            body: vec![b' '; 4096],
            include_length,
            ..Reply::json(result())
        });
        let mut settings = config(&mock.url);
        settings.max_response_bytes = 1024;
        assert!(
            BraveClient::new(&settings)
                .unwrap()
                .search("Example")
                .unwrap_err()
                .to_string()
                .contains("byte limit")
        );
        assert_eq!(mock.requests().len(), 1);
    }
}

#[test]
fn headers_and_body_share_the_search_deadline() {
    for delayed_body in [false, true] {
        let mock = Mock::new(move |_, _| Reply {
            header_delay: if delayed_body {
                Duration::ZERO
            } else {
                Duration::from_millis(1200)
            },
            body_delay: if delayed_body {
                Duration::from_millis(1200)
            } else {
                Duration::ZERO
            },
            ..Reply::json(result())
        });
        let mut settings = config(&mock.url);
        settings.timeout_seconds = 1;
        let start = Instant::now();
        let error = BraveClient::new(&settings)
            .unwrap()
            .search("Example")
            .unwrap_err()
            .to_string();
        assert!(error.contains("deadline"));
        assert!(start.elapsed() < Duration::from_millis(1150));
        assert_eq!(mock.requests().len(), 1);
    }
}

#[test]
fn deadline_is_not_reset_between_queries() {
    let mock = Mock::new(|_, i| Reply {
        body_delay: if i == 0 {
            Duration::from_millis(300)
        } else {
            Duration::from_millis(900)
        },
        ..Reply::json(result())
    });
    let mut settings = config(&mock.url);
    settings.timeout_seconds = 1;
    let start = Instant::now();
    assert!(
        BraveClient::new(&settings)
            .unwrap()
            .search("Example")
            .unwrap_err()
            .to_string()
            .contains("deadline")
    );
    assert!(start.elapsed() < Duration::from_millis(1150));
    assert_eq!(mock.requests().len(), 2);
}

#[test]
fn malformed_empty_api_error_and_credential_echo_responses_are_rejected() {
    for bad in [
        json!({"web":{"results":[]}}),
        json!({"error":{"detail":SECRET}}),
        json!({"web":{"results":[{"title":"Wrong","url":"file:///etc/passwd","description":"Not a source"}]}}),
        json!({"web":{"results":[{"title":format!("echo {SECRET}"),"url":"https://example.com/game","description":"secret"}]}}),
        json!({"unrelated":"response"}),
    ] {
        let mock = Mock::new(move |_, _| Reply::json(bad.clone()));
        let error = BraveClient::new(&config(&mock.url))
            .unwrap()
            .search("Example")
            .unwrap_err()
            .to_string();
        assert!(!error.contains(SECRET));
        assert_eq!(mock.requests().len(), 1);
    }
    let mock = Mock::new(|_, _| Reply {
        body: b"invalid json secret".to_vec(),
        ..Reply::json(result())
    });
    assert!(
        BraveClient::new(&config(&mock.url))
            .unwrap()
            .search("Example")
            .is_err()
    );
}

#[test]
fn nullable_optional_snippets_and_descriptions_preserve_usable_evidence() {
    for (description, snippets) in [
        (json!("Grounded genre and gameplay"), Value::Null),
        (Value::Null, json!(["Grounded genre and gameplay"])),
        (json!(""), json!(["Grounded genre and gameplay"])),
    ] {
        let mock = Mock::new(move |_, _| {
            let mut response = result();
            response["web"]["results"][0]["description"] = description.clone();
            response["web"]["results"][0]["extra_snippets"] = snippets.clone();
            response
                .pointer_mut("/web/results").unwrap().as_array_mut().unwrap()
                .push(json!({"title":"No text result","url":"https://example.com/no-text","description":null,"extra_snippets":null}));
            Reply::json(response)
        });
        let found = BraveClient::new(&config(&mock.url))
            .unwrap()
            .search("Example")
            .unwrap();
        assert_eq!(found.results.len(), 2);
        build_context_request("Example", &found).unwrap();
        assert_eq!(mock.requests().len(), 2);
    }
}

#[test]
fn a_query_with_no_usable_text_stops_before_more_search_or_generation() {
    let mock = Mock::new(|_, _| {
        Reply::json(
            json!({"web":{"results":[{"title":"No text","url":"https://example.com/no-text",
        "description":null,"extra_snippets":null}]}}),
        )
    });
    assert!(
        BraveClient::new(&config(&mock.url))
            .unwrap()
            .search("Example")
            .unwrap_err()
            .to_string()
            .contains("usable text")
    );
    assert_eq!(mock.requests().len(), 1);
}

#[test]
fn invalid_or_secret_title_fails_before_network() {
    let mock = Mock::new(|_, _| Reply::json(result()));
    let client = BraveClient::new(&config(&mock.url)).unwrap();
    for title in ["", "\nExample", " Example ", SECRET] {
        assert!(client.search(title).is_err());
    }
    assert!(mock.requests().is_empty());
}

#[test]
fn context_requires_identity_evidence_and_all_selection_fields() {
    let evidence = evidence();
    let request = build_context_request("Example", &evidence).unwrap();
    assert!(request.prompt.contains("untrusted external data"));
    let value = context();
    assert_eq!(
        serde_json::from_str::<Value>(&validate_context("Example", &evidence, &value).unwrap())
            .unwrap(),
        value
    );
    for field in [
        "identification",
        "genre",
        "progression",
        "representative_screens",
        "selection_priorities",
        "source_urls",
    ] {
        let mut bad = value.clone();
        bad.as_object_mut().unwrap().remove(field);
        assert!(validate_context("Example", &evidence, &bad).is_err());
    }
    for bad in [json!(""), json!(" ")] {
        let mut response = value.clone();
        response["genre"] = bad;
        assert!(validate_context("Example", &evidence, &response).is_err());
    }
}

#[test]
fn long_source_urls_use_the_same_bound_in_evidence_and_context() {
    let mut evidence = evidence();
    let long_url = format!("https://example.com/game?ref={}", "a".repeat(3000));
    evidence.results[0].url = long_url.clone();
    let mut response = context();
    response["source_urls"] = json!([long_url]);
    validate_context("Example", &evidence, &response).unwrap();
}

#[test]
fn ambiguous_conflicting_wrong_game_and_unknown_source_contexts_fail() {
    let evidence = evidence();
    let value = context();
    for (field, bad) in [
        ("ambiguous", json!(true)),
        (
            "contradictions",
            json!(["Sources disagree on the game edition"]),
        ),
        ("game_title", json!("Namesake")),
        ("source_urls", json!(["https://unknown.example/game"])),
        ("selection_priorities", json!([" "])),
    ] {
        let mut response = value.clone();
        response[field] = bad;
        assert!(validate_context("Example", &evidence, &response).is_err());
    }
    let mut response = value;
    response["raw_snippets"] = json!("not an allowed field");
    assert!(validate_context("Example", &evidence, &response).is_err());
}
