#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
fixture_build=$(mktemp -d)
trap 'rm -rf "$fixture_build"' EXIT HUP INT TERM
rustfmt --edition 2024 --check tests/fixture_inputs.rs
rustc --edition=2024 --deny warnings --test tests/fixture_inputs.rs -o "$fixture_build/check-fixtures"
"$fixture_build/check-fixtures"
fixture_root=${GSP_FIXTURES:-tests/fixtures}
jq -se 'length == 1 and (.[0] |
  type == "object" and
  (.matrix | type == "object" and length > 0 and
    all(.[]; type == "array" and length > 0)) and
  (.cases | type == "array" and length > 0 and
    all(.[]; all(.name, .setup, .expect; type == "string" and length > 0))))' \
  "$fixture_root/scenarios.json" > /dev/null
jq -se --argjson sources '["01-blocks.mkv", "02-mirrored.mkv"]' \
  'length == 1 and (.[0] | . as $script |
  (.candidates | map(.source) | unique) == $sources and
  (.semantic_video | type == "array" and length > 0) and
  (.semantic_video | map(.source) | sort) == $sources and
  all(.semantic_video[]; . as $video |
    (.events | type == "array" and length > 0) and
    all(.events[]; . as $event |
      .start_seconds <= .representative_seconds and .representative_seconds < .end_seconds and
      any($script.candidates[];
        .source == $video.source and .time_seconds == $event.representative_seconds))))' \
  "$fixture_root/responses.json" > /dev/null
