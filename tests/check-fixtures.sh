#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
fixture_build=$(mktemp -d)
trap 'rm -rf "$fixture_build"' EXIT HUP INT TERM
rustfmt --edition 2024 --check tests/fixture_inputs.rs
rustc --edition=2024 --deny warnings --test tests/fixture_inputs.rs -o "$fixture_build/check-fixtures"
"$fixture_build/check-fixtures"
fixture_root=${GSP_FIXTURES:-tests/fixtures}
jq empty "$fixture_root/responses.json" "$fixture_root/scenarios.json"
jq -e '. as $script |
  (.semantic_video | type == "array" and length > 0) and
  all(.semantic_video[]; . as $video |
    (.events | type == "array" and length > 0) and
    all(.events[]; . as $event |
    .start_seconds <= .representative_seconds and .representative_seconds < .end_seconds and
    any($script.candidates[];
      .source == $video.source and .time_seconds == $event.representative_seconds)))' \
  "$fixture_root/responses.json" > /dev/null
