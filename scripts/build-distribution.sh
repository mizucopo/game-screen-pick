#!/bin/sh
# Development/CI wrapper; the precompiled test harness is never in the archive.
set -eu
# Native jq.exe otherwise translates LF to CRLF inside shell values/files.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) jq() { command jq --binary "$@"; } ;;
esac
cd "$(dirname "$0")/.."
target=$1
archive=${2:-}
if [ -z "$archive" ]; then archive=$(sh scripts/package-release.sh "$target"); fi
if command -v cygpath > /dev/null 2>&1; then archive=$(cygpath -u "$archive"); fi
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT HUP INT TERM
cargo test --locked --target "$target" --test workflow --no-run --message-format=json > "$work/build.jsonl"
runner=$(jq -serf scripts/test-runner.jq "$work/build.jsonl")
if command -v cygpath > /dev/null 2>&1; then runner=$(cygpath -u "$runner"); fi
test -x "$runner"
case "$target" in
  aarch64-apple-darwin|x86_64-pc-windows-msvc|aarch64-pc-windows-msvc)
    GSP_ARTIFACT_TEST_RUNNER="$runner" sh tests/artifact-smoke.sh "$archive"
    ;;
  x86_64-unknown-linux-gnu)
    mkdir -p "$work/validation/tests"
    cp tests/artifact-smoke.sh "$work/validation/tests/"
    cp -R tests/fixtures "$work/validation/tests/fixtures"
    mkdir -p "$work/runtime-tools/bin"
    cp "$(command -v ffmpeg)" "$(command -v ffprobe)" "$work/runtime-tools/bin/"
    docker build --platform linux/amd64 --tag game-screen-pick-artifact-runtime \
      --file scripts/Dockerfile.artifact-runtime "$work/runtime-tools"
    docker run --rm --platform linux/amd64 --network none \
      --mount "type=bind,src=$(dirname "$archive"),dst=/dist,readonly" \
      --mount "type=bind,src=$runner,dst=/artifact-runner,readonly" \
      --mount "type=bind,src=$work/validation,dst=/validation,readonly" \
      --env GSP_ARTIFACT_TEST_RUNNER=/artifact-runner \
      --env GSP_ARTIFACT_FIXTURES=/validation/tests/fixtures \
      --workdir /validation game-screen-pick-artifact-runtime \
      sh tests/artifact-smoke.sh "/dist/$(basename "$archive")"
    ;;
  *) printf 'Unsupported smoke target: %s\n' "$target" >&2; exit 1 ;;
esac
