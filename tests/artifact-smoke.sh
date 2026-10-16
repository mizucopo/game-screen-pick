#!/bin/sh
# Development-side harness; only the unpacked binary and FFmpeg run in its clean environment.
set -eu
cd "$(dirname "$0")/.."
test "$#" -eq 1 || { echo 'usage: sh tests/artifact-smoke.sh <archive>' >&2; exit 2; }
archive=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT HUP INT TERM
mkdir "$scratch/archive" "$scratch/tools"
case "$archive" in
  *.zip)
    "${PYTHON:-python3}" -c 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' \
      "$(cygpath -w "$archive")" "$(cygpath -w "$scratch/archive")"
    binary=$scratch/archive/game-screen-pick.exe
    ;;
  *.tar.gz)
    tar -xzf "$archive" -C "$scratch/archive"
    binary=$scratch/archive/game-screen-pick
    ;;
  *) echo 'Unsupported artifact archive' >&2; exit 1 ;;
esac
test -n "$binary" && test -x "$binary"
for tool in ffmpeg ffprobe; do
    tool_path=$(command -v "$tool")
    case "$binary" in
      *.exe) cp "$tool_path" "$scratch/tools/$tool.exe" ;;
      *) ln -s "$tool_path" "$scratch/tools/$tool" ;;
    esac
done
if command -v cygpath > /dev/null 2>&1; then
    binary=$(cygpath -w "$binary")
    artifact_tools=$(cygpath -w "$scratch/tools")
else
    artifact_tools=$scratch/tools
fi
export GSP_ARTIFACT_BINARY="$binary"
export GSP_ARTIFACT_TOOLS="$artifact_tools"
# A precompiled runner allows the Linux runtime container to have no toolchain.
run_test() {
    if [ -n "${GSP_ARTIFACT_TEST_RUNNER:-}" ]; then
        case "$("$GSP_ARTIFACT_TEST_RUNNER" --list)" in
            *"$1: test"*) ;;
            *) echo "missing artifact test: $1" >&2; exit 1 ;;
        esac
        "$GSP_ARTIFACT_TEST_RUNNER" "$1" --exact --nocapture
    else
        cargo test --locked --test workflow "$1" -- --exact --nocapture
    fi
}
for test_name in \
    cli_help_version_validation_extraction_and_unowned_output \
    cli_evidence_matrix_checks_pixels_labels_events_and_zero_call_warm \
    cli_rotated_multitrack_artifacts_keep_full_resolution_display_orientation \
    cli_resume_reuses_committed_media_and_context_and_matches_clean_cold \
    cli_shortage_http_and_schema_failures_preserve_previous_complete_output; do
    run_test "$test_name"
done
