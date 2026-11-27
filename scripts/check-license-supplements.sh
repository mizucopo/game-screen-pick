#!/bin/sh
# Resolve the same target graphs as stage-release.sh without compiling binaries.
set -eu
cd "$(dirname "$0")/.."
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT HUP INT TERM
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) jq() { command jq --binary "$@"; } ;;
esac
jq -er '.publication.rust.targets[]' .github/release.json > "$scratch/targets"
status=0
while IFS= read -r target; do
  cargo metadata --locked --format-version 1 --filter-platform "$target" > "$scratch/metadata.json"
  if ! sh scripts/collect-licenses.sh --check-supplements "$target" "$scratch/metadata.json" "$scratch/licenses"; then
    status=1
  fi
done < "$scratch/targets"
exit "$status"
