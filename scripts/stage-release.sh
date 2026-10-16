#!/bin/sh
# Product-owned evidence hook. Build/archive/checksum belong to release.py.
set -eu
# Native jq.exe otherwise translates LF to CRLF inside shell values/files.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) jq() { command jq --binary "$@"; } ;;
esac
cd "$(dirname "$0")/.."
target=${RUST_RELEASE_TARGET:?}
stage=${RUST_RELEASE_STAGE_DIR:?}
if command -v cygpath > /dev/null 2>&1; then stage=$(cygpath -u "$stage"); fi
case "$target" in
  aarch64-apple-darwin|x86_64-unknown-linux-gnu) binary=game-screen-pick ;;
  x86_64-pc-windows-msvc|aarch64-pc-windows-msvc) binary=game-screen-pick.exe ;;
  *) printf 'Unsupported distribution target: %s\n' "$target" >&2; exit 1 ;;
esac
minimum_os=$(jq -er --arg target "$target" '.[$target].minimum_os' scripts/distribution-targets.json)
test -s "$stage/$binary"
commit=$(git rev-parse HEAD)
dirty=false
if [ -n "$(git status --porcelain)" ]; then dirty=true; fi
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT HUP INT TERM
cargo metadata --locked --offline --format-version 1 --filter-platform "$target" > "$work/metadata.json"
version=$(jq -er '. as $metadata | .packages[] | select(.id == $metadata.resolve.root) | .version' "$work/metadata.json")
test "$RUST_RELEASE_TAG" = "$version"
test "$("$stage/$binary" --version)" = "game-screen-pick $version"
sh scripts/inspect-binary.sh "$stage/$binary" "$target" "$stage/DEPENDENCIES.txt"
{
  printf '\nExternal runtime commands (not bundled): ffmpeg, ffprobe\n'
  printf '\nBuild platform: '; uname -s -m
  printf '\nBuild host FFmpeg:\n'; ffmpeg -version
  printf '\nBuild host FFprobe:\n'; ffprobe -version
} >> "$stage/DEPENDENCIES.txt"
cp README.md CONTRIBUTING.md LICENSE config.example.toml "$stage/"
mkdir -p "$stage/docs" "$stage/assets"
for document in release technical-reference ai-client brave-search acceptance verification; do
  cp "docs/$document.md" "$stage/docs/"
done
cp assets/OFL.txt "$stage/assets/OFL.txt"
sh scripts/collect-licenses.sh "$work/metadata.json" "$stage/licenses"
test "$(git rev-parse HEAD)" = "$commit"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then dirty=true; fi
binary_sha256=$("${PYTHON:-python3}" -c 'import hashlib,sys; sys.stdout.write(hashlib.file_digest(open(sys.argv[1],"rb"),"sha256").hexdigest())' "$stage/$binary")
jq -n --arg version "$version" --arg target "$target" --arg commit "$commit" \
  --argjson dirty "$dirty" --arg minimum_os "$minimum_os" \
  --arg rustc "$(rustc --version)" --arg cargo "$(cargo --version)" \
  --arg binary_sha256 "$binary_sha256" \
  '{schema:1, version:$version, target:$target, commit:$commit, dirty:$dirty,
    minimum_os:$minimum_os, rustc:$rustc, cargo:$cargo,
    binary_sha256:$binary_sha256, external_commands:["ffmpeg","ffprobe"]}' > "$stage/BUILD-INFO.json"
