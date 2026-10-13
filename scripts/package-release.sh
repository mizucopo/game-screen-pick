#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
target=$1
case "$target" in
  aarch64-apple-darwin)
    test "$(uname -s):$(uname -m)" = Darwin:arm64
    export MACOSX_DEPLOYMENT_TARGET=14.0
    minimum_os='macOS 14.0'
    tar_command=gtar
    ;;
  x86_64-unknown-linux-gnu)
    test "$(uname -s):$(uname -m)" = Linux:x86_64
    minimum_os='Linux x86_64, glibc 2.35'
    tar_command=tar
    ;;
  *) printf 'Unsupported distribution target: %s\n' "$target" >&2; exit 1 ;;
esac
destination=${2:-dist}
mkdir -p "$destination"
destination=$(cd "$destination" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT HUP INT TERM
cargo build --release --locked --target "$target"
cargo metadata --locked --offline --format-version 1 --filter-platform "$target" > "$work/metadata.json"
version=$(jq -er '. as $metadata | .packages[] | select(.id == $metadata.resolve.root) | .version' "$work/metadata.json")
stem=game-screen-pick-$version-$target
root=$work/$stem
mkdir -p "$root/bin" "$root/licenses" "$root/assets"
cargo_target=$(jq -er '.target_directory' "$work/metadata.json")
cp "$cargo_target/$target/release/game-screen-pick" "$root/bin/game-screen-pick"
test "$("$root/bin/game-screen-pick" --version)" = "game-screen-pick $version"
sh scripts/inspect-binary.sh "$root/bin/game-screen-pick" "$target" "$root/DEPENDENCIES.txt"
{
  printf '\nExternal runtime commands (not bundled): ffmpeg, ffprobe\n'
  printf '\nBuild platform: '; uname -s -m
  printf '\nBuild host FFmpeg:\n'; ffmpeg -version
  printf '\nBuild host FFprobe:\n'; ffprobe -version
} >> "$root/DEPENDENCIES.txt"
cp README.md CONTRIBUTING.md LICENSE config.example.toml "$root/"
mkdir "$root/docs"
for document in release technical-reference ai-client brave-search acceptance verification; do
  cp "docs/$document.md" "$root/docs/"
done
cp assets/OFL.txt "$root/assets/OFL.txt"
sh scripts/collect-licenses.sh "$work/metadata.json" "$root/licenses"
commit=$(git rev-parse HEAD)
dirty=false
if [ -n "$(git status --porcelain)" ]; then dirty=true; fi
binary_sha256=$(shasum -a 256 "$root/bin/game-screen-pick" | awk '{print $1}')
jq -n --arg version "$version" --arg target "$target" --arg commit "$commit" \
  --argjson dirty "$dirty" --arg minimum_os "$minimum_os" \
  --arg rustc "$(rustc --version)" --arg cargo "$(cargo --version)" \
  --arg binary_sha256 "$binary_sha256" \
  '{schema:1, version:$version, target:$target, commit:$commit, dirty:$dirty,
    minimum_os:$minimum_os, rustc:$rustc, cargo:$cargo,
    binary_sha256:$binary_sha256, external_commands:["ffmpeg","ffprobe"]}' > "$root/BUILD-INFO.json"
# Stable metadata and gzip headers make reruns comparable. Changed compilers or
# inputs can still change bytes; publication refuses to overwrite those assets.
epoch=$(git show -s --format=%ct HEAD)
"$tar_command" --sort=name --mtime="@$epoch" --owner=0 --group=0 --numeric-owner \
  -C "$work" -cf "$work/archive.tar" "$stem"
gzip -n -c "$work/archive.tar" > "$work/$stem.tar.gz"
(cd "$work" && shasum -a 256 "$stem.tar.gz" > "$stem.tar.gz.sha256")
mv "$work/$stem.tar.gz" "$destination/$stem.tar.gz"
mv "$work/$stem.tar.gz.sha256" "$destination/$stem.tar.gz.sha256"
printf '%s\n' "$destination/$stem.tar.gz"
