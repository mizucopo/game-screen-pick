#!/bin/sh
# Asset/provenance and publication state contracts, using local archives and a
# fake gh. No tags, releases, credentials, network, or remote assets are changed.
set -eu
cd "$(dirname "$0")/.."
repository=$(pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT HUP INT TERM
printf '%s\n' \
  '{"reason":"compiler-artifact","target":{"name":"workflow","kind":["test"]},"executable":"/tmp/fixture-runner"}' \
  '{"reason":"build-finished","success":true}' > "$scratch/cargo.jsonl"
runner=$(jq -serf scripts/test-runner.jq "$scratch/cargo.jsonl")
test "$runner" = /tmp/fixture-runner
printf '%s\n' '{"reason":"build-finished","success":true}' > "$scratch/no-runner.jsonl"
if jq -serf scripts/test-runner.jq "$scratch/no-runner.jsonl" > /dev/null 2>&1; then
  printf 'A missing test runner must fail\n' >&2; exit 1
fi
cat "$scratch/cargo.jsonl" "$scratch/cargo.jsonl" > "$scratch/duplicate-runner.jsonl"
if jq -serf scripts/test-runner.jq "$scratch/duplicate-runner.jsonl" > /dev/null 2>&1; then
  printf 'Multiple test runners must fail\n' >&2; exit 1
fi
version=1.0.0
commit=0123456789012345678901234567890123456789
mkdir "$scratch/dist" "$scratch/roots" "$scratch/tools" "$scratch/remote"
archive() {
  target=$1
  stem=game-screen-pick-$version-$target
  (cd "$scratch/roots" && tar -czf "$scratch/dist/$stem.tar.gz" "$stem")
  (cd "$scratch/dist" && shasum -a 256 "$stem.tar.gz" > "$stem.tar.gz.sha256")
}
for target in aarch64-apple-darwin x86_64-unknown-linux-gnu; do
  root=$scratch/roots/game-screen-pick-$version-$target
  mkdir -p "$root/bin" "$root/licenses/rust" "$root/docs" "$root/assets"
  for file in DEPENDENCIES.txt LICENSE config.example.toml README.md CONTRIBUTING.md \
    assets/OFL.txt licenses/NotoSansJP-OFL.txt licenses/THIRD-PARTY-LICENSES.txt \
    licenses/rust/COPYRIGHT.html licenses/rust/SOURCE-LICENSES.txt \
    docs/release.md docs/technical-reference.md docs/ai-client.md docs/brave-search.md \
    docs/acceptance.md docs/verification.md; do
    printf 'fixture\n' > "$root/$file"
  done
  printf '#!/bin/sh\nexit 0\n' > "$root/bin/game-screen-pick"
  chmod +x "$root/bin/game-screen-pick"
  printf '[{"name":"fixture","version":"1.0.0","license":"MIT"}]\n' > "$root/licenses/dependencies.json"
  case "$target" in
    aarch64-apple-darwin) minimum_os='macOS 14.0' ;;
    *) minimum_os='Linux x86_64, glibc 2.35' ;;
  esac
  binary_sha256=$(shasum -a 256 "$root/bin/game-screen-pick" | awk '{print $1}')
  jq -n --arg version "$version" --arg commit "$commit" --arg target "$target" \
    --arg minimum_os "$minimum_os" --arg binary_sha256 "$binary_sha256" \
    '{schema:1,version:$version,commit:$commit,target:$target,dirty:false,
      minimum_os:$minimum_os,binary_sha256:$binary_sha256,rustc:"fixture",cargo:"fixture",
      external_commands:["ffmpeg","ffprobe"]}' > "$root/BUILD-INFO.json"
  archive "$target"
done
verify() { sh scripts/verify-release-assets.sh "$scratch/dist" "$version" "$commit"; }
reject() {
  if "$@" > "$scratch/rejected.log" 2>&1; then
    printf 'Expected rejection: %s\n' "$*" >&2; exit 1
  fi
}
verify
cp -R "$scratch/dist" "$scratch/good"
mac=game-screen-pick-$version-aarch64-apple-darwin
root=$scratch/roots/$mac
cp "$root/BUILD-INFO.json" "$scratch/good-info.json"
rm "$scratch/dist/$mac.tar.gz"
reject verify
cp "$scratch/good/$mac.tar.gz" "$scratch/dist/"
printf 'extra\n' > "$scratch/dist/extra"
reject verify
rm "$scratch/dist/extra"
cat "$scratch/good/$mac.tar.gz.sha256" >> "$scratch/dist/$mac.tar.gz.sha256"
reject verify
cp "$scratch/good/$mac.tar.gz.sha256" "$scratch/dist/"
printf 'corrupt\n' >> "$scratch/dist/$mac.tar.gz"
reject verify
cp "$scratch/good/$mac.tar.gz" "$scratch/dist/"
for edit in '.commit="wrong"' '.version="2.0.0"' '.target="x86_64-unknown-linux-gnu"' \
  '.dirty=true' '.minimum_os="macOS 15.0"' '.binary_sha256="wrong"'; do
  jq "$edit" "$scratch/good-info.json" > "$root/BUILD-INFO.json"
  archive aarch64-apple-darwin
  reject verify
done
cp "$scratch/good-info.json" "$root/BUILD-INFO.json"
printf 'changed\n' >> "$root/bin/game-screen-pick"
archive aarch64-apple-darwin
reject verify
printf '#!/bin/sh\nexit 0\n' > "$root/bin/game-screen-pick"
ln -s /tmp "$root/escape"
archive aarch64-apple-darwin
reject verify
rm "$root/escape"
archive aarch64-apple-darwin
verify

# rustc's GNU binary may explicitly depend on glibc's loader as well as libc.
# Exercise that observed dependency while keeping non-system links and a higher
# glibc requirement forbidden. These commands simulate inspection, not execution.
cat > "$scratch/tools/uname" <<'EOF'
#!/bin/sh
printf 'Linux\n'
EOF
cat > "$scratch/tools/file" <<'EOF'
#!/bin/sh
printf 'ELF 64-bit LSB pie executable, x86-64\n'
EOF
cat > "$scratch/tools/readelf" <<'EOF'
#!/bin/sh
case "$1" in
  -d)
    for library in libc.so.6 libgcc_s.so.1 ld-linux-x86-64.so.2 ${GSP_FAKE_EXTRA_LIBRARY:-}; do
      printf ' 0x1 (NEEDED) Shared library: [%s]\n' "$library"
    done
    if [ "${GSP_FAKE_RPATH:-false}" = true ]; then printf ' 0x1 (RPATH) [/developer/lib]\n'; fi
    ;;
  --version-info) printf 'Name: GLIBC_%s\n' "${GSP_FAKE_GLIBC:-2.35}" ;;
  *) exit 1 ;;
esac
EOF
cat > "$scratch/tools/ldd" <<'EOF'
#!/bin/sh
if [ "${GSP_FAKE_UNRESOLVED:-false}" = true ]; then
  printf 'libc.so.6 => not found\n'
else
  printf 'libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x1234)\n'
fi
EOF
chmod +x "$scratch/tools/uname" "$scratch/tools/file" "$scratch/tools/readelf" "$scratch/tools/ldd"
inspect_linux() {
  PATH="$scratch/tools:$PATH" sh scripts/inspect-binary.sh "$root/bin/game-screen-pick" \
    x86_64-unknown-linux-gnu "$scratch/dependencies.txt"
}
inspect_linux
export GSP_FAKE_EXTRA_LIBRARY=libopencv_core.so
reject inspect_linux
unset GSP_FAKE_EXTRA_LIBRARY
export GSP_FAKE_GLIBC=2.36
reject inspect_linux
unset GSP_FAKE_GLIBC
export GSP_FAKE_RPATH=true
reject inspect_linux
unset GSP_FAKE_RPATH
export GSP_FAKE_UNRESOLVED=true
reject inspect_linux
unset GSP_FAKE_UNRESOLVED
rm "$scratch/tools/uname" "$scratch/tools/file" "$scratch/tools/readelf" "$scratch/tools/ldd"

# The publication script's external effects are simulated. Only these fake gh
# upload/edit operations can touch the scratch state below.
cat > "$scratch/tools/git" <<'EOF'
#!/bin/sh
case "$1" in rev-parse|rev-list) printf '%s\n' "$GSP_FAKE_COMMIT" ;; *) exit 1 ;; esac
EOF
cat > "$scratch/tools/gh" <<'EOF'
#!/bin/sh
set -eu
case "$1:$2" in
  api:--paginate) jq -s '[.]' "$GSP_FAKE_STATE" ;;
  release:download)
    shift 3
    while [ "$#" -gt 0 ]; do
      case "$1" in --pattern) name=$2 ;; --dir) directory=$2 ;; esac
      shift 2
    done
    cp "$GSP_FAKE_REMOTE/$name" "$directory/$name"
    ;;
  release:upload)
    file=$4
    name=${file##*/}
    printf 'upload %s\n' "$name" >> "$GSP_FAKE_LOG"
    cp "$file" "$GSP_FAKE_REMOTE/$name"
    jq --arg name "$name" --argjson size "$(wc -c < "$file")" \
      '.assets += [{name:$name,state:"uploaded",size:$size}]' "$GSP_FAKE_STATE" > "$GSP_FAKE_STATE.new"
    mv "$GSP_FAKE_STATE.new" "$GSP_FAKE_STATE"
    ;;
  release:edit)
    printf 'edit\n' >> "$GSP_FAKE_LOG"
    if [ "${GSP_FAKE_KEEP_DRAFT:-false}" != true ]; then
      jq '.draft=false' "$GSP_FAKE_STATE" > "$GSP_FAKE_STATE.new"
      mv "$GSP_FAKE_STATE.new" "$GSP_FAKE_STATE"
    fi
    ;;
  *) printf 'Unexpected fake gh invocation\n' >&2; exit 1 ;;
esac
EOF
chmod +x "$scratch/tools/gh" "$scratch/tools/git"
export GSP_FAKE_COMMIT="$commit" GSP_FAKE_STATE="$scratch/release.json" \
  GSP_FAKE_REMOTE="$scratch/remote" GSP_FAKE_LOG="$scratch/effects.log"
export GITHUB_REPOSITORY=fixture/repository
reset_release() {
  jq -n --arg version "$version" --arg commit "$commit" \
    '{id:1,tag_name:$version,target_commitish:$commit,draft:true,prerelease:false,assets:[]}' > "$GSP_FAKE_STATE"
  rm -f "$scratch/remote/"* "$GSP_FAKE_LOG"
  : > "$GSP_FAKE_LOG"
}
publish() {
  PATH="$scratch/tools:$PATH" sh "$repository/scripts/publish-release-assets.sh" \
    "$scratch/dist" "$version" "$commit" "$version" false
}
verify_published() {
  PATH="$scratch/tools:$PATH" sh "$repository/scripts/verify-published-release.sh" \
    "$scratch/dist" "$version" "$commit" "$version" false
}
modify_release() { jq "$1" "$GSP_FAKE_STATE" > "$GSP_FAKE_STATE.new"; mv "$GSP_FAKE_STATE.new" "$GSP_FAKE_STATE"; }
reset_release
publish
test "$(grep -c '^upload ' "$GSP_FAKE_LOG")" = 4
test "$(grep -c '^edit$' "$GSP_FAKE_LOG")" = 1
cp "$GSP_FAKE_STATE" "$scratch/complete.json"
: > "$GSP_FAKE_LOG"
verify_published
publish
test ! -s "$GSP_FAKE_LOG"
for edit in '.target_commitish="wrong"' '.tag_name="wrong"' '.prerelease=true' \
  '.assets += [.assets[0]]' '.assets[0].size=0' '.assets[0].state="new"' \
  '.assets += [{name:"extra",state:"uploaded",size:1}]' '.assets=.assets[1:]'; do
  cp "$scratch/complete.json" "$GSP_FAKE_STATE"
  modify_release "$edit"
  reject publish
  reject verify_published
  test ! -s "$GSP_FAKE_LOG"
done
cp "$scratch/complete.json" "$GSP_FAKE_STATE"
printf 'changed\n' >> "$scratch/remote/$mac.tar.gz"
reject publish
reject verify_published
test ! -s "$GSP_FAKE_LOG"
cp "$scratch/dist/$mac.tar.gz" "$scratch/remote/"
modify_release '.draft=true | .assets=.assets[0:2]'
publish
test "$(grep -c '^upload ' "$GSP_FAKE_LOG")" = 2
test "$(grep -c '^edit$' "$GSP_FAKE_LOG")" = 1
reset_release
export GSP_FAKE_KEEP_DRAFT=true
reject publish
unset GSP_FAKE_KEEP_DRAFT
printf 'Distribution asset and publication contracts passed\n'
