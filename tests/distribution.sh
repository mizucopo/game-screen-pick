#!/bin/sh
# Asset/provenance and publication state contracts, using local archives and a
# fake gh. No tags, releases, credentials, network, or remote assets are changed.
set -eu
# Native jq.exe otherwise translates LF to CRLF inside shell values/files.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) jq() { command jq --binary "$@"; } ;;
esac
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
python=${PYTHON:-python3}
# Resolve before adding the fixture's python wrapper to PATH, including CI's
# PYTHON=python. Otherwise that wrapper would recursively execute itself.
python=$("$python" -c 'import sys; sys.stdout.write(sys.executable)')
if command -v cygpath > /dev/null 2>&1; then python=$(cygpath -u "$python"); fi
# The shared builder must read the Japanese Cargo metadata even on a Windows
# installation whose default text encoding is not UTF-8.
"$python" - <<'LOCALE_CONTRACT'
import contextlib, importlib.util, io, sys, tempfile
from pathlib import Path
read_text = Path.read_text
def legacy_read(self, encoding=None, *args, **kwargs):
    return read_text(self, encoding or "cp1252", *args, **kwargs)
Path.read_text = legacy_read
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("product", "scripts/distribution.py")
product = importlib.util.module_from_spec(spec)
spec.loader.exec_module(product)
builds = []
product.controller.build_rust_distribution = lambda *args: builds.append(args)
with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
    sys.argv = ["distribution.py", "build", "x86_64-pc-windows-msvc", directory]
    product.main()
assert len(builds) == 1 and builds[0][1]["release_tag"]
LOCALE_CONTRACT
sha256() {
  "$python" -c 'import hashlib,sys; sys.stdout.write(hashlib.file_digest(open(sys.argv[1],"rb"),"sha256").hexdigest())' "$1"
}
archive_name() {
  case "$1" in
    aarch64-apple-darwin) printf 'game-screen-pick-%s-macos-arm64.tar.gz\n' "$version" ;;
    x86_64-unknown-linux-gnu) printf 'game-screen-pick-%s-linux-x86_64.tar.gz\n' "$version" ;;
    x86_64-pc-windows-msvc) printf 'game-screen-pick-%s-windows-x86_64.zip\n' "$version" ;;
    aarch64-pc-windows-msvc) printf 'game-screen-pick-%s-windows-arm64.zip\n' "$version" ;;
  esac
}
archive() {
  target=$1
  name=$(archive_name "$target")
  "$python" - "$scratch/roots/$target" "$scratch/dist/$name" <<'ARCHIVE'
import sys, tarfile, zipfile
from pathlib import Path
root, archive = map(Path, sys.argv[1:])
if archive.name.endswith('.zip'):
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                bundle.write(path, path.relative_to(root).as_posix())
else:
    with tarfile.open(archive, 'w:gz') as bundle:
        for path in sorted(root.rglob('*')):
            if path.is_file() or path.is_symlink():
                entry = bundle.gettarinfo(path, arcname=path.relative_to(root).as_posix())
                if entry.name == 'game-screen-pick': entry.mode = 0o755
                if entry.isfile():
                    with path.open('rb') as stream: bundle.addfile(entry, stream)
                else:
                    bundle.addfile(entry)
ARCHIVE
  printf '%s  %s\n' "$(sha256 "$scratch/dist/$name")" "$name" > "$scratch/dist/$name.sha256"
}
for target in aarch64-apple-darwin x86_64-unknown-linux-gnu x86_64-pc-windows-msvc aarch64-pc-windows-msvc; do
  root=$scratch/roots/$target
  mkdir -p "$root/licenses/rust" "$root/docs" "$root/assets"
  for file in DEPENDENCIES.txt LICENSE config.example.toml README.md CONTRIBUTING.md \
    assets/OFL.txt licenses/NotoSansJP-OFL.txt licenses/THIRD-PARTY-LICENSES.txt \
    licenses/rust/COPYRIGHT.html licenses/rust/SOURCE-LICENSES.txt \
    docs/release.md docs/technical-reference.md docs/ai-client.md docs/brave-search.md \
    docs/acceptance.md docs/verification.md; do
    printf 'fixture\n' > "$root/$file"
  done
  case "$target" in *windows*) binary=game-screen-pick.exe ;; *) binary=game-screen-pick ;; esac
  printf '#!/bin/sh\nexit 0\n' > "$root/$binary"
  chmod +x "$root/$binary"
  printf '[{"name":"fixture","version":"1.0.0","license":"MIT"}]\n' > "$root/licenses/dependencies.json"
  minimum_os=$(jq -er --arg target "$target" '.[$target].minimum_os' scripts/distribution-targets.json)
  binary_sha256=$(sha256 "$root/$binary")
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
mac=game-screen-pick-$version-macos-arm64
root=$scratch/roots/aarch64-apple-darwin
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
printf 'changed\n' >> "$root/game-screen-pick"
archive aarch64-apple-darwin
reject verify
printf '#!/bin/sh\nexit 0\n' > "$root/game-screen-pick"
archive aarch64-apple-darwin
"$python" - "$scratch/dist/$mac.tar.gz" <<'TAR_LINK'
import io, sys, tarfile
archive = sys.argv[1]
with tarfile.open(archive) as bundle:
    entries = [(entry, bundle.extractfile(entry).read()) for entry in bundle.getmembers()]
with tarfile.open(archive, 'w:gz') as bundle:
    for entry, data in entries:
        bundle.addfile(entry, io.BytesIO(data))
    link = tarfile.TarInfo('escape')
    link.type, link.linkname = tarfile.SYMTYPE, '/tmp'
    bundle.addfile(link)
TAR_LINK
printf '%s  %s\n' "$(sha256 "$scratch/dist/$mac.tar.gz")" "$mac.tar.gz" > "$scratch/dist/$mac.tar.gz.sha256"
reject verify
archive aarch64-apple-darwin
verify

# Both native Windows artifacts must keep their own CPU, executable suffix,
# license inventory and minimum-OS evidence. ZIP path/link attacks are rejected
# before a downloaded release can be accepted.
for target in x86_64-pc-windows-msvc aarch64-pc-windows-msvc; do
  windows_root=$scratch/roots/$target
  name=$(archive_name "$target")
  cp "$windows_root/BUILD-INFO.json" "$scratch/windows-info.json"
  for edit in '.target="aarch64-apple-darwin"' '.minimum_os="wrong"' '.dirty=true'; do
    jq "$edit" "$scratch/windows-info.json" > "$windows_root/BUILD-INFO.json"
    archive "$target"
    reject verify
  done
  cp "$scratch/windows-info.json" "$windows_root/BUILD-INFO.json"
  mv "$windows_root/game-screen-pick.exe" "$windows_root/game-screen-pick"
  archive "$target"
  reject verify
  mv "$windows_root/game-screen-pick" "$windows_root/game-screen-pick.exe"
  mv "$windows_root/licenses/THIRD-PARTY-LICENSES.txt" "$scratch/windows-license.txt"
  archive "$target"
  reject verify
  mv "$scratch/windows-license.txt" "$windows_root/licenses/THIRD-PARTY-LICENSES.txt"
  for attack in traversal duplicate symlink; do
    archive "$target"
    "$python" - "$scratch/dist/$name" "$attack" <<'ZIP_ATTACK'
import stat, sys, zipfile
with zipfile.ZipFile(sys.argv[1], 'a') as bundle:
    if sys.argv[2] == 'traversal':
        bundle.writestr('../escape', 'not a product file')
    elif sys.argv[2] == 'duplicate':
        bundle.writestr('game-screen-pick.exe', 'substituted')
    else:
        entry = zipfile.ZipInfo('escape')
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(entry, '/tmp')
ZIP_ATTACK
    printf '%s  %s\n' "$(sha256 "$scratch/dist/$name")" "$name" > "$scratch/dist/$name.sha256"
    reject verify
  done
  archive "$target"
  verify
done

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
  PATH="$scratch/tools:$PATH" sh scripts/inspect-binary.sh "$root/game-screen-pick" \
    x86_64-unknown-linux-gnu "$scratch/dependencies.txt"
}
inspect_linux
export GSP_FAKE_EXTRA_LIBRARY=libunexpected.so
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
# Native jq.exe otherwise translates LF to CRLF inside shell values/files.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) jq() { command jq --binary "$@"; } ;;
esac
upload() {
    file=$1
  if command -v cygpath > /dev/null 2>&1; then file=$(cygpath -u "$file"); fi
    name=${file##*/}
    printf 'upload %s\n' "$name" >> "$GSP_FAKE_LOG"
    cp "$file" "$GSP_FAKE_REMOTE/$name"
    jq --arg name "$name" --argjson size "$(wc -c < "$file")" \
      '.assets += [{id:(100+(.assets|length)),name:$name,state:"uploaded",size:$size,
        browser_download_url:("https://fixture.invalid/"+$name)}]' "$GSP_FAKE_STATE" > "$GSP_FAKE_STATE.new"
    mv "$GSP_FAKE_STATE.new" "$GSP_FAKE_STATE"
}
case "$1:$2" in
  api:--paginate) jq -s '[.]' "$GSP_FAKE_STATE" ;;
  api:--method)
    if [ "$3" = POST ]; then
      test "$7" = --input
      upload "$8"
      exit 0
    fi
    test "$3" = DELETE
    id=${4##*/}
    name=$(jq -er --argjson id "$id" '.assets[] | select(.id == $id) | .name' "$GSP_FAKE_STATE")
    printf 'delete %s\n' "$name" >> "$GSP_FAKE_LOG"
    if [ "${GSP_FAKE_KEEP_STARTER:-false}" != true ]; then
      jq --argjson id "$id" '.assets |= map(select(.id != $id))' "$GSP_FAKE_STATE" > "$GSP_FAKE_STATE.new"
      mv "$GSP_FAKE_STATE.new" "$GSP_FAKE_STATE"
    fi
    ;;
  release:download)
    shift 3
    while [ "$#" -gt 0 ]; do
      case "$1" in --pattern) name=$2 ;; --dir) directory=$2 ;; esac
      shift 2
    done
    cp "$GSP_FAKE_REMOTE/$name" "$directory/$name"
    ;;
  release:upload)
    upload "$4"
    ;;
  release:edit)
    printf 'edit\n' >> "$GSP_FAKE_LOG"
    if [ "${GSP_FAKE_KEEP_DRAFT:-false}" != true ]; then
      jq '.draft=false' "$GSP_FAKE_STATE" > "$GSP_FAKE_STATE.new"
      mv "$GSP_FAKE_STATE.new" "$GSP_FAKE_STATE"
    fi
    ;;
  api:*)
    id=${2##*/}
    name=$(jq -er --argjson id "$id" '.assets[] | select(.id == $id) | .name' "$GSP_FAKE_STATE")
    cat "$GSP_FAKE_REMOTE/$name"
    ;;
  *) printf 'Unexpected fake gh invocation\n' >&2; exit 1 ;;
esac
EOF
chmod +x "$scratch/tools/gh" "$scratch/tools/git"
native_path() {
  if command -v cygpath > /dev/null 2>&1; then cygpath -m "$1"; else printf '%s\n' "$1"; fi
}
export GSP_FAKE_COMMIT="$commit" GSP_FAKE_STATE="$(native_path "$scratch/release.json")" \
  GSP_FAKE_REMOTE="$(native_path "$scratch/remote")" GSP_FAKE_LOG="$(native_path "$scratch/effects.log")"
export GITHUB_REPOSITORY=fixture/repository
export GITHUB_REF=refs/heads/main RELEASE_SHA="$commit"
export GSP_REAL_PYTHON="$python" GSP_FAKE_REPOSITORY="$(native_path "$repository")" \
  GSP_FAKE_ARTIFACTS="$(native_path "$scratch/artifacts")" RUST_RELEASE_DIR="$(native_path "$scratch/artifacts")" \
  GSP_FAKE_VERSION="$version" GSP_FAKE_HANDOFF="$(native_path "$scratch/common-handoff.py")" \
  GSP_FAKE_GH="$(native_path "$scratch/tools/gh")"
# Exercise the unchanged common uploader/distribution functions with local API
# fixtures. Only numbering/API access are substituted; no credential or network
# is used. Product preflight still controls whether these functions may run.
cat > "$GSP_FAKE_HANDOFF" <<'PY_HANDOFF'
import importlib.util, json, os, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('product', Path(os.environ['GSP_FAKE_REPOSITORY']) / 'scripts/distribution.py')
product = importlib.util.module_from_spec(spec)
spec.loader.exec_module(product)
common = product.controller
real_run = common.run
def local_run(*args, **kwargs):
    assert args[0] == 'gh', args
    return real_run('bash', os.environ['GSP_FAKE_GH'], *args[1:], **kwargs)
common.run = local_run
state_file = Path(os.environ['GSP_FAKE_STATE'])
plan = {'release_tag': os.environ['GSP_FAKE_VERSION'], 'is_prerelease': False,
        'images': [], 'rust': {'targets': list(product.TARGETS)}}
class Git:
    root = os.environ['GSP_FAKE_REPOSITORY']
    def __init__(self, *args): pass
    def text(self, *args): return os.environ['GSP_FAKE_COMMIT']
class GitHub:
    repository = os.environ['GITHUB_REPOSITORY']
    def pages(self, path):
        state = json.loads(state_file.read_text())
        return [state] if path == '/releases' else state['assets']
    def repo(self, path, method='GET', payload=None):
        state = json.loads(state_file.read_text())
        if method == 'DELETE':
            common.run('gh', 'api', '--method', 'DELETE', path)
            return None
        if method == 'PATCH':
            payload = dict(payload)
            if 'draft' in payload:
                with open(os.environ['GSP_FAKE_LOG'], 'a', newline='') as log: log.write('edit\n')
                if os.environ.get('GSP_FAKE_KEEP_DRAFT') == 'true': payload.pop('draft')
            state.update(payload)
            state_file.write_text(json.dumps(state))
        return state
common.Git, common.GitHub = Git, GitHub
common.checkout_plan = lambda git: {'publication': plan}
if sys.argv[1] == 'upload':
    sys.argv = ['distribution.py', 'upload']
    product.main()
else:
    common.distribution(Git(), GitHub(), plan)
PY_HANDOFF
cat > "$scratch/tools/python" <<'EOF'
#!/bin/sh
case "${2##*/}:$3" in
  distribution.py:upload) exec "$GSP_REAL_PYTHON" "$GSP_FAKE_HANDOFF" upload ;;
  release.py:distribution) exec "$GSP_REAL_PYTHON" "$GSP_FAKE_HANDOFF" publish ;;
  *) exec "$GSP_REAL_PYTHON" "$@" ;;
esac
EOF
chmod +x "$scratch/tools/python"
export PYTHON="$scratch/tools/python"
"$python" - "$scratch/dist" <<'MANIFESTS'
import importlib.util, json, os, shutil, sys
from pathlib import Path
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('product', Path(os.environ['GSP_FAKE_REPOSITORY']) / 'scripts/distribution.py')
product = importlib.util.module_from_spec(spec)
spec.loader.exec_module(product)
version, commit = os.environ['GSP_FAKE_VERSION'], os.environ['GSP_FAKE_COMMIT']
for target in product.TARGETS:
    name = product.controller.rust_name('game-screen-pick', version, target)
    root = Path(os.environ['GSP_FAKE_ARTIFACTS']) / ('rust-release-' + target)
    root.mkdir(parents=True)
    for suffix in ('', '.sha256'): shutil.copy2(Path(sys.argv[1]) / (name + suffix), root / (name + suffix))
    (root / 'manifest.json').write_text(json.dumps({
        'release_sha': commit, 'release_tag': version, 'target': target,
        'package': 'game-screen-pick', 'binaries': ['game-screen-pick' + ('.exe' if 'windows' in target else '')],
        'archive': name, 'sha256': product.controller.file_sha256(root / name), 'smoke': 'native'}))
MANIFESTS
"$python" -I scripts/distribution.py flatten "$RUST_RELEASE_DIR" "$scratch/flattened" "$version" "$commit"
for asset in "$scratch/dist/"*; do cmp "$asset" "$scratch/flattened/${asset##*/}"; done
reject "$python" -I scripts/distribution.py flatten "$RUST_RELEASE_DIR" "$scratch/flattened" "$version" "$commit"
manifest=$RUST_RELEASE_DIR/rust-release-aarch64-pc-windows-msvc/manifest.json
cp "$manifest" "$scratch/good-manifest.json"
jq '.smoke="cross-compiled"' "$manifest" > "$manifest.new"
mv "$manifest.new" "$manifest"
reject "$python" -I scripts/distribution.py flatten "$RUST_RELEASE_DIR" "$scratch/bad-flattened" "$version" "$commit"
cp "$scratch/good-manifest.json" "$manifest"
reset_release() {
  jq -n --arg version "$version" --arg commit "$commit" \
    '{id:1,tag_name:$version,target_commitish:$commit,draft:true,prerelease:false,
      upload_url:"https://uploads.fixture.invalid/assets{?name,label}",assets:[]}' > "$GSP_FAKE_STATE"
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
test "$(grep -c '^upload ' "$GSP_FAKE_LOG")" = 8
test "$(grep -c '^edit$' "$GSP_FAKE_LOG")" = 1
jq -e '.body | contains("## Downloads") and contains("windows-x86_64.zip") and contains("windows-arm64.zip")' \
  "$GSP_FAKE_STATE" > /dev/null
cp "$GSP_FAKE_STATE" "$scratch/complete.json"
: > "$GSP_FAKE_LOG"
verify_published
publish
test ! -s "$GSP_FAKE_LOG"
export RELEASE_ID=2
reject publish
reject verify_published
test ! -s "$GSP_FAKE_LOG"
export RELEASE_ID=1
publish
verify_published
unset RELEASE_ID
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
test "$(grep -c '^upload ' "$GSP_FAKE_LOG")" = 6
test "$(grep -c '^edit$' "$GSP_FAKE_LOG")" = 1
# GitHub 502 upload failures can leave a zero-byte starter record. Recovery may
# delete only that expected-name placeholder from a draft, after checking all
# already uploaded bytes, then upload its replacement and the missing assets.
starter_release() {
  reset_release
  cp "$scratch/dist/"* "$scratch/remote/"
  cp "$scratch/complete.json" "$GSP_FAKE_STATE"
  modify_release '.draft=true | .assets=.assets[0:2] | .assets[1] |= (.id=9 | .state="starter" | .size=0)'
}
starter_release
publish
test "$(grep -c '^delete ' "$GSP_FAKE_LOG")" = 1
test "$(grep -c '^upload ' "$GSP_FAKE_LOG")" = 7
test "$(grep -c '^edit$' "$GSP_FAKE_LOG")" = 1
: > "$GSP_FAKE_LOG"
publish
test ! -s "$GSP_FAKE_LOG"
for edit in '.draft=false' '.assets[1].size=1' '.assets[1].state="uploaded"' \
  '.assets[1].id=null' '.assets[1].id=-1' '.assets[1].name="unexpected"'; do
  starter_release
  modify_release "$edit"
  reject publish
  test ! -s "$GSP_FAKE_LOG"
done
starter_release
first=$(jq -r '.assets[0].name' "$GSP_FAKE_STATE")
printf 'changed\n' >> "$scratch/remote/$first"
reject publish
test ! -s "$GSP_FAKE_LOG"
cp "$scratch/dist/$first" "$scratch/remote/"
export GSP_FAKE_KEEP_STARTER=true
reject publish
unset GSP_FAKE_KEEP_STARTER
test "$(grep -c '^delete ' "$GSP_FAKE_LOG")" = 1
test "$(grep -c '^upload ' "$GSP_FAKE_LOG" || true)" = 0
reset_release
export GSP_FAKE_KEEP_DRAFT=true
reject publish
unset GSP_FAKE_KEEP_DRAFT
printf 'Distribution asset and publication contracts passed\n'
