#!/bin/sh
set -eu
# Native jq.exe otherwise translates LF to CRLF inside shell values/files.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) jq() { command jq --binary "$@"; } ;;
esac
cd "$(dirname "$0")/.."
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT HUP INT TERM
mkdir -p "$scratch/crate" "$scratch/tools" "$scratch/sysroot/share/doc/rust/licenses" \
  "$scratch/sysroot/lib/rustlib/src/rust/library"
printf 'fixture\n' > "$scratch/sysroot/share/doc/rust/COPYRIGHT.html"
printf 'fixture\n' > "$scratch/sysroot/lib/rustlib/src/rust/library/LICENSE"
cat > "$scratch/tools/rustc" <<'EOF'
#!/bin/sh
test "$1:$2" = '--print:sysroot'
printf '%s\n' "$GSP_LICENSE_TEST_SYSROOT"
EOF
chmod +x "$scratch/tools/rustc"
printf '{"git":{"sha1":"55ac1664384793c5d3b20745d3f9c169dfe2c06f"},"path_in_vcs":"crates/jsonschema-regex"}\n' \
  > "$scratch/crate/.cargo_vcs_info.json"
jq -n --arg manifest "$scratch/crate/Cargo.toml" '
  {resolve:{root:"root",nodes:[
    {id:"root",deps:[{pkg:"dependency",dep_kinds:[{kind:null}]}]},
    {id:"dependency",deps:[]}]},
   packages:[{id:"root",source:null},
    {id:"dependency",name:"jsonschema-regex",version:"0.58.6",license:"MIT",
     source:"registry+https://github.com/rust-lang/crates.io-index",
     manifest_path:$manifest,license_file:null,
     repository:"https://github.com/Stranger6667/jsonschema"}]}
' > "$scratch/original.json"
cp "$scratch/crate/.cargo_vcs_info.json" "$scratch/original-vcs.json"
cp "$scratch/original.json" "$scratch/template.json"
attempt=0
collect() {
  attempt=$((attempt + 1))
  GSP_LICENSE_TEST_SYSROOT="${test_sysroot:-$scratch/sysroot}" PATH="$scratch/tools:$PATH" \
    sh scripts/collect-licenses.sh "$scratch/metadata.json" "$scratch/result-$attempt"
}
reject() {
  if collect > "$scratch/rejected.log" 2>&1; then
    printf 'Unexpected license supplement acceptance\n' >&2
    exit 1
  fi
}
for version in 0.58.6 0.59.0; do
  case "$version" in
    0.58.6) commit=55ac1664384793c5d3b20745d3f9c169dfe2c06f; other_version=0.59.0 ;;
    0.59.0) commit=d59ab79cb6eab73bb2d3caa6d1b4e538ca81567f; other_version=0.58.6 ;;
  esac
  jq --arg version "$version" '.packages[1].version=$version' "$scratch/template.json" > "$scratch/original.json"
  jq --arg commit "$commit" '.git.sha1=$commit' "$scratch/original-vcs.json" > "$scratch/version-vcs.json"
  cp "$scratch/version-vcs.json" "$scratch/original-vcs.json"
  for name in jsonschema-regex jsonschema-value; do
    jq --arg name "$name" '.packages[1].name=$name' "$scratch/original.json" > "$scratch/metadata.json"
    jq --arg name "$name" '.path_in_vcs="crates/"+$name' "$scratch/original-vcs.json" \
      > "$scratch/crate/.cargo_vcs_info.json"
    collect
    grep -F 'Copyright (c) 2020-2026 Dmitry Dygalo' "$scratch/result-$attempt/THIRD-PARTY-LICENSES.txt" > /dev/null
    grep -F "https://github.com/Stranger6667/jsonschema/blob/$commit/LICENSE" \
      "$scratch/result-$attempt/THIRD-PARTY-LICENSES.txt" > /dev/null
  done
  cp "$scratch/original-vcs.json" "$scratch/crate/.cargo_vcs_info.json"
  for edit in '.name="unknown"' '.version="0.58.7"' '.version="0.59.1"' \
    ".version=\"$other_version\"" '.license="Apache-2.0"' '.repository="https://example.invalid/other"'; do
    jq ".packages[1] |= ($edit)" "$scratch/original.json" > "$scratch/metadata.json"
    reject
  done
  cp "$scratch/original.json" "$scratch/metadata.json"
  for edit in '.git.sha1="wrong"' '.path_in_vcs="crates/other"'; do
    jq "$edit" "$scratch/original-vcs.json" > "$scratch/crate/.cargo_vcs_info.json"
    reject
  done
done
cp "$scratch/original-vcs.json" "$scratch/crate/.cargo_vcs_info.json"
# Cargo and rustc return native Windows paths under Git Bash. Convert both the
# manifest/license path and sysroot before collecting real files, preserving the
# same license-text requirement as on macOS/Linux.
cat > "$scratch/tools/cygpath" <<'EOF'
#!/bin/sh
test "$1" = -u
case "$2" in
  C:/*) printf '%s/%s\n' "$GSP_LICENSE_DRIVE_ROOT" "${2#C:/}" ;;
  *) printf '%s\n' "$2" ;;
esac
EOF
chmod +x "$scratch/tools/cygpath"
export GSP_LICENSE_DRIVE_ROOT="$scratch"
test_sysroot=C:/sysroot
printf 'native Windows crate license\n' > "$scratch/crate/LICENSE"
jq '.packages[1].manifest_path="C:/crate/Cargo.toml" |
    .packages[1].license_file="C:/crate/LICENSE"' "$scratch/original.json" > "$scratch/metadata.json"
collect
grep -F 'native Windows crate license' "$scratch/result-$attempt/THIRD-PARTY-LICENSES.txt" > /dev/null
rm "$scratch/crate/LICENSE"
reject
printf 'License supplement contracts passed\n'
