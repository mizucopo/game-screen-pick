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
# Isolate supplement mutations so rejection tests cannot alter the checkout.
mkdir -p "$scratch/project/scripts" "$scratch/project/assets"
cp scripts/collect-licenses.sh "$scratch/project/scripts/"
cp -R scripts/licenses "$scratch/project/scripts/"
cp assets/OFL.txt "$scratch/project/assets/"
supplements=$scratch/project/scripts/licenses
cp "$supplements/supplements.json" "$scratch/supplements.json"
attempt=0
collect() {
  attempt=$((attempt + 1))
  GSP_LICENSE_TEST_SYSROOT="${test_sysroot:-$scratch/sysroot}" PATH="$scratch/tools:${checksum_test_path:-$PATH}" \
    sh "$scratch/project/scripts/collect-licenses.sh" "$scratch/metadata.json" "$scratch/result-$attempt"
}
reject() {
  if collect > "$scratch/rejected.log" 2>&1; then
    printf 'Unexpected license supplement acceptance\n' >&2
    exit 1
  fi
}
for version in 0.58.6 0.59.0 0.60.1 0.60.2; do
  case "$version" in
    0.58.6) commit=55ac1664384793c5d3b20745d3f9c169dfe2c06f; other_version=0.59.0 ;;
    0.59.0) commit=d59ab79cb6eab73bb2d3caa6d1b4e538ca81567f; other_version=0.58.6 ;;
    0.60.1) commit=4eb7a64d9cf45bcbf38c7a69e544477294460751; other_version=0.59.0 ;;
    0.60.2) commit=0b657fa991c9333908e8c51e8a582d602aac5391; other_version=0.60.1 ;;
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
    jq -en --rawfile expected "$supplements/jsonschema-MIT.txt" \
      --rawfile actual "$scratch/result-$attempt/THIRD-PARTY-LICENSES.txt" \
      '$actual | contains($expected)' > /dev/null
    cp "$scratch/metadata.json" "$scratch/accepted.json"
    cp "$scratch/crate/.cargo_vcs_info.json" "$scratch/accepted-vcs.json"
    for edit in '.name="unknown"' '.version="0.58.7"' '.version="0.59.1"' \
      '.version="0.60.0"' '.version="0.60.3"' ".version=\"$other_version\"" \
      '.license="Apache-2.0"' '.repository="https://example.invalid/other"'; do
      jq ".packages[1] |= ($edit)" "$scratch/accepted.json" > "$scratch/metadata.json"
      reject
    done
    cp "$scratch/accepted.json" "$scratch/metadata.json"
    for edit in '.git.sha1="wrong"' '.path_in_vcs="crates/other"'; do
      jq "$edit" "$scratch/accepted-vcs.json" > "$scratch/crate/.cargo_vcs_info.json"
      reject
    done
  done
done
# A recognized release still requires one unambiguous, intact supplement.
cp "$scratch/accepted.json" "$scratch/metadata.json"
cp "$scratch/accepted-vcs.json" "$scratch/crate/.cargo_vcs_info.json"
# macOS can provide shasum without GNU sha256sum.
mkdir "$scratch/hash-tools"
for tool in sh uname dirname mkdir jq find sort mv cp cat rm shasum; do
  ln -s "$(command -v "$tool")" "$scratch/hash-tools/$tool"
done
checksum_test_path=$scratch/hash-tools
collect
unset checksum_test_path
for edit in '. + [.[1]]' 'map(select(.repository != "https://github.com/Stranger6667/jsonschema"))' \
  '.[1].sha256="0000000000000000000000000000000000000000000000000000000000000000"'; do
  jq "$edit" "$scratch/supplements.json" > "$supplements/supplements.json"
  reject
done
cp "$scratch/supplements.json" "$supplements/supplements.json"
mv "$supplements/jsonschema-MIT.txt" "$scratch/shared-license.txt"
reject
printf 'altered license text\n' > "$supplements/jsonschema-MIT.txt"
reject
cp "$scratch/shared-license.txt" "$supplements/jsonschema-MIT.txt"
for edit in 'del(.git)' 'del(.path_in_vcs)'; do
  jq "$edit" "$scratch/accepted-vcs.json" > "$scratch/crate/.cargo_vcs_info.json"
  reject
done
rm "$scratch/crate/.cargo_vcs_info.json"
reject

# The existing simd exception also uses the shared lookup, without gaining a
# path restriction that the original exception did not require.
for name in uuid-simd vsimd; do
  jq --arg name "$name" '.packages[1] |= (.name=$name | .version="0.8.0" |
    .repository="https://github.com/Nugine/simd")' "$scratch/accepted.json" > "$scratch/metadata.json"
  printf '{"git":{"sha1":"d74c030d9dc4f3cae02146d1f497ff62726ef09a"}}\n' \
    > "$scratch/crate/.cargo_vcs_info.json"
  collect
  jq -en --rawfile expected "$supplements/Nugine-simd-MIT.txt" \
    --rawfile actual "$scratch/result-$attempt/THIRD-PARTY-LICENSES.txt" \
    '$actual | contains($expected)' > /dev/null
  grep -F 'https://github.com/Nugine/simd/blob/d74c030d9dc4f3cae02146d1f497ff62726ef09a/LICENSE' \
    "$scratch/result-$attempt/THIRD-PARTY-LICENSES.txt" > /dev/null
  cp "$scratch/metadata.json" "$scratch/simd.json"
  for edit in '.version="0.8.1"' '.license="Apache-2.0"' '.repository="https://example.invalid/other"'; do
    jq ".packages[1] |= ($edit)" "$scratch/simd.json" > "$scratch/metadata.json"
    reject
  done
  cp "$scratch/simd.json" "$scratch/metadata.json"
  printf '{"git":{"sha1":"wrong"}}\n' > "$scratch/crate/.cargo_vcs_info.json"
  reject
done

# Bundled legal text takes precedence over supplements and keeps every notice,
# even for a version that has not been added to the supplement allowlist.
jq '.packages[1].version="99.0.0"' "$scratch/accepted.json" > "$scratch/metadata.json"
printf 'bundled license\n' > "$scratch/crate/LICENSE"
printf 'bundled notice\n' > "$scratch/crate/NOTICE"
collect
for text in 'bundled license' 'bundled notice'; do
  grep -F "$text" "$scratch/result-$attempt/THIRD-PARTY-LICENSES.txt" > /dev/null
done
if grep -F 'License supplement source:' "$scratch/result-$attempt/THIRD-PARTY-LICENSES.txt"; then
  printf 'Bundled licenses unexpectedly used a supplement\n' >&2
  exit 1
fi
rm "$scratch/crate/LICENSE" "$scratch/crate/NOTICE"
reject
# A declared relative license_file outside the usual top-level names is kept.
mkdir "$scratch/crate/legal"
printf 'declared nested license\n' > "$scratch/crate/legal/terms.txt"
jq '.packages[1].license_file="legal/terms.txt"' "$scratch/metadata.json" > "$scratch/relative.json"
mv "$scratch/relative.json" "$scratch/metadata.json"
collect
grep -F 'declared nested license' "$scratch/result-$attempt/THIRD-PARTY-LICENSES.txt" > /dev/null
rm "$scratch/crate/legal/terms.txt"
reject

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
