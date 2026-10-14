#!/bin/sh
set -eu
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
attempt=0
collect() {
  attempt=$((attempt + 1))
  GSP_LICENSE_TEST_SYSROOT="$scratch/sysroot" PATH="$scratch/tools:$PATH" \
    sh scripts/collect-licenses.sh "$scratch/metadata.json" "$scratch/result-$attempt"
}
reject() {
  if collect > "$scratch/rejected.log" 2>&1; then
    printf 'Unexpected license supplement acceptance\n' >&2
    exit 1
  fi
}
for name in jsonschema-regex jsonschema-value; do
  jq --arg name "$name" '.packages[1].name=$name' "$scratch/original.json" > "$scratch/metadata.json"
  jq --arg name "$name" '.path_in_vcs="crates/"+$name' "$scratch/original-vcs.json" \
    > "$scratch/crate/.cargo_vcs_info.json"
  collect
  grep -F 'Copyright (c) 2020-2026 Dmitry Dygalo' "$scratch/result-$attempt/THIRD-PARTY-LICENSES.txt" > /dev/null
  grep -F 'https://github.com/Stranger6667/jsonschema/blob/55ac1664384793c5d3b20745d3f9c169dfe2c06f/LICENSE' \
    "$scratch/result-$attempt/THIRD-PARTY-LICENSES.txt" > /dev/null
done
cp "$scratch/original-vcs.json" "$scratch/crate/.cargo_vcs_info.json"
for edit in '.name="unknown"' '.version="0.58.7"' '.license="Apache-2.0"' '.repository="https://example.invalid/other"'; do
  jq ".packages[1] |= ($edit)" "$scratch/original.json" > "$scratch/metadata.json"
  reject
done
cp "$scratch/original.json" "$scratch/metadata.json"
for edit in '.git.sha1="wrong"' '.path_in_vcs="crates/other"'; do
  jq "$edit" "$scratch/original-vcs.json" > "$scratch/crate/.cargo_vcs_info.json"
  reject
done
printf 'License supplement contracts passed\n'
