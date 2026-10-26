#!/bin/sh
# Collect the locked target's normal/build dependency licenses, without a network
# license lookup. Missing license material is a packaging failure.
set -eu
# Native jq.exe otherwise translates LF to CRLF inside shell values/files.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) jq() { command jq --binary "$@"; } ;;
esac
cd "$(dirname "$0")/.."
metadata=$1
destination=$2
shell_path() {
  if command -v cygpath > /dev/null 2>&1; then cygpath -u "$1"; else printf '%s\n' "$1"; fi
}
mkdir -p "$destination"
jq '
  def closure($nodes; $seen):
    ([$nodes[] | select(.id as $id | $seen | index($id)) | .deps[] |
      select(any(.dep_kinds[]; .kind != "dev")) | .pkg] + $seen | unique) as $next |
    if $next == $seen then $seen else closure($nodes; $next) end;
  closure(.resolve.nodes; [.resolve.root]) as $ids |
  [.packages[] | select(.id as $id | $ids | index($id)) |
    select(.source != null)] | sort_by(.name, .version)
' "$metadata" > "$destination/packages.json"
jq '[.[] | {name, version, license, source, repository}]' \
  "$destination/packages.json" > "$destination/dependencies.json"
: > "$destination/THIRD-PARTY-LICENSES.txt"
jq -r '.[] | [.name, .version, (.license // ""), .manifest_path,
  (.license_file // ""), (.repository // "")] | @tsv' \
  "$destination/packages.json" > "$destination/packages.tsv"
tab=$(printf '\t')
while IFS="$tab" read -r name version license manifest license_file repository; do
  # TSV empty fields are collapsed by shell IFS. Read path-sensitive fields from
  # JSON, preserving an absent license_file instead of shifting the repository.
  package=$(jq -c --arg name "$name" --arg version "$version" \
    '.[] | select(.name == $name and .version == $version)' "$destination/packages.json")
  license=$(printf '%s' "$package" | jq -er '.license | select(length > 0)')
  manifest=$(shell_path "$(printf '%s' "$package" | jq -er '.manifest_path')")
  license_file=$(printf '%s' "$package" | jq -r '.license_file // ""')
  repository=$(printf '%s' "$package" | jq -r '.repository // ""')
  crate_dir=${manifest%/*}
  {
    printf '\n===== %s %s =====\nDeclared license: %s\nSource: %s\n' \
      "$name" "$version" "$license" "$repository"
  } >> "$destination/THIRD-PARTY-LICENSES.txt"
  find "$crate_dir" -maxdepth 1 -type f \( -iname '*license*' -o \
    -iname '*notice*' -o -iname '*copying*' -o -iname 'unlicense' \) \
    | LC_ALL=C sort > "$destination/license-files.txt"
  if [ -n "$license_file" ]; then
    case "$license_file" in /*|[A-Za-z]:[\\/]*) declared=$(shell_path "$license_file") ;; *) declared=$crate_dir/$license_file ;; esac
    test -s "$declared"
    printf '%s\n' "$declared" >> "$destination/license-files.txt"
    LC_ALL=C sort -u "$destination/license-files.txt" > "$destination/license-files.sorted"
    mv "$destination/license-files.sorted" "$destination/license-files.txt"
  fi
  if [ ! -s "$destination/license-files.txt" ]; then
    case "$name:$version:$license:$repository" in
      uuid-simd:0.8.0:MIT:https://github.com/Nugine/simd|vsimd:0.8.0:MIT:https://github.com/Nugine/simd)
        jq -e '.git.sha1 == "d74c030d9dc4f3cae02146d1f497ff62726ef09a"' \
          "$crate_dir/.cargo_vcs_info.json" > /dev/null
        printf '%s\n' scripts/licenses/Nugine-simd-MIT.txt > "$destination/license-files.txt"
        printf 'License supplement source: https://github.com/Nugine/simd/blob/d74c030d9dc4f3cae02146d1f497ff62726ef09a/LICENSE\n' \
          >> "$destination/THIRD-PARTY-LICENSES.txt"
        ;;
      jsonschema-regex:0.58.6:MIT:https://github.com/Stranger6667/jsonschema|jsonschema-value:0.58.6:MIT:https://github.com/Stranger6667/jsonschema)
        jq -e --arg name "$name" '
          .git.sha1 == "55ac1664384793c5d3b20745d3f9c169dfe2c06f" and
          .path_in_vcs == ("crates/" + $name)
        ' "$crate_dir/.cargo_vcs_info.json" > /dev/null
        printf '%s\n' scripts/licenses/jsonschema-0.58.6-MIT.txt > "$destination/license-files.txt"
        printf 'License supplement source: https://github.com/Stranger6667/jsonschema/blob/55ac1664384793c5d3b20745d3f9c169dfe2c06f/LICENSE\n' \
          >> "$destination/THIRD-PARTY-LICENSES.txt"
        ;;
      jsonschema-regex:0.59.0:MIT:https://github.com/Stranger6667/jsonschema|jsonschema-value:0.59.0:MIT:https://github.com/Stranger6667/jsonschema)
        jq -e --arg name "$name" '
          .git.sha1 == "d59ab79cb6eab73bb2d3caa6d1b4e538ca81567f" and
          .path_in_vcs == ("crates/" + $name)
        ' "$crate_dir/.cargo_vcs_info.json" > /dev/null
        printf '%s\n' scripts/licenses/jsonschema-0.59.0-MIT.txt > "$destination/license-files.txt"
        printf 'License supplement source: https://github.com/Stranger6667/jsonschema/blob/d59ab79cb6eab73bb2d3caa6d1b4e538ca81567f/LICENSE\n' \
          >> "$destination/THIRD-PARTY-LICENSES.txt"
        ;;
      *) printf 'Missing license text: %s %s\n' "$name" "$version" >&2; exit 1 ;;
    esac
  fi
  while IFS= read -r file; do
    test -s "$file"
    {
      printf '\n--- %s ---\n' "${file##*/}"
      cat "$file"
      printf '\n'
    } >> "$destination/THIRD-PARTY-LICENSES.txt"
  done < "$destination/license-files.txt"
done < "$destination/packages.tsv"
cp assets/OFL.txt "$destination/NotoSansJP-OFL.txt"
rm "$destination/packages.json" "$destination/packages.tsv" "$destination/license-files.txt"
# The statically linked Rust standard library is outside Cargo's package graph.
# Preserve the installed toolchain's copyright inventory and license texts,
# including compiler-builtins/libunwind exceptions from its source component.
sysroot=$(shell_path "$(rustc --print sysroot)")
documentation=
for candidate in "$sysroot/share/doc/rust" "$sysroot/share/doc/rustc"; do
  if [ -s "$candidate/COPYRIGHT.html" ] && [ -d "$candidate/licenses" ]; then
    documentation=$candidate
    break
  fi
done
if [ -z "$documentation" ]; then
  printf 'Rust license inventory missing; install rust-docs for this toolchain\n' >&2; exit 1
fi
mkdir "$destination/rust"
cp "$documentation/COPYRIGHT.html" "$destination/rust/"
if [ -s "$documentation/COPYRIGHT-library.html" ]; then
  cp "$documentation/COPYRIGHT-library.html" "$destination/rust/"
fi
cp -R "$documentation/licenses" "$destination/rust/licenses"
source=$sysroot/lib/rustlib/src/rust/library
if [ ! -d "$source" ]; then
  printf 'Rust source license material missing; install rust-src for this toolchain\n' >&2; exit 1
fi
find "$source" -type f \( -iname '*license*' -o -iname '*notice*' -o -iname '*copying*' -o -iname '*copyright*' \) \
  | LC_ALL=C sort > "$destination/rust/source-files.txt"
test -s "$destination/rust/source-files.txt"
: > "$destination/rust/SOURCE-LICENSES.txt"
while IFS= read -r file; do
  {
    printf '\n--- %s ---\n' "${file#"$source"/}"
    cat "$file"
    printf '\n'
  } >> "$destination/rust/SOURCE-LICENSES.txt"
done < "$destination/rust/source-files.txt"
rm "$destination/rust/source-files.txt"
