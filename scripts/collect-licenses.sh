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
    # Share identical license text, but authorize each exact release separately.
    supplement=$(jq -ce --arg name "$name" --arg version "$version" \
      --arg license "$license" --arg repository "$repository" '
      [.[] | select(.license == $license and .repository == $repository) |
        select(.packages | has($name)) | select(.versions[$version] != null) |
        . + {commit: .versions[$version], path: .packages[$name]}] |
      if length == 1 then .[0] else error("Missing or ambiguous license supplement") end
    ' scripts/licenses/supplements.json) || {
      printf 'Missing or ambiguous license supplement: %s %s\n' "$name" "$version" >&2
      exit 1
    }
    printf '%s' "$supplement" | jq -e --slurpfile vcs "$crate_dir/.cargo_vcs_info.json" '
      .commit == $vcs[0].git.sha1 and
      (.path == null or .path == $vcs[0].path_in_vcs)
    ' > /dev/null
    file=scripts/licenses/$(printf '%s' "$supplement" | jq -er '.file')
    checksum=$(printf '%s' "$supplement" | jq -er '.sha256')
    if command -v sha256sum > /dev/null 2>&1; then
      printf '%s  %s\n' "$checksum" "$file" | sha256sum -c - > /dev/null
    else
      printf '%s  %s\n' "$checksum" "$file" | shasum -a 256 -c - > /dev/null
    fi
    printf '%s\n' "$file" > "$destination/license-files.txt"
    commit=$(printf '%s' "$supplement" | jq -er '.commit')
    printf 'License supplement source: %s/blob/%s/LICENSE\n' "$repository" "$commit" \
      >> "$destination/THIRD-PARTY-LICENSES.txt"
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
