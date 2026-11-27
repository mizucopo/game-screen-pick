#!/bin/sh
# Collect the locked target's normal/build dependency licenses, without a network
# license lookup. Missing license material is a packaging failure.
set -eu
# Native jq.exe otherwise translates LF to CRLF inside shell values/files.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) jq() { command jq --binary "$@"; } ;;
esac
cd "$(dirname "$0")/.."
check_target=
if [ "$1" = --check-supplements ]; then
  check_target=${2:?distribution target required}
  shift 2
fi
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
  [.packages[] | select(.id as $id | $ids | index($id)) | select(.source != null) |
    {name, version, license, source, repository, manifest_path, license_file}] | sort_by(.name, .version)
' "$metadata" > "$destination/packages.json"
if [ -z "$check_target" ]; then
  jq '[.[] | {name, version, license, source, repository}]' \
    "$destination/packages.json" > "$destination/dependencies.json"
  : > "$destination/THIRD-PARTY-LICENSES.txt"
fi
jq -r '.[] | [.name, .version, (.license // ""), .manifest_path,
  (.license_file // ""), (.repository // "")] | @tsv' \
  "$destination/packages.json" > "$destination/packages.tsv"
tab=$(printf '\t')
missing_supplement=0
while IFS="$tab" read -r name version license manifest license_file repository; do
  dependency="$name $version${check_target:+ ($check_target)}"
  # TSV empty fields are collapsed by shell IFS. Read path-sensitive fields from
  # JSON, preserving an absent license_file instead of shifting the repository.
  package=$(jq -c --arg name "$name" --arg version "$version" \
    '.[] | select(.name == $name and .version == $version)' "$destination/packages.json")
  license=$(printf '%s' "$package" | jq -er '.license | select(length > 0)') || {
    printf 'Missing declared license: %s\n' "$dependency" >&2
    exit 1
  }
  manifest=$(shell_path "$(printf '%s' "$package" | jq -er '.manifest_path')")
  license_file=$(printf '%s' "$package" | jq -r '.license_file // ""')
  repository=$(printf '%s' "$package" | jq -r '.repository // ""')
  crate_dir=${manifest%/*}
  supplement_source=
  find "$crate_dir" -maxdepth 1 -type f \( -iname '*license*' -o \
    -iname '*notice*' -o -iname '*copying*' -o -iname 'unlicense' \) \
    | LC_ALL=C sort > "$destination/license-files.txt"
  if [ -n "$license_file" ]; then
    case "$license_file" in /*|[A-Za-z]:[\\/]*) declared=$(shell_path "$license_file") ;; *) declared=$crate_dir/$license_file ;; esac
    printf '%s\n' "$declared" >> "$destination/license-files.txt"
    LC_ALL=C sort -u "$destination/license-files.txt" > "$destination/license-files.sorted"
    mv "$destination/license-files.sorted" "$destination/license-files.txt"
  fi
  if [ ! -s "$destination/license-files.txt" ]; then
    if [ -n "$check_target" ]; then
      printf 'License supplement required: %s\n' "$dependency"
    fi
    # Share identical license text, but authorize each exact release separately.
    supplement=$(jq -ce --arg name "$name" --arg version "$version" \
      --arg license "$license" --arg repository "$repository" '
      [.[] | select(.license == $license and .repository == $repository) |
        select(.packages | has($name)) | select(.versions[$version] != null) |
        . + {commit: .versions[$version], path: .packages[$name]}] |
      if length == 1 then .[0] else empty end
    ' scripts/licenses/supplements.json) || {
      printf 'Missing or ambiguous license supplement: %s\n' "$dependency" >&2
      if [ -n "$check_target" ]; then
        missing_supplement=1
        continue
      fi
      exit 1
    }
    printf '%s' "$supplement" | jq -e --slurpfile vcs "$crate_dir/.cargo_vcs_info.json" '
      .commit == $vcs[0].git.sha1 and
      (.path == null or .path == $vcs[0].path_in_vcs)
    ' > /dev/null || { printf 'Invalid license supplement provenance: %s\n' "$dependency" >&2; exit 1; }
    file=scripts/licenses/$(printf '%s' "$supplement" | jq -er '.file')
    checksum=$(printf '%s' "$supplement" | jq -er '.sha256')
    if command -v sha256sum > /dev/null 2>&1; then
      printf '%s  %s\n' "$checksum" "$file" | sha256sum -c - > /dev/null
    else
      printf '%s  %s\n' "$checksum" "$file" | shasum -a 256 -c - > /dev/null
    fi || { printf 'Invalid license supplement checksum: %s\n' "$dependency" >&2; exit 1; }
    printf '%s\n' "$file" > "$destination/license-files.txt"
    commit=$(printf '%s' "$supplement" | jq -er '.commit')
    supplement_source=$repository/blob/$commit/LICENSE
  fi
  while IFS= read -r file; do
    if [ ! -f "$file" ] || [ ! -s "$file" ] || [ ! -r "$file" ]; then
      printf 'Missing or unreadable license text for %s: %s\n' "$dependency" "$file" >&2
      exit 1
    fi
  done < "$destination/license-files.txt"
  if [ -n "$check_target" ]; then continue; fi
  {
    printf '\n===== %s %s =====\nDeclared license: %s\nSource: %s\n' "$name" "$version" "$license" "$repository"
    if [ -n "$supplement_source" ]; then printf 'License supplement source: %s\n' "$supplement_source"; fi
    while IFS= read -r file; do
      printf '\n--- %s ---\n' "${file##*/}"
      cat "$file"
      printf '\n'
    done < "$destination/license-files.txt"
  } >> "$destination/THIRD-PARTY-LICENSES.txt"
done < "$destination/packages.tsv"
if [ -n "$check_target" ]; then
  exit "$missing_supplement"
fi
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
