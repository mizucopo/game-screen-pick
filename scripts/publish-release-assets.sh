#!/bin/sh
# Called only by release.yml after all tests pass. No clobber or tag mutation.
set -eu
# Native jq.exe otherwise translates LF to CRLF inside shell values/files.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) jq() { command jq --binary "$@"; } ;;
esac
cd "$(dirname "$0")/.."
destination=$1
version=$2
commit=$3
tag=$4
is_prerelease=$5
case "$is_prerelease" in true|false) ;; *) exit 1 ;; esac
test "$tag" = "$version"
test "$(git rev-parse HEAD)" = "$commit"
test "$(git rev-list -n 1 "refs/tags/$tag")" = "$commit"
test "${RELEASE_SHA:?}" = "$commit"
sh scripts/verify-release-assets.sh "$destination" "$version" "$commit"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT HUP INT TERM
repository=${GITHUB_REPOSITORY:?}
inspect() {
  gh api --paginate --slurp "/repos/$repository/releases?per_page=100" \
    | jq -e --arg tag "$tag" --arg commit "$commit" --argjson is_prerelease "$is_prerelease" \
      --arg retained_id "${RELEASE_ID:-}" '
      [.[][] | select(.tag_name == $tag)] |
      if length != 1 then error("Expected one GitHub Release") else .[0] end |
      if .target_commitish != $commit or .prerelease != $is_prerelease or (.id | type) != "number" or
         ($retained_id != "" and (.id | tostring) != $retained_id) or
         (.draft | type) != "boolean" or (.assets | type) != "array" or
         ([.assets[].name] | length != (unique | length))
      then error("Release provenance or asset metadata differs") else . end
    ' > "$work/release.json"
}
inspect
draft=$(jq -r '.draft' "$work/release.json")
find "$destination" -mindepth 1 -maxdepth 1 -type f -exec basename {} \; | LC_ALL=C sort > "$work/expected"
jq -r '.assets[].name' "$work/release.json" | LC_ALL=C sort > "$work/remote"
if [ -n "$(comm -13 "$work/expected" "$work/remote")" ]; then
  printf 'Unexpected assets on Release %s\n' "$tag" >&2; exit 1
fi
mkdir "$work/downloaded"
: > "$work/starters"
# Validate every existing byte before uploading any missing asset. Rebuilding
# with a different compiler is not authority to replace an immutable asset.
while IFS= read -r name; do
  if jq -e --arg name "$name" 'any(.assets[]; .name == $name and .state == "starter" and .size == 0)' \
    "$work/release.json" > /dev/null; then
    test "$draft" = true
    id=$(jq -er --arg name "$name" '.assets[] | select(.name == $name) | .id |
      select(type == "number" and . > 0 and . == floor)' "$work/release.json")
    printf '%s\t%s\n' "$id" "$name" >> "$work/starters"
    continue
  fi
  jq -e --arg name "$name" 'any(.assets[]; .name == $name and .state == "uploaded" and .size > 0)' \
    "$work/release.json" > /dev/null
  gh release download "$tag" --repo "$repository" --pattern "$name" --dir "$work/downloaded"
  if ! cmp -s "$destination/$name" "$work/downloaded/$name"; then
    printf 'Existing asset differs: %s\n' "$name" >&2; exit 1
  fi
done < "$work/remote"
# GitHub can leave an empty starter after an upstream 502. It has no uploaded
# bytes to preserve. Recheck draft provenance and the exact placeholder before
# deleting it; uploaded, nonempty and unexpected-name assets are never deleted.
while IFS="$(printf '\t')" read -r id name; do
  inspect
  jq -e --argjson id "$id" --arg name "$name" '.draft == true and
    any(.assets[]; .id == $id and .name == $name and .state == "starter" and .size == 0)' \
    "$work/release.json" > /dev/null
  gh api --method DELETE "/repos/$repository/releases/assets/$id"
done < "$work/starters"
if [ -s "$work/starters" ]; then
  inspect
  jq -e '.draft == true and all(.assets[]; .state == "uploaded" and .size > 0)' \
    "$work/release.json" > /dev/null
  jq -r '.assets[].name' "$work/release.json" | LC_ALL=C sort > "$work/remote"
  test -z "$(comm -13 "$work/expected" "$work/remote")"
fi
if [ "$draft" = false ] && ! cmp -s "$work/expected" "$work/remote"; then
  printf 'Published Release %s is incomplete; inspect its state before recovery\n' "$tag" >&2; exit 1
fi
# The common controller owns immutable upload. Product cleanup above confirms
# that a failed starter deletion cannot be followed by a replacement upload.
# Retain its manifests alongside the flattened, product-verified asset set.
"${PYTHON:-python3}" -I scripts/distribution.py upload
inspect
jq -r '.assets[].name' "$work/release.json" | LC_ALL=C sort > "$work/remote"
cmp "$work/expected" "$work/remote"
rm -rf "$work/downloaded"
mkdir "$work/downloaded"
while IFS= read -r name; do
  jq -e --arg name "$name" 'any(.assets[]; .name == $name and .state == "uploaded" and .size > 0)' \
    "$work/release.json" > /dev/null
  gh release download "$tag" --repo "$repository" --pattern "$name" --dir "$work/downloaded"
  cmp "$destination/$name" "$work/downloaded/$name"
done < "$work/expected"
sh scripts/verify-release-assets.sh "$work/downloaded" "$version" "$commit"
# Every uploaded archive has been downloaded and revalidated. The shared
# distribution action now owns the Downloads block and draft publication;
# disabling asset building prevents a second mutation after this byte check.
BUILD_RUST_BINARY=false "${PYTHON:-python3}" -I .github/scripts/release.py distribution
inspect
jq -e '.draft == false and all(.assets[]; .state == "uploaded" and .size > 0)' \
  "$work/release.json" > /dev/null
jq -r '.assets[].name' "$work/release.json" | LC_ALL=C sort > "$work/remote"
cmp "$work/expected" "$work/remote"
