#!/bin/sh
# Called only by release.yml after all tests pass. No clobber or tag mutation.
set -eu
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
sh scripts/verify-release-assets.sh "$destination" "$version" "$commit"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT HUP INT TERM
repository=${GITHUB_REPOSITORY:?}
inspect() {
  gh api --paginate --slurp "/repos/$repository/releases?per_page=100" \
    | jq -e --arg tag "$tag" --arg commit "$commit" --argjson is_prerelease "$is_prerelease" '
      [.[][] | select(.tag_name == $tag)] |
      if length != 1 then error("Expected one GitHub Release") else .[0] end |
      if .target_commitish != $commit or .prerelease != $is_prerelease or (.id | type) != "number" or
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
# Validate every existing byte before uploading any missing asset. Rebuilding
# with a different compiler is not authority to replace an immutable asset.
while IFS= read -r name; do
  jq -e --arg name "$name" 'any(.assets[]; .name == $name and .state == "uploaded" and .size > 0)' \
    "$work/release.json" > /dev/null
  gh release download "$tag" --repo "$repository" --pattern "$name" --dir "$work/downloaded"
  if ! cmp -s "$destination/$name" "$work/downloaded/$name"; then
    printf 'Existing asset differs: %s\n' "$name" >&2; exit 1
  fi
done < "$work/remote"
if [ "$draft" = false ] && ! cmp -s "$work/expected" "$work/remote"; then
  printf 'Published Release %s is incomplete; inspect its state before recovery\n' "$tag" >&2; exit 1
fi
comm -23 "$work/expected" "$work/remote" > "$work/missing"
while IFS= read -r name; do
  gh release upload "$tag" "$destination/$name" --repo "$repository"
done < "$work/missing"
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
if [ "$draft" = true ]; then
  gh release edit "$tag" --repo "$repository" --draft=false --latest=false
fi
inspect
jq -e '.draft == false and all(.assets[]; .state == "uploaded" and .size > 0)' \
  "$work/release.json" > /dev/null
jq -r '.assets[].name' "$work/release.json" | LC_ALL=C sort > "$work/remote"
cmp "$work/expected" "$work/remote"
