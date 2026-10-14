#!/bin/sh
# Read-only guard immediately before the shared controller may promote Latest.
set -eu
destination=$1
version=$2
commit=$3
tag=$4
is_prerelease=$5
case "$is_prerelease" in true|false) ;; *) exit 1 ;; esac
test "$tag" = "$version"
sh "$(dirname "$0")/verify-release-assets.sh" "$destination" "$version" "$commit"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT HUP INT TERM
repository=${GITHUB_REPOSITORY:?}
gh api --paginate --slurp "/repos/$repository/releases?per_page=100" \
  | jq -e --arg tag "$tag" --arg commit "$commit" --argjson is_prerelease "$is_prerelease" '
    [.[][] | select(.tag_name == $tag)] |
    if length != 1 then error("Expected one GitHub Release") else .[0] end |
    if .target_commitish != $commit or .draft != false or .prerelease != $is_prerelease or
      (.assets | type) != "array" or
      ([.assets[].name] | length != (unique | length)) or
      (all(.assets[]; .state == "uploaded" and .size > 0) | not)
    then error("Published Release is incomplete or provenance differs") else . end
  ' > "$work/release.json"
find "$destination" -mindepth 1 -maxdepth 1 -type f -exec basename {} \; | LC_ALL=C sort > "$work/expected"
jq -r '.assets[].name' "$work/release.json" | LC_ALL=C sort > "$work/remote"
cmp "$work/expected" "$work/remote"
mkdir "$work/downloaded"
while IFS= read -r name; do
  gh release download "$tag" --repo "$repository" --pattern "$name" --dir "$work/downloaded"
  cmp "$destination/$name" "$work/downloaded/$name"
done < "$work/expected"
sh "$(dirname "$0")/verify-release-assets.sh" "$work/downloaded" "$version" "$commit"
