# Crate license supplements

Keep one complete text per byte-distinct upstream license, not per crate version
or SPDX identifier. `supplements.json` maps verified crate releases to the shared
text, SHA-256, repository and source commit. Package paths are checked when listed;
`null` preserves the original simd supplement's commit-only check. Each source is
`<repository>/blob/<commit>/LICENSE`. Git keeps these texts LF-only on every
platform so their checksums remain stable.

`Nugine-simd-MIT.txt` covers uuid-simd/vsimd; `jsonschema-MIT.txt` covers
jsonschema-regex/jsonschema-value. These published crates omit their license text.
The collector prefers bundled license/notice files and fails closed for an unknown
release, mismatched provenance or missing/changed supplement. Every dependency's
attribution and full license text remain in the distribution.

When updating a dependency:

1. Verify the downloaded crate checksum against `Cargo.lock`, then inspect its
   `.cargo_vcs_info.json`, declared license, repository and package path.
2. Read the root LICENSE at that exact upstream commit. Reuse the existing text
   only if its bytes match, including copyright notices. Otherwise add a distinct
   complete text and record its SHA-256; never shorten or rewrite legal text.
3. Add the verified version/commit mapping to `supplements.json`. For a new text,
   use a separate record with only the releases that it covers. Keep existing
   release mappings without copying text or adding per-version prose here.
4. Add the version/commit fixture in `tests/licenses.sh` and run `sh tests/check.sh`.
