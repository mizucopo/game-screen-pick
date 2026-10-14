# Crate license supplements

`uuid-simd 0.8.0` and `vsimd 0.8.0` omit their MIT license file from the
published crate. `Nugine-simd-MIT.txt` preserves the upstream file at the commit
recorded in both crates' `.cargo_vcs_info.json`:

https://github.com/Nugine/simd/blob/d74c030d9dc4f3cae02146d1f497ff62726ef09a/LICENSE

`jsonschema-regex 0.58.6` and `jsonschema-value 0.58.6` also omit their MIT
license file. `jsonschema-0.58.6-MIT.txt` preserves the shared workspace's root
LICENSE at the commit recorded in both published crates' `.cargo_vcs_info.json`:

https://github.com/Stranger6667/jsonschema/blob/55ac1664384793c5d3b20745d3f9c169dfe2c06f/LICENSE

Its SHA-256 is `117829c3ca21efb132d81a44b55363d395ab8eea18526873bc828da4c0e5f038`.
The published crate checksums match `Cargo.lock`; their source paths are
`crates/jsonschema-regex` and `crates/jsonschema-value`, respectively. The
packaging script checks those paths along with the source commit.

The packaging script checks the crate name, version, declared license, repository,
and source commit before using this supplement. A dependency update with a missing
license file fails packaging and requires a new, verified supplement.
