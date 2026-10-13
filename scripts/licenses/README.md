# Crate license supplements

`uuid-simd 0.8.0` and `vsimd 0.8.0` omit their MIT license file from the
published crate. `Nugine-simd-MIT.txt` preserves the upstream file at the commit
recorded in both crates' `.cargo_vcs_info.json`:

https://github.com/Nugine/simd/blob/d74c030d9dc4f3cae02146d1f497ff62726ef09a/LICENSE

The packaging script checks the crate name, version, declared license, repository,
and source commit before using this supplement. A dependency update with a missing
license file fails packaging and requires a new, verified supplement.
