# Project guidance

- This application is the `game-screen-pick` Rust CLI. Follow
  `docs/acceptance.md` and `CONTEXT.md` for product behavior and responsibility
  boundaries. Unimplemented selection and publication paths must fail clearly.
- Keep local configuration and generated runtime files out of Git. Configuration
  may contain API keys; never print those values in logs or diagnostics.
- Run the single quality gate with `sh tests/check.sh`. Cargo tests include the
  real input fixture checks and product extraction tests; fixture verification
  alone does not verify extraction, selection, cache, or live AI quality.
- Install FFmpeg and ffprobe before both PR and release quality gates. Tests
  that require these commands must fail when they are absent.
- The Copier-managed release controller is shared release infrastructure. Keep
  its version source at `Cargo.toml` and its package lock entry at `Cargo.lock`.
  Version numbering remains automatic; do not hand-edit a new product version.
- Repository-specific template differences install video tools, use the single
  locked-dependency quality command, list Cargo and Copier release inputs, and
  build and smoke-test all selected supported binary archives before publication.
  The release controller and prerelease numbering use the template directly.
