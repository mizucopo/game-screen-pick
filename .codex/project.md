# Project guidance

- This application is the `game-screen-pick` Rust CLI. Follow
  `docs/acceptance.md` and ADR 0010 for product behavior and responsibility
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
- Repository-specific template differences are limited to installing video
  tools, using the single locked-dependency quality command, and listing its
  Cargo and Copier release inputs. The release controller and prerelease
  behavior use the template directly.
