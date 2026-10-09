# Project guidance

- This application is installed as the `game-screen-pick` CLI. Preserve the
  Hatchling build settings, `[project.scripts]` entry point, and
  `[tool.uv] package = true` unless an intentional packaging migration replaces
  them. The template's `application` output does not install this CLI.
- Preserve `.gitignore` entries for `config/*` (except `config/.gitkeep`) and
  `.game-screen-pick/`. They exclude local configuration that may contain API
  keys and generated runtime cache.
- Preserve the `src` package initializer and import contract test, and the
  mypy `files`/`mypy_path` mapping. The CLI imports `src.main` and its modules
  use package-relative imports; the template's flat application layout does
  not match this project.
- Keep direct pytest execution in the quality task: this project has an
  established test suite, so empty collection must remain a failing gate.
- Run Ruff over the whole repository in quality and fix tasks so the release
  controller remains covered when CI delegates to `task check`.
- Preserve FFmpeg/ffprobe installation before both PR and release quality
  gates. Input fixtures require real video decoding and must fail rather than
  skip when the tools are absent.
- For Rust product decisions, follow `docs/acceptance.md` and ADR 0010.
  Existing packaging/import and Python quality instructions apply to the
  current CLI until #339 replaces its development setup. They do not require
  keeping Python code, comparison tests, old cache/config formats, or a dual
  quality gate. Do not add compatibility layers or preserve unused files.
- Run `sh tests/check-fixtures.sh` for the Rust input corpus. #339 will connect
  these facts to extraction tests and its Cargo quality gate; fixture checks
  alone do not verify the product CLI, selection, cache, or live AI quality.
- The SemVer release workflow marks prereleases with `--prerelease`, using
  the prepared version before build metadata, and skips their Latest lookup.
  This is a temporary template exception until the generic release workflow
  supplies the same behavior (tracked in mizucopo/repo-template#167).
