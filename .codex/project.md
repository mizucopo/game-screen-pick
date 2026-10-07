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
