# Project guidance

- This application is installed as the `game-screen-pick` CLI. Preserve the
  Hatchling build settings, `[project.scripts]` entry point, and
  `[tool.uv] package = true` unless an intentional packaging migration replaces
  them. The template's `application` output does not install this CLI.
- Preserve `.gitignore` entries for `config/*` (except `config/.gitkeep`) and
  `.game-screen-pick/`. They exclude local configuration that may contain API
  keys and generated runtime cache.
- The PR quality workflow retains the Dependabot-updated `setup-uv` v10.1.0
  pin until the template provides it. Refs mizucopo/repo-template#104.
