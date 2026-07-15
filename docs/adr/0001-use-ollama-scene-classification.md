# Use Ollama Scene Classification

> **Status: Superseded for v2.0.** Retained as screenshot-selector history. The
> replacement migration is [ADR 0007](0007-migrate-to-video-set-selector-through-gated-cutover.md).

The tool now optimizes for choosing screenshots that are useful in blog articles, so scene labels must describe image content rather than density-based `play` / `event` buckets. We decided to make Ollama scene classification the required core flow, replacing the previous density-based scene assignment, because dynamic game-specific scenes and duplicate-variant handling are more valuable for blog image selection than preserving the old fixed bucket model.
