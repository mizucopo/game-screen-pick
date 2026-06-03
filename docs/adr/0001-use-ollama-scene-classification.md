# Use Ollama Scene Classification

The tool now optimizes for choosing screenshots that are useful in blog articles, so scene labels must describe image content rather than density-based `play` / `event` buckets. We decided to make Ollama scene classification the required core flow, replacing the previous density-based scene assignment, because dynamic game-specific scenes and duplicate-variant handling are more valuable for blog image selection than preserving the old fixed bucket model.
