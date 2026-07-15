# Classify Selection Shortlist With Ollama

> **Status: Superseded for v2.0.** Retained as screenshot-selector history. The
> replacement migration is [ADR 0007](0007-migrate-to-video-set-selector-through-gated-cutover.md).

The tool may process tens of thousands of screenshots, and classifying every blog candidate with Ollama makes runtime scale with the full input size. We decided to build a selection shortlist from neutral image analysis, content filtering, quality score, and visual diversity first, then use Ollama scene catalog creation and scene classification only on that shortlist. This trades complete scene coverage across all blog candidates for much faster runs while preserving Ollama's role in the final blog-image selection.
