# Select Images Across Multiple Videos

Issue #272 extends the Issue #271 pipeline from one input video to an ordered set of one or more videos. The CLI keeps its existing positional shape by treating every path except the last as an input video and the last path as the output folder; a single input remains valid. Each video is sampled across its own full timeline. The original combined-run 4,000-frame candidate limit is superseded by Issue #300: automatic and explicit-interval sampling are not capped by candidate count.

Candidates retain their source video through both Ollama stages and final artifact generation. When enough output slots and valid candidates exist, secondary and final selection cover every source at least once, then balance source, scene, visual, and within-video timeline diversity globally. The report and contact sheet identify the source of every selected image.

Resumability remains a product guarantee, but the run-level full-SHA-256 identity and all-or-nothing reuse decision from this paragraph are superseded by ADR 0008. Adding an Input Video now preserves reusable per-video phases while rerunning input-set-wide selection and artifacts.
