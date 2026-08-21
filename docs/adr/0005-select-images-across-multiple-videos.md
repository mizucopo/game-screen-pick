# Select Images Across Multiple Videos

Issue #272 extends the Issue #271 pipeline from one input video to an ordered set of one or more videos. The CLI keeps its existing positional shape by treating every path except the last as an input video and the last path as the output folder; a single input remains valid. Each video is sampled across its own full timeline, while the 4,000-frame candidate limit applies to the combined run.

Candidates retain their source video through both Ollama stages and final artifact generation. When enough output slots and valid candidates exist, secondary and final selection cover every source at least once, then balance source, scene, visual, and within-video timeline diversity globally. The report and contact sheet identify the source of every selected image.

Resumability remains a run-level guarantee. The manifest binds the ordered input list, full SHA-256 and metadata for every video, per-video sample positions, model metadata, prompts, and selection settings; changing or reordering any input requires a new output folder.
