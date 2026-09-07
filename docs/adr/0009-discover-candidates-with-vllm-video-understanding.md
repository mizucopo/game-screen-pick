# Discover Candidates With vLLM Video Understanding

Issue #321 adds `semantic_video`: understand the full Input Video in overlapping,
bounded video chunks, then extract original-video frames around important events.
The user's implementation scope prioritizes this working path over the issue's
earlier remote-runtime orchestration design. The application uses an already
running vLLM server for video understanding and both image-assessment stages;
server installation, GPU placement and lifecycle remain external.

This extends ADR 0004's equal-interval/Ollama path, retained as the default
`sampled_frames` strategy. Both strategies use the existing mechanical quality,
transition context, diversity selection and atomic artifact publication. Temporal
event importance affects primary candidate ranking, and event explanations inform
both image assessments. Failures never select another strategy implicitly.

ADR 0008's input-video identity and phase-cache contracts remain. Each semantic
chunk is checkpointed independently; its conditions exclude output count and the
other input videos. Changed analysis results invalidate downstream work even when
candidate times happen to stay equal. The model endpoint does not expose immutable
weights, so `vllm_cache_revision` explicitly identifies the deployment's weights,
quantization, processor and runtime settings. Operators change it with the server;
the stored digest is a configuration fingerprint, not evidence of weight identity.
