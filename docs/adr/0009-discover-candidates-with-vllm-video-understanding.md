# Discover Candidates With vLLM Video Understanding

Issue #321 adds `semantic_video`: understand the full Input Video in overlapping,
bounded video chunks, then extract original-video frames around important events.
The user's implementation scope prioritizes this working path over the issue's
earlier remote-runtime orchestration design. The application uses an already
running vLLM server for video understanding and both image-assessment stages;
server installation and GPU placement remain external.

The approved follow-up adds optional lifecycle commands because Ollama and vLLM
cannot both retain models on the user's limited VRAM. Unset operations remain
passive: only an explicit `ollama_unload_before_vllm` uses the existing Ollama host
and credential to unload its currently loaded models, and only paired start/stop
argv commands manage vLLM. Commands run locally; explicit SSH argv supports remote
control without adding a second host configuration for Ollama. Operators provide
finite control commands for an exclusively used server and its workers.

A lazy runtime session wraps the full selection run. It acquires the server before
the first live vLLM request, after any required Game Context generation; a managed
endpoint already responding is rejected before Ollama context inference or unload.
Unload must be confirmed before startup. Every attempted startup has a corresponding
cleanup attempt, including partial startup and interruption. Cleanup failure is
reported without replacing an existing inference error or deleting artifacts.
The application does not reload Ollama models automatically. Forceful process death
and cross-client GPU exclusion require the operator's service supervision.

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
Lifecycle commands and timeout settings are not inference conditions and do not
invalidate caches. Fully cached runs perform no lifecycle communication or commands.
