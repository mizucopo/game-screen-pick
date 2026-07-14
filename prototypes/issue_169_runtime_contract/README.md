# PROTOTYPE — 24-image Scene Catalog capability probe

This throwaway probe answers one question for Issue #169:

> Can the installed `qwen3-vl:8b-instruct` model on the reference RTX 5090 runtime accept the maximum 24 representative images, return the Issue #165 Scene Catalog schema three times, and stay fully GPU-loaded with an explicit 32K context?

It is not production code. It deterministically samples two temporary 960 px JPEG frames from each of the 12 supplied videos, sends the same 24 images to the Windows Ollama server three times, validates the response locally, and prints only aggregate metrics plus scene slugs and roles. Temporary images and full model responses remain under the remote process temporary directory and are deleted on exit.

Run from the repository root:

```bash
ssh winpc python3 - \
  --video-dir '/mnt/g/Captures/14_冒険家エリオットの千年物語/movie' \
  --ollama-host http://172.20.32.1:11434 \
  --model qwen3-vl:8b-instruct \
  --runs 3 \
  < prototypes/issue_169_runtime_contract/scene_catalog_probe.py
```

The observed gateway is intentionally an argument, not a built-in default. It can change after a WSL restart.

## Observed result

Verified on 2026-07-14 with Windows Ollama `0.31.2`, RTX 5090, and installed model `qwen3-vl:8b-instruct` (`Q4_K_M`, digest `0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319`).

- All three 24-image requests passed the JSON Schema and local domain validation.
- The explicit context length was 32,768 for every request.
- The first request took 14.261 seconds including 7.409 seconds of model load.
- The two warm requests took 3.795 and 3.742 seconds.
- Prompt evaluation used 12,514 tokens.
- Ollama reported 10,210,393,456 bytes for both model size and `size_vram`, so the model was fully GPU-loaded.
- `nvidia-smi` global memory usage peaked at 14,629 MiB during the probe.
- Warm runs 2 and 3 returned the same complete output. Run 1 used slightly different scene slug wording, so temperature zero is not treated as a bit-for-bit stability guarantee.

No extracted frame or full model response is committed.
