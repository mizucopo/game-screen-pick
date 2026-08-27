# Generate Game Context Before Video Selection

Image evaluation needs concise game facts, but a user may know only an informal
title such as an abbreviation. We decided that every new run accepts exactly one
of a directly supplied Game Context or a Game Title used to generate that context.
Game Title resolution, edition disambiguation, and Web search are confined to the
generation boundary. The resulting Game Context is the only game-specific text
used by image evaluation and selection.

Dynamic generation uses one explicitly selected provider: Ollama Web Search with
a local Ollama model, OpenAI Responses `web_search`, Gemini Google Search, or xAI
Responses `web_search`. Search content is treated as untrusted data, official
sites and stores are preferred, and unresolved identity, insufficient evidence,
or contradictory sources fail instead of producing guessed facts. Provider
and model must both be configured explicitly. Neither has an implicit default,
and provider failure never falls back to another provider.

The final Game Context is resolved before video probing and long-running frame
processing. It is logged and stored in the Run Manifest and report. Dynamically
generated contexts also store the provider and actual model. Game Title and raw
search results are not selection inputs and are not stored in those artifacts.
Resume uses the stored Game Context without searching again; legacy manifests
that already contain a Game Context remain resumable while their old Game Title
field is ignored.
