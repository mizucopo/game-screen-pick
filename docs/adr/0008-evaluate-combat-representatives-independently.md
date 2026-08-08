# Evaluate Combat Representative Frames Independently

## Status

Accepted.

## Context

Neutral Image Analysis can choose a high-scoring attack-effect frame even when a
nearby frame in the same Candidate Moment shows the enemy clearly. Passing several
frames to one model request was previously rejected because the response could mix
their semantics and identifiers, while accepting only a successful subset after an
inference failure would make output depend on transient runtime behavior.

## Decision

Candidate Annotation first evaluates the Primary Representative Frame in a
single-image Ollama request. Only when that successful observation shows combat but
has Explanation Value `none`, it evaluates the remaining two or fewer Frame
Candidates from the same Candidate Moment, with every image in its own request and
conversation context. Independent requests may run concurrently up to
`ollama.max_parallel_requests`; the built-in default remains `1`, while the supported
RTX 5090 target profile uses `2`.

Each main Candidate Annotation request also returns finite Combat Subject Evidence
for the opponent visible in that image. It must not use another frame, Context Cue,
Scene Slug, opponent name, screen text, status UI, background, or player appearance.
This preserves the independent-evaluation boundary when downstream selection groups
the same major combat subject across videos.

Different Candidate Moments may also be processed concurrently under the same
limit. Each Moment still completes its Primary Representative Frame before deciding
whether its own fallback frames are required. Scheduling never combines images or
conversation state, and results are restored to deterministic shortlist and frame
order rather than completion order.

The scheduler does not reserve worker capacity for cache hits or wait for a fixed
batch's slowest request. It continuously assigns the next unfinished Moment when a
worker becomes free. On user interruption it cancels queued Moments and propagates a
cooperative cancellation request to the active VisionRuntime operations. The default
Ollama transport aborts the active connection during response-header or body waits,
and Candidate Annotation retry delays wait on the same cancellation event.

The shortlist reserves every Primary Representative Frame before assigning fallback
frames in deterministic shortlist order. A Frame Candidate ID therefore belongs to
only one annotation request, without allowing an earlier fallback to displace a
later moment's primary.

Every per-frame result is an atomic Completed Stage. If any required inference fails,
the Candidate Moment is not aggregated: successful sibling results remain reusable,
and a resumed run retries only the failed or unfinished frame before comparing all
observations. The final Representative Frame is selected locally and deterministically
by explanation value, blog-relevant content, subject visibility, transient
obstruction, Neutral Image Analysis quality, and Frame Candidate ID. No alternative
is forced when every observation has Explanation Value `none`, and a non-combat
observation is never eligible to replace the combat primary.

The per-frame cache identity includes the image, context, model, prompt, schema, and
runtime contract, but not the fallback aggregation policy or scheduling order. A
fallback-policy change therefore invalidates aggregation and downstream selection,
not unchanged per-frame annotations.

Changing the Candidate Annotation schema to add Combat Subject Evidence does
invalidate that per-frame annotation and downstream selection. It does not invalidate
Video Identity, Video Stage, Context Cue, or other unchanged upstream artifacts.

## Consequences

- Combat fallback spends extra inference only after an unusable combat primary.
- Parallel scheduling improves target utilization without combining image semantics.
- The configured limit bounds all in-flight Candidate Annotation stages, including
  primaries and fallback siblings from different Moments.
- A slow request does not leave capacity idle after another worker finishes.
- Interruption, request completion order, and transient partial failure do not change
  the selected Representative Frame.
- Valid existing one-frame annotations can be reused when their semantic identity is
  unchanged.
