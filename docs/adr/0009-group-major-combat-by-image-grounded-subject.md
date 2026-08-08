# Group Major Combat by Image-Grounded Subject

## Status

Accepted.

## Context

A Video Set can contain several images of the same boss or other major opponent in
different recordings, encounters, camera states, Scene Slugs, and public labels. A
name-based rule is unsafe because image descriptions can be misspelled or inferred
from text, while the existing Combat Encounter Group only represents one local
timeline run. Selecting several images of the same opponent crowds out ordinary
combat and other distinct opponents even when each image is individually useful.

The selector must keep only the best image of one major combat subject without
merging every boss battle or depending on request completion order. The evidence must
remain reusable after interruption and safe to expose in diagnostics.

## Decision

Candidate Annotation returns a structured Combat Subject Evidence object from each
single image. It contains only finite enums for body plan, scale, surface, at most two
colors, at most four visible traits, and `distinctive`, `generic`, or `unclear`
distinctiveness. It observes the visible opponent body only. It does not use the
opponent name, Scene Slug, screen text, HP or status UI, background, player, Context
Cue, adjacent images, or another model response.

Only a `major` candidate with complete `distinctive` evidence can identify a combat
subject. Two candidates match when body plan, scale, and surface are equal, their
color sets and trait sets each intersect, and their Neutral Image Analysis cosine
similarity is at least 0.80. A group uses complete linkage: every pair must match.
The group can cross videos, encounter times, Scene Slugs, public descriptions, and
wrong names. Generic, unclear, incomplete, or visibly different subjects remain
separate.

Combat Encounter Group remains auxiliary. Within one encounter, identifiable
evidence is compatible without the 0.80 Neutral threshold when body plan, scale, and
surface match and the color sets and trait sets each intersect. When all identifiable
evidence is compatible, the existing chronological group remains useful, including
unclear members. When multiple clearly different subjects are present, only mutually
compatible repeated subjects use the encounter basis; unclear members are not
attached across that distinction.

Overlapping semantic groups are merged deterministically, with published basis
priority `combat_subject_appearance`, `combat_encounter_sequence`, `title_semantics`,
then `visual_role_similarity`. A basis participates in that priority only when its
public contract can describe the merged component. In particular,
`combat_subject_appearance` requires at least one finite evidence token common to
every member; without one, publication falls back to the next applicable basis while
the originating subject groups remain internal evidence. The pre-merge Combat
Encounter Group edges remain available to Shortlist boundary observation even when
the published basis becomes `combat_subject_appearance`. Selection cannot stop while
an unannotated Candidate Moment remains between members of one of those encounter
edges.

Every Combat Subject Group has the existing Semantic Duplicate Group hard maximum of
one selected image, including during Selection Shortfall. Marginal Selection Utility
and the stable selector tie-break choose the representative. Input order, parallel
completion order, public description, and naming errors do not affect the result.

The selected representative and rejected duplicates keep the same deterministic
group ID, basis, and finite evidence tokens common to every member. Each token is
validated as a complete field-and-value enum rather than a prefix pattern, so a value
from one category cannot be relabeled as another. Diagnostics do not contain free
text, names, raw model output, or inferred identity.

## Contract and cache versions

The single-image contract becomes `candidate-annotation-prompt-v18`,
`candidate-annotation-schema-v13`, and `candidate-annotation-stage-v35`, stored as
`game-screen-pick/candidate-annotation@5.0.0`. The deterministic selector becomes
`video-set-selection-v6`, stored as
`game-screen-pick/video-set-selection@3.0.0`. The canonical report becomes
`game-screen-pick/report@2.2.0`.

Old Candidate Annotation and Selection artifacts are not restored into these new
contracts. Only the affected annotation and downstream selection/report are
recomputed. Video Identity, whole-file digest, Video Stage, frame extraction,
Context Cue, and other unchanged upstream stages remain reusable when their own
fingerprints match.

## Rejected alternatives

- Scene Slug or opponent names: labels can vary or be wrong and may expose inferred
  identity rather than image-grounded evidence.
- Encounter sequence alone: it cannot recognize the same subject across recordings
  and can merge distinct subjects inside one encounter.
- One global visual threshold: effects and arena composition can make different
  opponents look similar, while camera variation can lower similarity for one
  opponent.
- Multi-image model comparison: it would mix image semantics, reduce cache
  granularity, and make retry or parallel completion behavior affect the result.

## Consequences

- Repeated images of one identifiable major opponent compete for one representative,
  leaving more room for ordinary combat and diverse opponents.
- Conservative `generic` and `unclear` results can leave duplicates unmerged rather
  than risk suppressing a distinct subject.
- Candidate Annotation inference and cache size grow by one small structured object
  per frame.
- A schema upgrade recomputes affected annotations once, but interruption-safe atomic
  frame artifacts prevent already completed new-contract work from being repeated.
