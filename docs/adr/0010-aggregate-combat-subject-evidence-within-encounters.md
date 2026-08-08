# Aggregate Combat Subject Evidence Within Encounters

## Status

Accepted. Supersedes ADR 0009.

## Context

Release acceptance showed that one major opponent can produce contradictory
single-image Combat Subject Evidence as attacks, occlusion, distance, and damage
effects change its visible body, surface, and colors. Complete linkage over those
observations split one known subject into several groups or left candidates
ungrouped, so repeated boss images crowded out ordinary combat and distinct scenes.
The earlier conservative choice to preserve a possible second subject after one
conflicting observation therefore violates the human requirement more often than it
protects useful diversity.

## Decision

Candidate Annotation remains a one-image request with an independent conversation
and an atomic per-frame artifact. The selector does not correct or replace any
single-image Combat Subject Evidence. Instead, it first derives a deterministic
Combat Encounter Subject Profile from observations in different Candidate Moments
of one chronological Combat Encounter Group.

A continuous major encounter is one subject and has a hard maximum of one selected
image by default. One conflicting or generic frame is treated as observation noise.
The encounter is split only when every resulting subject is corroborated by at least
two clear observations from different Candidate Moments and the corroborated
profiles are mutually incompatible. Two sibling frames from one Candidate Moment do
not independently corroborate a split.

Combat Subject Group matching across encounters compares the encounter profiles,
not every pair of raw frame observations. A profile match needs image-grounded common
features and supporting Neutral Image Analysis; a name, Scene Slug, screen text, or
generic boss category cannot establish identity. An isolated incompatible
observation does not defeat a profile match, while incompatible values corroborated
in at least two Candidate Moments keep distinct subjects separate.

The resulting group, representative, rejection reason, blocking selected ID, and
privacy-safe evidence are deterministic over the ordered semantic inputs. Worker
count, request completion order, interruption, and resume do not change them. Public
evidence describes only corroborated finite profile values and never exposes names,
raw model output, or media paths.

## Rejected alternatives

- Keep single-frame complete linkage: real effects and partial views fragment one
  opponent and reproduce the acceptance failure.
- Trust opponent names or Scene Slugs: one known acceptance image carried the same
  wrong subject label as a different opponent.
- Split after one incompatible frame: it makes transient effects stronger than the
  chronological encounter evidence.
- Compare several images in one model request: it breaks independent evaluation,
  retry granularity, and interruption-safe cache reuse.

## Consequences

- One noisy frame no longer allows repeated images from a continuous boss encounter.
- A genuine multi-opponent encounter can retain one representative per subject only
  after each subject has independent visual corroboration.
- Existing Candidate Annotation artifacts remain reusable. Selection changes to
  `video-set-selection-v7`, invalidating prior Selection Stages and recomputing the
  canonical report with the new policy provenance. The unchanged selection artifact
  and report JSON structures remain `game-screen-pick/video-set-selection@3.0.0` and
  `game-screen-pick/report@2.2.0`.
