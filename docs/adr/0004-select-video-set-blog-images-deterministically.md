# Select Video Set Blog Images Deterministically

## Context

The video-input design produces annotated Blog Candidates across an ordered Video Set. Candidate Annotation owns semantic judgments, but it deliberately does not decide quality, a final score, soft coverage, diversity, or selection. The final selector therefore needs one deterministic policy that preserves useful late-game images, applies spoiler preferences softly, maintains a practical blog-image mix, avoids visual and temporal repetition, and explains why each candidate won or lost.

This ADR defines that policy. It applies to the future Video Set selector and does not implement or change the existing screenshot-input flow.

## Semantic boundaries

Blog Image Type describes the candidate's primary explanatory role in a blog, not whether the player can currently provide input:

- `normal_gameplay`: exploration, combat, puzzles, or other play remains primary; short dialogue or HUD text may be overlaid.
- `event`: conversation, a cutscene, or a scripted presentation is itself the primary subject.
- `menu`: an inventory, equipment, map, settings, shop, or similar interface is primary.
- `title`: a title, logo, or landing screen is primary.
- `other`: a valid candidate whose primary role is none of the above.

Scene Selection Role remains separate. `recurring_gameplay` controls variant expansion, while `ordinary` and `cinematic` use normal visual-diversity behavior. The Video Set selector does not apply ADR 0003's Cinematic Soft Cap because Blog Image Type soft coverage replaces it.

## Base utility

Quality remains the primary signal. Candidate Annotation's ordinal values are converted as follows:

| Value | Explanation Value | Context Cue Relevance |
|---|---:|---:|
| unavailable | - | 0 |
| none | 0 | 0 |
| low / weak | 1/3 | 0.5 |
| medium | 2/3 | - |
| high / strong | 1 | 1 |

For Quality Score `Q`, converted Explanation Value `E`, and converted Context Cue Relevance `C`:

```text
Selection Base Utility = 0.70 * Q + 0.25 * E + 0.05 * C
```

Context Cue can therefore add at most 0.05. Video position, Blog Image Type, diversity, and spoiler handling are not part of this candidate-local value.

An annotated candidate whose Explanation Value is `none` remains available for diagnostics and receives a Counterfactual Selection Score, but is deterministically ineligible for final selection. The selector reports it as `lower_marginal_utility` and accepts a Selection Shortfall instead of filling the request with an image that cannot explain play or an event. Candidate Annotation still does not return an eligibility or selected flag; it returns the ordinal semantic value and the deterministic selector applies this boundary.

## Spoiler handling

Spoiler Sensitivity is a run setting with `low`, `medium`, and `high`; the default is `medium`. The selector subtracts the following Spoiler Penalty from utility:

| Spoiler Risk | low | medium | high |
|---|---:|---:|---:|
| `none` | 0 | 0 | 0 |
| `low` | 0 | 0.01 | 0.02 |
| `medium` | 0.02 | 0.04 | 0.08 |
| `high` | 0.05 | 0.10 | 0.18 |

All table values are soft penalties; no table entry is a per-candidate hard
rejection. Because greedy coverage and diversity interactions can otherwise make
a higher sensitivity select more major spoilers, the selector also applies a
deterministic monotonicity guard: `medium` may select no more `high`-risk
candidates than the same pool selected at `low`, and `high` may select no more
than `medium`. Each guarded profile is recomputed from an empty selected set
under its own penalty values and the immediately lower profile's count limit.

Risk boundaries are:

- `none`: exploration, combat, and UI that reveal no meaningful story information.
- `low`: minor progression facts such as an ordinary place, item, or quest.
- `medium`: a named boss, unique late-game area, important quest outcome, or new form when the image reveals meaningful story progress.
- `high`: an ending, final-boss identity or form, major-character fate, betrayal, culprit or true identity, or central story reveal.

`high` requires concrete semantic evidence from the image, screen-text role, or Context Cue. Enemy names, HP/status bars, Combat Encounter Kind, Video Order, and late Video Set Progress alone never raise risk.

## Blog Image Type soft coverage

For requested output count `N`, normal targets are:

| Blog Image Type | Target share |
|---|---:|
| `normal_gameplay` | 70% |
| `event` | 25% |
| `menu` | 5% |
| `title` | no reserved target |
| `other` | no reserved target |

Fractional targets use the largest-remainder method. Equal remainders are resolved in `normal_gameplay`, `event`, `menu` order. Examples are 7/3/0 for `N=10`, 22/8/2 for `N=32`, and 70/25/5 for `N=100`.

An eligible `normal_gameplay`, `event`, or `menu` candidate receives a `+0.10` Blog Image Type Coverage Bonus while the selected count for its type is below target. The bonus becomes zero at target; exceeding a target has no penalty. `other` never receives a coverage bonus. `title` receives `+0.05` until one title is selected, after which every further title candidate is ineligible. This is a hard maximum of one title, not a guaranteed title slot.

## Conditional minimum coverage

The public Effective Configuration requires `N >= 10`. At that size, the selector applies a conditional minimum of one selected image to each of these facets when an eligible candidate exists:

- `ordinary_combat`: `normal_gameplay` whose Combat Encounter Kind is `ordinary` and whose Combat Encounter Basis positively identifies an ordinary opponent group or ordinary encounter presentation. The absence of major-encounter evidence is insufficient. A visible combat frame with only an enemy name or HP/status bar, with neither positive ordinary nor major evidence, is `uncertain` and does not qualify. `major` combat, exploration, movement, and obstacle interactions do not qualify. Spoiler Risk evaluates story disclosure independently and does not classify the encounter.
- `event`: Blog Image Type `event`.

Explanation Value, visual eligibility, the title maximum, existing duplicate boundaries, and the Spoiler Monotonicity Guard remain stronger than these minimums. A missing or still-ineligible facet releases its slot rather than filling it with a poor image. After satisfiable minimums are met, all remaining slots use the normal utility policy; the resulting proportions are therefore dynamic rather than a fixed quota.

## Temporal and visual diversity

Video Set Progress concatenates Video Durations in Video Order and normalizes a Candidate Moment's cumulative time to `[0, 1)`. It is used only for temporal diversity, not as quality or spoiler evidence.

For requested count `N`, let `d` be the nearest absolute Video Set Progress distance to a selected candidate. With no selected candidate, the penalty is zero. Otherwise:

```text
Temporal Diversity Penalty = 0.08 * max(0, 1 - d / (1 / N))
```

There are no chronological buckets and no per-video minimums.

Visual similarity is an eligibility rule rather than another numeric penalty:

- Normal selection begins at the configured similarity ceiling, whose default is 0.72.
- If the current annotated pool cannot fill the request, the ceiling is relaxed through deterministic configured steps. A configured base at or below 0.97 ends at 0.97; an explicitly configured base above 0.97 remains the terminal ceiling.
- The built-in relaxation deltas are `+0.03`, `+0.06`, `+0.10`, and `+0.15`; duplicate capped values are removed and the terminal ceiling is appended.
- A `recurring_gameplay` scene may use the same terminal ceiling after eligible Variant Groups have each had their first opportunity, allowing state variants without automatically admitting the observed 0.973 near-repetition boundary.
- A pair with cosine similarity greater than 0.995 is a Visual Near-Duplicate and can never be selected together.

Classification boundaries are not sufficient to prevent semantic repetition, so the selector also assigns Semantic Duplicate Groups before greedy selection:

- `title_semantics` groups every candidate whose Blog Image Type, Screen Text Kind, or Representative Frame Evidence identifies a title screen. This preserves the one-title limit when Blog Image Type is wrong.
- `combat_encounter_sequence` orders every candidate within one Video Source, splits encounters at non-`major` candidates, and then groups chronological `major`-combat Scene Slug runs. A one-candidate slug blip is bridged only when matching slug runs surround it and both adjacent gaps are at most 15 seconds. The same slug after a different major-combat run or a non-major scene starts a different encounter.
- `visual_role_similarity` groups non-title, non-major candidates only when every pair is from the same source, no more than 30 seconds apart, has the same image-grounded content kind and Combat Encounter Kind, and has Neutral visual similarity of at least 0.93. Recurring-gameplay candidates additionally require equal normalized independent image summaries so distinct techniques, enemies, and outcomes remain available.

Every Semantic Duplicate Group has a hard maximum of one selected image, even during Selection Shortfall. The first candidate selected by the normal Marginal Selection Utility ordering is its representative. This maximum outranks conditional coverage and Variant Expansion; an unrepresented encounter or role remains preferable to a second image from the same group. Neutral visual similarity never establishes a group by itself.

The selector-level ceiling never exceeds the explicitly configured maximum of 0.98. The user-facing configuration and validation contract belongs to the CLI/config design.

## Greedy selection

At each selection step, every currently eligible unselected candidate receives:

```text
Marginal Selection Utility =
    Selection Base Utility
  - Spoiler Penalty
  + Blog Image Type Coverage Bonus
  - Temporal Diversity Penalty
```

While an applicable conditional minimum is unmet, the selector compares eligible candidates belonging to unmet facets before the unrestricted pool. When multiple facets remain unmet, it first retains only candidates that participate in at least one jointly compatible terminal-ceiling choice containing one candidate per unmet facet and all Variant Group representatives required by that choice within the remaining output capacity; a locally higher-utility choice cannot make another feasible minimum unsatisfiable. If a candidate in an already represented recurring-gameplay Variant Group first requires another group from that scene to be represented, that eligible prerequisite advances while enough output capacity remains reserved for every unmet minimum. If a minimum candidate is blocked only by the current similarity pass, it preserves room through later passes. At the terminal pass an unsatisfiable minimum is released; selected images are retained and unrestricted selection restarts from the configured base similarity ceiling. Satisfying the final minimum on a relaxed pass performs the same base-ceiling restart before filling unrestricted slots. The selector otherwise chooses the highest value and recomputes all coverage and temporal terms before choosing the next image. Ties are resolved by:

1. lower Spoiler Penalty;
2. higher Quality Score;
3. lower maximum visual similarity to the selected set;
4. Video Order, Video Time, and Frame Candidate ID.

The last keys provide stable ordering only for exact ties and do not add a preference for earlier progress.

Selection starts at the normal similarity ceiling and preserves selected images while moving through relaxation steps. If the Selection Shortlist grows, selection is recomputed from an empty selected set at the base ceiling against the expanded annotated pool so a previously relaxed choice cannot lock out a newly annotated diverse candidate.

## Shortfall and failure

If the current annotated Selection Shortlist cannot produce `N` images at the terminal similarity ceiling, the Video Set Stage extends it in deterministic local shortlist order, completes Candidate Annotation for the added Candidate Moments, and recomputes selection. For `N >= 10`, it also extends while a known conditional facet has not yet been discovered or its minimum remains unsatisfied; a candidate identified as a title by Screen Text Kind or Representative Frame Evidence cannot satisfy the event facet even when Blog Image Type says `event`. When two annotated major-combat candidates currently share a Combat Encounter Group, selection also waits while any unannotated Candidate Moment remains between them in the complete source timeline. This observes a later-batch non-major boundary without forcing annotation of every remaining candidate. Expansion stops when the output, known minima, and observed encounter boundaries are complete or all valid Candidate Moments are exhausted. Batch sizing and operational limits belong to the runtime-capacity design.

After all valid Candidate Moments are exhausted, selecting fewer than `N` images is a Selection Shortfall. The run publishes the selected images, completes successfully with a warning, and reports requested count, selected count, the final similarity pass, and reason counts. It never fills a shortfall with:

- an invalid Frame Candidate;
- an incomplete Candidate Annotation;
- a Candidate Annotation whose Explanation Value is `none`;
- a second `title` image;
- a second member of a Semantic Duplicate Group;
- a Visual Near-Duplicate.

Candidate Annotation failure is not a shortfall. The failure contract from Issue 165 remains fatal, and no output is published until the failed annotation succeeds on a later run.

## Ranking examples

With `N=10`, default Spoiler Sensitivity `medium`, no selected images, and all candidates visually eligible:

| Candidate | Description | Base | Coverage | Spoiler | Marginal |
|---|---|---:|---:|---:|---:|
| B | useful late normal gameplay | 0.916 | +0.10 | 0 | **1.016** |
| D | explanatory mid-game event | 0.821 | +0.10 | 0 | **0.921** |
| C | late major-spoiler event | 0.902 | +0.10 | -0.10 | **0.902** |
| A | ordinary early combat | 0.741 | +0.10 | 0 | **0.841** |

Because `N=10`, D and A satisfy the two conditional facets and are selected before the unrestricted pool, in D then A utility order. B remains the strongest unrestricted candidate. Late position does not lower B, while C's soft spoiler penalty does not make it a hard rejection.

If seven gameplay and one event image have already been selected for `N=10`, a gameplay candidate at 0.86 remains 0.86, an event candidate at 0.79 becomes 0.89 from coverage, and a gameplay candidate at 0.92 remains 0.92. Coverage helps the under-target event without overruling a clearly stronger gameplay image.

## Consequences and diagnostics

Changing requested count, Spoiler Sensitivity, penalty weights, or coverage targets can reuse Candidate Annotation because these inputs affect only deterministic final selection. The selection-policy version must still be part of the final selection stage fingerprint.

The report must retain enough diagnostic data to reproduce each decision: utility components, type targets and actuals, conditional-facet eligible/minimum/actual counts and reallocation, spoiler setting and penalty, the monotonicity count limit, progress distance and temporal penalty, visual threshold/pass, nearest selected similarity, Variant Group behavior, Semantic Duplicate Group ID and privacy-safe basis, blocking selected ID, tie-break use, and Selection Shortfall reasons. Unselected candidates retain their best observed counterfactual utility and one stable rejection code, including `semantic_duplicate` when the selected representative blocks another group member. The exact public report schema is decided with the CLI/config/report contract.

Greedy selection is intentionally preferred over a global optimizer: it is deterministic, incremental, and reportable, while still allowing coverage and temporal effects to react after each selected image.
