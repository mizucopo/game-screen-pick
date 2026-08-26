# Cache Video Selection Phases By Input Video

Issue #298 replaces the output-folder, all-input manifest cache with a visible
`cache-game-screen-pick/` directory directly under the Input Video Directory.
Selected Images, Selected Contact Sheet, and `report.json` remain in the Output
Folder. `CACHE_INFO.txt` identifies the cache as regenerable and explains that
the whole directory can be deleted while the application is not running.

An Input Video is identified across runs only by its relative filename and file
size. Absolute path, mtime, and whole-file SHA-256 are not identity inputs. This
allows an Input Video Directory and its cache to be moved or copied without
losing completed work and avoids reading every source byte before resume. The
accepted trade-off is that a content replacement preserving the same relative
filename and size is not detected; deleting the cache is the explicit way to
force regeneration in that case.

The cache is divided into versioned per-video phases: video probe, candidate
frame extraction, mechanical analysis, primary assessment, transition context,
and secondary assessment. Each phase key includes its own version, semantic
inputs, and the keys of upstream phases. Assessment keys additionally include
the relevant model digest, prompt version, Game Context, selection settings,
GPU requirement, and the ordered candidate image digests for the evaluation
unit. Invalid, corrupt, schema-mismatched, or legacy entries are cache misses.
An assessment checkpoint is reusable only when it contains the complete prefix
of batches that the current evaluation unit could have saved; a hole or a
regrouped partial batch invalidates the whole phase checkpoint. Cached scene and
reason text must also match the live normalized 80- and 300-character limits.
Changing one phase version invalidates that phase and dependent later phases
while preserving compatible earlier phases.

Automatic Sample Positions and Frame Candidate IDs are derived independently
for each Input Video. They do not depend on the video's position in the current
input set, so adding another video does not rename or resample unchanged videos.
Per-video primary and secondary candidate pools are evaluated and cached before
the current input set is combined. Global candidate selection, final selection,
Selected Images, Selected Contact Sheet, and report are rerun whenever the input
set or run conditions change. This preserves global diversity without repeating
available per-video extraction or model evaluation.

Run manifests and Output Folder registration live below the visible cache root.
They contain relative Input Video identities and phase conditions, never source
absolute paths, mtimes, or source SHA-256 values. Output completion records may
record the current Input Video Directory and Output Folder locations because
those values control report regeneration and safe ownership of generated
artifacts, not Input Video identity.

Output ownership is accepted only from a structurally valid registration for
the exact resolved Output Folder or from a completion record whose artifact
size and SHA-256 values still match. The report also records its schema and the
size and SHA-256 of every JPEG artifact, allowing an intact application output
set to reestablish ownership after the visible cache is deliberately deleted.
That report-derived set must exactly cover every managed-looking artifact in the
Output Folder.
When changed conditions reuse an owned Output Folder, the next artifacts are
completed in staging within that Output Folder first. The old completed
artifacts remain usable if probing, model validation, assessment, or staging
fails, and are replaced atomically on the same filesystem only at publication.
Abandoned application publication staging directories are removed after output
ownership is established; staging symlinks remain unmanaged and are rejected.
An Output Folder equal to or below `cache-game-screen-pick/` is rejected so
deleting regenerable state cannot delete user artifacts. Managed cache
directory components must be real directories, and cached candidate or context
image leaf symlinks, non-JPEG files, images wider than the 960-pixel extraction
contract, and Pillow decompression bombs are regenerated before use. The cache
lock uses no-follow semantics and accepts only a regular file. JSON, contact
sheet, and extracted-frame publication use exclusive unpredictable temporary
files with the normal process-umask-derived mode, so fixed temporary leaf
symlinks cannot redirect writes outside `cache-game-screen-pick/` or make
published artifacts unexpectedly private. Mechanical analysis records both
usable and rejected frame IDs and is reusable only when they cover every source
frame.
