# Discover Input Videos From a Directory

Issue #281 replaces the public CLI's variable-length video path list with one
input video directory. This supersedes the CLI-shape decision in ADR 0005; the
ordered multi-video pipeline and run-level resumability from that ADR remain in
effect.

The CLI scans only regular files directly inside the input directory. It
recognizes common ffmpeg-supported video container extensions case-insensitively
and sorts matching paths by filename before creating `Input Videos`. It does not
scan subdirectories. This makes repeated discovery deterministic, avoids
silently expanding the run when unrelated nested folders appear, and keeps the
selected source order visible through the existing manifest, report, and
contact sheet.

A missing or non-directory input and a directory with no matching videos are
rejected before the application pipeline starts. A single video file is not a
valid public CLI input. The internal `VideoSelectionRequest` continues to carry
an ordered tuple of video paths. The selector requires those paths to share one
direct parent so ADR 0008 can place the visible phase cache in the corresponding
Input Video Directory.
