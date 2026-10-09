def text: type == "string" and length > 0;
def inventory($expected): type == "array" and sort == ($expected | sort);
def score: type == "number" and isfinite and . >= 0 and . <= 100;
def interval($outer):
  type == "array" and length == 2 and all(.[]; type == "number" and isfinite) and
  .[0] >= $outer[0] and .[0] < .[1] and .[1] <= $outer[1];
def assessment:
  (.id | text) and (.blog_score | score) and (.transition | type == "boolean") and
  (.scene | text) and (.reason | text);

["01-blocks.mkv", "02-mirrored.mkv"] as $sources |
["A01", "A02", "A03", "B01"] as $ids |
["A01", "A03", "B01"] as $useful_ids |
[
  "insufficient_candidates", "corrupt_payload", "missing_media", "wrong_reference",
  "input_added_moved_count_changed", "inference_changed", "context_generation_changed",
  "context_changed", "unowned_output", "modified_output", "cache_removed", "unsafe_paths",
  "concurrent_run", "interrupt", "staging_failure", "publication_failure", "http_failure",
  "runtime_failure"
] as $cases |
[
  "malformed_json", "unknown_id", "missing_id", "duplicate_id", "non_finite_score",
  "timestamp_outside_chunk", "reversed_event_interval", "refusal", "truncated_response",
  "unsupported_media"
] as $mutations |

($scenarios | length == 1) and ($scenarios[0] |
  .matrix == {
    methods: ["sampled_frames", "semantic_video"],
    inputs: [[$sources[0]], $sources],
    contexts: ["direct", "generated_mock"],
    runs: ["cold", "warm", "interrupted_after_first_batch_or_chunk"]
  } and
  .count == 2 and
  .extraction_cases == [{input: "03-multitrack-rotated.mov", stream_index: 0,
    raw_dimensions: [160, 96], display_dimensions: [96, 160], counter_clockwise_degrees: 90}] and
  (.artifact_expectations |
    .image_count == 2 and .dimensions == [160, 96] and .black_interval_excluded == [2, 3] and
    .source_and_actual_pts_match == true and .sheet_and_report_rank_match == true and
    .warm_calls == {inference: 0, search: 0, runtime: 0} and
    (.warm_processing |
      .media_decode == 0 and .source_content_hash == 0 and .unused_candidate_content_hash == 0 and
      (.io_bound | text)) and (.resume | text)) and
  (.cases | map(.name) | inventory($cases)) and
  all(.cases[]; (.setup | text) and (.expect | text))) and

length == 1 and (.[0] | . as $script |
  (.game_context | text) and
  (.candidates | map(.id) | inventory($ids)) and
  (.candidates | map(.source) | unique | inventory($sources)) and
  all(.candidates[];
    (.time_seconds | type == "number" and isfinite and . >= 0 and . < 6) and
    (.context_seconds | type == "array" and length == 3 and . == (sort | unique) and
      all(.[]; type == "number" and isfinite and . >= 0 and . < 6)) and
    .context_seconds[1] == .time_seconds) and
  (.primary.frames | map(.id) | inventory($ids)) and all(.primary.frames[]; assessment) and
  (.secondary.frames | map(.id) | inventory($useful_ids)) and
  all(.secondary.frames[]; assessment and .transition == false and .blog_score > 0) and
  (.primary.frames | map(select(.transition == false)) | map(.id) | inventory($useful_ids)) and
  any(.primary.frames[]; .id == "A02" and .transition == true and .blog_score == 0) and
  (.primary.frames | map(select(.id == "A01" or .id == "A03")) | map(.blog_score) | unique | length == 1) and
  (.secondary.frames | map(select(.id == "A01" or .id == "A03")) | map(.blog_score) | unique | length == 1) and
  (.invalid_response_mutations | inventory($mutations)) and
  (.semantic_video | map(.source) | inventory($sources)) and
  all(.semantic_video[]; . as $video |
    (.chunk_seconds | interval([0, 6])) and
    (.events | type == "array" and length > 0) and
    (.events | map(.representative_seconds) | sort) ==
      ([$script.candidates[] | select(.source == $video.source and .id != "A02") | .time_seconds] | sort) and
    all(.events[];
      ([.start_seconds, .end_seconds] | interval($video.chunk_seconds)) and
      .start_seconds <= .representative_seconds and .representative_seconds < .end_seconds and
      (.importance | score) and (.scene | text) and (.reason | text)) and
    (.transitions | type == "array" and length == 1) and
    all(.transitions[];
      [.start_seconds, .end_seconds] == [2, 3] and
      ([.start_seconds, .end_seconds] | interval($video.chunk_seconds)) and (.reason | text))))
