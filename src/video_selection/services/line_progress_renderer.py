"""Progress Eventの一行renderer。"""

import json

from ..models.progress_event import ProgressEvent


class LineProgressRenderer:
    """redirectとCI向けにProgress Eventを一行へ描画する。"""

    def render(self, event: ProgressEvent) -> str:
        """制御文字をescapeしたstable field列を返す。"""
        parts = [f"[{event.severity}]", f"event={event.kind}"]
        if event.reason_code is not None:
            parts.append(f"reason={event.reason_code}")
        if event.stage is not None:
            parts.append(f"stage={event.stage.value}")
        if event.stage_index is not None:
            parts.append(
                f"stage_index={_position(event.stage_index, event.stage_count)}"
            )
        if event.video_order is not None:
            parts.append(f"video={_position(event.video_order, event.video_count)}")
        if event.video_relative_path is not None:
            escaped_path = json.dumps(
                event.video_relative_path,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            parts.append(f"path={escaped_path}")
        if event.processed_count is not None:
            parts.append(
                f"progress={_position(event.processed_count, event.total_count)}"
            )
        if any(
            (
                event.cache_hit_count,
                event.cache_miss_count,
                event.reuse_count,
                event.recompute_count,
            )
        ):
            parts.extend(
                (
                    f"cache_hit={event.cache_hit_count}",
                    f"cache_miss={event.cache_miss_count}",
                    f"reuse={event.reuse_count}",
                    f"recompute={event.recompute_count}",
                )
            )
        if event.elapsed_seconds is not None:
            parts.append(f"elapsed={event.elapsed_seconds:.1f}s")
        if event.eta_seconds is not None:
            parts.append(f"eta={event.eta_seconds:.1f}s")
        elif event.estimation_state == "estimating":
            parts.append("eta=estimating")
        if event.work_unit_kind is not None:
            parts.append(f"unit={event.work_unit_kind}")
        return " ".join(parts)


def _position(index: int, total: int | None) -> str:
    return str(index) if total is None else f"{index}/{total}"
