"""Video処理とSelected Image公開を担うMediaRuntime port。"""

from typing import Protocol

from .selected_frame_media_runtime import SelectedFrameMediaRuntime
from .video_stage_media_runtime import VideoStageMediaRuntime


class SelectionMediaRuntime(
    VideoStageMediaRuntime,
    SelectedFrameMediaRuntime,
    Protocol,
):
    """実Video Selection Applicationが必要とするmedia能力の合成port。"""
