"""旧単一動画selector import pathの互換shim."""

from .video_selector import VideoSelector as SingleVideoSelector

__all__ = ["SingleVideoSelector"]
