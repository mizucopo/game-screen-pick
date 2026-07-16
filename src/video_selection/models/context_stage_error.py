"""Context Collection Stageのreason-coded failure。"""

from .context_stage_failure_reason import ContextStageFailureReason


class ContextStageError(RuntimeError):
    """安全な説明とstable reasonを持つContext Stage error。"""

    def __init__(self, reason: ContextStageFailureReason) -> None:
        messages = {
            ContextStageFailureReason.AMBIGUOUS_AUDIO_STREAM: (
                "audio streamを一意に選択できませんでした"
            ),
            ContextStageFailureReason.AMBIGUOUS_SUBTITLE_STREAM: (
                "subtitle streamを一意に選択できませんでした"
            ),
            ContextStageFailureReason.CHUNK_FAILED: (
                "speech-to-textの一部chunkを完了できませんでした"
            ),
            ContextStageFailureReason.INVALID_AUDIO_STREAM: (
                "明示されたaudio streamを利用できません"
            ),
            ContextStageFailureReason.INVALID_SUBTITLE_STREAM: (
                "明示されたsubtitle streamを利用できません"
            ),
            ContextStageFailureReason.STT_ANALYSIS_FAILED: (
                "speech-to-text処理を完了できませんでした"
            ),
            ContextStageFailureReason.TIMESTAMP_DRIFT: (
                "Context CueのtimestampをVideo Timeへ対応付けられませんでした"
            ),
            ContextStageFailureReason.UNSUPPORTED_BITMAP_SUBTITLE: (
                "選択されたbitmap subtitleは利用できません"
            ),
        }
        super().__init__(messages[reason])
        self.reason = reason
