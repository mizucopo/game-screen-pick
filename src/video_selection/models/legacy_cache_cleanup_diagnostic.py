"""Legacy Cache cleanupのstructured diagnostic。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LegacyCacheCleanupDiagnostic:
    """削除された認識済みlegacy entryの件数と内容byte。"""

    removed_entry_count: int
    removed_bytes: int

    def __post_init__(self) -> None:
        """diagnosticのcountを非負に保つ。"""
        if self.removed_entry_count < 0 or self.removed_bytes < 0:
            msg = "Legacy Cache cleanup diagnosticは非負である必要があります"
            raise ValueError(msg)
