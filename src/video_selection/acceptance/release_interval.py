"""release acceptance suiteの一つのprivate source interval。"""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import PurePosixPath


@dataclass(frozen=True)
class ReleaseInterval:
    """source相対path、期待区間、scenario roleをtarget内だけで保持する。"""

    relative_video_path: str
    start: Fraction
    end: Fraction
    scenario_role: str

    def __post_init__(self) -> None:
        """安全な相対pathと正の有限区間だけを受理する。"""
        path = PurePosixPath(self.relative_video_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != self.relative_video_path
            or self.start < 0
            or self.end <= self.start
            or not self.scenario_role.strip()
            or any(character in self.scenario_role for character in "\r\n")
        ):
            msg = "Release intervalのpath、時刻、scenario roleが不正です"
            raise ValueError(msg)

    @property
    def expected_duration(self) -> Fraction:
        """profileで指定された期待区間長を返す。"""
        return self.end - self.start
