"""公開直前までstagingされたOutput Folder。"""

import shutil
from pathlib import Path


class PreparedOutput:
    """最終renameだけを残したOutput Folderを所有する。"""

    def __init__(
        self,
        staging_folder: Path,
        output_folder: Path,
        report: dict[str, object],
    ) -> None:
        self._staging_folder = staging_folder
        self._output_folder = output_folder
        self._report = report

    @property
    def report(self) -> dict[str, object]:
        """staging済みのcanonical reportを返す。"""
        return self._report

    def publish(self) -> None:
        """staging directoryを一回のrenameで公開する。"""
        try:
            self._staging_folder.replace(self._output_folder)
        except BaseException:
            self.discard()
            raise

    def discard(self) -> None:
        """未公開のstaging directoryを取り除く。"""
        shutil.rmtree(self._staging_folder, ignore_errors=True)
