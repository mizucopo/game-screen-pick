"""ゲーム画面ピッカーの統合オーケストレーション."""

import re
from pathlib import Path

from ..models.analyzed_image import AnalyzedImage
from ..models.picker_statistics import PickerStatistics
from ..models.scored_candidate import ScoredCandidate
from ..models.selection_config import SelectionConfig
from ..protocols.analyzer_like import AnalyzerLike
from .analyzed_image_selector import AnalyzedImageSelector


class GameScreenPicker:
    """画像解析と選定を統合する.

    フォルダ走査、Analyzer実行、解析済み画像選定moduleへの委譲を
    ひとつの入り口として提供する。
    """

    SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
        {".jpg", ".jpeg", ".png", ".bmp"},
    )

    def __init__(
        self,
        analyzer: AnalyzerLike,
        config: SelectionConfig | None = None,
    ):
        """ピッカーを初期化する.

        Args:
            analyzer: 中立解析結果を返すAnalyzer実装。
            config: scene mix、類似度しきい値、profile指定を含む選択設定。
        """
        self.analyzer = analyzer
        self.config = config or SelectionConfig()
        self._analyzed_image_selector = AnalyzedImageSelector(
            config=self.config,
            metric_calculator=self.analyzer.metric_calculator,
        )

    @staticmethod
    def load_image_files(
        folder: str,
        recursive: bool,
    ) -> list[Path]:
        """フォルダから画像ファイルを取得する.

        Args:
            folder: 入力フォルダのパス。
            recursive: サブフォルダも含めて探索するかどうか。

        Returns:
            対応拡張子を持つ画像パスの一覧。
        """
        path_obj = Path(folder)
        files = [
            path
            for path in (path_obj.rglob("*") if recursive else path_obj.glob("*"))
            if path.suffix.lower() in GameScreenPicker.SUPPORTED_EXTENSIONS
        ]
        files = sorted(
            files,
            key=lambda path: [
                int(chunk) if chunk.isdigit() else chunk.lower()
                for chunk in re.split(r"(\d+)", path.relative_to(path_obj).as_posix())
            ],
        )
        return files

    def _analyze_images(
        self,
        files: list[Path],
        show_progress: bool = False,
    ) -> list[AnalyzedImage]:
        """画像を解析して中立特徴を取得する.

        AnalyzerのバッチAPIを使って画像群を一括解析し、
        読み込み失敗や解析失敗で `None` になった要素をここで除外する。
        返却値はまだscene判定や選定スコアを持たない中立データである。

        Args:
            files: 解析対象の画像パス一覧。
            show_progress: 解析進捗ログを出すかどうか。

        Returns:
            正常に解析できた `AnalyzedImage` のみを含むリスト。
        """
        paths = [str(file_path) for file_path in files]
        results = self.analyzer.analyze_batch(
            paths,
            batch_size=self.config.batch_size,
            show_progress=show_progress,
        )
        return [result for result in results if result is not None]

    def select(
        self,
        folder: str,
        num: int,
        recursive: bool,
        show_progress: bool = True,
    ) -> tuple[list[ScoredCandidate], list[ScoredCandidate], PickerStatistics]:
        """フォルダから画像を選択する.

        入力フォルダから対象画像を自然順で集め、
        中立解析とcontent filterを通した後にscene mix選定を実行する。
        フォルダ単位のI/Oを伴う高水準APIとして使うことを想定している。

        Args:
            folder: 入力画像フォルダ。
            num: 選択したい画像枚数。
            recursive: サブフォルダも探索対象に含めるかどうか。
            show_progress: 解析進捗ログを出すかどうか。

        Returns:
            1. 選択された候補
            2. 非選択になった候補
            3. 実行統計をまとめた `PickerStatistics`
        """
        files = GameScreenPicker.load_image_files(folder, recursive)
        total_files = len(files)

        analyzed_images = self._analyze_images(files, show_progress)
        analyzed_ok = len(analyzed_images)
        analyzed_fail = total_files - analyzed_ok

        return self.select_from_analyzed(
            analyzed_images=analyzed_images,
            total_files=total_files,
            analyzed_fail=analyzed_fail,
            num=num,
        )

    def select_from_analyzed(
        self,
        analyzed_images: list[AnalyzedImage],
        num: int,
        total_files: int | None = None,
        analyzed_fail: int = 0,
    ) -> tuple[list[ScoredCandidate], list[ScoredCandidate], PickerStatistics]:
        """解析済み画像から候補を選択する.

        このメソッドはファイルI/Oを行わず、ドメインロジックだけを扱う。
        scene判定、profile解決、候補採点、scene mix選定、統計生成は
        解析済み画像選定moduleへ委譲される。

        Args:
            analyzed_images: 解析済みの中立画像データ。
            num: 選択したい画像枚数。
            total_files: 元の入力総数。 `None` の場合は解析済み件数を使う。
            analyzed_fail: 解析失敗件数。フォルダ起点の統計で使う。

        Returns:
            1. 選択された候補
            2. 非選択候補を選定スコア順に並べたリスト
            3. scene mix目標値と実績を含む `PickerStatistics`
        """
        return self._analyzed_image_selector.select(
            analyzed_images=analyzed_images,
            num=num,
            total_files=total_files,
            analyzed_fail=analyzed_fail,
        )
