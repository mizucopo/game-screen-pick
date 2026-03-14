"""GameScreenPickerの単体テスト.

scene mix ベースへ再設計されたピッカーについて、
解析済み入力からの選定とフォルダ起点の統計集計を公開API経由で確認する。
"""

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import torch

from src.analyzers.metric_calculator import MetricCalculator
from src.models.analyzed_image import AnalyzedImage
from src.models.analyzer_config import AnalyzerConfig
from src.models.selection_config import SelectionConfig
from src.services.game_screen_picker import GameScreenPicker
from tests.conftest import create_analyzed_image


class DummyModelManager:
    """固定テキスト埋め込みを返すダミーモデル."""

    def get_text_embeddings(self, texts: Sequence[str]) -> torch.Tensor:
        """promptに応じた固定ベクトルを返す."""
        embeddings = []
        for text in texts:
            lowered = text.lower()
            if "dialogue" in lowered or "cutscene" in lowered or "event" in lowered:
                embeddings.append(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32))
            elif (
                "menu" in lowered
                or "title" in lowered
                or "game over" in lowered
                or "result" in lowered
                or "reward" in lowered
                or "loading" in lowered
            ):
                embeddings.append(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))
            else:
                embeddings.append(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32))
        return torch.stack(embeddings)


class FakeAnalyzer:
    """GameScreenPicker向けの軽量アナライザー.

    事前に組み立てた `AnalyzedImage` を返し、
    実ファイル解析なしでピッカーのドメインロジックだけをテストする。
    """

    def __init__(self, analyzed_images: list[AnalyzedImage]) -> None:
        self._analyzed_images = analyzed_images
        self.metric_calculator = MetricCalculator(AnalyzerConfig())
        self.model_manager = DummyModelManager()

    def analyze_batch(
        self,
        paths: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[AnalyzedImage | None]:
        del batch_size, show_progress
        return cast(list[AnalyzedImage | None], self._analyzed_images[: len(paths)])


def _make_analyzed_images() -> list[AnalyzedImage]:
    """scene mix テスト用の解析済み画像群を作成する.

    Returns:
        gameplay 5件、event 4件、other 1件の `AnalyzedImage` 一覧。
    """
    images = []
    for idx in range(5):
        features = torch.tensor([1.0, 0.0, 0.0]).numpy()
        combined = torch.zeros(576)
        combined[idx] = 1.0
        images.append(
            create_analyzed_image(
                path=f"/tmp/gameplay_{idx}.jpg",
                clip_features=features,
                combined_features=combined.numpy(),
                normalized_metrics_dict={"action_intensity": 0.6, "ui_density": 0.5},
            )
        )
    for idx in range(4):
        features = torch.tensor([0.0, 1.0, 0.0]).numpy()
        combined = torch.zeros(576)
        combined[100 + idx] = 1.0
        images.append(
            create_analyzed_image(
                path=f"/tmp/event_{idx}.jpg",
                clip_features=features,
                combined_features=combined.numpy(),
                normalized_metrics_dict={"action_intensity": 0.4, "ui_density": 0.4},
                layout_dict={"dialogue_overlay_score": 0.5},
            )
        )
    features = torch.tensor([0.0, 0.0, 1.0]).numpy()
    combined = torch.zeros(576)
    combined[200] = 1.0
    images.append(
        create_analyzed_image(
            path="/tmp/other_0.jpg",
            clip_features=features,
            combined_features=combined.numpy(),
            normalized_metrics_dict={"action_intensity": 0.2, "ui_density": 0.7},
            layout_dict={"menu_layout_score": 0.6, "title_layout_score": 0.4},
        )
    )
    return images


def test_select_from_analyzed_returns_scene_mix() -> None:
    """解析済み画像から50/40/10のscene mixで選ばれること.

    Given:
        - gameplay / event / other が既定比率ぶん揃った解析済み画像群がある
    When:
        - `select_from_analyzed` で10件を選択する
    Then:
        - 既定の 50 / 40 / 10 に一致する目標値と実績が返ること
    """
    # Arrange
    analyzed_images = _make_analyzed_images()
    analyzer = FakeAnalyzer(analyzed_images)
    picker = GameScreenPicker(analyzer=analyzer, config=SelectionConfig())

    # Act
    selected, rejected, stats = picker.select_from_analyzed(analyzed_images, num=10)

    # Assert
    assert len(selected) == 10
    assert len(rejected) == 0
    assert stats.scene_mix_target == {"gameplay": 5, "event": 4, "other": 1}
    assert stats.scene_mix_actual == {"gameplay": 5, "event": 4, "other": 1}
    assert stats.selected_count == 10


def test_select_from_folder_processes_images_and_handles_failures() -> None:
    """フォルダ選択時に統計情報が正しく計算されること.

    Given:
        - 入力フォルダには5件の画像パスがある
        - Analyzer はそのうち先頭3件分だけ解析結果を返す
    When:
        - `select` でフォルダ起点の選定を行う
    Then:
        - 選択結果は3件となり、未解析2件が失敗数として集計されること
    """
    # Arrange
    analyzed_images = _make_analyzed_images()[:3]
    analyzer = FakeAnalyzer(analyzed_images)

    with tempfile.TemporaryDirectory() as temp_dir:
        for idx in range(5):
            Path(temp_dir, f"image{idx}.jpg").touch()

        picker = GameScreenPicker(analyzer=analyzer, config=SelectionConfig())

        # Act
        selected, _rejected, stats = picker.select(
            folder=temp_dir,
            num=5,
            recursive=False,
            show_progress=False,
        )

        # Assert
        assert len(selected) == 3
        assert stats.total_files == 5
        assert stats.analyzed_ok == 3
        assert stats.analyzed_fail == 2
