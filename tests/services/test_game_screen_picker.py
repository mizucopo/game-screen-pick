"""GameScreenPickerの単体テスト.

scene mix ベースへ再設計されたピッカーについて、
解析済み入力からの選定とフォルダ起点の統計集計を公開API経由で確認する。
"""

import tempfile
from pathlib import Path

import torch

from src.models.analyzed_image import AnalyzedImage
from src.models.scene_mix import SceneMix
from src.models.selection_config import SelectionConfig
from src.services.game_screen_picker import GameScreenPicker
from tests.conftest import create_analyzed_image
from tests.fake_analyzer import FakeAnalyzer


def _make_feature(index: int) -> torch.Tensor:
    """類似度判定用の one-hot 特徴を作る."""
    feature = torch.zeros(576)
    feature[index] = 1.0
    return feature


def _make_near_duplicate(base: torch.Tensor, index: int) -> torch.Tensor:
    """base に非常に近い特徴を作る."""
    feature = base.clone()
    feature[index] = 0.01
    return feature


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


def _make_homogeneous_scene_mix_images() -> list[AnalyzedImage]:
    """多様性不足を再現する解析済み画像群を作成する."""
    gameplay_base = _make_feature(0)
    event_base = _make_feature(100)
    return [
        create_analyzed_image(
            path="/tmp/gameplay_0.jpg",
            clip_features=torch.tensor([1.0, 0.0, 0.0]).numpy(),
            combined_features=gameplay_base.numpy(),
            normalized_metrics_dict={"action_intensity": 0.6, "ui_density": 0.5},
        ),
        create_analyzed_image(
            path="/tmp/gameplay_1.jpg",
            clip_features=torch.tensor([1.0, 0.0, 0.0]).numpy(),
            combined_features=_make_near_duplicate(gameplay_base, 1).numpy(),
            normalized_metrics_dict={"action_intensity": 0.6, "ui_density": 0.5},
        ),
        create_analyzed_image(
            path="/tmp/event_0.jpg",
            clip_features=torch.tensor([0.0, 1.0, 0.0]).numpy(),
            combined_features=event_base.numpy(),
            normalized_metrics_dict={"action_intensity": 0.4, "ui_density": 0.4},
            layout_dict={"dialogue_overlay_score": 0.5},
        ),
        create_analyzed_image(
            path="/tmp/event_1.jpg",
            clip_features=torch.tensor([0.0, 1.0, 0.0]).numpy(),
            combined_features=_make_near_duplicate(event_base, 101).numpy(),
            normalized_metrics_dict={"action_intensity": 0.4, "ui_density": 0.4},
            layout_dict={"dialogue_overlay_score": 0.5},
        ),
    ]


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


def test_select_from_analyzed_allows_short_result() -> None:
    """多様性不足なら要求枚数未満でも正常に返すこと."""
    # Arrange
    analyzed_images = _make_homogeneous_scene_mix_images()
    analyzer = FakeAnalyzer(analyzed_images)
    picker = GameScreenPicker(
        analyzer=analyzer,
        config=SelectionConfig(scene_mix=SceneMix(gameplay=0.5, event=0.5, other=0.0)),
    )

    # Act
    selected, rejected, stats = picker.select_from_analyzed(analyzed_images, num=4)

    # Assert
    assert len(selected) == 2
    assert len(rejected) == 2
    assert stats.selected_count == 2
    assert stats.scene_mix_target == {"gameplay": 2, "event": 2, "other": 0}
    assert stats.scene_mix_actual == {"gameplay": 1, "event": 1, "other": 0}


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
