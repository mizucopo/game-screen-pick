"""GameScreenPickerの単体テスト.

scene mix ベースへ再設計されたピッカーについて、
解析済み入力からの選定とフォルダ起点の統計集計を公開API経由で確認する。
"""

import tempfile
from pathlib import Path

import numpy as np
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
    """多様性不足なら要求枚数未満でも正常に返すこと.

    Given:
        - 各scene内で候補が互いに非常に似ている画像群がある
        - gameplay/event 50%ずつの設定で選択が行われる
    When:
        - 4件の選択が要求される
    Then:
        - 多様性判定により各scene 1件ずつ計2件のみ選択されること
        - 残り2件が除外としてカウントされること
    """
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


def test_select_from_analyzed_filters_content_before_scene_mix() -> None:
    """content filter通過後の候補だけでscene mixが適用されること.

    Given:
        - blackout、whiteout、単色、フェード遷移などの低情報量フレームを含む画像群がある
        - 高情報量の暗いgameplay/eventフレームもある
        - gameplay/event 50%ずつのscene mix設定がある
    When:
        - select_from_analyzedで選択する
    Then:
        - content filterで低情報量フレームが除外されること
        - 残った候補に対してscene mixが適用されること
    """

    # Arrange
    def feature(
        index: int,
        delta_index: int | None = None,
        delta: float = 0.0,
    ) -> np.ndarray:
        vector = np.zeros(101, dtype=np.float32)
        vector[index] = 1.0
        if delta_index is not None:
            vector[delta_index] = delta
        return vector

    analyzed_images = [
        create_analyzed_image(
            path="/tmp/dark_gameplay.jpg",
            raw_metrics_dict={
                "brightness": 28.0,
                "contrast": 22.0,
                "edge_density": 0.34,
                "action_intensity": 24.0,
                "luminance_entropy": 1.5,
                "luminance_range": 48.0,
                "near_black_ratio": 0.12,
                "dominant_tone_ratio": 0.62,
            },
            clip_features=torch.tensor([1.0, 0.0, 0.0]).numpy(),
            combined_features=np.pad(feature(0), (0, 475)),
            content_features=feature(0),
        ),
        create_analyzed_image(
            path="/tmp/dark_event.jpg",
            raw_metrics_dict={
                "brightness": 26.0,
                "contrast": 18.0,
                "edge_density": 0.28,
                "action_intensity": 20.0,
                "luminance_entropy": 1.2,
                "luminance_range": 40.0,
                "near_black_ratio": 0.18,
                "dominant_tone_ratio": 0.67,
            },
            clip_features=torch.tensor([0.0, 1.0, 0.0]).numpy(),
            combined_features=np.pad(feature(1), (0, 475)),
            content_features=feature(1),
            layout_dict={"dialogue_overlay_score": 0.6},
        ),
        create_analyzed_image(
            path="/tmp/blackout.jpg",
            raw_metrics_dict={
                "brightness": 1.0,
                "contrast": 0.2,
                "edge_density": 0.001,
                "action_intensity": 0.1,
                "luminance_entropy": 0.05,
                "luminance_range": 1.0,
                "near_black_ratio": 0.99,
                "dominant_tone_ratio": 1.0,
            },
            clip_features=torch.tensor([1.0, 0.0, 0.0]).numpy(),
            combined_features=np.pad(feature(2), (0, 475)),
            content_features=feature(2),
        ),
        create_analyzed_image(
            path="/tmp/whiteout.jpg",
            raw_metrics_dict={
                "brightness": 250.0,
                "contrast": 0.3,
                "edge_density": 0.001,
                "action_intensity": 0.1,
                "luminance_entropy": 0.05,
                "luminance_range": 1.0,
                "near_white_ratio": 0.99,
                "dominant_tone_ratio": 1.0,
            },
            clip_features=torch.tensor([0.0, 0.0, 1.0]).numpy(),
            combined_features=np.pad(feature(2, 3, 0.01), (0, 475)),
            content_features=feature(2, 3, 0.01),
        ),
        create_analyzed_image(
            path="/tmp/single_tone.jpg",
            raw_metrics_dict={
                "brightness": 120.0,
                "contrast": 0.2,
                "edge_density": 0.001,
                "action_intensity": 0.1,
                "luminance_entropy": 0.08,
                "luminance_range": 3.0,
                "dominant_tone_ratio": 0.96,
            },
            clip_features=torch.tensor([0.0, 0.0, 1.0]).numpy(),
            combined_features=np.pad(feature(2, 4, 0.02), (0, 475)),
            content_features=feature(2, 4, 0.02),
        ),
        create_analyzed_image(
            path="/tmp/fade_transition.jpg",
            raw_metrics_dict={
                "brightness": 12.0,
                "contrast": 0.0,
                "edge_density": 0.0,
                "action_intensity": 0.0,
                "luminance_entropy": 0.0,
                "luminance_range": 0.0,
                "near_black_ratio": 0.85,
                "dominant_tone_ratio": 0.90,
            },
            clip_features=torch.tensor([1.0, 0.0, 0.0]).numpy(),
            combined_features=np.pad(feature(2, 5, 0.005), (0, 475)),
            content_features=feature(2, 5, 0.005),
        ),
    ]
    analyzer = FakeAnalyzer(analyzed_images)
    picker = GameScreenPicker(
        analyzer=analyzer,
        config=SelectionConfig(scene_mix=SceneMix(gameplay=0.5, event=0.5, other=0.0)),
    )

    # Act
    selected, rejected, stats = picker.select_from_analyzed(analyzed_images, num=4)

    # Assert
    assert len(selected) == 2
    assert len(rejected) == 0
    assert {candidate.path for candidate in selected} == {
        "/tmp/dark_gameplay.jpg",
        "/tmp/dark_event.jpg",
    }
    assert stats.rejected_by_content_filter == 4
    assert stats.content_filter_breakdown == {
        "blackout": 1,
        "whiteout": 1,
        "single_tone": 1,
        "fade_transition": 1,
        "temporal_transition": 0,
    }
    assert stats.scene_distribution == {"gameplay": 1, "event": 1, "other": 0}
    assert stats.scene_mix_actual == {"gameplay": 1, "event": 1, "other": 0}


def test_select_from_analyzed_rejects_mid_fade_regression() -> None:
    """従来残っていた fade70 が候補から外れること.

    Given:
        - 通常フレームと70%以上の暗転途中フレームを含む画像群がある
        - gameplay/event 75/25 のscene mix設定がある
    When:
        - select_from_analyzedで選択する
    Then:
        - fade70、fade75、fade82がfade_transitionとして除外されること
        - 通常フレームのみが選択されること
    """

    # Arrange
    def feature(index: int) -> np.ndarray:
        vector = np.zeros(101, dtype=np.float32)
        vector[index] = 1.0
        return vector

    analyzed_images = [
        create_analyzed_image(
            path="/tmp/good1.jpg",
            raw_metrics_dict={
                "contrast": 18.0,
                "edge_density": 0.28,
                "action_intensity": 20.0,
                "luminance_entropy": 1.2,
                "luminance_range": 40.0,
                "near_black_ratio": 0.18,
                "dominant_tone_ratio": 0.67,
            },
            clip_features=torch.tensor([1.0, 0.0, 0.0]).numpy(),
            combined_features=np.pad(feature(0), (0, 475)),
            content_features=feature(0),
        ),
        create_analyzed_image(
            path="/tmp/good2.jpg",
            raw_metrics_dict={
                "contrast": 16.0,
                "edge_density": 0.22,
                "action_intensity": 17.0,
                "luminance_entropy": 1.0,
                "luminance_range": 34.0,
                "near_black_ratio": 0.22,
                "dominant_tone_ratio": 0.70,
            },
            clip_features=torch.tensor([0.0, 1.0, 0.0]).numpy(),
            combined_features=np.pad(feature(1), (0, 475)),
            content_features=feature(1),
            layout_dict={"dialogue_overlay_score": 0.5},
        ),
        create_analyzed_image(
            path="/tmp/fade70.jpg",
            raw_metrics_dict={
                "contrast": 3.5,
                "edge_density": 0.015,
                "action_intensity": 1.2,
                "luminance_entropy": 0.42,
                "luminance_range": 9.0,
                "near_black_ratio": 0.70,
                "dominant_tone_ratio": 0.88,
            },
            clip_features=torch.tensor([1.0, 0.0, 0.0]).numpy(),
            combined_features=np.pad(feature(2), (0, 475)),
            content_features=feature(2),
        ),
        create_analyzed_image(
            path="/tmp/fade75.jpg",
            raw_metrics_dict={
                "contrast": 2.5,
                "edge_density": 0.010,
                "action_intensity": 0.8,
                "luminance_entropy": 0.36,
                "luminance_range": 8.0,
                "near_black_ratio": 0.75,
                "dominant_tone_ratio": 0.89,
            },
            clip_features=torch.tensor([1.0, 0.0, 0.0]).numpy(),
            combined_features=np.pad(feature(3), (0, 475)),
            content_features=feature(3),
        ),
        create_analyzed_image(
            path="/tmp/fade82.jpg",
            raw_metrics_dict={
                "contrast": 2.0,
                "edge_density": 0.008,
                "action_intensity": 0.4,
                "luminance_entropy": 0.32,
                "luminance_range": 6.0,
                "near_black_ratio": 0.82,
                "dominant_tone_ratio": 0.90,
            },
            clip_features=torch.tensor([1.0, 0.0, 0.0]).numpy(),
            combined_features=np.pad(feature(4), (0, 475)),
            content_features=feature(4),
        ),
    ]
    analyzer = FakeAnalyzer(analyzed_images)
    config = SelectionConfig(scene_mix=SceneMix(gameplay=0.75, event=0.25, other=0.0))
    picker = GameScreenPicker(analyzer=analyzer, config=config)

    selected, rejected, stats = picker.select_from_analyzed(analyzed_images, num=3)

    assert [candidate.path for candidate in selected] == [
        "/tmp/good1.jpg",
        "/tmp/good2.jpg",
    ]
    assert rejected == []
    assert stats.rejected_by_content_filter == 3
    assert stats.content_filter_breakdown["fade_transition"] == 3
