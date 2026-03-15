"""画面種別判定器.

CLIPのゼロショット類似度とレイアウトヒューリスティクスを組み合わせ、
画像を gameplay / event / other の3系統へ寄せて評価する。
"""

from typing import Any

import numpy as np

from ..constants.scene_label import SceneLabel
from ..models.analyzed_image import AnalyzedImage
from ..models.scene_assessment import SceneAssessment
from ..protocols.text_embedding_provider import TextEmbeddingProvider


class SceneScorer:
    """画像を gameplay / event / other に分類する.

    タイトル固有の知識は使わず、複数プロンプトのCLIP類似度と
    UI量・会話オーバーレイ・メニューらしさなどの
    汎用ヒューリスティクスだけで画面種別を推定する。
    """

    AMBIGUITY_MARGIN = 0.05
    PROMPT_GROUPS: dict[str, tuple[str, ...]] = {
        "gameplay": (
            "video game gameplay screenshot with heads-up display",
            "player controlling a character during combat gameplay",
            "in-game exploration screen with interface elements",
            "gameplay action scene with status bars and HUD",
        ),
        "event": (
            "video game dialogue event scene",
            "story cutscene from a video game",
            "boss introduction cutscene from a video game",
            "scripted dramatic event in a video game",
            "stage intro cinematic from a video game",
            "character close-up event scene in a video game",
        ),
        "other": (
            "video game main menu screen",
            "game title screen",
            "game over screen",
            "full-screen world map from a video game",
            "inventory or equipment menu from a video game",
            "skill tree or upgrade menu from a video game",
            "shop or merchant screen from a video game",
            "pause or settings menu from a video game",
            "video game result or reward screen",
            "loading screen from a video game",
        ),
    }

    def __init__(self, model_manager: TextEmbeddingProvider):
        """SceneScorerを初期化する.

        Args:
            model_manager: テキスト埋め込みを取得できるモデル管理器。
                起動時にプロンプト埋め込みをまとめてキャッシュする。
        """
        self._prompt_embeddings = self._build_prompt_embeddings(model_manager)

    @classmethod
    def _build_prompt_embeddings(
        cls,
        model_manager: TextEmbeddingProvider,
    ) -> dict[str, np.ndarray[Any, Any]]:
        """各プロンプト群のテキスト埋め込みをキャッシュする.

        scene labelごとに複数のゼロショットプロンプトを用意し、
        初期化時にまとめてベクトル化して再利用する。

        Args:
            model_manager: テキスト埋め込みを返すオブジェクト。

        Returns:
            scene labelごとの埋め込み行列。
        """
        result: dict[str, np.ndarray[Any, Any]] = {}
        for label, prompts in cls.PROMPT_GROUPS.items():
            embeddings = model_manager.get_text_embeddings(prompts).cpu().numpy()
            result[label] = embeddings
        return result

    @staticmethod
    def _mean_top_two(scores: np.ndarray[Any, Any]) -> float:
        """上位2件の平均値を返す.

        単一プロンプトの偶然の当たりに引っ張られすぎないよう、
        各グループで最も強い2件の平均を代表値として使う。

        Args:
            scores: 1次元の類似度配列。

        Returns:
            上位2件の平均値。要素数が1以下の場合は利用可能な値だけで返す。
        """
        if scores.size == 0:
            return 0.0
        if scores.size == 1:
            return float(scores[0])
        top_two = np.partition(scores, -2)[-2:]
        return float(np.mean(top_two))

    @staticmethod
    def _clamp(value: float) -> float:
        """0..1にクリップする.

        Args:
            value: 補正後スコア。

        Returns:
            0.0以上1.0以下へ丸めた値。
        """
        return max(0.0, min(1.0, value))

    def assess(
        self,
        analyzed_image: AnalyzedImage,
        distinctiveness_score: float = 0.5,
    ) -> SceneAssessment:
        """単一画像の画面種別を評価する.

        まずCLIP特徴と各プロンプト群の内積から
        gameplay / event / other の基礎スコアを作り、
        その後で入力全体に対する頻出度、UI量、会話オーバーレイ、
        メニュー配置、タイトル画面らしさ、ゲームオーバー画面らしさを
        加減算して補正する。
        最終的な `scene_label` は最大スコアを基本にしつつ、
        gameplay と other が僅差なら other へ倒す。
        `scene_confidence` は採用ラベルと次点候補の差分で表す。

        Args:
            analyzed_image: scene判定前の中立解析結果。
            distinctiveness_score: 入力全体に対する相対差分量スコア。
                0に近いほど頻出、1に近いほど珍しい画像を表す。

        Returns:
            3系統のスコア、最終ラベル、信頼度を持つ `SceneAssessment` 。
        """
        clip_features = analyzed_image.clip_features
        heuristics = analyzed_image.layout_heuristics
        norm = analyzed_image.normalized_metrics
        gameplay_typicality = self._clamp(1.0 - distinctiveness_score)
        support_ui_score = self._clamp(
            0.65 * norm.ui_density + 0.35 * (1.0 - norm.action_intensity)
        )

        gameplay_base = self._mean_top_two(
            self._prompt_embeddings["gameplay"] @ clip_features
        )
        event_base = self._mean_top_two(
            self._prompt_embeddings["event"] @ clip_features
        )
        other_base = self._mean_top_two(
            self._prompt_embeddings["other"] @ clip_features
        )

        gameplay_score = self._clamp(
            gameplay_base
            + 0.18 * gameplay_typicality
            + 0.10 * norm.ui_density
            + 0.08 * norm.action_intensity
            - 0.10 * support_ui_score
            - 0.08 * heuristics.menu_layout_score
            - 0.06 * heuristics.title_layout_score
            - 0.06 * heuristics.game_over_layout_score
            - 0.05 * heuristics.dialogue_overlay_score
        )
        event_score = self._clamp(
            event_base
            + 0.14 * heuristics.dialogue_overlay_score
            + 0.10 * distinctiveness_score
            + 0.08 * norm.dramatic_score
            + 0.04 * norm.color_richness
            - 0.06 * support_ui_score
            - 0.04 * heuristics.menu_layout_score
        )
        other_score = self._clamp(
            other_base
            + 0.18 * support_ui_score
            + 0.14 * heuristics.menu_layout_score
            + 0.12 * heuristics.title_layout_score
            + 0.12 * heuristics.game_over_layout_score
            + 0.06 * distinctiveness_score
            - 0.06 * gameplay_typicality
        )

        label_scores = {
            SceneLabel.GAMEPLAY: gameplay_score,
            SceneLabel.EVENT: event_score,
            SceneLabel.OTHER: other_score,
        }
        ordered_scores = sorted(
            label_scores.items(), key=lambda item: item[1], reverse=True
        )
        scene_label, top_score = ordered_scores[0]
        second_score = ordered_scores[1][1] if len(ordered_scores) > 1 else 0.0
        if (
            top_score - second_score <= self.AMBIGUITY_MARGIN
            and other_score >= top_score - self.AMBIGUITY_MARGIN
        ):
            scene_label = SceneLabel.OTHER

        chosen_score = label_scores[scene_label]
        unchosen_scores = [
            score for label, score in label_scores.items() if label != scene_label
        ]
        scene_confidence = self._clamp(chosen_score - max(unchosen_scores, default=0.0))

        return SceneAssessment(
            gameplay_score=gameplay_score,
            event_score=event_score,
            other_score=other_score,
            scene_label=scene_label,
            scene_confidence=scene_confidence,
        )
