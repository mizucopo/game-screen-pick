"""画面種別判定器.

CLIPのゼロショット類似度とレイアウトヒューリスティクスを組み合わせ、
画像を gameplay / event / other の3系統へ寄せて評価する。
"""

from typing import Any

import numpy as np

from ..constants.scene_label import SceneLabel
from ..models.adaptive_scores import AdaptiveScores
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
    EVENT_PROMOTION_MARGIN = 0.01
    EVENT_PROMOTION_MIN_SCORE = 0.40
    EVENT_PROMOTION_MIN_SIGNAL = 0.50
    EVENT_PROMOTION_MIN_DISTINCTIVENESS = 0.60
    EVENT_PROMOTION_MAX_ACTION = 0.45
    EVENT_PROMOTION_MAX_UI = 0.45
    EVENT_PROMOTION_MAX_SUPPORT_UI = 0.55
    EVENT_PROMOTION_MIN_OTHER_GAP = 0.01
    TRANSITION_RISK_EVENT_PENALTY = 0.06
    TRANSITION_SUPPRESS_MIN_RISK = 0.72
    TRANSITION_SUPPRESS_MAX_VISIBILITY = 0.42
    TRANSITION_SUPPRESS_MAX_INFORMATION = 0.35
    TRANSITION_SUPPRESS_MAX_MARGIN = 0.02
    TRANSITION_SUPPRESS_MAX_DIALOGUE_OVERLAY = 0.15
    TRANSITION_SUPPRESS_MAX_DRAMATIC = 0.45
    TRANSITION_SUPPRESS_MAX_OTHER_GAP = 0.01
    TRANSITION_RISK_LUMINANCE_RANGE_SCALE = 48.0
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
            "in-engine cinematic scene from a video game",
            "boss introduction cutscene from a video game",
            "scripted dramatic event in a video game",
            "stage intro cinematic from a video game",
            "character close-up event scene in a video game",
            "story scene with character portraits in a video game",
            "villain introduction scene from a video game",
            "special scripted sequence in a video game",
            "conversation scene with subtitles in a video game",
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
        adaptive_scores: AdaptiveScores,
    ) -> SceneAssessment:
        """単一画像の画面種別を評価する.

        まずCLIP特徴と各プロンプト群の内積から
        gameplay / event / other の基礎スコアを作り、
        その後で入力全体に対する頻出度、可視性、情報量、UI量、会話オーバーレイ、
        メニュー配置、タイトル画面らしさ、ゲームオーバー画面らしさを
        加減算して補正する。
        最終的な `scene_label` は最大スコアを基本にしつつ、
        gameplay と other が僅差なら other へ倒し、
        低可視性の遷移フレームらしい raw event は event から外す。
        `scene_confidence` は採用ラベルと次点候補の差分で表す。

        Args:
            analyzed_image: scene判定前の中立解析結果。
            adaptive_scores: 入力全体に対する相対情報量・差分量・可視性スコア。

        Returns:
            3系統のスコア、最終ラベル、信頼度を持つ `SceneAssessment` 。
        """
        clip_features = analyzed_image.clip_features
        raw = analyzed_image.raw_metrics
        heuristics = analyzed_image.layout_heuristics
        norm = analyzed_image.normalized_metrics
        distinctiveness_score = adaptive_scores.distinctiveness_score
        information_score = adaptive_scores.information_score
        visibility_score = adaptive_scores.visibility_score
        gameplay_typicality = self._clamp(1.0 - distinctiveness_score)
        support_ui_score = self._clamp(
            0.65 * norm.ui_density + 0.35 * (1.0 - norm.action_intensity)
        )
        rare_cinematic_score = self._clamp(
            0.55 * distinctiveness_score
            + 0.30 * norm.dramatic_score
            + 0.15 * (1.0 - support_ui_score)
            - 0.20
        )
        transition_risk_score = self._calculate_transition_risk(raw, adaptive_scores)

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
            - 0.10 * rare_cinematic_score
            - 0.08 * heuristics.menu_layout_score
            - 0.06 * heuristics.title_layout_score
            - 0.06 * heuristics.game_over_layout_score
            - 0.05 * heuristics.dialogue_overlay_score
        )
        event_score = self._clamp(
            1.18 * event_base
            + 0.14 * heuristics.dialogue_overlay_score
            + 0.28 * rare_cinematic_score
            + 0.10 * distinctiveness_score
            + 0.06 * norm.color_richness
            - 0.02 * heuristics.menu_layout_score
            - 0.04 * support_ui_score
            - self.TRANSITION_RISK_EVENT_PENALTY * transition_risk_score
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
        argmax_scene_label, argmax_score = ordered_scores[0]
        second_score = ordered_scores[1][1] if len(ordered_scores) > 1 else 0.0
        argmax_margin = argmax_score - second_score
        scene_label = argmax_scene_label
        transition_suppressed_event = False
        if (
            argmax_scene_label == SceneLabel.EVENT
            and transition_risk_score >= self.TRANSITION_SUPPRESS_MIN_RISK
            and visibility_score <= self.TRANSITION_SUPPRESS_MAX_VISIBILITY
            and information_score <= self.TRANSITION_SUPPRESS_MAX_INFORMATION
            and argmax_margin <= self.TRANSITION_SUPPRESS_MAX_MARGIN
            and heuristics.dialogue_overlay_score
            <= self.TRANSITION_SUPPRESS_MAX_DIALOGUE_OVERLAY
            and norm.dramatic_score <= self.TRANSITION_SUPPRESS_MAX_DRAMATIC
            and other_score >= event_score - self.TRANSITION_SUPPRESS_MAX_OTHER_GAP
        ):
            scene_label = (
                SceneLabel.GAMEPLAY
                if gameplay_score >= other_score
                else SceneLabel.OTHER
            )
            transition_suppressed_event = True
        elif (
            argmax_scene_label == SceneLabel.GAMEPLAY
            and argmax_margin <= self.AMBIGUITY_MARGIN
            and other_score >= argmax_score - self.AMBIGUITY_MARGIN
        ):
            scene_label = SceneLabel.OTHER
        elif (
            argmax_scene_label == SceneLabel.GAMEPLAY
            and event_score >= gameplay_score - self.EVENT_PROMOTION_MARGIN
            and event_score >= self.EVENT_PROMOTION_MIN_SCORE
            and rare_cinematic_score >= self.EVENT_PROMOTION_MIN_SIGNAL
            and distinctiveness_score >= self.EVENT_PROMOTION_MIN_DISTINCTIVENESS
            and norm.action_intensity <= self.EVENT_PROMOTION_MAX_ACTION
            and norm.ui_density <= self.EVENT_PROMOTION_MAX_UI
            and support_ui_score <= self.EVENT_PROMOTION_MAX_SUPPORT_UI
            and event_score >= other_score + self.EVENT_PROMOTION_MIN_OTHER_GAP
        ):
            scene_label = SceneLabel.EVENT

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
            transition_risk_score=transition_risk_score,
            transition_suppressed_event=transition_suppressed_event,
        )

    @classmethod
    def _calculate_transition_risk(
        cls,
        raw_metrics: Any,
        adaptive_scores: AdaptiveScores,
    ) -> float:
        """暗転・明転・露出過多/不足の遷移フレームらしさを返す."""
        exposure_extreme = max(
            raw_metrics.near_black_ratio, raw_metrics.near_white_ratio
        )
        compressed_range = 1.0 - min(
            1.0,
            raw_metrics.luminance_range / cls.TRANSITION_RISK_LUMINANCE_RANGE_SCALE,
        )
        return cls._clamp(
            0.35 * (1.0 - adaptive_scores.visibility_score)
            + 0.25 * (1.0 - adaptive_scores.information_score)
            + 0.15 * exposure_extreme
            + 0.15 * raw_metrics.dominant_tone_ratio
            + 0.10 * compressed_range
        )
