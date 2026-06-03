"""結果フォーマットユーティリティクラス."""

import logging

from ..models.output_record import OutputRecord

logger = logging.getLogger(__name__)


class ResultFormatter:
    """選択結果と統計情報をコンソールに出力する."""

    @staticmethod
    def display_results(output_record: OutputRecord) -> None:
        """選択結果を表示する."""
        logger.info("--- 選択された画像一覧 ---")
        for i, result in enumerate(output_record.selected):
            logger.info(
                f"[{i + 1}] {result.filename} "
                f"(カテゴリ: {result.scene_label}, "
                f"band: {result.score_band}, "
                f"play: {result.play_score:.3f}, "
                f"event: {result.event_score:.3f}, "
                f"density: {result.density_score:.3f})"
            )

        logger.info("--- 統計情報 ---")
        logger.info(f"総ファイル数: {output_record.total_files}")
        logger.info(f"解析成功: {output_record.analyzed_ok}")
        logger.info(f"解析失敗: {output_record.analyzed_fail}")
        logger.info(
            f"コンテンツフィルターで除外: {output_record.rejected_by_content_filter}"
        )
        logger.info(
            f"コンテンツフィルター内訳: {output_record.content_filter_breakdown}"
        )
        logger.info(f"類似度で除外: {output_record.rejected_by_similarity}")
        logger.info(f"選択数: {output_record.selected_count}")
        logger.info(f"プロファイル: {output_record.resolved_profile}")
        logger.info(f"画面分布(候補): {output_record.scene_distribution}")
        logger.info(f"画面分布(目標): {output_record.scene_mix_target}")
        logger.info(f"画面分布(実績): {output_record.scene_mix_actual}")
