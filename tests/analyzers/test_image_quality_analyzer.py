"""ImageQualityAnalyzerの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1.「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. CLIPモデルのみ戦略的にモック化（700MB、10-30秒のロード時間）
3. OpenCV操作、NumPy計算、MetricNormalizerはモック化しない
4. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
5. 高速実行（約2-5秒） - 重いモデルロードなし
"""

from pathlib import Path


from src.analyzers.image_quality_analyzer import ImageQualityAnalyzer
from src.models.image_metrics import ImageMetrics


def test_analyze_returns_valid_metrics_with_default_weights(
    sample_image_path: str,
) -> None:
    """デフォルトの重みで有効なメトリックが返されること.

    Given:
        - デフォルト設定のアナライザインスタンスがある
        - 有効なテスト画像がある
    When:
        - 画像が分析される
    Then:
        - 有効なImageMetricsが返されること
        - スコアが有効範囲内にあること
    """
    # Arrange
    # キャッシュを使用しないようにデバイスをCPUに指定
    # （MPS環境でキャッシュ未使用時のテストを安定させるため）
    analyzer = ImageQualityAnalyzer(device="cpu")

    # Act
    result = analyzer.analyze(sample_image_path)

    # Assert
    assert result is not None
    assert isinstance(result, ImageMetrics)
    assert result.path == sample_image_path
    assert 0 <= result.total_score <= 100
    assert -1.0 <= result.semantic_score <= 1.0 + 1e-5
    assert len(result.normalized_metrics) > 0


def test_analyze_returns_none_for_invalid_inputs(
    tmp_path: Path,
) -> None:
    """無効な入力に対してNoneが返されること.

    Given:
        - アナライザインスタンスがある
        - 存在しないファイルパスと破損した画像がある
    When:
        - 無効な入力が分析される
    Then:
        - すべてのケースでNoneが返されること
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()
    corrupted_path = tmp_path / "corrupted.jpg"
    corrupted_path.write_text("This is not a valid image file")

    # Act
    result_nonexistent = analyzer.analyze("/path/that/does/not/exist.jpg")
    result_corrupted = analyzer.analyze(str(corrupted_path))

    # Assert
    assert result_nonexistent is None
    assert result_corrupted is None
