from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.services.stage_eta_estimator import StageEtaEstimator


def test_stage_eta_estimator_converges_for_stable_workload() -> None:
    """安定した5 sampleから最終実績20%以内のStage ETAが返されること。

    Arrange:
        - 同じStageとwork unitに10秒のrecompute sampleが5件記録される
    Act:
        - 30秒経過後に残り5件のETAが要求される
    Assert:
        - 50秒のETAがavailableとして返されること
    """
    # Arrange
    estimator = StageEtaEstimator()
    for _ in range(5):
        estimator.record_sample(
            ProcessingStage.ANNOTATE_CANDIDATE,
            "candidate",
            "recompute",
            10.0,
        )

    # Act
    estimate = estimator.estimate(
        ProcessingStage.ANNOTATE_CANDIDATE,
        "candidate",
        remaining_reuse_count=0,
        remaining_recompute_count=5,
        stage_elapsed_seconds=50.0,
    )

    # Assert
    assert estimate == ("available", 50.0)


def test_stage_eta_estimator_hides_eta_before_thirty_seconds() -> None:
    """5 sampleがあってもStage開始30秒未満ではETAが隠されること。

    Arrange:
        - 同じComparable Work Seriesへsampleが5件記録される
    Act:
        - Stage開始29.9秒時点のETAが要求される
    Assert:
        - estimatingとなりETAが返されないこと
    """
    # Arrange
    estimator = StageEtaEstimator()
    for _ in range(5):
        estimator.record_sample(
            ProcessingStage.ANNOTATE_CANDIDATE,
            "candidate",
            "recompute",
            10.0,
        )

    # Act
    estimate = estimator.estimate(
        ProcessingStage.ANNOTATE_CANDIDATE,
        "candidate",
        remaining_reuse_count=0,
        remaining_recompute_count=1,
        stage_elapsed_seconds=29.9,
    )

    # Assert
    assert estimate == ("estimating", None)


def test_stage_eta_estimator_resets_series_after_large_swing() -> None:
    """予測が50%を超えて変動した系列が5件の新sampleまで隠されること。

    Arrange:
        - 1秒で安定したrecompute sampleが5件記録される
    Act:
        - 10秒の急変sampleと、その後4件の10秒sampleが順次記録される
    Assert:
        - 急変直後はestimatingとなり、新系列5件でETAが再表示されること
    """
    # Arrange
    estimator = StageEtaEstimator()
    for _ in range(5):
        estimator.record_sample(
            ProcessingStage.ANNOTATE_CANDIDATE,
            "candidate",
            "recompute",
            1.0,
        )

    # Act
    before_swing = estimator.estimate(
        ProcessingStage.ANNOTATE_CANDIDATE,
        "candidate",
        remaining_reuse_count=0,
        remaining_recompute_count=5,
        stage_elapsed_seconds=30.0,
    )
    estimator.record_sample(
        ProcessingStage.ANNOTATE_CANDIDATE,
        "candidate",
        "recompute",
        10.0,
    )
    after_swing = estimator.estimate(
        ProcessingStage.ANNOTATE_CANDIDATE,
        "candidate",
        remaining_reuse_count=0,
        remaining_recompute_count=5,
        stage_elapsed_seconds=31.0,
    )
    for _ in range(4):
        estimator.record_sample(
            ProcessingStage.ANNOTATE_CANDIDATE,
            "candidate",
            "recompute",
            10.0,
        )
    after_recovery = estimator.estimate(
        ProcessingStage.ANNOTATE_CANDIDATE,
        "candidate",
        remaining_reuse_count=0,
        remaining_recompute_count=5,
        stage_elapsed_seconds=35.0,
    )

    # Assert
    assert (before_swing, after_swing, after_recovery) == (
        ("available", 5.0),
        ("estimating", None),
        ("available", 50.0),
    )


def test_reuse_and_recompute_samples_are_required_separately() -> None:
    """reuseとrecomputeの残件が各系列のsampleだけで見積もられること。

    Arrange:
        - reuse 5件とrecompute 4件の異なる所要時間sampleが用意される
    Act:
        - 両方の残件を含むETAがrecompute 5件目の前後で要求される
    Assert:
        - 一方のsampleで代用されず、両系列5件後だけ加算ETAが返されること
    """
    # Arrange
    estimator = StageEtaEstimator()
    for _ in range(5):
        estimator.record_sample(
            ProcessingStage.ANNOTATE_CANDIDATE,
            "candidate",
            "reuse",
            1.0,
        )
    for _ in range(4):
        estimator.record_sample(
            ProcessingStage.ANNOTATE_CANDIDATE,
            "candidate",
            "recompute",
            10.0,
        )

    # Act
    before_recompute_series_is_ready = estimator.estimate(
        ProcessingStage.ANNOTATE_CANDIDATE,
        "candidate",
        remaining_reuse_count=5,
        remaining_recompute_count=1,
        stage_elapsed_seconds=30.0,
    )
    estimator.record_sample(
        ProcessingStage.ANNOTATE_CANDIDATE,
        "candidate",
        "recompute",
        10.0,
    )
    after_both_series_are_ready = estimator.estimate(
        ProcessingStage.ANNOTATE_CANDIDATE,
        "candidate",
        remaining_reuse_count=5,
        remaining_recompute_count=1,
        stage_elapsed_seconds=30.0,
    )
    unknown_future_cache_results = estimator.estimate(
        ProcessingStage.ANNOTATE_CANDIDATE,
        "candidate",
        remaining_reuse_count=None,
        remaining_recompute_count=None,
        stage_elapsed_seconds=30.0,
    )

    # Assert
    assert (
        before_recompute_series_is_ready,
        after_both_series_are_ready,
        unknown_future_cache_results,
    ) == (
        ("estimating", None),
        ("available", 15.0),
        ("unavailable", None),
    )
