from src.video_selection.models.run_failure import RunFailure


def test_run_failure_keeps_original_error_out_of_safe_representation() -> None:
    """元例外がcauseとして保持されても安全な表現へ漏れないこと。

    Arrange:
        - absolute pathとraw model textを含む未知の例外が用意される
    Act:
        - stable codeとallowlist値だけでRun Failureが生成される
    Assert:
        - 公開表現には安全なfieldだけが含まれ、元例外はcauseで保持されること
    """
    # Arrange
    cause = RuntimeError("/private/model-store: raw model response")

    # Act
    failure = RunFailure(
        reason_code="internal_error",
        exit_code=1,
        remediation_code="report_internal_error",
        resume_guidance="completed_stages_reusable",
        observed_values=(("attempt_count", 2),),
        cause=cause,
    )

    # Assert
    assert (
        str(failure),
        repr(failure),
        failure.cause,
    ) == (
        "internal_error",
        (
            "RunFailure(reason_code='internal_error', exit_code=1, "
            "remediation_code='report_internal_error', "
            "resume_guidance='completed_stages_reusable', "
            "observed_values=(('attempt_count', 2),))"
        ),
        cause,
    )
