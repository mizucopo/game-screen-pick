"""target-only human review worksheetのtest。"""

from pathlib import Path

import pytest

from src.video_selection.acceptance.human_review import (
    complete_review_metadata,
    ensure_review_worksheet,
    evaluate_human_review,
    review_candidate_digest,
)


def test_pending_worksheet_contains_private_candidate_ids_and_stable_enums(
    tmp_path: Path,
) -> None:
    """初回worksheetがcandidate IDとstable enum欄をpendingで生成すること。

    Arrange:
        - selected/rejectedを持つcanonical reportとselection artifactが用意される
    Act:
        - target-only review worksheetが生成・評価される
    Assert:
        - selected/rejected IDとstable reasonがworksheetだけに保持されること
        - 未記入worksheetがpending_human_reviewになること
    """
    # Arrange / Act
    path = tmp_path / "review.json"
    worksheet = ensure_review_worksheet(
        path,
        suite="release",
        suite_fingerprint="a" * 64,
        canonical_report=_report(),
        selection_artifact=_selection_artifact(),
    )
    result = evaluate_human_review(
        worksheet,
        suite="release",
        suite_fingerprint="a" * 64,
        expected_candidate_digest=review_candidate_digest(worksheet),
    )

    # Assert
    selected = worksheet["selected"]
    rejected = worksheet["rejected"]
    assert isinstance(selected, list)
    assert isinstance(rejected, list)
    assert selected[0]["candidate_id"] == "frm_" + "1" * 64
    assert rejected[0]["reason_code"] == "lower_marginal_utility"
    assert result["status"] == "pending_human_review"
    assert path.is_file()


def test_existing_generated_worksheet_resumes_with_same_candidate_binding(
    tmp_path: Path,
) -> None:
    """state確定前に生成済みのworksheetが同じcold evidenceで再利用されること。

    Arrange:
        - pending review fieldを持つworksheetがatomicに生成済みである
    Act:
        - 同じcold reportとselection artifactでworksheet生成が再試行される
    Assert:
        - mutable review fieldを除く同じcandidate bindingとして再利用されること
    """
    # Arrange
    path = tmp_path / "review.json"
    first = ensure_review_worksheet(
        path,
        suite="release",
        suite_fingerprint="a" * 64,
        canonical_report=_report(),
        selection_artifact=_selection_artifact(),
    )

    # Act
    resumed = ensure_review_worksheet(
        path,
        suite="release",
        suite_fingerprint="a" * 64,
        canonical_report=_report(),
        selection_artifact=_selection_artifact(),
    )

    # Assert
    assert resumed == first
    assert review_candidate_digest(resumed) == review_candidate_digest(first)


def test_completed_review_is_aggregated_without_candidate_ids() -> None:
    """記入済みworksheetがquality gateのaggregateだけへ変換されること。

    Arrange:
        - 10件中9件usable、矛盾0、visual/context問題0のworksheetが用意される
    Act:
        - human quality gateが評価される
    Assert:
        - 全gateが合格しaggregateへcandidate IDが含まれないこと
    """
    # Arrange
    selected = [
        {
            "candidate_id": "frm_" + f"{index:x}" * 64,
            "output_relative_path": f"images/{index}.webp",
            "visual_quality": "pass",
            "blog_usable": "yes" if index < 9 else "no",
            "annotation_consistency": "consistent",
            "context_overrode_visual_invalidity": "no",
        }
        for index in range(10)
    ]
    worksheet: dict[str, object] = {
        "schema": "game-screen-pick/human-review-worksheet@1.0.0",
        "suite": "full",
        "suite_fingerprint": "b" * 64,
        "reviewer": "reviewer",
        "completed_at": None,
        "selected": selected,
        "rejected": [
            {
                "candidate_id": "frm_" + "a" * 64,
                "reason_code": "similarity_ceiling",
            }
        ],
        "suite_checks": {"spoiler_monotonicity": "pass"},
    }
    complete_review_metadata(worksheet)

    # Act
    result = evaluate_human_review(
        worksheet,
        suite="full",
        suite_fingerprint="b" * 64,
        expected_candidate_digest=review_candidate_digest(worksheet),
    )

    # Assert
    assert result["status"] == "passed"
    assert result["blog_usable_ratio"] == 0.9
    assert result["stable_rejection_reason_count"] == 1
    assert "candidate_id" not in str(result)


def test_zero_selected_candidates_fail_completed_human_gate() -> None:
    """0候補shortfallがproduction正常でもhuman acceptanceでは不合格になること。

    Arrange:
        - review metadataだけが完了したselected 0件worksheetが用意される
    Act:
        - human quality gateが評価される
    Assert:
        - pendingではなくquality failureになること
    """
    # Arrange
    worksheet: dict[str, object] = {
        "schema": "game-screen-pick/human-review-worksheet@1.0.0",
        "suite": "release",
        "suite_fingerprint": "c" * 64,
        "reviewer": "reviewer",
        "completed_at": "2026-07-17T00:00:00+00:00",
        "selected": [],
        "rejected": [],
        "suite_checks": {"spoiler_monotonicity": "pass"},
    }

    # Act
    result = evaluate_human_review(
        worksheet,
        suite="release",
        suite_fingerprint="c" * 64,
        expected_candidate_digest=review_candidate_digest(worksheet),
    )

    # Assert
    assert result["status"] == "failed"


@pytest.mark.parametrize(
    "completed_at",
    [False, "", "not-a-timestamp", "2026-07-17T00:00:00"],
)
def test_invalid_completion_timestamp_keeps_review_pending(
    completed_at: object,
) -> None:
    """timezone-awareでない完了時刻ではreviewが完了扱いされないこと。

    Arrange:
        - review enumとreviewerは完了し不正なcompleted_atだけを持つ
    Act:
        - human quality gateが評価される
    Assert:
        - audit metadata未完了としてpendingになること
    """
    # Arrange
    worksheet: dict[str, object] = {
        "schema": "game-screen-pick/human-review-worksheet@1.0.0",
        "suite": "release",
        "suite_fingerprint": "d" * 64,
        "reviewer": "reviewer",
        "completed_at": completed_at,
        "selected": [
            {
                "candidate_id": "frm_" + "1" * 64,
                "output_relative_path": "images/0001.webp",
                "visual_quality": "pass",
                "blog_usable": "yes",
                "annotation_consistency": "consistent",
                "context_overrode_visual_invalidity": "no",
            }
        ],
        "rejected": [],
        "suite_checks": {"spoiler_monotonicity": "pass"},
    }

    # Act
    result = evaluate_human_review(
        worksheet,
        suite="release",
        suite_fingerprint="d" * 64,
        expected_candidate_digest=review_candidate_digest(worksheet),
    )

    # Assert
    assert result["status"] == "pending_human_review"


def test_review_rejects_candidate_set_changed_after_generation(tmp_path: Path) -> None:
    """cold evidenceから生成されたcandidate集合の欠落が拒否されること。

    Arrange:
        - selectedとrejectedへ結び付いたworksheetが生成される
        - reviewer入力を装ってrejected candidateが削除される
    Act:
        - cold evidenceのcandidate digestを使ってreviewが評価される
    Assert:
        - candidate集合不一致として受理されないこと
    """
    # Arrange
    worksheet = ensure_review_worksheet(
        tmp_path / "review.json",
        suite="release",
        suite_fingerprint="d" * 64,
        canonical_report=_report(),
        selection_artifact=_selection_artifact(),
    )
    expected_digest = review_candidate_digest(worksheet)
    worksheet["rejected"] = []

    # Act / Assert
    with pytest.raises(ValueError, match="candidate集合"):
        evaluate_human_review(
            worksheet,
            suite="release",
            suite_fingerprint="d" * 64,
            expected_candidate_digest=expected_digest,
        )


def _report() -> dict[str, object]:
    """一つのselected recordを持つcanonical reportを返す。"""
    return {
        "selected": [
            {
                "image_id": "frm_" + "1" * 64,
                "output": {"relative_path": "images/0001_gameplay.webp"},
            }
        ]
    }


def _selection_artifact() -> dict[str, object]:
    """一つのstable rejectionを持つselection artifactを返す。"""
    return {
        "rejected": [
            {
                "candidate_id": "frm_" + "2" * 64,
                "reason_code": "lower_marginal_utility",
            }
        ]
    }
