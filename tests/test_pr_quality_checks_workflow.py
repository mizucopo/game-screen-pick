import re
from pathlib import Path

_WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "pr-quality-checks.yml"
)


def test_quality_check_job_fails_on_setup_errors() -> None:
    """setupエラーがquality check jobの失敗として扱われること.

    Arrange:
        - PR quality check workflowが読み込まれる
    Act:
        - quality-checks jobのstep前設定が取り出される
    Assert:
        - job全体のcontinue-on-errorが設定されていないこと
    """
    # Arrange
    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")

    # Act
    job_settings = re.search(
        r"jobs:\n  quality-checks:\n(?P<settings>.*?)\n    steps:",
        workflow,
        re.DOTALL,
    )
    assert job_settings is not None

    # Assert
    assert "continue-on-error:" not in job_settings.group("settings")
