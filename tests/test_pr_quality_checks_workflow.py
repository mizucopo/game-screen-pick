import re
from pathlib import Path

_WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "pr-quality-checks.yml"
)
_PROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"


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


def test_ffmpeg_integration_is_a_separate_ubuntu_required_check() -> None:
    """real FFmpeg suiteが通常testと分離したCI jobで実行されること.

    Arrange:
        - PR workflowとtask定義が読み込まれる
    Act:
        - ffmpeg-integration jobとtest-ffmpeg taskが取り出される
    Assert:
        - Ubuntu 24.04上の独立jobから専用taskだけが実行されること
    """
    # Arrange
    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
    project = _PROJECT_PATH.read_text(encoding="utf-8")

    # Act
    ffmpeg_job = re.search(
        r"\n  ffmpeg-integration:\n(?P<job>.*?)(?=\n  [a-z][a-z-]+:\n|\Z)",
        workflow,
        re.DOTALL,
    )

    # Assert
    assert ffmpeg_job is not None
    assert "runs-on: ubuntu-24.04" in ffmpeg_job.group("job")
    assert "uv run task test-ffmpeg" in ffmpeg_job.group("job")
    assert 'test-ffmpeg = "pytest tests_ffmpeg"' in project
