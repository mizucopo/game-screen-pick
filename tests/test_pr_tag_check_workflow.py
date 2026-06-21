import re
from pathlib import Path

_WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pr-tag-check.yml"
)


def test_existing_version_tag_publishes_failure_check() -> None:
    """既存のtagと同じversionが検出されると失敗checkが公開されること.

    Arrange:
        - PR tag check workflowが読み込まれる
    Act:
        - 既存tag分岐のgithub-scriptが取り出される
    Assert:
        - 既存tag分岐でcheck conclusionがfailureに設定されること
    """
    # Arrange
    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")

    # Act
    duplicate_tag_branch = re.search(
        r"else if \(tagExists\) \{\n(?P<body>.*?)\n\s+\}",
        workflow,
        re.DOTALL,
    )
    assert duplicate_tag_branch is not None
    branch_body = duplicate_tag_branch.group("body")

    # Assert
    assert 'checkConclusion = "failure";' in branch_body
    assert 'checkTitle = "Version tag already exists";' in branch_body


def test_existing_version_tag_fails_workflow_job() -> None:
    """既存のtagと同じversionが検出されるとworkflow jobが失敗されること.

    Arrange:
        - PR tag check workflowが読み込まれる
    Act:
        - 既存tagを失敗扱いにするstepが検索される
    Assert:
        - 既存tagの場合にexit 1が実行されること
    """
    # Arrange
    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")

    # Act
    fail_step = re.search(
        r"- name: Fail if tag exists\n"
        r"\s+if: steps\.tag\.outputs\.exists == 'true'\n"
        r"\s+run: exit 1",
        workflow,
    )

    # Assert
    assert fail_step is not None
