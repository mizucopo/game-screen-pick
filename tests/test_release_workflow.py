import re
from pathlib import Path

_WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"
)


def test_existing_release_tag_skips_duplicate_release() -> None:
    """既存のversion tagがある場合に重複releaseがスキップされること。

    Arrange:
        - Release workflowが読み込まれる
    Act:
        - tag存在判定とrelease作成stepの条件が検索される
    Assert:
        - 既存tagではtagとGitHub Releaseが作成されないこと
    """
    # Arrange
    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")

    # Act
    tag_check = re.search(
        r"- name: Check if tag exists\n"
        r"\s+id: tag\n"
        r"(?P<body>.*?)\n\s+- name: Create and push git tag",
        workflow,
        re.DOTALL,
    )

    # Assert
    assert tag_check is not None
    assert 'echo "exists=true" >> "$GITHUB_OUTPUT"' in tag_check.group("body")
    assert workflow.count("if: steps.tag.outputs.exists != 'true'") == 2
