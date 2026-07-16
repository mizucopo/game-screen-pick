import pytest

from src.video_selection.model_runtime.canonicalize_ollama_model_selector import (
    canonicalize_ollama_model_selector,
)


@pytest.mark.parametrize(
    ("configured_name", "expected"),
    (
        ("qwen3-vl", "qwen3-vl:latest"),
        ("qwen3-vl:latest", "qwen3-vl:latest"),
        ("org/qwen3-vl", "org/qwen3-vl:latest"),
        (
            "registry.example:5000/org/qwen3-vl",
            "registry.example:5000/org/qwen3-vl:latest",
        ),
        (
            "registry.example:5000/org/qwen3-vl:v1",
            "registry.example:5000/org/qwen3-vl:v1",
        ),
    ),
)
def test_omitted_ollama_tag_is_canonicalized_to_latest(
    configured_name: str,
    expected: str,
) -> None:
    """省略tagがregistry portと混同されずlatestへ正規化されること。

    Arrange:
        - 省略または明示tagを持つOllama selectorが用意される
    Act:
        - selectorがcanonical化される
    Assert:
        - model末尾の省略tagだけにlatestが補われること
    """
    # Arrange
    selector = configured_name

    # Act
    actual = canonicalize_ollama_model_selector(selector)

    # Assert
    assert actual == expected
