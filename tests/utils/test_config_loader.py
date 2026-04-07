"""ConfigLoader のテスト."""

from pathlib import Path

from src.utils.config_loader import ConfigLoader


def test_load_returns_empty_dict_when_path_is_none() -> None:
    """pathがNoneの場合、空辞書が返されること."""
    result = ConfigLoader.load(None)
    assert result == {}


def test_load_reads_profile_from_selection_section(tmp_path: Path) -> None:
    """selection.profile キーが正しく読み込まれること."""
    config_file = tmp_path / "test.toml"
    config_file.write_text('[selection]\nprofile = "active"\n')
    result = ConfigLoader.load(str(config_file))
    assert result["profile"] == "active"


def test_load_reads_scene_mix(tmp_path: Path) -> None:
    """scene_mix セクションから SceneMix が生成されること."""
    config_file = tmp_path / "test.toml"
    config_file.write_text("[scene_mix]\nplay = 0.8\nevent = 0.2\n")
    result = ConfigLoader.load(str(config_file))
    assert result["scene_mix"].play == 0.8
    assert result["scene_mix"].event == 0.2


def test_load_reads_similarity_threshold(tmp_path: Path) -> None:
    """thresholds.similarity キーが正しく読み込まれること."""
    config_file = tmp_path / "test.toml"
    config_file.write_text("[thresholds]\nsimilarity = 0.75\n")
    result = ConfigLoader.load(str(config_file))
    assert result["similarity_threshold"] == 0.75


def test_load_ignores_empty_sections(tmp_path: Path) -> None:
    """空セクションは結果に含まれないこと."""
    config_file = tmp_path / "test.toml"
    config_file.write_text("[selection]\n[thresholds]\n")
    result = ConfigLoader.load(str(config_file))
    assert result == {}
