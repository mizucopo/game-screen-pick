"""ConfigLoader の単体テスト."""

from pathlib import Path

from src.utils.config_loader import ConfigLoader


def test_load_returns_empty_dict_when_path_is_none() -> None:
    """pathがNoneの場合、空辞書が返されること.

    Arrange:
        - pathにNoneを指定する
    Act:
        - ConfigLoader.loadを呼び出す
    Assert:
        - 空辞書が返されること
    """
    # Arrange
    input_path = None

    # Act
    result = ConfigLoader.load(input_path)

    # Assert
    assert result == {}


def test_load_reads_profile_from_selection_section(tmp_path: Path) -> None:
    """selection.profile キーが正しく読み込まれること.

    Arrange:
        - TOMLファイルに selection.profile = "active" を書き込む
    Act:
        - ConfigLoader.loadを呼び出す
    Assert:
        - result["profile"]が "active" であること
    """
    # Arrange
    config_file = tmp_path / "test.toml"
    config_file.write_text('[selection]\nprofile = "active"\n')

    # Act
    result = ConfigLoader.load(str(config_file))

    # Assert
    assert result["profile"] == "active"


def test_load_reads_scene_mix(tmp_path: Path) -> None:
    """scene_mix セクションから SceneMix が生成されること.

    Arrange:
        - TOMLファイルに scene_mix.play=0.8, event=0.2 を書き込む
    Act:
        - ConfigLoader.loadを呼び出す
    Assert:
        - scene_mix.playが0.8であること
        - scene_mix.eventが0.2であること
    """
    # Arrange
    config_file = tmp_path / "test.toml"
    config_file.write_text("[scene_mix]\nplay = 0.8\nevent = 0.2\n")

    # Act
    result = ConfigLoader.load(str(config_file))

    # Assert
    assert result["scene_mix"].play == 0.8
    assert result["scene_mix"].event == 0.2


def test_load_reads_similarity_threshold(tmp_path: Path) -> None:
    """thresholds.similarity キーが正しく読み込まれること.

    Arrange:
        - TOMLファイルに thresholds.similarity = 0.75 を書き込む
    Act:
        - ConfigLoader.loadを呼び出す
    Assert:
        - result["similarity_threshold"]が0.75であること
    """
    # Arrange
    config_file = tmp_path / "test.toml"
    config_file.write_text("[thresholds]\nsimilarity = 0.75\n")

    # Act
    result = ConfigLoader.load(str(config_file))

    # Assert
    assert result["similarity_threshold"] == 0.75


def test_load_ignores_empty_sections(tmp_path: Path) -> None:
    """空セクションは結果に含まれないこと.

    Arrange:
        - TOMLファイルに空の selection / thresholds セクションを書き込む
    Act:
        - ConfigLoader.loadを呼び出す
    Assert:
        - 結果が空辞書であること
    """
    # Arrange
    config_file = tmp_path / "test.toml"
    config_file.write_text("[selection]\n[thresholds]\n")

    # Act
    result = ConfigLoader.load(str(config_file))

    # Assert
    assert result == {}
