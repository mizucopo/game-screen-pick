"""VideoRunConfigLoaderの単体テスト."""

from pathlib import Path

from src.utils.video_run_config_loader import VideoRunConfigLoader


def test_load_returns_only_explicit_run_values(tmp_path: Path) -> None:
    """明示したrun設定を公開型へ正規化して返すこと."""
    config_path = tmp_path / "picker.toml"
    config_path.write_text(
        """[run]
num = 12
game_context = "探索を含む"
ollama_timeout = 120
allow_cpu = true
sample_interval_seconds = 2.5
""",
        encoding="utf-8",
    )

    assert VideoRunConfigLoader.load(str(config_path)) == {
        "num": 12,
        "game_context": "探索を含む",
        "ollama_timeout": 120.0,
        "allow_cpu": True,
        "sample_interval_seconds": 2.5,
    }
