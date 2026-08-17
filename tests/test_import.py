from importlib import import_module


def test_python_package_is_importable() -> None:
    """Pythonパッケージがimportされること。

    Arrange: import対象のパッケージ名が用意されること
    Act: 指定されたパッケージがimportされること
    Assert: importされたmodule名が一致すること
    """
    # Arrange
    module_name = "src"

    # Act
    module = import_module(module_name)

    # Assert
    assert module.__name__ == module_name
