"""プロファイル解決の重み定数。"""


class ProfileWeights:
    """ProfileResolver で使う重みを集約する."""

    # active スコアの重み
    ACTION: float = 0.45
    EDGE: float = 0.35
    UI_INVERSE: float = 0.20

    # static スコアの重み
    UI: float = 0.50
    ACTION_INVERSE: float = 0.25
    EDGE_INVERSE: float = 0.25
