"""model artifactの完全性・capability failure。"""


class ModelArtifactInvalidError(RuntimeError):
    """artifactを実行identityとして利用できないことを表す。"""
