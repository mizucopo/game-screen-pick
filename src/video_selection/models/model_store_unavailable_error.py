"""model storeの一時的または運用上の利用不能。"""


class ModelStoreUnavailableError(RuntimeError):
    """local/remote store operationが完了できないことを表す。"""
