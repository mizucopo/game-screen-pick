class WhisperModel:
    def __init__(
        self,
        model_size_or_path: str,
        *,
        device: str,
        compute_type: str,
        local_files_only: bool,
    ) -> None: ...
