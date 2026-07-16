"""Ollama model selectorのcanonical化。"""


def canonicalize_ollama_model_selector(configured_name: str) -> str:
    """model末尾で省略されたtagだけへlatestを補う。"""
    model_segment = configured_name.rsplit("/", maxsplit=1)[-1]
    if ":" in model_segment:
        return configured_name
    return f"{configured_name}:latest"
