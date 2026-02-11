"""Analyzers package for image analysis and metric normalization.

This package uses lazy imports to reduce import overhead.
Modules are only loaded when their attributes are accessed.
"""

from typing import Any

__all__ = [
    "MetricNormalizer",
    "ImageQualityAnalyzer",
    "ImageQualityAnalyzerPool",
]


def __getattr__(name: str) -> Any:
    """Lazy import for submodules to reduce startup overhead.

    This function is called when an attribute is not found in the module's
    __dict__, allowing us to defer importing heavy modules until they are
    actually needed.

    Args:
        name: The attribute name being accessed

    Returns:
        The requested module/class

    Raises:
        AttributeError: If the requested name is not in __all__
    """
    if name == "MetricNormalizer":
        from .metric_normalizer import MetricNormalizer

        return MetricNormalizer
    if name == "ImageQualityAnalyzer":
        from .image_quality_analyzer import ImageQualityAnalyzer

        return ImageQualityAnalyzer
    if name == "ImageQualityAnalyzerPool":
        from .analyzer_pool import ImageQualityAnalyzerPool

        return ImageQualityAnalyzerPool

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
