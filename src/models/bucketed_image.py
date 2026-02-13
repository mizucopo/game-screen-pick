"""BucketedImageクラス."""

from dataclasses import dataclass

from .activity_bucket import ActivityBucket
from .image_metrics import ImageMetrics


@dataclass
class BucketedImage:
    """バケット付けされた画像."""

    image: ImageMetrics
    bucket: ActivityBucket
    activity_score: float
