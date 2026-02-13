"""活動量バケット付けされた画像データ."""

from dataclasses import dataclass

from .activity_bucket import ActivityBucket
from .image_metrics import ImageMetrics


@dataclass(eq=False)
class BucketedImage:
    """画像とその活動量バケット・スコアを紐付けるデータクラス.

    Attributes:
        image: 画像メトリクス
        bucket: 活動量バケット（LOW/MID/HIGH）
        activity_score: 活動量スコア（0.0-1.0、高いほど活動的）
    """

    image: ImageMetrics
    bucket: ActivityBucket
    activity_score: float
