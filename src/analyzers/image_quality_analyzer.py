"""Image quality analyzer using CLIP and computer vision metrics."""

import logging
from typing import Optional, List, Tuple

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from PIL import UnidentifiedImageError

from ..models.image_metrics import ImageMetrics
from ..models.genre_weights import GenreWeights
from .metric_normalizer import MetricNormalizer

logger = logging.getLogger(__name__)

# 型エイリアス（行長制限対応）
_PreprocessResult = Tuple[
    Optional[Image.Image], Optional[dict[str, float]], Optional[np.ndarray]
]


class ImageQualityAnalyzer:
    """画像品質アナライザー."""

    def __init__(self, genre: str = "mixed"):
        """アナライザーを初期化する."""
        self.weights = GenreWeights.get_weights(genre)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        # テキスト埋め込みの事前計算とキャッシュ
        self._target_text = "epic game scenery"
        self._text_embeddings = self._precompute_text_embeddings()

    def _precompute_text_embeddings(self) -> torch.Tensor:
        """テキスト埋め込みを事前計算してキャッシュする."""
        with torch.no_grad():
            inputs = self.processor(
                text=[self._target_text],
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            return self.model.get_text_features(**inputs)  # type: ignore[no-any-return]

    def _extract_diversity_features(self, img: np.ndarray) -> np.ndarray:
        """見た目の特徴を抽出（色と構造）."""
        small = cv2.resize(img, (128, 128))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def analyze(self, path: str) -> Optional[ImageMetrics]:
        """画像を解析して品質スコアを計算する."""
        try:
            # PILで1回だけ読み込み、ファイル記述子のリークを防止
            with Image.open(path) as pil_img:
                # OpenCV形式（BGR）に変換
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                features = self._extract_diversity_features(img)
                raw = self._calculate_raw_metrics(img)
                norm = MetricNormalizer.normalize_all(raw)
                semantic = self._calculate_semantic_score(pil_img)
                total = self._calculate_total_score(raw, norm, semantic)

                return ImageMetrics(path, raw, norm, semantic, total, features)
        except self._get_expected_errors() as e:
            logger.warning(
                f"画像分析をスキップしました: {path}, 理由: {type(e).__name__}: {e}"
            )
            return None
        except self._get_unexpected_errors():
            logger.error(f"予期しないエラーが発生しました: {path}", exc_info=True)
            raise

    def _calculate_raw_metrics(self, img: np.ndarray) -> dict[str, float]:
        """生の画像メトリクスを計算する.

        メトリクス計算用に画像を長辺720pxに縮小して処理することで、
        計算コストを削減する。アスペクト比は保持する。
        """
        # メトリクス計算用に画像を縮小（長辺720px、アスペクト比保持）
        h, w = img.shape[:2]
        max_dim = 720
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray_size = gray.size
        gray_mean = np.mean(gray)
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        return {
            "blur_score": cv2.Laplacian(gray, cv2.CV_64F).var(),
            "brightness": gray_mean,
            "contrast": np.std(gray),
            "edge_density": np.sum(cv2.Canny(gray, 50, 150) > 0) / gray_size,
            "color_richness": np.std(hsv[:, :, 1]),
            "ui_density": (
                np.sum(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0))) / gray_size
            ),
            "action_intensity": np.std(cv2.filter2D(gray, -1, kernel)),
            "visual_balance": max(0, 100 - abs(gray_mean - 128) * 0.5),
            "dramatic_score": (
                np.sum((hsv[:, :, 1] > 180) & (hsv[:, :, 2] > 180)) / gray_size
            )
            * 1000,
        }

    def _calculate_semantic_score(self, pil_img: Image.Image) -> float:
        """CLIPモデルを使用してセマンティックスコアを計算する."""
        with torch.no_grad():
            inputs = self.processor(
                images=pil_img,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            image_features = self.model.get_image_features(**inputs)

            # キャッシュされたテキスト埋め込みを使用
            logits = torch.matmul(image_features, self._text_embeddings.T)
            return float(logits[0][0]) / 100.0

    def _calculate_total_score(
        self, raw: dict[str, float], norm: dict[str, float], semantic: float
    ) -> float:
        """総合スコアを計算する."""
        weighted_sum = sum(
            norm[k] * self.weights.get(k, 0.0) for k in norm if k in self.weights
        )
        penalty = 0.6 if raw["brightness"] < 40 else 0.0
        return max(0.0, (weighted_sum + (semantic * 0.2) - penalty) * 100.0)

    def analyze_batch(
        self,
        paths: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> List[Optional[ImageMetrics]]:
        """複数の画像をバッチ処理で解析する.

        2段パイプライン構成:
        1. I/O+CV前処理ステージを並列（ThreadPoolExecutor）
        2. CLIPステージはバッチで直列

        Args:
            paths: 画像ファイルパスのリスト
            batch_size: CLIP推論のバッチサイズ（デフォルト32）
            show_progress: 進捗表示をするかどうか

        Returns:
            解析結果のリスト（失敗した画像はNone）
        """
        # ステージ1: I/O + CV前処理を並列実行
        preprocessed = self._load_and_preprocess_images(paths)

        # PIL画像とメトリクスを分離
        pil_images: List[Optional[Image.Image]] = []
        raw_metrics_list: List[Optional[dict[str, float]]] = []
        features_list: List[Optional[np.ndarray]] = []

        for pil_img, raw, features in preprocessed:
            pil_images.append(pil_img)
            raw_metrics_list.append(raw)
            features_list.append(features)

        # ステージ2: バッチCLIP推論を実行
        semantic_scores = self._calculate_semantic_scores_batch(
            pil_images, initial_batch_size=batch_size
        )

        # 結果を構築
        results: List[Optional[ImageMetrics]] = []
        for i, (path, pil_img, raw, features) in enumerate(
            zip(paths, pil_images, raw_metrics_list, features_list)
        ):
            if pil_img is None or raw is None or features is None:
                results.append(None)
                continue

            # semantic_scoresがNoneの場合もスキップ
            semantic_score = semantic_scores[i]
            if semantic_score is None:
                results.append(None)
                continue

            if show_progress and i % 50 == 0:
                print(f"解析済み: {i}/{len(paths)}")

            try:
                norm = MetricNormalizer.normalize_all(raw)
                total = self._calculate_total_score(raw, norm, semantic_score)

                results.append(
                    ImageMetrics(path, raw, norm, semantic_score, total, features)
                )
            except self._get_expected_errors() as e:
                logger.warning(
                    f"画像分析をスキップしました: {path}, 理由: {type(e).__name__}: {e}"
                )
                results.append(None)

        return results

    def _load_and_preprocess_images(
        self, paths: List[str], max_workers: Optional[int] = None
    ) -> List[_PreprocessResult]:
        """複数のパスからPIL画像を読み込み、OpenCV前処理まで並列実行する.

        I/O（画像読み込み）とCPU-bound処理（OpenCV前処理）をThreadPoolExecutorで並列化.

        Args:
            paths: 画像ファイルパスのリスト
            max_workers: スレッドプールの最大ワーカー数（Noneで自動設定）

        Returns:
            タプルのリスト: (PIL画像, 生メトリクス, 多様性特徴)
            失敗したパスは (None, None, None)
        """

        def process_single(path: str) -> _PreprocessResult:
            """単一の画像を読み込み、前処理する."""
            try:
                # PILで画像を読み込み
                with Image.open(path) as img_file:
                    # RGBモードに変換（必要な場合）
                    if img_file.mode != "RGB":
                        pil_img: Image.Image = img_file.convert("RGB")
                        rgb_img = pil_img.copy()
                    else:
                        rgb_img = img_file.copy()

                # OpenCV形式（BGR）に変換
                img = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)

                # 多様性特徴を抽出
                features = self._extract_diversity_features(img)

                # 生メトリクスを計算
                raw = self._calculate_raw_metrics(img)

                return rgb_img, raw, features

            except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError):
                return None, None, None

        # ThreadPoolExecutorで並列処理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single, paths))

        return results

    @staticmethod
    def _load_pil_images(paths: List[str]) -> List[Optional[Image.Image]]:
        """複数のパスからPIL画像を読み込む.

        失敗した画像に対してはNoneを返し、後続処理でスキップできるようにする.

        Args:
            paths: 画像ファイルパスのリスト

        Returns:
            PIL画像のリスト（失敗したパスはNone）
        """
        images: List[Optional[Image.Image]] = []
        for path in paths:
            try:
                with Image.open(path) as img_file:
                    # RGBモードに変換（必要な場合）
                    if img_file.mode != "RGB":
                        rgb_img: Image.Image = img_file.convert("RGB")
                        # コピーを作成してwithブロック外でも使用可能にする
                        images.append(rgb_img.copy())
                    else:
                        images.append(img_file.copy())
            except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError):
                images.append(None)
        return images

    def _calculate_semantic_scores_batch(
        self,
        pil_images: List[Optional[Image.Image]],
        initial_batch_size: int = 32,
    ) -> List[Optional[float]]:
        """複数のPIL画像に対してCLIP推論をバッチ実行する.

        OOM対策:
        - initial_batch_sizeから開始（デフォルト32）
        - torch.cuda.OutOfMemoryError発生時にバッチサイズを半分にしてリトライ
        - 最小バッチサイズ1まで試行（32→16→8→4→2→1）
        - それでも失敗した画像はNoneとして返す

        Args:
            pil_images: PIL画像のリスト（失敗した画像はNone）
            initial_batch_size: 初期バッチサイズ

        Returns:
            セマンティックスコアのリスト（失敗した画像はNone）
        """
        # 有効な画像のインデックスと画像を収集
        valid_indices = [i for i, img in enumerate(pil_images) if img is not None]
        # Type narrowing: valid_imagesはNoneを含まないことを保証
        valid_images: List[Image.Image] = [img for img in pil_images if img is not None]

        if not valid_images:
            return [None] * len(pil_images)

        # 結果を格納する配列（初期値はNone）
        results: List[Optional[float]] = [None] * len(pil_images)

        # バッチサイズをinitial_batch_sizeから開始
        current_batch_size = initial_batch_size

        # バッチ処理
        while current_batch_size >= 1:
            try:
                for i in range(0, len(valid_images), current_batch_size):
                    batch: List[Image.Image] = valid_images[i : i + current_batch_size]

                    with torch.no_grad():
                        inputs = self.processor(
                            images=batch,
                            return_tensors="pt",
                            padding=True,
                        ).to(self.device)
                        image_features = self.model.get_image_features(**inputs)

                        # キャッシュされたテキスト埋め込みを使用
                        logits = torch.matmul(image_features, self._text_embeddings.T)
                        batch_scores = (logits[:, 0] / 100.0).cpu().tolist()

                    # 結果を元のインデックスにマッピング
                    for j, score in enumerate(batch_scores):
                        results[valid_indices[i + j]] = score

                # 成功したらループを抜ける
                break

            except torch.cuda.OutOfMemoryError:
                # バッチサイズを半分にしてリトライ
                current_batch_size = current_batch_size // 2
                # まだリトライ余地があれば警告
                if current_batch_size >= 1:
                    logger.warning(
                        f"CUDA OOM発生。バッチサイズを{current_batch_size}"
                        "に縮小してリトライします。"
                    )
                else:
                    logger.error(
                        "バッチサイズ1でもOOMが発生しました。"
                        "一部画像の処理をスキップします。"
                    )
                    # 処理済みの結果は残すが、未処理の画像はNoneのまま
                    break

        return results

    @staticmethod
    def _get_expected_errors() -> tuple[type[Exception], ...]:
        """正常な失敗として扱うエラー型."""
        return (
            FileNotFoundError,
            UnidentifiedImageError,
            OSError,
            cv2.error,
            ValueError,
        )

    @staticmethod
    def _get_unexpected_errors() -> tuple[type[Exception], ...]:
        """異常な失敗として扱うエラー型（実装バグ）."""
        return (
            AttributeError,
            TypeError,
            KeyError,
            IndexError,
            RuntimeError,
            MemoryError,
        )
