"""Image quality analyzer using CLIP and computer vision metrics."""

import contextlib
import logging
from typing import Optional, List

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from PIL import UnidentifiedImageError

from ..models.analyzer_config import AnalyzerConfig
from ..models.image_metrics import ImageMetrics
from ..models.genre_weights import GenreWeights
from .metric_normalizer import MetricNormalizer

logger = logging.getLogger(__name__)

# 型エイリアス（行長制限対応）
_PreprocessResult = Optional[Image.Image]


def _safe_l2_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """ゼロ割れ安全なL2正規化を行う.

    Args:
        vec: 正規化するベクトル
        eps: ゼロ割れ防止用の微小値

    Returns:
        L2正規化されたベクトル（元のノルムが0の場合はゼロベクトル）
    """
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return np.zeros_like(vec)
    return vec / norm


class ImageQualityAnalyzer:
    """画像品質アナライザー."""

    def __init__(self, genre: str = "mixed", config: AnalyzerConfig | None = None):
        """アナライザーを初期化する.

        Args:
            genre: ジャンル（重み付け用）
            config: アナライザー設定（Noneの場合はデフォルト値を使用）
        """
        self.config = config or AnalyzerConfig()
        self.weights = GenreWeights.get_weights(genre)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        # 推論モードに設定（ドロップアウト等を無効化）
        self.model.eval()
        # GPU最適化: TF32を許可（Ampere GPU以上で有効）
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        # テキスト埋め込みの事前計算とキャッシュ
        self._target_text = "epic game scenery"
        self._text_embeddings = self._precompute_text_embeddings()

    def _precompute_text_embeddings(self) -> torch.Tensor:
        """テキスト埋め込みを事前計算してキャッシュする."""
        with torch.inference_mode():
            inputs = self.processor(
                text=[self._target_text],
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            return self.model.get_text_features(**inputs)  # type: ignore[no-any-return]

    def _extract_hsv_features(self, img: np.ndarray) -> np.ndarray:
        """HSV色空間のヒストグラム特徴を抽出する.

        Returns:
            正規化されたHSVヒストグラム特徴（64次元）
        """
        small = cv2.resize(img, (128, 128))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def _extract_clip_features(self, pil_img: Image.Image) -> np.ndarray:
        """CLIP画像埋め込みを抽出する.

        Args:
            pil_img: PIL画像（RGB形式）

        Returns:
            正規化されたCLIP画像埋め込み（512次元）
        """
        with torch.inference_mode():
            inputs = self.processor(
                images=pil_img,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            image_features = self.model.get_image_features(**inputs)
            # L2正規化して返す
            features = image_features[0].cpu().numpy()
            return _safe_l2_normalize(features)

    def _extract_combined_features(
        self, img: np.ndarray, clip_features: np.ndarray
    ) -> np.ndarray:
        """HSV特徴とCLIP特徴を結合する.

        Args:
            img: OpenCV画像（BGR形式）
            clip_features: CLIP画像埋め込み（512次元、正規化済み）

        Returns:
            結合された特徴ベクトル（576次元）
        """
        hsv_features = self._extract_hsv_features(img)
        # L2正規化（既に正規化されているが、安全のため再正規化）
        hsv_normalized = _safe_l2_normalize(hsv_features)
        # 結合
        return np.concatenate([hsv_normalized, clip_features])

    def analyze(self, path: str) -> Optional[ImageMetrics]:
        """画像を解析して品質スコアを計算する."""
        try:
            # PILで1回だけ読み込み、ファイル記述子のリークを防止
            with Image.open(path) as pil_img:
                # RGBモードに変換（必要な場合）
                if pil_img.mode != "RGB":
                    pil_img_rgb: Image.Image = pil_img.convert("RGB")
                    pil_img_copy = pil_img_rgb.copy()
                else:
                    pil_img_copy = pil_img.copy()

                # OpenCV形式（BGR）に変換
                img = cv2.cvtColor(np.array(pil_img_copy), cv2.COLOR_RGB2BGR)

                # CLIP特徴を抽出
                clip_features = self._extract_clip_features(pil_img_copy)

                # HSV特徴とCLIP特徴を結合
                features = self._extract_combined_features(img, clip_features)

                raw = self._calculate_raw_metrics(img)
                norm = MetricNormalizer.normalize_all(raw)
                semantic = self._calculate_semantic_score_from_features(clip_features)
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

        メトリクス計算用に画像を長辺max_dim pxに縮小して処理することで、
        計算コストを削減する。アスペクト比は保持する。
        """
        # メトリクス計算用に画像を縮小（長辺max_dim px、アスペクト比保持）
        h, w = img.shape[:2]
        if max(h, w) > self.config.max_dim:
            scale = self.config.max_dim / max(h, w)
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
        with torch.inference_mode():
            inputs = self.processor(
                images=pil_img,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            image_features = self.model.get_image_features(**inputs)

            # キャッシュされたテキスト埋め込みを使用
            logits = torch.matmul(image_features, self._text_embeddings.T)
            return float(logits[0][0]) / 100.0

    def _calculate_semantic_score_from_features(
        self, clip_features: np.ndarray
    ) -> float:
        """既に計算済みのCLIP特徴からセマンティックスコアを計算する.

        Args:
            clip_features: 正規化済みのCLIP画像特徴（512次元）

        Returns:
            セマンティックスコア
        """
        # NumPy配列をtorch.Tensorに変換
        with torch.inference_mode():
            image_features = (
                torch.from_numpy(clip_features).unsqueeze(0).to(self.device)
            )
            # キャッシュされたテキスト埋め込みとの類似度を計算
            logits = torch.matmul(image_features, self._text_embeddings.T)
            return float(logits[0][0]) / 100.0

    def _calculate_total_score(
        self, raw: dict[str, float], norm: dict[str, float], semantic: float
    ) -> float:
        """総合スコアを計算する."""
        weighted_sum = sum(
            norm[k] * self.weights.get(k, 0.0) for k in norm if k in self.weights
        )
        penalty = (
            self.config.brightness_penalty_value
            if raw["brightness"] < self.config.brightness_penalty_threshold
            else 0.0
        )
        return max(
            0.0,
            (weighted_sum + (semantic * self.config.semantic_weight) - penalty)
            * self.config.score_multiplier,
        )

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

        メモリ効率化のためチャンク単位でストリーミング処理:
        - チャンク単位で「前処理→CLIP→結果確定→解放」を実行
        - 大規模画像時のスワップ回避で速度安定化

        Args:
            paths: 画像ファイルパスのリスト
            batch_size: CLIP推論のバッチサイズ（デフォルト32）
            show_progress: 進捗表示をするかどうか

        Returns:
            解析結果のリスト（失敗した画像はNone）
        """
        results: List[Optional[ImageMetrics]] = []

        for chunk_start in range(0, len(paths), self.config.chunk_size):
            chunk_end = min(chunk_start + self.config.chunk_size, len(paths))
            chunk_paths = paths[chunk_start:chunk_end]

            # ステージ1: チャンク単位でI/O + 前処理を並列実行
            pil_images = self._load_and_preprocess_images(chunk_paths)

            # ステージ2: チャンク単位でバッチCLIP推論を実行
            clip_features_list = self._calculate_clip_features_batch(
                pil_images, initial_batch_size=batch_size
            )

            # ステージ3: チャンク単位で結果を構築
            for i, (path, pil_img, clip_features) in enumerate(
                zip(chunk_paths, pil_images, clip_features_list)
            ):
                if pil_img is None or clip_features is None:
                    results.append(None)
                    continue

                if show_progress and (chunk_start + i) % 50 == 0:
                    print(f"解析済み: {chunk_start + i}/{len(paths)}")

                try:
                    # OpenCV形式（BGR）に変換
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                    # 生メトリクスを計算
                    raw = self._calculate_raw_metrics(img)

                    # HSV特徴とCLIP特徴を結合
                    features = self._extract_combined_features(img, clip_features)

                    # 正規化メトリクスとセマンティックスコアを計算
                    norm = MetricNormalizer.normalize_all(raw)
                    semantic = self._calculate_semantic_score_from_features(
                        clip_features
                    )
                    total = self._calculate_total_score(raw, norm, semantic)

                    results.append(
                        ImageMetrics(path, raw, norm, semantic, total, features)
                    )
                except self._get_expected_errors() as e:
                    logger.warning(
                        f"画像分析をスキップしました: {path}, "
                        f"理由: {type(e).__name__}: {e}"
                    )
                    results.append(None)

            # チャンク処理完了後にメモリを解放
            del pil_images, clip_features_list

        return results

    def _load_and_preprocess_images(
        self, paths: List[str], max_workers: Optional[int] = None
    ) -> List[_PreprocessResult]:
        """複数のパスからPIL画像を読み込み、前処理まで並列実行する.

        I/O（画像読み込み）とCPU-bound処理（RGB変換）をThreadPoolExecutorで並列化.

        Args:
            paths: 画像ファイルパスのリスト
            max_workers: スレッドプールの最大ワーカー数（Noneで自動設定）

        Returns:
            PIL画像のリスト（失敗したパスはNone）
        """

        def process_single(path: str) -> _PreprocessResult:
            """単一の画像を読み込み、前処理する."""
            try:
                # PILで画像を読み込み
                with Image.open(path) as img_file:
                    # RGBモードに変換（必要な場合）
                    if img_file.mode != "RGB":
                        pil_img: Image.Image = img_file.convert("RGB")
                        return pil_img.copy()
                    else:
                        return img_file.copy()

            except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError):
                return None

        # ThreadPoolExecutorで並列処理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single, paths))

        return results

    def _calculate_clip_features_batch(
        self,
        pil_images: List[Optional[Image.Image]],
        initial_batch_size: int = 32,
    ) -> List[Optional[np.ndarray]]:
        """複数のPIL画像に対してCLIP推論をバッチ実行して特徴を抽出.

        OOM対策（失敗したバッチのみ再試行）:
        - initial_batch_sizeから開始（デフォルト32）
        - torch.cuda.OutOfMemoryError発生時に失敗したバッチのみを分割してリトライ
        - 未処理のバッチは縮小されたバッチサイズで処理
        - 最小バッチサイズ1まで試行（32→16→8→4→2→1）
        - それでも失敗した画像はNoneとして返す

        Args:
            pil_images: PIL画像のリスト（失敗した画像はNone）
            initial_batch_size: 初期バッチサイズ

        Returns:
            CLIP画像埋め込みのリスト（512次元、正規化済み、失敗した画像はNone）
        """
        # 有効な画像のインデックスと画像を収集
        valid_indices = [i for i, img in enumerate(pil_images) if img is not None]
        # Type narrowing: valid_imagesはNoneを含まないことを保証
        valid_images: List[Image.Image] = [img for img in pil_images if img is not None]

        if not valid_images:
            return [None] * len(pil_images)

        # 結果を格納する配列（初期値はNone）
        results: List[Optional[np.ndarray]] = [None] * len(pil_images)

        # 現在のバッチサイズ
        current_batch_size = initial_batch_size

        # 処理位置を追跡
        i = 0
        while i < len(valid_images):
            # 現在のバッチを取得
            batch_start = i
            batch_end = min(i + current_batch_size, len(valid_images))
            batch: List[Image.Image] = valid_images[batch_start:batch_end]

            try:
                # GPUの場合はautocastでfp16推論を使用（高速化）
                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.device == "cuda"
                    else contextlib.nullcontext()
                )
                with autocast_context, torch.inference_mode():
                    inputs = self.processor(
                        images=batch,
                        return_tensors="pt",
                        padding=True,
                    ).to(self.device)
                    image_features = self.model.get_image_features(**inputs)

                    # L2正規化してNumPy配列に変換
                    batch_features = []
                    for j in range(image_features.shape[0]):
                        features = image_features[j].cpu().numpy()
                        normalized = _safe_l2_normalize(features)
                        batch_features.append(normalized)

                # 結果を元のインデックスにマッピング
                for j, features in enumerate(batch_features):
                    original_idx = valid_indices[batch_start + j]
                    results[original_idx] = features

                # バッチ処理成功、次へ進む
                i = batch_end

            except torch.cuda.OutOfMemoryError:
                # バッチサイズを半分にしてリトライ
                new_batch_size = current_batch_size // 2
                if new_batch_size >= 1:
                    logger.warning(
                        f"CUDA OOM発生（位置{batch_start}/{len(valid_images)}）。"
                        f"バッチサイズを{new_batch_size}に縮小してリトライします。"
                    )
                    current_batch_size = new_batch_size
                    # i は変更せず、同じ位置から小さいバッチサイズでリトライ
                else:
                    logger.error(
                        f"バッチサイズ1でもOOMが発生しました（位置{batch_start}）。"
                        "残りの画像の処理をスキップします。"
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
