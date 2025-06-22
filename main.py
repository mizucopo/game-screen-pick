"""
game_screen_pick.py - ゲーム画像品質分析・自動選択ツール（リファクタリング版）
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import shutil
import re

# GPU関連のオプショナルインポート
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


@dataclass
class ImageMetrics:
    """画像の評価指標を格納するデータクラス"""
    path: str
    blur_score: float
    brightness: float
    contrast: float
    exposure_score: float
    edge_density: float
    color_richness: float
    ui_density: float
    action_intensity: float
    visual_balance: float
    dramatic_score: float
    total_score: float
    features: Optional[np.ndarray] = None
    genre_hints: Optional[Dict[str, float]] = None


class GenreWeights:
    """ジャンル別の評価重みを管理するクラス"""

    DEFAULT_WEIGHTS = {
        "rpg": {
            "blur_score": 0.15,
            "contrast": 0.10,
            "color_richness": 0.20,
            "visual_balance": 0.15,
            "edge_density": 0.10,
            "action_intensity": 0.10,
            "ui_density": 0.05,
            "dramatic_score": 0.15
        },
        "fps": {
            "blur_score": 0.25,
            "contrast": 0.20,
            "color_richness": 0.10,
            "visual_balance": 0.10,
            "edge_density": 0.10,
            "action_intensity": 0.15,
            "ui_density": 0.00,
            "dramatic_score": 0.10
        },
        "tps": {
            "blur_score": 0.20,
            "contrast": 0.15,
            "color_richness": 0.15,
            "visual_balance": 0.15,
            "edge_density": 0.10,
            "action_intensity": 0.15,
            "ui_density": 0.00,
            "dramatic_score": 0.10
        },
        "2d_action": {
            "blur_score": 0.15,
            "contrast": 0.15,
            "color_richness": 0.20,
            "visual_balance": 0.10,
            "edge_density": 0.05,
            "action_intensity": 0.15,
            "ui_density": 0.00,
            "dramatic_score": 0.20
        },
        "2d_shooting": {
            "blur_score": 0.20,
            "contrast": 0.20,
            "color_richness": 0.15,
            "visual_balance": 0.10,
            "edge_density": 0.05,
            "action_intensity": 0.10,
            "ui_density": 0.00,
            "dramatic_score": 0.20
        },
        "puzzle": {
            "blur_score": 0.25,
            "contrast": 0.20,
            "color_richness": 0.15,
            "visual_balance": 0.25,
            "edge_density": 0.10,
            "action_intensity": 0.00,
            "ui_density": 0.05,
            "dramatic_score": 0.00
        },
        "racing": {
            "blur_score": 0.15,
            "contrast": 0.15,
            "color_richness": 0.15,
            "visual_balance": 0.15,
            "edge_density": 0.10,
            "action_intensity": 0.20,
            "ui_density": 0.00,
            "dramatic_score": 0.10
        },
        "strategy": {
            "blur_score": 0.20,
            "contrast": 0.15,
            "color_richness": 0.15,
            "visual_balance": 0.20,
            "edge_density": 0.15,
            "action_intensity": 0.05,
            "ui_density": 0.10,
            "dramatic_score": 0.00
        },
        "adventure": {
            "blur_score": 0.15,
            "contrast": 0.15,
            "color_richness": 0.20,
            "visual_balance": 0.20,
            "edge_density": 0.10,
            "action_intensity": 0.05,
            "ui_density": 0.00,
            "dramatic_score": 0.15
        },
        "mixed": {
            "blur_score": 0.20,
            "contrast": 0.15,
            "color_richness": 0.15,
            "visual_balance": 0.15,
            "edge_density": 0.10,
            "action_intensity": 0.10,
            "ui_density": 0.05,
            "dramatic_score": 0.10
        }
    }

    @classmethod
    def get_weights(cls, genre: str) -> Dict[str, float]:
        """指定ジャンルの重みを取得"""
        return cls.DEFAULT_WEIGHTS.get(genre.lower(), cls.DEFAULT_WEIGHTS["mixed"])

    @classmethod
    def save_weights(cls, filepath: str):
        """重み設定をファイルに保存"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cls.DEFAULT_WEIGHTS, f, indent=2)

    @classmethod
    def load_weights(cls, filepath: str):
        """重み設定をファイルから読み込み"""
        with open(filepath, 'r', encoding='utf-8') as f:
            cls.DEFAULT_WEIGHTS = json.load(f)


class ImageProcessor(ABC):
    """画像処理の基底クラス"""

    @abstractmethod
    def process(self, image: np.ndarray) -> Union[float, Tuple[float, ...], np.ndarray]:
        """画像を処理して結果を返す"""
        pass


class BlurDetector(ImageProcessor):
    """ブレ検出器"""

    def process(self, image: np.ndarray) -> float:
        """ラプラシアン分散でブレスコアを計算"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()


class BrightnessContrastAnalyzer(ImageProcessor):
    """明度・コントラスト分析器"""

    def process(self, image: np.ndarray) -> Tuple[float, float]:
        """明度とコントラストを計算"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        return brightness, contrast

    @staticmethod
    def calculate_exposure_score(brightness: float) -> float:
        """露出の適正度を評価"""
        return max(0, 100 - abs(brightness - 128) * 2)


class EdgeAnalyzer(ImageProcessor):
    """エッジ分析器"""

    def process(self, image: np.ndarray) -> float:
        """エッジ密度を計算"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return np.sum(edges > 0) / edges.size


class ColorAnalyzer(ImageProcessor):
    """色彩分析器"""

    def process(self, image: np.ndarray) -> float:
        """色の豊かさ（彩度の分散）を計算"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        return np.std(saturation)


class UIDetector(ImageProcessor):
    """UI要素検出器"""

    def process(self, image: np.ndarray) -> float:
        """UI要素の密度を検出"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        ui_density = (np.sum(np.abs(sobelx)) + np.sum(np.abs(sobely))) / gray.size
        return ui_density


class ActionIntensityAnalyzer(ImageProcessor):
    """アクション強度分析器"""

    def process(self, image: np.ndarray) -> float:
        """アクション強度を推定"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_diag1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_diag2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        diag_edges = cv2.filter2D(gray, -1, kernel_diag1) + cv2.filter2D(gray, -1, kernel_diag2)
        return np.std(diag_edges)


class DramaticSceneAnalyzer(ImageProcessor):
    """ドラマチックなシーン（ボス戦、重要シーン）を検出する分析器"""

    def process(self, image: np.ndarray) -> float:
        """ドラマチック度を評価"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]

        # 1. 色の強度と分布（ボス戦は派手な色使いが多い）
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # 高彩度・高明度の領域
        high_impact_mask = (saturation > 180) & (value > 180)
        color_intensity = np.sum(high_impact_mask) / (h * w)

        # 2. 赤・オレンジ系の色（炎、爆発、警告色）の検出
        # 赤系: 0-10, 170-180
        red_mask1 = (hsv[:, :, 0] <= 10) & (saturation > 100)
        red_mask2 = (hsv[:, :, 0] >= 170) & (saturation > 100)
        red_orange_mask = red_mask1 | red_mask2
        # オレンジ系: 10-25
        orange_mask = (hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 25) & (saturation > 100)

        fire_color_ratio = np.sum(red_orange_mask | orange_mask) / (h * w)

        # 3. 紫・青系の色（魔法、エネルギー）の検出
        # 紫系: 130-150
        purple_mask = (hsv[:, :, 0] >= 130) & (hsv[:, :, 0] <= 150) & (saturation > 100)
        # 青系: 100-130
        blue_mask = (hsv[:, :, 0] >= 100) & (hsv[:, :, 0] <= 130) & (saturation > 100)

        magic_color_ratio = np.sum(purple_mask | blue_mask) / (h * w)

        # 4. コントラストの局所的な高さ（エフェクトや光の表現）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        local_std = cv2.blur(gray, (50, 50))
        contrast_map = np.abs(gray.astype(float) - local_std.astype(float))
        high_contrast_ratio = np.sum(contrast_map > 50) / (h * w)

        # 5. 画面中央の重要度（ボスは中央に配置されることが多い）
        center_y, center_x = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        center_mask = ((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2) <= (min(h, w) // 3) ** 2

        center_intensity = np.mean(value[center_mask])
        edge_intensity = np.mean(value[~center_mask])
        center_importance = (center_intensity - edge_intensity) / 255.0

        # 6. 大きなオブジェクトの存在（ボスキャラクター）
        # エッジ検出して大きな輪郭を探す
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        large_object_score = 0
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            max_area = max(areas) if areas else 0
            large_object_score = min(max_area / (h * w), 0.3) * 3.33  # 0-1に正規化

        # 7. 動的な要素（エフェクトライン、スピード線）
        # 放射状のエッジパターンを検出
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # 中心からの放射状パターンの強度
        angles = np.arctan2(y_coords - center_y, x_coords - center_x)
        radial_edges = np.abs(np.cos(angles) * sobel_x + np.sin(angles) * sobel_y)
        radial_score = np.mean(radial_edges) / 255.0

        # 総合スコア計算（各要素を重み付けして合計）
        dramatic_score = (
            color_intensity * 15.0 +          # 派手な色の重要度
            fire_color_ratio * 25.0 +         # 炎・爆発色の重要度（高め）
            magic_color_ratio * 20.0 +        # 魔法色の重要度
            high_contrast_ratio * 15.0 +      # コントラストの重要度
            max(center_importance, 0) * 10.0 + # 中央集中度
            large_object_score * 10.0 +       # 大きなオブジェクト
            radial_score * 5.0                # 動的要素
        )

        return dramatic_score


class VisualBalanceAnalyzer(ImageProcessor):
    """視覚的バランス分析器"""

    def process(self, image: np.ndarray) -> float:
        """構図の安定感を評価"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 画面を9分割して重心バランスを計算
        center_region = gray[h//3:2*h//3, w//3:2*w//3]
        edge_regions = np.concatenate([
            gray[0:h//3, :].flatten(),
            gray[2*h//3:h, :].flatten(),
            gray[:, 0:w//3].flatten(),
            gray[:, 2*w//3:w].flatten()
        ])

        center_intensity = np.mean(center_region)
        edge_intensity = np.mean(edge_regions)

        balance_score = 100 - abs(center_intensity - edge_intensity) * 2
        return max(0, balance_score)


class FeatureExtractor:
    """画像特徴抽出器"""

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and GPU_AVAILABLE

    def extract(self, image: np.ndarray) -> np.ndarray:
        """画像から特徴ベクトルを抽出"""
        if self.use_gpu:
            return self._extract_gpu(image)
        return self._extract_cpu(image)

    def _extract_cpu(self, image: np.ndarray) -> np.ndarray:
        """CPU版特徴抽出"""
        small_img = cv2.resize(image, (64, 64))
        features = []

        # 色ヒストグラム（HSV）
        hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
        for i, bins in enumerate([[32, 180], [32, 256], [32, 256]]):
            hist = cv2.calcHist([hsv], [i], None, [bins[0]], [0, bins[1]])
            features.append(hist.flatten())

        # エッジヒストグラム
        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_hist = cv2.calcHist([edges], [0], None, [32], [0, 256])
        features.append(edge_hist.flatten())

        # 平均色とテクスチャ特徴
        mean_colors = np.mean(small_img.reshape(-1, 3), axis=0)
        features.append(mean_colors)

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features.append([laplacian_var])

        feature_vector = np.concatenate(features)
        return feature_vector / (np.linalg.norm(feature_vector) + 1e-7)

    def _extract_gpu(self, image: np.ndarray) -> np.ndarray:
        """GPU版特徴抽出"""
        try:
            small_img = cv2.resize(image, (64, 64))
            gpu_img = cp.asarray(small_img)
            features = []

            # 色ヒストグラム
            for channel in range(3):
                hist, _ = cp.histogram(gpu_img[:, :, channel], bins=32, range=(0, 256))
                features.append(hist)

            # 統計量
            mean_colors = cp.mean(gpu_img.reshape(-1, 3), axis=0)
            features.append(mean_colors)

            std_colors = cp.std(gpu_img.reshape(-1, 3), axis=0)
            features.append(std_colors)

            feature_vector = cp.concatenate(features)
            feature_vector = cp.asnumpy(feature_vector)
            return feature_vector / (np.linalg.norm(feature_vector) + 1e-7)

        except Exception as e:
            print(f"GPU処理エラー、CPUに切り替え: {e}")
            return self._extract_cpu(image)


class SimilarityCalculator:
    """類似度計算器"""

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and GPU_AVAILABLE

    def calculate(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """2つの特徴ベクトル間の類似度を計算"""
        if self.use_gpu:
            return self._calculate_gpu(features1, features2)
        return self._calculate_cpu(features1, features2)

    @staticmethod
    def _calculate_cpu(features1: np.ndarray, features2: np.ndarray) -> float:
        """CPU版類似度計算"""
        return cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]

    def _calculate_gpu(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """GPU版類似度計算"""
        try:
            gpu_f1 = cp.asarray(features1.reshape(1, -1))
            gpu_f2 = cp.asarray(features2.reshape(1, -1))

            dot_product = cp.dot(gpu_f1, gpu_f2.T)
            norm1 = cp.linalg.norm(gpu_f1)
            norm2 = cp.linalg.norm(gpu_f2)
            similarity = dot_product / (norm1 * norm2 + 1e-7)

            return float(cp.asnumpy(similarity[0, 0]))
        except:
            return self._calculate_cpu(features1, features2)


class ImageQualityAnalyzer:
    """ゲーム画像品質分析器"""

    # 対応画像形式
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    MAX_IMAGE_SIZE = 1000  # リサイズ時の最大サイズ

    def __init__(self, genre: str = "mixed", use_gpu: bool = False):
        self.genre = genre.lower()
        self.weights = GenreWeights.get_weights(self.genre)
        self.use_gpu = use_gpu and GPU_AVAILABLE

        # 各種プロセッサの初期化
        self.blur_detector = BlurDetector()
        self.brightness_analyzer = BrightnessContrastAnalyzer()
        self.edge_analyzer = EdgeAnalyzer()
        self.color_analyzer = ColorAnalyzer()
        self.ui_detector = UIDetector()
        self.action_analyzer = ActionIntensityAnalyzer()
        self.balance_analyzer = VisualBalanceAnalyzer()
        self.dramatic_analyzer = DramaticSceneAnalyzer()
        self.feature_extractor = FeatureExtractor(use_gpu)

        if self.use_gpu:
            print("GPU加速モードで動作します")
        else:
            print("CPUモードで動作します")

    def _resize_image_if_needed(self, image: np.ndarray) -> np.ndarray:
        """必要に応じて画像をリサイズ"""
        height, width = image.shape[:2]
        if max(height, width) > self.MAX_IMAGE_SIZE:
            scale = self.MAX_IMAGE_SIZE / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height))
        return image

    def _calculate_total_score(self, metrics: Dict[str, float]) -> float:
        """重み付きスコアを計算"""
        return (
            metrics['blur_score'] * self.weights['blur_score'] +
            metrics['contrast'] * self.weights['contrast'] +
            metrics['color_richness'] * self.weights['color_richness'] +
            metrics['visual_balance'] * self.weights['visual_balance'] +
            metrics['edge_density'] * 1000 * self.weights['edge_density'] +
            metrics['action_intensity'] * self.weights['action_intensity'] +
            metrics['ui_density'] * self.weights['ui_density'] +
            metrics['dramatic_score'] * self.weights['dramatic_score']
        )

    def analyze_image(self, image_path: str) -> Optional[ImageMetrics]:
        """画像を総合分析"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            # リサイズ
            image = self._resize_image_if_needed(image)

            # 各種メトリクスを計算
            blur_score = self.blur_detector.process(image)
            brightness, contrast = self.brightness_analyzer.process(image)
            exposure_score = BrightnessContrastAnalyzer.calculate_exposure_score(brightness)
            edge_density = self.edge_analyzer.process(image)
            color_richness = self.color_analyzer.process(image)
            ui_density = self.ui_detector.process(image)
            action_intensity = self.action_analyzer.process(image)
            visual_balance = self.balance_analyzer.process(image)
            dramatic_score = self.dramatic_analyzer.process(image)

            # 特徴ベクトル抽出
            features = self.feature_extractor.extract(image)

            # スコア計算
            metrics_dict = {
                'blur_score': blur_score,
                'contrast': contrast,
                'color_richness': color_richness,
                'visual_balance': visual_balance,
                'edge_density': edge_density,
                'action_intensity': action_intensity,
                'ui_density': ui_density,
                'dramatic_score': dramatic_score
            }
            total_score = self._calculate_total_score(metrics_dict)

            return ImageMetrics(
                path=image_path,
                blur_score=blur_score,
                brightness=brightness,
                contrast=contrast,
                exposure_score=exposure_score,
                edge_density=edge_density,
                color_richness=color_richness,
                ui_density=ui_density,
                action_intensity=action_intensity,
                visual_balance=visual_balance,
                dramatic_score=dramatic_score,
                total_score=total_score,
                features=features
            )

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None


class ImageSelector:
    """画像選択器"""

    def __init__(self, similarity_threshold: float = 0.85, use_gpu: bool = False):
        self.similarity_threshold = similarity_threshold
        self.similarity_calculator = SimilarityCalculator(use_gpu)

    def remove_similar_images(self, results: List[ImageMetrics]) -> List[ImageMetrics]:
        """類似画像を除去"""
        if not results:
            return results

        print(f"類似画像除去中... (閾値: {self.similarity_threshold})")

        # スコア順にソート
        sorted_results = sorted(results, key=lambda x: x.total_score, reverse=True)
        selected = []

        for candidate in sorted_results:
            is_similar = False

            for selected_item in selected:
                similarity = self.similarity_calculator.calculate(
                    candidate.features,
                    selected_item.features
                )

                if similarity > self.similarity_threshold:
                    is_similar = True
                    print(f"  類似画像を除外: {Path(candidate.path).name} (類似度: {similarity:.3f})")
                    break

            if not is_similar:
                selected.append(candidate)
                print(f"  選択: {Path(candidate.path).name} (スコア: {candidate.total_score:.1f})")

        return selected


def analyze_image_worker(args: Tuple[str, str, bool]) -> Optional[ImageMetrics]:
    """並列処理用のワーカー関数"""
    image_path, genre, use_gpu = args
    analyzer = ImageQualityAnalyzer(genre, use_gpu)
    return analyzer.analyze_image(image_path)


class ImageCopier:
    """選択された画像を別ディレクトリにコピーするクラス"""

    @staticmethod
    def create_safe_filename(source_path: Path, base_folder: Path) -> str:
        """ファイル名の衝突を回避する安全なファイル名を生成"""
        # ベースフォルダからの相対パスを取得
        try:
            relative_path = source_path.relative_to(base_folder)
        except ValueError:
            # 相対パスが取得できない場合は絶対パスを使用
            relative_path = source_path

        # パスの各部分をアンダースコアで結合
        parts = list(relative_path.parent.parts) if relative_path.parent.parts else []

        # Windowsのドライブレター対応
        parts = [part.replace(':', '') for part in parts]

        # 安全なファイル名に変換（特殊文字を除去）
        safe_parts = []
        for part in parts:
            safe_part = re.sub(r'[<>:"/\\|?*]', '_', part)
            safe_parts.append(safe_part)

        # ディレクトリパスとファイル名を結合
        if safe_parts:
            prefix = '_'.join(safe_parts) + '_'
        else:
            prefix = ''

        return prefix + source_path.name

    @staticmethod
    def copy_images(image_paths: List[str], output_dir: str, base_folder: Path,
                   preserve_structure: bool = False) -> Dict[str, str]:
        """画像を出力ディレクトリにコピー"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        copied_files = {}

        for i, src_path in enumerate(image_paths):
            src = Path(src_path)

            if preserve_structure:
                # ディレクトリ構造を保持
                try:
                    relative = src.relative_to(base_folder)
                    dst = output_path / relative
                    dst.parent.mkdir(parents=True, exist_ok=True)
                except ValueError:
                    # 相対パスが取得できない場合
                    dst = output_path / src.name
            else:
                # フラットな構造でコピー（パスをファイル名に含める）
                safe_filename = ImageCopier.create_safe_filename(src, base_folder)
                dst = output_path / safe_filename

            # 既存ファイルとの衝突を回避
            if dst.exists():
                stem = dst.stem
                suffix = dst.suffix
                counter = 1
                while dst.exists():
                    dst = dst.parent / f"{stem}_{counter}{suffix}"
                    counter += 1

            try:
                shutil.copy2(src, dst)
                copied_files[str(src)] = str(dst)
                print(f"  [{i+1}] コピー: {src.name} -> {dst.name}")
            except Exception as e:
                print(f"  エラー: {src} のコピーに失敗: {e}")

        return copied_files


class GameScreenPicker:
    """ゲーム画像選択のメインクラス"""

    def __init__(self, genre: str = "mixed", use_gpu: bool = False,
                 num_workers: Optional[int] = None):
        self.genre = genre
        self.use_gpu = use_gpu
        self.num_workers = num_workers or min(multiprocessing.cpu_count(), 8)
        self.analyzer = ImageQualityAnalyzer(genre, use_gpu)
    """ゲーム画像選択のメインクラス"""

    def __init__(self, genre: str = "mixed", use_gpu: bool = False,
                 num_workers: Optional[int] = None):
        self.genre = genre
        self.use_gpu = use_gpu
        self.num_workers = num_workers or min(multiprocessing.cpu_count(), 8)
        self.analyzer = ImageQualityAnalyzer(genre, use_gpu)

    def get_image_files(self, folder_path: str, recursive: bool = False) -> List[Path]:
        """フォルダ内の画像ファイルを取得"""
        image_files = []
        folder = Path(folder_path)

        if recursive:
            # サブディレクトリも含めて検索
            for ext in ImageQualityAnalyzer.SUPPORTED_EXTENSIONS:
                image_files.extend(folder.rglob(f'*{ext}'))
                image_files.extend(folder.rglob(f'*{ext.upper()}'))
        else:
            # 現在のディレクトリのみ
            for ext in ImageQualityAnalyzer.SUPPORTED_EXTENSIONS:
                image_files.extend(folder.glob(f'*{ext}'))
                image_files.extend(folder.glob(f'*{ext.upper()}'))

        return sorted(list(set(image_files)))  # 重複を除去してソート

    def analyze_images(self, image_files: List[Path]) -> List[ImageMetrics]:
        """画像を分析"""
        results = []
        total_files = len(image_files)
        print(f"見つかった画像: {total_files}枚")

        if self.num_workers > 1 and not self.use_gpu:
            # 並列処理
            print(f"並列処理で分析中... (ワーカー数: {self.num_workers})")
            worker_args = [(str(img), self.genre, self.use_gpu) for img in image_files]

            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(analyze_image_worker, worker_args))

            results = [r for r in results if r is not None]
        else:
            # 順次処理
            for i, image_path in enumerate(image_files):
                if i % 50 == 0:
                    print(f"進捗: {i}/{total_files}")

                result = self.analyzer.analyze_image(str(image_path))
                if result:
                    results.append(result)

        return results

    def select_best_images(self, folder_path: str, num_select: int = 10,
                          similarity_threshold: float = 0.85,
                          recursive: bool = False,
                          copy_to_dir: Optional[str] = None,
                          preserve_structure: bool = False) -> List[str]:
        """フォルダ内の画像から上位N枚を選択"""
        start_time = time.time()

        print(f"ジャンル設定: {self.genre.upper()}")
        print(f"処理モード: {'GPU' if self.use_gpu else 'CPU'}")
        print(f"検索モード: {'再帰的' if recursive else '単一ディレクトリ'}")

        # 画像ファイル取得
        image_files = self.get_image_files(folder_path, recursive)
        if not image_files:
            print("画像ファイルが見つかりませんでした")
            return []

        # 画像分析
        print("画像を分析中...")
        results = self.analyze_images(image_files)

        if not results:
            print("分析可能な画像が見つかりませんでした")
            return []

        analysis_time = time.time() - start_time
        print(f"分析完了: {analysis_time:.1f}秒")

        # 類似画像除去
        selector = ImageSelector(similarity_threshold, self.use_gpu)
        unique_results = selector.remove_similar_images(results)

        # 必要な枚数まで選択
        selected_results = unique_results[:num_select]

        # 選択された画像のパスリスト
        selected_paths = [result.path for result in selected_results]

        # 画像をコピー
        if copy_to_dir and selected_paths:
            print(f"\n画像を {copy_to_dir} にコピー中...")
            base_folder = Path(folder_path)
            ImageCopier.copy_images(selected_paths, copy_to_dir, base_folder, preserve_structure)

        # 結果表示
        self._print_results(selected_results)

        print(f"\n総処理時間: {time.time() - start_time:.1f}秒")

        return selected_paths

    @staticmethod
    def _print_results(results: List[ImageMetrics]):
        """結果を表示"""
        print(f"\n=== 選択された上位{len(results)}枚 ===")
        for i, result in enumerate(results):
            print(f"{i+1}. {Path(result.path).name}")
            print(f"   スコア: {result.total_score:.1f}")
            print(f"   (鮮明度: {result.blur_score:.1f}, "
                  f"ドラマ: {result.dramatic_score:.1f}, "
                  f"アクション: {result.action_intensity:.1f})")


def print_genre_info():
    """ジャンル情報を表示"""
    print("=== game-screen-pick ===")
    print("選択可能ジャンル:")
    genre_descriptions = {
        "rpg": "美しさと構図重視",
        "fps": "鮮明さと視認性重視",
        "tps": "鮮明さ重視、キャラと背景のバランス考慮",
        "2d_action": "ピクセルアートの鮮明さと色彩美重視",
        "2d_shooting": "弾幕の鮮明さと視認性最重要",
        "puzzle": "バランスと整理された画面重視",
        "racing": "スピード感重視",
        "strategy": "情報量とUI重視",
        "adventure": "美しさと構図重視",
        "mixed": "バランス型（複数ジャンル混在時）"
    }

    for genre, desc in genre_descriptions.items():
        print(f"  {genre}: {desc}")

    print("\n類似度閾値について:")
    print("  0.9以上: 非常に厳しく除外（ほぼ同じ画像のみ除外）")
    print("  0.85: 標準（推奨）")
    print("  0.8以下: 緩い除外（バラエティ重視）")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='game-screen-pick: ゲーム画像品質分析・自動選択ツール'
    )
    parser.add_argument('folder', help='画像フォルダのパス')
    parser.add_argument('-n', '--num', type=int, default=10,
                       help='選択する画像数 (デフォルト: 10)')
    parser.add_argument('-g', '--genre',
                       choices=['rpg', 'fps', 'tps', '2d_action', '2d_shooting',
                               'puzzle', 'racing', 'strategy', 'adventure', 'mixed'],
                       default='mixed',
                       help='ゲームジャンル (デフォルト: mixed)')
    parser.add_argument('-s', '--similarity', type=float, default=0.85,
                       help='類似度閾値 (0.0-1.0, デフォルト: 0.85)')
    parser.add_argument('--gpu', action='store_true',
                       help='GPU加速を有効にする（CuPy必須）')
    parser.add_argument('-w', '--workers', type=int,
                       help='並列処理のワーカー数 (デフォルト: CPU数)')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='サブディレクトリも含めて検索')
    parser.add_argument('-c', '--copy-to',
                       help='選択した画像をコピーする出力ディレクトリ')
    parser.add_argument('--preserve-structure', action='store_true',
                       help='コピー時にディレクトリ構造を保持（デフォルト: フラット構造）')
    parser.add_argument('--save-weights', help='重み設定をJSONファイルに保存')
    parser.add_argument('--load-weights', help='重み設定をJSONファイルから読み込み')

    args = parser.parse_args()

    # 重み設定の保存/読み込み
    if args.save_weights:
        GenreWeights.save_weights(args.save_weights)
        print(f"重み設定を {args.save_weights} に保存しました")
        return

    if args.load_weights:
        try:
            GenreWeights.load_weights(args.load_weights)
            print(f"重み設定を {args.load_weights} から読み込みました")
        except Exception as e:
            print(f"重み設定の読み込みに失敗しました: {e}")
            return

    # 入力検証
    if not Path(args.folder).exists():
        print(f"エラー: フォルダ '{args.folder}' が見つかりません")
        return

    if not (0.0 <= args.similarity <= 1.0):
        print("エラー: 類似度閾値は0.0-1.0の範囲で指定してください")
        return

    if args.gpu and not GPU_AVAILABLE:
        print("警告: GPU加速が要求されましたが、CuPyがインストールされていません")
        print("pip install cupy-cuda11x または cupy-cuda12x でインストールしてください")
        print("CPUモードで続行します...")
        args.gpu = False

    # ジャンル情報表示
    print_genre_info()

    # メイン処理
    picker = GameScreenPicker(
        genre=args.genre,
        use_gpu=args.gpu,
        num_workers=args.workers
    )

    selected_images = picker.select_best_images(
        folder_path=args.folder,
        num_select=args.num,
        similarity_threshold=args.similarity,
        recursive=args.recursive,
        copy_to_dir=args.copy_to,
        preserve_structure=args.preserve_structure
    )

    print(f"\n選択完了: {len(selected_images)}枚")

    if args.copy_to:
        print(f"画像は {args.copy_to} にコピーされました")


if __name__ == "__main__":
    main()
