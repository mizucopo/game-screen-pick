"""
game_screen_pick.py - 高精度・コンテンツ多様性重視選択ツール
"""

import cv2
import numpy as np
import math
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import time
from dataclasses import dataclass
import shutil

# CLIP (AI分析) 関連
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

@dataclass
class ImageMetrics:
    path: str
    raw_metrics: Dict[str, float]
    normalized_metrics: Dict[str, float]
    semantic_score: float
    total_score: float
    features: np.ndarray

class GenreWeights:
    """全10ジャンルの重み定義"""
    DEFAULT_WEIGHTS = {
        "rpg": {"blur_score": 0.15, "contrast": 0.10, "color_richness": 0.20, "visual_balance": 0.15, "edge_density": 0.10, "action_intensity": 0.10, "ui_density": 0.05, "dramatic_score": 0.15},
        "fps": {"blur_score": 0.25, "contrast": 0.20, "color_richness": 0.10, "visual_balance": 0.10, "edge_density": 0.10, "action_intensity": 0.15, "ui_density": 0.00, "dramatic_score": 0.10},
        "tps": {"blur_score": 0.20, "contrast": 0.15, "color_richness": 0.15, "visual_balance": 0.15, "edge_density": 0.10, "action_intensity": 0.15, "ui_density": 0.00, "dramatic_score": 0.10},
        "2d_action": {"blur_score": 0.15, "contrast": 0.15, "color_richness": 0.20, "visual_balance": 0.10, "edge_density": 0.05, "action_intensity": 0.15, "ui_density": 0.00, "dramatic_score": 0.20},
        "2d_shooting": {"blur_score": 0.20, "contrast": 0.20, "color_richness": 0.15, "visual_balance": 0.10, "edge_density": 0.05, "action_intensity": 0.10, "ui_density": 0.00, "dramatic_score": 0.20},
        "3d_action": {"blur_score": 0.18, "contrast": 0.18, "color_richness": 0.18, "visual_balance": 0.12, "edge_density": 0.08, "action_intensity": 0.18, "ui_density": 0.00, "dramatic_score": 0.08},
        "puzzle": {"blur_score": 0.25, "contrast": 0.20, "color_richness": 0.15, "visual_balance": 0.25, "edge_density": 0.10, "action_intensity": 0.00, "ui_density": 0.05, "dramatic_score": 0.00},
        "racing": {"blur_score": 0.15, "contrast": 0.15, "color_richness": 0.15, "visual_balance": 0.15, "edge_density": 0.10, "action_intensity": 0.20, "ui_density": 0.00, "dramatic_score": 0.10},
        "strategy": {"blur_score": 0.20, "contrast": 0.15, "color_richness": 0.15, "visual_balance": 0.20, "edge_density": 0.15, "action_intensity": 0.05, "ui_density": 0.10, "dramatic_score": 0.00},
        "adventure": {"blur_score": 0.15, "contrast": 0.15, "color_richness": 0.20, "visual_balance": 0.20, "edge_density": 0.10, "action_intensity": 0.05, "ui_density": 0.00, "dramatic_score": 0.15},
        "mixed": {"blur_score": 0.20, "contrast": 0.15, "color_richness": 0.15, "visual_balance": 0.15, "edge_density": 0.10, "action_intensity": 0.10, "ui_density": 0.05, "dramatic_score": 0.10}
    }

    @classmethod
    def get_weights(cls, genre: str) -> Dict[str, float]:
        return cls.DEFAULT_WEIGHTS.get(genre.lower(), cls.DEFAULT_WEIGHTS["mixed"])

class MetricNormalizer:
    @staticmethod
    def sigmoid(x: float, center: float, steepness: float = 0.1) -> float:
        try: return 1 / (1 + math.exp(-steepness * (x - center)))
        except: return 1.0 if x > center else 0.0

    @classmethod
    def normalize_all(cls, raw: Dict[str, float]) -> Dict[str, float]:
        return {
            "blur_score": cls.sigmoid(raw['blur_score'], center=500, steepness=0.005),
            "contrast": cls.sigmoid(raw['contrast'], center=50, steepness=0.1),
            "color_richness": cls.sigmoid(raw['color_richness'], center=40, steepness=0.1),
            "edge_density": min(1.0, raw['edge_density'] * 5.0),
            "dramatic_score": min(1.0, raw['dramatic_score'] / 100.0),
            "visual_balance": raw['visual_balance'] / 100.0,
            "action_intensity": cls.sigmoid(raw['action_intensity'], center=30, steepness=0.2),
            "ui_density": cls.sigmoid(raw['ui_density'], center=10, steepness=0.3)
        }

class ImageQualityAnalyzer:
    def __init__(self, genre: str = "mixed", use_clip: bool = False):
        self.weights = GenreWeights.get_weights(genre)
        self.model = None
        if use_clip and HAS_CLIP:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

    def _extract_diversity_features(self, img: np.ndarray) -> np.ndarray:
        """見た目の特徴を抽出（色と構造）"""
        small = cv2.resize(img, (128, 128))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def analyze(self, path: str) -> Optional[ImageMetrics]:
        try:
            img = cv2.imread(path)
            if img is None: return None
            features = self._extract_diversity_features(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            raw = {
                "blur_score": cv2.Laplacian(gray, cv2.CV_64F).var(),
                "brightness": np.mean(gray),
                "contrast": np.std(gray),
                "edge_density": np.sum(cv2.Canny(gray, 50, 150) > 0) / gray.size,
                "color_richness": np.std(hsv[:, :, 1]),
                "ui_density": (np.sum(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0))) / gray.size),
                "action_intensity": np.std(cv2.filter2D(gray, -1, np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))),
                "visual_balance": max(0, 100 - abs(np.mean(gray) - 128) * 0.5),
                "dramatic_score": (np.sum((hsv[:,:,1]>180) & (hsv[:,:,2]>180)) / img.size) * 1000
            }
            norm = MetricNormalizer.normalize_all(raw)
            semantic = 0.0
            if self.model:
                with torch.no_grad():
                    inputs = self.processor(text=["epic game scenery"], images=Image.open(path), return_tensors="pt", padding=True).to(self.device)
                    semantic = float(self.model(**inputs).logits_per_image[0][0]) / 100.0

            weighted_sum = sum(norm[k] * self.weights.get(k, 0.0) for k in norm if k in self.weights)
            # ペナルティ（暗すぎる画像）
            penalty = 0.6 if raw['brightness'] < 40 else 0.0
            total = max(0.0, (weighted_sum + (semantic * 0.2) - penalty) * 100.0)
            return ImageMetrics(path, raw, norm, semantic, total, features)
        except: return None

class GameScreenPicker:
    def __init__(self, genre: str, use_clip: bool):
        self.analyzer = ImageQualityAnalyzer(genre, use_clip)

    def select(self, folder: str, num: int, similarity_threshold: float, recursive: bool):
        path_obj = Path(folder)
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        # 全フォルダからファイルを一括取得
        files = [p for p in (path_obj.rglob('*') if recursive else path_obj.glob('*')) if p.suffix.lower() in exts]

        # 1. 完全にランダムにシャッフル（フォルダやファイル名のバイアスを破壊）
        random.shuffle(files)

        print(f"合計 {len(files)} 枚を解析中...")
        all_results = []
        for i, f in enumerate(files):
            if i % 50 == 0: print(f"解析済み: {i}/{len(files)}")
            res = self.analyzer.analyze(str(f))
            if res: all_results.append(res)

        # 2. スコア順にソート（この時点では最高画質が上にくる）
        all_results.sort(key=lambda x: x.total_score, reverse=True)

        # 3. コンテンツの多様性に基づいたフィルタリング
        selected = []
        for candidate in all_results:
            if len(selected) >= num: break

            # 既に選ばれた画像たちと「見た目」を比較
            is_similar = False
            for s in selected:
                # コサイン類似度で「似すぎていないか」チェック
                sim = cosine_similarity(candidate.features.reshape(1, -1), s.features.reshape(1, -1))[0][0]
                if sim > similarity_threshold:
                    is_similar = True
                    break

            if not is_similar:
                selected.append(candidate)

        return selected

def main():
    parser = argparse.ArgumentParser(description='Diverse Game Screen Picker')
    parser.add_argument('input', help='入力フォルダ')
    parser.add_argument('-c', '--copy-to', help='出力フォルダ')
    parser.add_argument('-n', '--num', type=int, default=10, help='選択枚数')
    parser.add_argument('-g', '--genre', default='mixed', choices=['rpg', 'fps', 'tps', '2d_action', '2d_shooting', '3d_action', 'puzzle', 'racing', 'strategy', 'adventure', 'mixed'])
    parser.add_argument('-s', '--similarity', type=float, default=0.82, help='類似度しきい値(0.7~0.85推奨)')
    parser.add_argument('-r', '--recursive', action='store_true', help='サブフォルダも検索')
    parser.add_argument('--clip', action='store_true', help='CLIP分析を有効化')
    args = parser.parse_args()

    picker = GameScreenPicker(args.genre, args.clip)
    # 多様性重視で選択
    best = picker.select(args.input, args.num, args.similarity, args.recursive)

    if args.copy_to and best:
        out = Path(args.copy_to)
        out.mkdir(parents=True, exist_ok=True)
        for res in best:
            shutil.copy2(res.path, out / Path(res.path).name)
        print(f"\n{len(best)} 枚を {args.copy_to} に保存しました（多様性確保済み）。")

    print("\n--- 選択された画像一覧 ---")
    for i, res in enumerate(best):
        print(f"[{i+1}] {Path(res.path).name} (Score: {res.total_score:.2f})")

if __name__ == "__main__":
    main()
