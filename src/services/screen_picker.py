"""Game screen picker for diverse image selection.
"""

from pathlib import Path
import random
from sklearn.metrics.pairwise import cosine_similarity

from ..analyzers.image_quality_analyzer import ImageQualityAnalyzer


class GameScreenPicker:
    """ゲーム画面選択クラス"""

    def __init__(self, genre: str):
        self.analyzer = ImageQualityAnalyzer(genre)

    def select(
        self,
        folder: str,
        num: int,
        similarity_threshold: float,
        recursive: bool
    ):
        path_obj = Path(folder)
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        # 全フォルダからファイルを一括取得
        files = [
            p for p in (
                path_obj.rglob('*') if recursive else path_obj.glob('*')
            )
            if p.suffix.lower() in exts
        ]

        # 1. 完全にランダムにシャッフル（フォルダやファイル名のバイアスを破壊）
        random.shuffle(files)

        print(f"合計 {len(files)} 枚を解析中...")
        all_results = []
        for i, f in enumerate(files):
            if i % 50 == 0:
                print(f"解析済み: {i}/{len(files)}")
            res = self.analyzer.analyze(str(f))
            if res:
                all_results.append(res)

        # 2. スコア順にソート（この時点では最高画質が上にくる）
        all_results.sort(key=lambda x: x.total_score, reverse=True)

        # 3. コンテンツの多様性に基づいたフィルタリング
        selected = []
        for candidate in all_results:
            if len(selected) >= num:
                break

            # 既に選ばれた画像たちと「見た目」を比較
            is_similar = False
            for s in selected:
                # コサイン類似度で「似すぎていないか」チェック
                sim = cosine_similarity(
                    candidate.features.reshape(1, -1),
                    s.features.reshape(1, -1)
                )[0][0]
                if sim > similarity_threshold:
                    is_similar = True
                    break

            if not is_similar:
                selected.append(candidate)

        return selected
