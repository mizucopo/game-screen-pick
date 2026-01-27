"""game_screen_pick - 高精度・コンテンツ多様性重視選択ツール"""

import argparse
import shutil
from pathlib import Path

from .services import GameScreenPicker


def main():
    parser = argparse.ArgumentParser(description='Diverse Game Screen Picker')
    parser.add_argument('input', help='入力フォルダ')
    parser.add_argument('-c', '--copy-to', help='出力フォルダ')
    parser.add_argument('-n', '--num', type=int, default=10, help='選択枚数')
    parser.add_argument(
        '-g', '--genre',
        default='mixed',
        choices=[
            'rpg', 'fps', 'tps', '2d_action', '2d_shooting', '3d_action',
            'puzzle', 'racing', 'strategy', 'adventure', 'mixed'
        ]
    )
    parser.add_argument(
        '-s', '--similarity',
        type=float,
        default=0.82,
        help='類似度しきい値(0.7~0.85推奨)'
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='サブフォルダも検索'
    )
    args = parser.parse_args()

    picker = GameScreenPicker(args.genre)
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
