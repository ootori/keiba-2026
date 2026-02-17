#!/usr/bin/env python3
"""予測実行スクリプト.

使用例:
    # 特定レースの予測
    python run_predict.py --year 2025 --monthday 0622 --jyocd 09 \
                          --kaiji 03 --nichiji 08 --racenum 11

    # 1日分の全レース予測
    python run_predict.py --year 2025 --monthday 0622 --all-day

    # 特定場の1日分
    python run_predict.py --year 2025 --monthday 0622 --jyocd 09 --all-day

    # オッズ込みモデルで予測
    python run_predict.py --year 2025 --monthday 0622 --all-day --with-odds
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.db import check_connection
from src.model.predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="競馬予測実行")
    parser.add_argument("--year", required=True, help="年（例: 2025）")
    parser.add_argument("--monthday", required=True, help="月日（例: 0622）")
    parser.add_argument("--jyocd", default=None, help="競馬場コード（例: 09=阪神）")
    parser.add_argument("--kaiji", default=None, help="開催回")
    parser.add_argument("--nichiji", default=None, help="開催日目")
    parser.add_argument("--racenum", default=None, help="レース番号")
    parser.add_argument(
        "--all-day",
        action="store_true",
        help="指定日の全レースを予測",
    )
    parser.add_argument(
        "--with-odds",
        action="store_true",
        help="オッズ込みモデルを使用",
    )
    parser.add_argument(
        "--model-name",
        default="model",
        help="モデル名（デフォルト: model）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # DB接続確認
    if not check_connection():
        logger.error("DB接続に失敗しました。")
        sys.exit(1)

    # 予測器の初期化
    predictor = Predictor(
        model_name=args.model_name,
        include_odds=args.with_odds,
    )
    predictor.load()

    if args.all_day:
        # 1日分の全レース予測
        results = predictor.predict_day(
            year=args.year,
            monthday=args.monthday,
            jyocd=args.jyocd,
        )

        if not results:
            logger.warning("予測結果がありません。")
            return

        for key, pred in results.items():
            # レースキーを復元して表示
            if not pred.empty:
                print()
                # 簡易表示
                jyocd = key.split("_")[0]
                racenum = key.split("_")[1] if "_" in key else ""
                race_key_for_format = {
                    "year": args.year,
                    "monthday": args.monthday,
                    "jyocd": jyocd,
                    "kaiji": "",
                    "nichiji": "",
                    "racenum": racenum.replace("R", ""),
                }
                output = predictor.format_prediction(race_key_for_format, pred)
                print(output)
                print()

    else:
        # 特定レースの予測
        if not all([args.jyocd, args.kaiji, args.nichiji, args.racenum]):
            logger.error(
                "特定レース予測には --jyocd, --kaiji, --nichiji, --racenum が必要です。"
                "\n全レース予測は --all-day を使用してください。"
            )
            sys.exit(1)

        race_key = {
            "year": args.year,
            "monthday": args.monthday,
            "jyocd": args.jyocd,
            "kaiji": args.kaiji,
            "nichiji": args.nichiji,
            "racenum": args.racenum,
        }

        prediction = predictor.predict_race(race_key)
        if prediction.empty:
            logger.warning("予測結果がありません。")
            return

        output = predictor.format_prediction(race_key, prediction)
        print()
        print(output)
        print()


if __name__ == "__main__":
    main()
