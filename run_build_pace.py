#!/usr/bin/env python3
"""ペース特徴量の差分構築スクリプト.

既存の年度別 parquet を再構築せずに、ペース特徴量だけを
差分 parquet として保存する。学習時に既存特徴量にマージして使う。

使用例:
    # 全年度（2015-2025）のペース特徴量を構築
    python run_build_pace.py

    # 期間指定
    python run_build_pace.py --start 2024 --end 2025

    # 並列構築
    python run_build_pace.py --workers 4

    # 既存を再構築
    python run_build_pace.py --force-rebuild

    # 特定年度だけ再構築
    python run_build_pace.py --start 2020 --end 2020 --force-rebuild
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    TRAIN_START_YEAR,
    TRAIN_END_YEAR,
    VALID_YEAR,
    DATA_DIR,
    RACE_KEY_COLS,
    JRA_JYO_CODES,
)
from src.db import check_connection, query_df
from src.features.pace import PaceFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def pace_parquet_path(year: str) -> Path:
    """ペース特徴量の年度別 parquet パスを返す."""
    return DATA_DIR / f"pace_features_{year}.parquet"


def _get_target_races(year: str) -> pd.DataFrame:
    """対象年のレース一覧を取得する."""
    sql = """
    SELECT DISTINCT year, monthday, jyocd, kaiji, nichiji, racenum
    FROM n_race
    WHERE datakubun = '7'
      AND year = %(year)s
      AND jyocd IN %(jyo_codes)s
    ORDER BY year, monthday, jyocd, racenum
    """
    return query_df(sql, {"year": year, "jyo_codes": tuple(JRA_JYO_CODES)})


def _get_horses(race_key: dict[str, str]) -> pd.DataFrame:
    """出走馬のkettonumとumabanを取得する."""
    sql = """
    SELECT kettonum, umaban
    FROM n_uma_race
    WHERE year = %(year)s AND monthday = %(monthday)s
      AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
      AND nichiji = %(nichiji)s AND racenum = %(racenum)s
      AND datakubun IN ('1','2','3','4','5','6','7')
      AND ijyocd = '0'
    ORDER BY CAST(umaban AS integer)
    """
    return query_df(sql, race_key)


def build_pace_year(year: str, force_rebuild: bool = False) -> str:
    """1年分のペース特徴量を構築する.

    Args:
        year: 対象年
        force_rebuild: 既存 parquet を無視して再構築

    Returns:
        完了メッセージ
    """
    path = pace_parquet_path(year)
    if not force_rebuild and path.exists():
        return f"{year}: スキップ（既存 pace parquet あり）"

    extractor = PaceFeatureExtractor()
    races = _get_target_races(year)

    if races.empty:
        return f"{year}: レースなし（0件）"

    all_features: list[pd.DataFrame] = []
    for _, race_row in tqdm(
        races.iterrows(), total=len(races), desc=f"ペース特徴量 {year}"
    ):
        race_key = {col: str(race_row[col]).strip() for col in RACE_KEY_COLS}
        try:
            uma_race_df = _get_horses(race_key)
            if uma_race_df.empty:
                continue

            feat_df = extractor.extract(race_key, uma_race_df)
            if feat_df.empty:
                continue

            # レースキーを付加（マージ用）
            for col in RACE_KEY_COLS:
                feat_df[f"_key_{col}"] = race_key[col]

            all_features.append(feat_df)
        except Exception as e:
            logger.warning("レース処理エラー (%s): %s", race_key, e)
            continue

    if not all_features:
        return f"{year}: 特徴量なし（0行）"

    result = pd.concat(all_features, ignore_index=False).reset_index()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    return f"{year}: {len(result)}行を保存 → {path}"


def _build_pace_worker(year: str, force_rebuild: bool) -> str:
    """並列用ワーカー関数."""
    return build_pace_year(year, force_rebuild=force_rebuild)


def build_pace_years(
    year_start: str,
    year_end: str,
    workers: int = 1,
    force_rebuild: bool = False,
) -> None:
    """複数年度のペース特徴量を構築する.

    Args:
        year_start: 開始年
        year_end: 終了年
        workers: 並列ワーカー数
        force_rebuild: 既存 parquet を無視して再構築
    """
    years = [str(y) for y in range(int(year_start), int(year_end) + 1)]
    logger.info(
        "ペース特徴量構築: %s〜%s (%d年分, workers=%d)",
        year_start, year_end, len(years), workers,
    )

    if workers <= 1:
        for year in years:
            msg = build_pace_year(year, force_rebuild=force_rebuild)
            logger.info(msg)
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for year in years:
                fut = executor.submit(_build_pace_worker, year, force_rebuild)
                futures[fut] = year

            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="年度別構築"
            ):
                year = futures[fut]
                try:
                    msg = fut.result()
                    logger.info(msg)
                except Exception as e:
                    logger.error("%s: エラー — %s", year, e)


def load_pace_years(
    year_start: str,
    year_end: str,
) -> pd.DataFrame | None:
    """年度別ペース parquet を結合してロードする.

    Returns:
        結合した DataFrame。ファイルが1つも見つからない場合は None
    """
    years = [str(y) for y in range(int(year_start), int(year_end) + 1)]
    dfs: list[pd.DataFrame] = []
    missing: list[str] = []

    for year in years:
        path = pace_parquet_path(year)
        if path.exists():
            dfs.append(pd.read_parquet(path))
        else:
            missing.append(year)

    if missing:
        logger.warning(
            "ペース特徴量が見つからない年度: %s。"
            " run_build_pace.py で構築してください。",
            missing,
        )

    if not dfs:
        return None

    result = pd.concat(dfs, ignore_index=True)
    logger.info(
        "ペース特徴量ロード: %s〜%s → %d行 (欠損年度: %s)",
        year_start, year_end, len(result), missing or "なし",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ペース特徴量の差分構築")
    parser.add_argument(
        "--start",
        default=TRAIN_START_YEAR,
        help=f"開始年（デフォルト: {TRAIN_START_YEAR}）",
    )
    parser.add_argument(
        "--end",
        default=VALID_YEAR,
        help=f"終了年（デフォルト: {VALID_YEAR}）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="並列ワーカー数（デフォルト: 1 = 直列）",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="既存 parquet を無視して再構築",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not check_connection():
        logger.error("DB接続に失敗しました")
        sys.exit(1)

    build_pace_years(
        year_start=args.start,
        year_end=args.end,
        workers=args.workers,
        force_rebuild=args.force_rebuild,
    )
    logger.info("完了!")


if __name__ == "__main__":
    main()
