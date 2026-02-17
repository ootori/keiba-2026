"""基準タイムテーブルの構築・管理.

スピード指数算出に必要な「距離×トラック種別×馬場状態別の平均走破タイム」を
過去データから集計して保持する。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.db import query_df
from src.config import DATA_DIR

logger = logging.getLogger(__name__)

# 基準タイムのキャッシュファイル
BASE_TIME_CSV = DATA_DIR / "base_time_table.csv"


def build_base_time_table(
    base_year_start: str = "2012",
    base_year_end: str = "2024",
    min_samples: int = 10,
) -> dict[tuple[str, str, str], float]:
    """基準タイムテーブルをDBから構築する.

    距離×トラック種別×馬場状態別に、1着馬の平均走破タイム（秒）を算出する。

    Args:
        base_year_start: 集計開始年
        base_year_end: 集計終了年
        min_samples: 最小サンプル数（これ未満は除外）

    Returns:
        (距離, track_type, baba_cd) → 平均秒数 の辞書
    """
    sql = """
    SELECT
        r.kyori,
        CASE WHEN CAST(r.trackcd AS int) BETWEEN 10 AND 22 THEN 'turf'
             WHEN CAST(r.trackcd AS int) BETWEEN 23 AND 29 THEN 'dirt'
             ELSE 'other' END AS track_type,
        CASE WHEN CAST(r.trackcd AS int) BETWEEN 10 AND 22 THEN r.sibababacd
             ELSE r.dirtbabacd END AS baba_cd,
        AVG(
            CAST(SUBSTRING(ur.time FROM 1 FOR 1) AS int) * 60
            + CAST(SUBSTRING(ur.time FROM 2 FOR 2) AS int)
            + CAST(SUBSTRING(ur.time FROM 4 FOR 1) AS numeric) * 0.1
        ) AS avg_time,
        COUNT(*) AS sample_count
    FROM n_uma_race ur
    JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
    WHERE ur.datakubun = '7'
      AND ur.ijyocd = '0'
      AND ur.kakuteijyuni = '01'
      AND r.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
      AND r.year >= %(base_year_start)s
      AND r.year <= %(base_year_end)s
      AND LENGTH(TRIM(ur.time)) >= 4
    GROUP BY r.kyori, track_type, baba_cd
    HAVING COUNT(*) >= %(min_samples)s
    ORDER BY r.kyori, track_type, baba_cd
    """
    params: dict[str, Any] = {
        "base_year_start": base_year_start,
        "base_year_end": base_year_end,
        "min_samples": min_samples,
    }
    df = query_df(sql, params)

    # CSV保存
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(BASE_TIME_CSV, index=False)
    logger.info("基準タイムテーブル: %d エントリ → %s", len(df), BASE_TIME_CSV)

    return _df_to_dict(df)


def load_base_time_table() -> dict[tuple[str, str, str], float]:
    """CSVファイルから基準タイムテーブルをロードする.

    Returns:
        (距離, track_type, baba_cd) → 平均秒数 の辞書

    Raises:
        FileNotFoundError: CSVが存在しない場合
    """
    if not BASE_TIME_CSV.exists():
        raise FileNotFoundError(
            f"基準タイムテーブルが見つかりません: {BASE_TIME_CSV}\n"
            "先に build_base_time_table() を実行してください。"
        )
    df = pd.read_csv(BASE_TIME_CSV, dtype=str)
    df["avg_time"] = df["avg_time"].astype(float)
    return _df_to_dict(df)


def get_or_build_base_time(
    base_year_start: str = "2012",
    base_year_end: str = "2024",
) -> dict[tuple[str, str, str], float]:
    """基準タイムテーブルをロード。なければ構築する.

    Args:
        base_year_start: 集計開始年
        base_year_end: 集計終了年

    Returns:
        基準タイム辞書
    """
    try:
        return load_base_time_table()
    except FileNotFoundError:
        logger.info("基準タイムテーブルをDBから構築します...")
        return build_base_time_table(base_year_start, base_year_end)


def calc_speed_index(
    time_sec: float,
    distance: str,
    track_type_str: str,
    baba_cd: str,
    base_time_dict: dict[tuple[str, str, str], float],
) -> float:
    """スピード指数を算出する.

    スピード指数 = (基準タイム - 走破タイム) / 基準タイム * 1000

    Args:
        time_sec: 走破タイム（秒）
        distance: 距離（文字列）
        track_type_str: 'turf' or 'dirt'
        baba_cd: 馬場状態コード
        base_time_dict: 基準タイム辞書

    Returns:
        スピード指数（高いほど速い）
    """
    key = (distance, track_type_str, baba_cd)
    base_time = base_time_dict.get(key)
    if base_time is None or base_time == 0:
        return 0.0
    return (base_time - time_sec) / base_time * 1000


def _df_to_dict(df: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    """DataFrameを辞書に変換する."""
    result: dict[tuple[str, str, str], float] = {}
    for _, row in df.iterrows():
        key = (str(row["kyori"]).strip(), str(row["track_type"]).strip(), str(row["baba_cd"]).strip())
        result[key] = float(row["avg_time"])
    return result
