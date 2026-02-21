"""Step 0: ベースデータセット構築.

条件戦の出走データに累計収得賞金・昇格リスクフラグ・人気情報を付与し、
分析用のベースデータセットを構築する。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RACE_KEY_COLS
from src.db import query_df

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# ---------------------------------------------------------------------------
# JRA収得賞金ルール
# ---------------------------------------------------------------------------
# 条件戦では1着のみ収得賞金が加算される（固定額）
# 重賞では1着: 本賞金の50%, 2着: 2着本賞金の50%
# OP特別/リステッドでは1着のみ: 本賞金の50%
# ここでは条件戦の固定加算額を使う（千円単位）
SYUTOKU_ADD_BY_CLASS = {
    "maiden": 4000,    # 新馬・未勝利: 400万円
    "1win": 5000,      # 1勝クラス: 500万円
    "2win": 6000,      # 2勝クラス: 600万円
    "3win": 9000,      # 3勝クラス: 900万円
}

# クラス昇格の収得賞金閾値（千円単位）
CLASS_THRESHOLDS = {
    "1win": 5010,      # 1勝 → 2勝: 501万円以上
    "2win": 10010,     # 2勝 → 3勝: 1001万円以上
    "3win": 16010,     # 3勝 → OP: 1601万円以上
}


def _map_class_label(jyokencd5: str, gradecd: str) -> str:
    """JyokenCD5 + GradeCD からクラスラベルを返す."""
    if gradecd in ("A", "B", "C", "D", "F", "G", "H"):
        return "graded"
    cd = int(jyokencd5) if jyokencd5.strip().isdigit() else 0
    if cd in (701, 702, 703):
        return "maiden"
    if 1 <= cd <= 5:
        return "1win"
    if 6 <= cd <= 10:
        return "2win"
    if 11 <= cd <= 16:
        return "3win"
    if cd == 999:
        return "open"
    return "other"


def build_base_dataset(year_start: int = 2015, year_end: int = 2025) -> pd.DataFrame:
    """ベースデータセットを構築する.

    Args:
        year_start: 開始年
        year_end: 終了年

    Returns:
        分析用ベースデータセット
    """
    logger.info("Step 0: ベースデータ構築 (%d-%d)", year_start, year_end)

    # ------------------------------------------------------------------
    # 1. 出走データ取得（n_uma_race + n_race）
    # ------------------------------------------------------------------
    logger.info("  出走データ取得中...")
    sql_entries = """
    SELECT
        ur.year, ur.monthday, ur.jyocd, ur.kaiji, ur.nichiji, ur.racenum,
        ur.umaban,
        ur.kettonum,
        ur.banusicode,
        ur.banusiname,
        ur.kisyucode,
        ur.sexcd,
        ur.barei,
        ur.kakuteijyuni,
        ur.honsyokin AS uma_honsyokin,
        ur.ijyocd,
        r.jyokencd5,
        r.gradecd,
        r.kyori,
        r.trackcd,
        r.sibababacd,
        r.syubetucd,
        r.syussotosu,
        r.honsyokin1 AS race_prize_1st,
        r.honsyokin2 AS race_prize_2nd,
        r.honsyokin3 AS race_prize_3rd
    FROM n_uma_race ur
    JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
    WHERE ur.datakubun = '7'
      AND ur.ijyocd = '0'
      AND ur.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
      AND CAST(ur.year AS integer) BETWEEN %(year_start)s AND %(year_end)s
      AND r.datakubun = '7'
    """
    df = query_df(sql_entries, {"year_start": year_start, "year_end": year_end})
    logger.info("  出走データ: %d 行", len(df))

    # ------------------------------------------------------------------
    # 2. 人気・オッズ取得
    # ------------------------------------------------------------------
    logger.info("  オッズデータ取得中...")
    sql_odds = """
    SELECT
        year, monthday, jyocd, kaiji, nichiji, racenum, umaban,
        tanninki, tanodds
    FROM n_odds_tanpuku
    WHERE CAST(year AS integer) BETWEEN %(year_start)s AND %(year_end)s
      AND tanninki ~ '^[0-9]+$'
      AND tanodds ~ '^[0-9]+$'
    """
    odds_df = query_df(sql_odds, {"year_start": year_start, "year_end": year_end})
    odds_df["tanninki"] = odds_df["tanninki"].astype(int)
    odds_df["tanodds"] = odds_df["tanodds"].astype(float) / 10.0  # 10倍値→実オッズ

    # 結合
    df = df.merge(
        odds_df,
        on=RACE_KEY_COLS + ["umaban"],
        how="left",
    )
    logger.info("  オッズ結合後: %d 行 (人気あり: %d)", len(df), df["tanninki"].notna().sum())

    # ------------------------------------------------------------------
    # 3. 数値変換
    # ------------------------------------------------------------------
    for col in ["kakuteijyuni", "uma_honsyokin", "race_prize_1st",
                "race_prize_2nd", "race_prize_3rd", "kyori", "syussotosu"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 年度を数値化
    df["year_int"] = df["year"].astype(int)

    # ------------------------------------------------------------------
    # 4. クラスラベル付与
    # ------------------------------------------------------------------
    df["class_label"] = df.apply(
        lambda r: _map_class_label(str(r["jyokencd5"]).strip(), str(r["gradecd"]).strip()),
        axis=1,
    )
    logger.info("  クラス分布:\n%s", df["class_label"].value_counts().to_string())

    # ------------------------------------------------------------------
    # 5. 累計収得賞金の推計（各馬のレース時点）
    # ------------------------------------------------------------------
    logger.info("  累計収得賞金を推計中...")
    df = _calc_cumulative_syutoku(df)

    # ------------------------------------------------------------------
    # 6. 昇格リスクフラグ
    # ------------------------------------------------------------------
    df["promotion_risk"] = df.apply(_check_promotion_risk, axis=1)

    # 条件戦に限定した昇格リスクの集計
    cond_mask = df["class_label"].isin(["1win", "2win", "3win"])
    cond_df = df[cond_mask]
    logger.info("  条件戦: %d 行", len(cond_df))
    logger.info("  昇格リスクあり: %d 行 (%.1f%%)",
                cond_df["promotion_risk"].sum(),
                100 * cond_df["promotion_risk"].mean())

    # ------------------------------------------------------------------
    # 7. レース内頭数を付与
    # ------------------------------------------------------------------
    race_key = RACE_KEY_COLS
    field_size = df.groupby(race_key)["umaban"].transform("count")
    df["field_size"] = field_size

    # ------------------------------------------------------------------
    # 8. 保存
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "base_dataset.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("  保存: %s (%d 行)", out_path, len(df))

    return df


def _calc_cumulative_syutoku(df: pd.DataFrame) -> pd.DataFrame:
    """各馬のレース時点での累計収得賞金を推計する.

    収得賞金ルール:
    - 条件戦1着: クラスごとの固定額を加算
    - 重賞1着: 本賞金の50%, 重賞2着: 2着本賞金の50%
    - OP特別/リステッド1着: 本賞金の50%
    - その他着順: 加算なし
    """
    # ソート: 時系列順
    df = df.sort_values(["kettonum", "year", "monthday"]).copy()

    # 各レースでの収得賞金加算額を計算
    syutoku_add = np.zeros(len(df), dtype=np.float64)

    for idx, row in df.iterrows():
        jyuni = row["kakuteijyuni"]
        if pd.isna(jyuni) or jyuni < 1:
            continue

        cl = row["class_label"]
        jyuni = int(jyuni)

        if cl in ("maiden", "1win", "2win", "3win"):
            # 条件戦: 1着のみ固定額
            if jyuni == 1:
                syutoku_add[df.index.get_loc(idx)] = SYUTOKU_ADD_BY_CLASS.get(cl, 0)
        elif cl == "graded":
            # 重賞: 1着=本賞金の50%, 2着=2着本賞金の50%
            if jyuni == 1:
                p1 = row["race_prize_1st"]
                if pd.notna(p1):
                    syutoku_add[df.index.get_loc(idx)] = p1 * 0.5
            elif jyuni == 2:
                p2 = row["race_prize_2nd"]
                if pd.notna(p2):
                    syutoku_add[df.index.get_loc(idx)] = p2 * 0.5
        elif cl == "open":
            # OP特別/リステッド: 1着のみ本賞金の50%
            if jyuni == 1:
                p1 = row["race_prize_1st"]
                if pd.notna(p1):
                    syutoku_add[df.index.get_loc(idx)] = p1 * 0.5

    df["syutoku_add"] = syutoku_add

    # 馬ごとの累計（当該レース時点: 自分自身のレースは含めない）
    df["cum_syutoku_before"] = (
        df.groupby("kettonum")["syutoku_add"]
        .transform(lambda s: s.cumsum().shift(1, fill_value=0))
    )
    # 当該レース結果を含む累計
    df["cum_syutoku_after"] = df["cum_syutoku_before"] + df["syutoku_add"]

    return df


def _check_promotion_risk(row: pd.Series) -> bool:
    """このレースで1着を取ると昇格するかを判定する."""
    cl = row["class_label"]
    if cl not in CLASS_THRESHOLDS:
        return False

    threshold = CLASS_THRESHOLDS[cl]
    cum_before = row.get("cum_syutoku_before", 0)
    add_amount = SYUTOKU_ADD_BY_CLASS.get(cl, 0)

    # 1着を取った場合の累計
    after_win = cum_before + add_amount
    return after_win >= threshold


def load_base_dataset() -> pd.DataFrame:
    """保存済みベースデータセットをロードする."""
    path = OUTPUT_DIR / "base_dataset.parquet"
    if not path.exists():
        raise FileNotFoundError(f"ベースデータが見つかりません: {path}")
    return pd.read_parquet(path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    df = build_base_dataset()
    print(f"\n完了: {len(df)} 行のベースデータセットを構築しました")

    # サマリー表示
    cond = df[df["class_label"].isin(["1win", "2win", "3win"])]
    print(f"\n条件戦: {len(cond)} 行")
    print(f"  昇格リスクあり: {cond['promotion_risk'].sum()} ({100*cond['promotion_risk'].mean():.1f}%)")
    print(f"\n人気分布:")
    print(cond["tanninki"].describe())
