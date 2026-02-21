"""種牡馬別 × 芝ダ × 馬場状態別の単勝回収率分析.

オッズ歪み補正への適用可否を検討するための分析スクリプト。
"""

from __future__ import annotations

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.db import query_df
from src.config import JRA_JYO_CODES

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.1f}".format)


def _jyo_filter(prefix: str = "") -> str:
    codes = ", ".join(f"'{c}'" for c in JRA_JYO_CODES)
    return f"{prefix}jyocd IN ({codes})"


def analyze_sire_track_baba_roi(
    year_start: str = "2019",
    year_end: str = "2024",
    min_samples: int = 100,
) -> pd.DataFrame:
    """種牡馬別 × 芝/ダ × 馬場状態別の単勝回収率を算出する.

    Args:
        year_start: 集計開始年
        year_end: 集計終了年
        min_samples: 表示最小サンプル数

    Returns:
        種牡馬 × tracktype × baba 別の集計DataFrame
    """
    # n_sanku から父馬の繁殖登録番号を取得し、n_hansyoku から名前を取得
    # n_race の trackcd で芝/ダ判定、sibababacd/dirtbabacd で馬場状態
    sql = f"""
    SELECT
        s.fnum AS sire_id,
        h.bamei AS sire_name,
        CASE
            WHEN CAST(r.trackcd AS int) BETWEEN 10 AND 22 THEN 'turf'
            WHEN CAST(r.trackcd AS int) BETWEEN 23 AND 29 THEN 'dirt'
            ELSE 'other'
        END AS track_type,
        CASE
            WHEN CAST(r.trackcd AS int) BETWEEN 10 AND 22 THEN r.sibababacd
            WHEN CAST(r.trackcd AS int) BETWEEN 23 AND 29 THEN r.dirtbabacd
            ELSE '0'
        END AS baba_cd,
        COUNT(*) AS runs,
        SUM(CASE WHEN ur.kakuteijyuni ~ '^[0-9]+$'
                      AND CAST(ur.kakuteijyuni AS int) = 1
            THEN 1 ELSE 0 END) AS wins,
        SUM(CASE WHEN ur.kakuteijyuni ~ '^[0-9]+$'
                      AND CAST(ur.kakuteijyuni AS int) <= 3
            THEN 1 ELSE 0 END) AS top3,
        SUM(CASE WHEN ur.kakuteijyuni ~ '^[0-9]+$'
                      AND CAST(ur.kakuteijyuni AS int) = 1
            THEN COALESCE(CAST(o.tanodds AS numeric) / 10.0, 0)
            ELSE 0 END) * 100 AS total_pay
    FROM n_uma_race ur
    JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
    JOIN n_sanku s ON ur.kettonum = s.kettonum
    JOIN n_hansyoku h ON s.fnum = h.hansyokunum
    LEFT JOIN n_odds_tanpuku o
        ON ur.year = o.year AND ur.monthday = o.monthday
        AND ur.jyocd = o.jyocd AND ur.kaiji = o.kaiji
        AND ur.nichiji = o.nichiji AND ur.racenum = o.racenum
        AND ur.umaban = o.umaban
    WHERE ur.datakubun = '7' AND ur.ijyocd = '0'
      AND ur.year BETWEEN %(start)s AND %(end)s
      AND {_jyo_filter("ur.")}
      AND r.trackcd ~ '^[0-9]+$'
    GROUP BY s.fnum, h.bamei, track_type, baba_cd
    HAVING COUNT(*) >= %(min_samples)s
    ORDER BY s.fnum, track_type, baba_cd
    """
    print(f"クエリ実行中... ({year_start}〜{year_end}年, min_samples={min_samples})")
    df = query_df(sql, {
        "start": year_start,
        "end": year_end,
        "min_samples": min_samples,
    })

    if df.empty:
        print("データなし")
        return df

    # 回収率・勝率・複勝率を計算
    df["win_rate"] = df["wins"] / df["runs"] * 100
    df["top3_rate"] = df["top3"] / df["runs"] * 100
    df["total_bet"] = df["runs"] * 100
    df["roi"] = df["total_pay"] / df["total_bet"] * 100

    # 馬場状態コードを名前に変換
    baba_names = {"1": "良", "2": "稍重", "3": "重", "4": "不良"}
    df["baba"] = df["baba_cd"].map(baba_names).fillna("不明")

    # 名前整形
    df["sire_name"] = df["sire_name"].str.strip()

    return df


def print_summary(df: pd.DataFrame) -> None:
    """サマリーを表示する."""
    if df.empty:
        return

    # --- 1. 全体の芝/ダート × 馬場状態別 回収率 ---
    print("\n" + "=" * 80)
    print("■ 全体: 芝/ダート × 馬場状態別 回収率")
    print("=" * 80)
    overall = df.groupby(["track_type", "baba"]).agg(
        runs=("runs", "sum"),
        wins=("wins", "sum"),
        top3=("top3", "sum"),
        total_pay=("total_pay", "sum"),
    ).reset_index()
    overall["win_rate"] = overall["wins"] / overall["runs"] * 100
    overall["top3_rate"] = overall["top3"] / overall["runs"] * 100
    overall["roi"] = overall["total_pay"] / (overall["runs"] * 100) * 100
    print(overall[["track_type", "baba", "runs", "win_rate", "top3_rate", "roi"]].to_string(index=False))

    # --- 2. 種牡馬別の全体回収率（上位30） ---
    print("\n" + "=" * 80)
    print("■ 種牡馬別 全体回収率 TOP30 (出走数500以上)")
    print("=" * 80)
    sire_total = df.groupby(["sire_id", "sire_name"]).agg(
        runs=("runs", "sum"),
        wins=("wins", "sum"),
        top3=("top3", "sum"),
        total_pay=("total_pay", "sum"),
    ).reset_index()
    sire_total["win_rate"] = sire_total["wins"] / sire_total["runs"] * 100
    sire_total["top3_rate"] = sire_total["top3"] / sire_total["runs"] * 100
    sire_total["roi"] = sire_total["total_pay"] / (sire_total["runs"] * 100) * 100
    sire_top = sire_total[sire_total["runs"] >= 500].sort_values("roi", ascending=False).head(30)
    print(sire_top[["sire_name", "runs", "win_rate", "top3_rate", "roi"]].to_string(index=False))

    # --- 3. 種牡馬×芝ダ×馬場別のピボット（主要種牡馬） ---
    # 出走数2000以上の主要種牡馬に絞る
    major_sires = sire_total[sire_total["runs"] >= 2000]["sire_id"].tolist()
    df_major = df[df["sire_id"].isin(major_sires)].copy()

    if df_major.empty:
        print("\n主要種牡馬データなし")
        return

    print("\n" + "=" * 80)
    print("■ 主要種牡馬（出走2000以上）× 芝/ダート × 馬場状態 回収率")
    print("=" * 80)

    # 芝
    print("\n--- 芝 ---")
    turf = df_major[df_major["track_type"] == "turf"].copy()
    if not turf.empty:
        pivot_turf = turf.pivot_table(
            index="sire_name",
            columns="baba",
            values=["roi", "runs"],
            aggfunc="first",
        )
        # フラット化
        pivot_turf.columns = [f"{c[0]}_{c[1]}" for c in pivot_turf.columns]
        # 全体ROIも追加
        turf_total = turf.groupby("sire_name").agg(
            total_runs=("runs", "sum"),
            total_pay=("total_pay", "sum"),
        )
        turf_total["total_roi"] = turf_total["total_pay"] / (turf_total["total_runs"] * 100) * 100
        pivot_turf = pivot_turf.join(turf_total[["total_runs", "total_roi"]])
        pivot_turf = pivot_turf.sort_values("total_roi", ascending=False)
        # 表示カラムを整理
        display_cols = []
        for baba in ["良", "稍重", "重", "不良"]:
            if f"roi_{baba}" in pivot_turf.columns:
                display_cols.extend([f"runs_{baba}", f"roi_{baba}"])
        display_cols.extend(["total_runs", "total_roi"])
        available = [c for c in display_cols if c in pivot_turf.columns]
        print(pivot_turf[available].to_string())

    # ダート
    print("\n--- ダート ---")
    dirt = df_major[df_major["track_type"] == "dirt"].copy()
    if not dirt.empty:
        pivot_dirt = dirt.pivot_table(
            index="sire_name",
            columns="baba",
            values=["roi", "runs"],
            aggfunc="first",
        )
        pivot_dirt.columns = [f"{c[0]}_{c[1]}" for c in pivot_dirt.columns]
        dirt_total = dirt.groupby("sire_name").agg(
            total_runs=("runs", "sum"),
            total_pay=("total_pay", "sum"),
        )
        dirt_total["total_roi"] = dirt_total["total_pay"] / (dirt_total["total_runs"] * 100) * 100
        pivot_dirt = pivot_dirt.join(dirt_total[["total_runs", "total_roi"]])
        pivot_dirt = pivot_dirt.sort_values("total_roi", ascending=False)
        display_cols = []
        for baba in ["良", "稍重", "重", "不良"]:
            if f"roi_{baba}" in pivot_dirt.columns:
                display_cols.extend([f"runs_{baba}", f"roi_{baba}"])
        display_cols.extend(["total_runs", "total_roi"])
        available = [c for c in display_cols if c in pivot_dirt.columns]
        print(pivot_dirt[available].to_string())

    # --- 4. 芝ダ間のROI差が大きい種牡馬 ---
    print("\n" + "=" * 80)
    print("■ 芝/ダートでROI差が大きい種牡馬（出走500以上 各コース）")
    print("=" * 80)
    track_pivot = df.groupby(["sire_name", "track_type"]).agg(
        runs=("runs", "sum"),
        total_pay=("total_pay", "sum"),
    ).reset_index()
    track_pivot["roi"] = track_pivot["total_pay"] / (track_pivot["runs"] * 100) * 100
    track_wide = track_pivot.pivot(index="sire_name", columns="track_type", values=["roi", "runs"])
    track_wide.columns = [f"{c[0]}_{c[1]}" for c in track_wide.columns]
    if "roi_turf" in track_wide.columns and "roi_dirt" in track_wide.columns:
        track_wide = track_wide.dropna(subset=["roi_turf", "roi_dirt"])
        min_runs = 500
        mask = True
        for c in ["runs_turf", "runs_dirt"]:
            if c in track_wide.columns:
                mask = mask & (track_wide[c] >= min_runs)
        track_wide = track_wide[mask].copy()
        track_wide["roi_diff"] = track_wide["roi_turf"] - track_wide["roi_dirt"]
        track_wide = track_wide.sort_values("roi_diff", ascending=False)
        available = [c for c in ["runs_turf", "roi_turf", "runs_dirt", "roi_dirt", "roi_diff"]
                     if c in track_wide.columns]
        print(track_wide[available].to_string())

    # --- 5. 重馬場（重+不良）でROI差が大きい種牡馬 ---
    print("\n" + "=" * 80)
    print("■ 良馬場 vs 重馬場（重+不良）でROI差が大きい種牡馬")
    print("=" * 80)
    df2 = df.copy()
    df2["baba_group"] = df2["baba_cd"].map(
        {"1": "good", "2": "good", "3": "heavy", "4": "heavy"}
    )
    baba_grp = df2.groupby(["sire_name", "baba_group"]).agg(
        runs=("runs", "sum"),
        total_pay=("total_pay", "sum"),
    ).reset_index()
    baba_grp["roi"] = baba_grp["total_pay"] / (baba_grp["runs"] * 100) * 100
    baba_wide = baba_grp.pivot(index="sire_name", columns="baba_group", values=["roi", "runs"])
    baba_wide.columns = [f"{c[0]}_{c[1]}" for c in baba_wide.columns]
    if "roi_good" in baba_wide.columns and "roi_heavy" in baba_wide.columns:
        baba_wide = baba_wide.dropna(subset=["roi_good", "roi_heavy"])
        mask = True
        for c in ["runs_good", "runs_heavy"]:
            if c in baba_wide.columns:
                mask = mask & (baba_wide[c] >= 200)
        baba_wide = baba_wide[mask].copy()
        baba_wide["roi_diff"] = baba_wide["roi_heavy"] - baba_wide["roi_good"]
        baba_wide = baba_wide.sort_values("roi_diff", ascending=False)
        available = [c for c in ["runs_good", "roi_good", "runs_heavy", "roi_heavy", "roi_diff"]
                     if c in baba_wide.columns]
        print(baba_wide[available].to_string())

    # --- 6. 芝×重馬場、ダート×重馬場での種牡馬別ROI ---
    print("\n" + "=" * 80)
    print("■ 芝×馬場状態別 ROI TOP20（各馬場状態で出走200以上）")
    print("=" * 80)
    for baba_name in ["良", "稍重", "重", "不良"]:
        subset = df[(df["track_type"] == "turf") & (df["baba"] == baba_name) & (df["runs"] >= 200)]
        if not subset.empty:
            top = subset.sort_values("roi", ascending=False).head(20)
            print(f"\n  [芝・{baba_name}]")
            print(top[["sire_name", "runs", "win_rate", "top3_rate", "roi"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("■ ダート×馬場状態別 ROI TOP20（各馬場状態で出走200以上）")
    print("=" * 80)
    for baba_name in ["良", "稍重", "重", "不良"]:
        subset = df[(df["track_type"] == "dirt") & (df["baba"] == baba_name) & (df["runs"] >= 200)]
        if not subset.empty:
            top = subset.sort_values("roi", ascending=False).head(20)
            print(f"\n  [ダート・{baba_name}]")
            print(top[["sire_name", "runs", "win_rate", "top3_rate", "roi"]].to_string(index=False))


if __name__ == "__main__":
    df = analyze_sire_track_baba_roi(
        year_start="2019",
        year_end="2024",
        min_samples=100,
    )
    print_summary(df)
