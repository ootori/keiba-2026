"""厩舎（調教師）の人気別成績分析 — 戦略的厩舎の特定.

人気別の全体統計（勝率・連対率・複勝率）を算出し、
人気以上の成績を安定的に出している「戦略的厩舎」を特定する。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.db import query_df


# ============================================================
# 1. データ取得
# ============================================================

def fetch_race_results(year_start: int = 2020, year_end: int = 2025) -> pd.DataFrame:
    """確定成績 + 人気順を結合して取得."""
    sql = """
    SELECT
        ur.year,
        ur.chokyosicode,
        ur.chokyosiryakusyo,
        ur.kakuteijyuni,
        o.tanninki,
        o.tanodds
    FROM n_uma_race ur
    JOIN n_odds_tanpuku o
      ON ur.year = o.year AND ur.monthday = o.monthday
     AND ur.jyocd = o.jyocd AND ur.kaiji = o.kaiji
     AND ur.nichiji = o.nichiji AND ur.racenum = o.racenum
     AND ur.umaban = o.umaban
    WHERE ur.datakubun = '7'
      AND ur.ijyocd = '0'
      AND ur.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
      AND CAST(ur.year AS integer) >= %(year_start)s
      AND CAST(ur.year AS integer) <= %(year_end)s
      AND ur.kakuteijyuni ~ '^[0-9]+$'
      AND o.tanninki ~ '^[0-9]+$'
      AND o.tanodds ~ '^[0-9]+$'
    """
    df = query_df(sql, {"year_start": year_start, "year_end": year_end})
    df["kakuteijyuni"] = df["kakuteijyuni"].astype(int)
    df["tanninki"] = df["tanninki"].astype(int)
    df["tanodds"] = df["tanodds"].astype(int) / 10.0
    df["chokyosiryakusyo"] = df["chokyosiryakusyo"].str.strip()
    return df


# ============================================================
# 2. 人気別 全体統計
# ============================================================

def calc_ninki_stats(df: pd.DataFrame) -> pd.DataFrame:
    """人気別の勝率・連対率・複勝率."""
    stats = []
    for ninki in range(1, 19):
        sub = df[df["tanninki"] == ninki]
        n = len(sub)
        if n == 0:
            continue
        win = (sub["kakuteijyuni"] == 1).sum()
        top2 = (sub["kakuteijyuni"] <= 2).sum()
        top3 = (sub["kakuteijyuni"] <= 3).sum()
        stats.append({
            "人気": ninki,
            "出走数": n,
            "勝率": win / n,
            "連対率": top2 / n,
            "複勝率": top3 / n,
            "平均着順": sub["kakuteijyuni"].mean(),
            "平均オッズ": sub["tanodds"].mean(),
        })
    return pd.DataFrame(stats)


# ============================================================
# 3. 厩舎別×人気帯別の成績
# ============================================================

def _ninki_band(ninki: int) -> str:
    """人気を帯域に分類."""
    if ninki <= 3:
        return "A_上位人気(1-3番人気)"
    elif ninki <= 6:
        return "B_中位人気(4-6番人気)"
    elif ninki <= 9:
        return "C_穴人気(7-9番人気)"
    else:
        return "D_大穴(10番人気以下)"


def calc_trainer_ninki_band_stats(df: pd.DataFrame, min_runs: int = 30) -> pd.DataFrame:
    """厩舎別×人気帯別の成績を算出."""
    df = df.copy()
    df["ninki_band"] = df["tanninki"].apply(_ninki_band)

    grouped = df.groupby(["chokyosicode", "chokyosiryakusyo", "ninki_band"])
    rows = []
    for (code, name, band), g in grouped:
        n = len(g)
        if n < min_runs:
            continue
        win = (g["kakuteijyuni"] == 1).sum()
        top2 = (g["kakuteijyuni"] <= 2).sum()
        top3 = (g["kakuteijyuni"] <= 3).sum()
        # 人気以上の着順を取った割合 (着順 <= 人気)
        beat_ninki = (g["kakuteijyuni"] <= g["tanninki"]).sum()
        rows.append({
            "調教師CD": code,
            "調教師名": name,
            "人気帯": band,
            "出走数": n,
            "勝率": win / n,
            "連対率": top2 / n,
            "複勝率": top3 / n,
            "平均着順": g["kakuteijyuni"].mean(),
            "人気以上率": beat_ninki / n,  # 着順 <= 人気 の割合
        })
    return pd.DataFrame(rows)


# ============================================================
# 4. 「戦略的厩舎」スコアリング
# ============================================================

def find_strategic_trainers(
    df: pd.DataFrame,
    ninki_stats: pd.DataFrame,
    min_runs_total: int = 100,
    min_runs_unpopular: int = 30,
) -> pd.DataFrame:
    """人気以上の成績を安定的に出している厩舎を特定.

    スコアリング基準:
    - 人気薄(7番人気以下)での複勝率が全体平均を有意に上回る
    - 「着順 ≤ 人気」の割合(人気以上率)が高い
    - 穴馬での単勝回収率が高い
    """
    df = df.copy()
    df["ninki_band"] = df["tanninki"].apply(_ninki_band)

    # 全体の人気別複勝率をルックアップ
    avg_fukusho = dict(zip(
        ninki_stats["人気"].astype(int),
        ninki_stats["複勝率"],
    ))

    # 厩舎別集計
    trainers = df.groupby(["chokyosicode", "chokyosiryakusyo"])
    results = []
    for (code, name), g in trainers:
        total = len(g)
        if total < min_runs_total:
            continue

        # ---- 全体成績 ----
        total_win = (g["kakuteijyuni"] == 1).sum()
        total_top3 = (g["kakuteijyuni"] <= 3).sum()

        # ---- 人気薄(7番人気以下)での成績 ----
        unpop = g[g["tanninki"] >= 7]
        n_unpop = len(unpop)
        if n_unpop < min_runs_unpopular:
            continue

        unpop_win = (unpop["kakuteijyuni"] == 1).sum()
        unpop_top2 = (unpop["kakuteijyuni"] <= 2).sum()
        unpop_top3 = (unpop["kakuteijyuni"] <= 3).sum()
        unpop_beat = (unpop["kakuteijyuni"] <= unpop["tanninki"]).sum()

        # 期待複勝率（人気分布の加重平均）
        expected_fukusho = unpop["tanninki"].map(
            lambda x: avg_fukusho.get(x, 0.05)
        ).mean()

        actual_fukusho = unpop_top3 / n_unpop if n_unpop > 0 else 0

        # 単勝回収率（人気薄）
        unpop_win_mask = unpop["kakuteijyuni"] == 1
        if unpop_win_mask.any():
            unpop_return = (unpop.loc[unpop_win_mask, "tanodds"] * 100).sum()
        else:
            unpop_return = 0
        unpop_roi = unpop_return / (n_unpop * 100) if n_unpop > 0 else 0

        # ---- 超人気薄(10番人気以下)での成績 ----
        very_unpop = g[g["tanninki"] >= 10]
        n_very_unpop = len(very_unpop)
        very_unpop_top3 = (very_unpop["kakuteijyuni"] <= 3).sum() if n_very_unpop > 0 else 0

        # ---- スコア算出 ----
        # 複勝率の超過率（実績/期待）
        fukusho_ratio = actual_fukusho / expected_fukusho if expected_fukusho > 0 else 1.0
        # 人気以上率
        beat_rate = unpop_beat / n_unpop if n_unpop > 0 else 0
        # 総合スコア: 複勝超過率 × 人気以上率 × 出走規模補正
        score = fukusho_ratio * beat_rate * np.log1p(n_unpop) / np.log1p(30)

        results.append({
            "調教師CD": code,
            "調教師名": name,
            "全出走数": total,
            "全勝率": total_win / total,
            "全複勝率": total_top3 / total,
            "穴馬出走数(7人気↓)": n_unpop,
            "穴馬勝率": unpop_win / n_unpop,
            "穴馬連対率": unpop_top2 / n_unpop,
            "穴馬複勝率": actual_fukusho,
            "穴馬期待複勝率": expected_fukusho,
            "穴馬複勝超過率": fukusho_ratio,
            "穴馬人気以上率": beat_rate,
            "穴馬単勝回収率": unpop_roi,
            "大穴出走数(10人気↓)": n_very_unpop,
            "大穴複勝率": very_unpop_top3 / n_very_unpop if n_very_unpop > 0 else np.nan,
            "戦略スコア": score,
        })

    result_df = pd.DataFrame(results).sort_values("戦略スコア", ascending=False)
    return result_df


# ============================================================
# 5. 人気帯別の「期待値を超える厩舎」
# ============================================================

def find_value_trainers_by_band(
    df: pd.DataFrame,
    ninki_stats: pd.DataFrame,
    min_runs: int = 50,
) -> dict[str, pd.DataFrame]:
    """人気帯ごとに、平均を上回る厩舎をリスト化."""
    df = df.copy()
    df["ninki_band"] = df["tanninki"].apply(_ninki_band)

    # 人気帯別の全体平均複勝率
    band_avg = df.groupby("ninki_band").apply(
        lambda g: (g["kakuteijyuni"] <= 3).mean()
    ).to_dict()

    results = {}
    for band in sorted(df["ninki_band"].unique()):
        avg = band_avg.get(band, 0)
        sub = df[df["ninki_band"] == band]
        grouped = sub.groupby(["chokyosicode", "chokyosiryakusyo"])
        rows = []
        for (code, name), g in grouped:
            n = len(g)
            if n < min_runs:
                continue
            win = (g["kakuteijyuni"] == 1).sum()
            top2 = (g["kakuteijyuni"] <= 2).sum()
            top3 = (g["kakuteijyuni"] <= 3).sum()
            fukusho = top3 / n
            beat = (g["kakuteijyuni"] <= g["tanninki"]).sum()

            # 単勝回収率
            win_mask = g["kakuteijyuni"] == 1
            roi = (g.loc[win_mask, "tanodds"] * 100).sum() / (n * 100) if n > 0 else 0

            rows.append({
                "調教師名": name,
                "出走数": n,
                "勝率": win / n,
                "連対率": top2 / n,
                "複勝率": fukusho,
                "全体平均複勝率": avg,
                "複勝率差": fukusho - avg,
                "人気以上率": beat / n,
                "単勝回収率": roi,
            })
        band_df = pd.DataFrame(rows).sort_values("複勝率差", ascending=False)
        results[band] = band_df
    return results


# ============================================================
# main
# ============================================================

def main():
    print("=" * 70)
    print("厩舎（調教師）人気別成績分析")
    print("=" * 70)

    print("\n[1/4] データ取得中 (2020-2025年)...")
    df = fetch_race_results(2020, 2025)
    print(f"  取得レコード数: {len(df):,}")

    # ---- 人気別全体統計 ----
    print("\n" + "=" * 70)
    print("[2/4] 人気別 全体統計（勝率・連対率・複勝率）")
    print("=" * 70)
    ns = calc_ninki_stats(df)
    pd.set_option("display.float_format", "{:.1%}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    # 平均着順とオッズだけフォーマットを変える
    display_ns = ns.copy()
    display_ns["平均着順"] = display_ns["平均着順"].map("{:.1f}".format)
    display_ns["平均オッズ"] = display_ns["平均オッズ"].map("{:.1f}倍".format)
    display_ns["勝率"] = display_ns["勝率"].map("{:.1%}".format)
    display_ns["連対率"] = display_ns["連対率"].map("{:.1%}".format)
    display_ns["複勝率"] = display_ns["複勝率"].map("{:.1%}".format)
    print(display_ns.to_string(index=False))

    # ---- 戦略的厩舎ランキング ----
    print("\n" + "=" * 70)
    print("[3/4] 戦略的厩舎ランキング TOP30")
    print("      (人気薄で期待以上の成績を出す厩舎)")
    print("=" * 70)
    strategic = find_strategic_trainers(df, ns)
    pd.set_option("display.float_format", "{:.1%}".format)
    top30 = strategic.head(30)
    display_cols = [
        "調教師名", "全出走数", "全勝率", "全複勝率",
        "穴馬出走数(7人気↓)", "穴馬勝率", "穴馬複勝率",
        "穴馬期待複勝率", "穴馬複勝超過率", "穴馬人気以上率",
        "穴馬単勝回収率", "大穴出走数(10人気↓)", "大穴複勝率",
        "戦略スコア",
    ]
    display_top30 = top30[display_cols].copy()
    # 超過率とスコアはパーセントではなく倍率表示
    for c in ["穴馬複勝超過率", "戦略スコア"]:
        display_top30[c] = display_top30[c].map("{:.2f}".format)
    for c in ["穴馬単勝回収率"]:
        display_top30[c] = display_top30[c].map("{:.0%}".format)
    print(display_top30.to_string(index=False))

    # ---- 人気帯別の「期待値を超える厩舎」 ----
    print("\n" + "=" * 70)
    print("[4/4] 人気帯別 — 期待値を超える厩舎 TOP15")
    print("=" * 70)
    value_by_band = find_value_trainers_by_band(df, ns)
    for band, band_df in value_by_band.items():
        if len(band_df) == 0:
            continue
        print(f"\n--- {band} ---")
        display_band = band_df.head(15).copy()
        for c in ["勝率", "連対率", "複勝率", "全体平均複勝率", "複勝率差", "人気以上率"]:
            if c in display_band.columns:
                display_band[c] = display_band[c].map("{:.1%}".format)
        display_band["単勝回収率"] = display_band["単勝回収率"].map("{:.0%}".format)
        print(display_band.to_string(index=False))

    # ---- CSV保存 ----
    os.makedirs("analysis", exist_ok=True)
    strategic.to_csv("analysis/strategic_trainers.csv", index=False, encoding="utf-8-sig")
    ns.to_csv("analysis/ninki_stats.csv", index=False, encoding="utf-8-sig")
    print(f"\n[完了] CSV保存: analysis/strategic_trainers.csv, analysis/ninki_stats.csv")


if __name__ == "__main__":
    main()
