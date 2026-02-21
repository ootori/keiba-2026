"""Step 3: 馬主別分析.

馬主ごとの「人気馬取りこぼし率」を算出し、
統計的に異常な馬主を検出する。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.owner_class_avoidance.step0_data_preparation import (
    load_base_dataset,
    OUTPUT_DIR,
)

logger = logging.getLogger(__name__)

MIN_ENTRIES = 50  # 馬主の最小出走数


def run_step3(df: pd.DataFrame | None = None) -> dict:
    """Step 3 を実行する."""
    if df is None:
        df = load_base_dataset()

    results = {}

    # 条件戦 + 人気1-3番に限定
    cond_classes = ["1win", "2win", "3win"]
    cond = df[
        (df["class_label"].isin(cond_classes))
        & (df["tanninki"].isin([1, 2, 3]))
    ].copy()

    cond["is_win"] = (cond["kakuteijyuni"] == 1).astype(int)
    cond["is_2nd_3rd"] = (cond["kakuteijyuni"].isin([2, 3])).astype(int)
    cond["is_4th_plus"] = (cond["kakuteijyuni"] >= 4).astype(int)

    # 全体の基準値
    pop_win_rate = cond["is_win"].mean()
    pop_2nd3rd_rate = cond["is_2nd_3rd"].mean()
    pop_drop_rate = cond["is_4th_plus"].mean()

    print("\n" + "=" * 70)
    print("Step 3.1: 全体基準値（条件戦 1-3番人気）")
    print("=" * 70)
    print(f"  N = {len(cond)}")
    print(f"  勝率: {pop_win_rate:.4f}")
    print(f"  2-3着率: {pop_2nd3rd_rate:.4f}")
    print(f"  4着以下率: {pop_drop_rate:.4f}")

    # ------------------------------------------------------------------
    # 3.2 馬主別集計
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 3.2: 馬主別の成績（条件戦 1-3番人気）")
    print("=" * 70)

    owner_stats = (
        cond.groupby(["banusicode", "banusiname"])
        .agg(
            N=("is_win", "count"),
            wins=("is_win", "sum"),
            place_2nd3rd=("is_2nd_3rd", "sum"),
            drops=("is_4th_plus", "sum"),
            promo_risk_count=("promotion_risk", "sum"),
        )
        .reset_index()
    )

    owner_stats["win_rate"] = owner_stats["wins"] / owner_stats["N"]
    owner_stats["rate_2nd3rd"] = owner_stats["place_2nd3rd"] / owner_stats["N"]
    owner_stats["drop_rate"] = owner_stats["drops"] / owner_stats["N"]

    # 最小出走数フィルタ
    owner_filtered = owner_stats[owner_stats["N"] >= MIN_ENTRIES].copy()
    print(f"\n  対象馬主: {len(owner_filtered)} (N>={MIN_ENTRIES})")

    # ------------------------------------------------------------------
    # 3.3 二項検定で異常馬主を検出
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 3.3: 勝率が有意に低い馬主（二項検定 + BH-FDR補正）")
    print("=" * 70)

    # 勝率が低い方向の検定
    p_values_win = []
    for _, row in owner_filtered.iterrows():
        # 片側検定: 勝率が全体平均より低いか
        result = stats.binomtest(
            int(row["wins"]), int(row["N"]), pop_win_rate, alternative="less"
        )
        p_values_win.append(result.pvalue)

    owner_filtered = owner_filtered.copy()
    owner_filtered["p_value_win"] = p_values_win

    # BH-FDR補正
    owner_filtered["p_bh_win"] = _benjamini_hochberg(owner_filtered["p_value_win"].values)

    # 有意な馬主（q < 0.05）
    sig_owners = owner_filtered[owner_filtered["p_bh_win"] < 0.05].sort_values("win_rate")

    if len(sig_owners) > 0:
        print(f"\n  勝率が有意に低い馬主: {len(sig_owners)}名")
        print("-" * 90)
        for _, row in sig_owners.head(20).iterrows():
            print(f"  {row['banusiname']:<20s} "
                  f"N={int(row['N']):>4d} "
                  f"勝率={row['win_rate']:.3f} "
                  f"2-3着率={row['rate_2nd3rd']:.3f} "
                  f"4着+率={row['drop_rate']:.3f} "
                  f"p_bh={row['p_bh_win']:.4f} "
                  f"リスクあり={int(row['promo_risk_count']):>3d}")
    else:
        print("  有意に低い勝率の馬主は検出されませんでした。")

    results["sig_low_win_owners"] = sig_owners

    # ------------------------------------------------------------------
    # 3.4 2-3着率が有意に高い馬主の検出
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 3.4: 2-3着率が有意に高い馬主（二項検定 + BH-FDR補正）")
    print("=" * 70)

    p_values_23 = []
    for _, row in owner_filtered.iterrows():
        result = stats.binomtest(
            int(row["place_2nd3rd"]), int(row["N"]), pop_2nd3rd_rate, alternative="greater"
        )
        p_values_23.append(result.pvalue)

    owner_filtered["p_value_2nd3rd"] = p_values_23
    owner_filtered["p_bh_2nd3rd"] = _benjamini_hochberg(
        owner_filtered["p_value_2nd3rd"].values
    )

    sig_23 = owner_filtered[owner_filtered["p_bh_2nd3rd"] < 0.05].sort_values(
        "rate_2nd3rd", ascending=False
    )

    if len(sig_23) > 0:
        print(f"\n  2-3着率が有意に高い馬主: {len(sig_23)}名")
        print("-" * 90)
        for _, row in sig_23.head(20).iterrows():
            print(f"  {row['banusiname']:<20s} "
                  f"N={int(row['N']):>4d} "
                  f"勝率={row['win_rate']:.3f} "
                  f"2-3着率={row['rate_2nd3rd']:.3f} "
                  f"4着+率={row['drop_rate']:.3f} "
                  f"p_bh={row['p_bh_2nd3rd']:.4f}")
    else:
        print("  有意に高い2-3着率の馬主は検出されませんでした。")

    results["sig_high_23_owners"] = sig_23

    # ------------------------------------------------------------------
    # 3.5 馬主×昇格リスクの交互作用
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 3.5: 異常馬主の昇格リスク集中度")
    print("=" * 70)

    # 3.3または3.4で検出された馬主について、昇格リスクあり/なしでの成績比較
    flagged_codes = set()
    if len(sig_owners) > 0:
        flagged_codes.update(sig_owners["banusicode"].values)
    if len(sig_23) > 0:
        flagged_codes.update(sig_23["banusicode"].values)

    if flagged_codes:
        flagged = cond[cond["banusicode"].isin(flagged_codes)]
        not_flagged = cond[~cond["banusicode"].isin(flagged_codes)]

        print(f"\n  異常馬主 {len(flagged_codes)}名の成績:")
        for label, group in [("昇格リスクあり", flagged[flagged["promotion_risk"]]),
                              ("昇格リスクなし", flagged[~flagged["promotion_risk"]])]:
            if len(group) > 0:
                wr = group["is_win"].mean()
                pr = group["is_2nd_3rd"].mean()
                print(f"    {label}: 勝率={wr:.4f}, 2-3着率={pr:.4f} (N={len(group)})")

        print(f"\n  非異常馬主の成績:")
        for label, group in [("昇格リスクあり", not_flagged[not_flagged["promotion_risk"]]),
                              ("昇格リスクなし", not_flagged[~not_flagged["promotion_risk"]])]:
            if len(group) > 0:
                wr = group["is_win"].mean()
                pr = group["is_2nd_3rd"].mean()
                print(f"    {label}: 勝率={wr:.4f}, 2-3着率={pr:.4f} (N={len(group)})")
    else:
        print("  異常馬主が検出されていないためスキップ")

    # ------------------------------------------------------------------
    # 3.6 全馬主の統計サマリーを保存
    # ------------------------------------------------------------------
    out_path = OUTPUT_DIR / "owner_stats.csv"
    owner_filtered.to_csv(out_path, index=False)
    print(f"\n  馬主統計を保存: {out_path}")
    results["owner_stats"] = owner_filtered

    return results


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR補正."""
    n = len(p_values)
    ranked = np.argsort(p_values)
    adjusted = np.empty(n)
    adjusted[ranked[-1]] = p_values[ranked[-1]]

    for i in range(n - 2, -1, -1):
        rank = i + 1
        adjusted[ranked[i]] = min(
            adjusted[ranked[i + 1]],
            p_values[ranked[i]] * n / rank,
        )

    return np.clip(adjusted, 0, 1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run_step3()
