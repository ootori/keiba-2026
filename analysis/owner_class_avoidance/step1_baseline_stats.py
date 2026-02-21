"""Step 1: 基礎統計.

クラス別の人気馬勝率、条件戦 vs OP/重賞の比較など、
全体像を把握するための記述統計を算出する。
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


def run_step1(df: pd.DataFrame | None = None) -> dict:
    """Step 1 を実行する.

    Returns:
        結果辞書
    """
    if df is None:
        df = load_base_dataset()

    results = {}

    # ------------------------------------------------------------------
    # 1.1 クラス別の1番人気勝率
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 1.1: クラス別 1番人気の成績")
    print("=" * 70)

    fav1 = df[df["tanninki"] == 1].copy()
    fav1["is_win"] = (fav1["kakuteijyuni"] == 1).astype(int)
    fav1["is_place"] = (fav1["kakuteijyuni"] <= 3).astype(int)
    fav1["is_2nd_3rd"] = (fav1["kakuteijyuni"].isin([2, 3])).astype(int)
    fav1["is_4th_plus"] = (fav1["kakuteijyuni"] >= 4).astype(int)

    class_order = ["maiden", "1win", "2win", "3win", "open", "graded"]
    summary_rows = []
    for cl in class_order:
        subset = fav1[fav1["class_label"] == cl]
        if len(subset) == 0:
            continue
        n = len(subset)
        win_rate = subset["is_win"].mean()
        place_rate = subset["is_place"].mean()
        rate_2nd_3rd = subset["is_2nd_3rd"].mean()
        rate_4th_plus = subset["is_4th_plus"].mean()
        mean_odds = subset["tanodds"].mean()
        # 95% CI for win_rate
        ci_low, ci_high = _proportion_ci(win_rate, n)

        summary_rows.append({
            "class": cl,
            "N": n,
            "win_rate": win_rate,
            "win_ci_low": ci_low,
            "win_ci_high": ci_high,
            "2nd_3rd_rate": rate_2nd_3rd,
            "4th+_rate": rate_4th_plus,
            "place_rate": place_rate,
            "mean_odds": mean_odds,
        })

    summary_df = pd.DataFrame(summary_rows)
    _print_table(summary_df, "1番人気の成績（クラス別）")
    results["fav1_by_class"] = summary_df

    # χ²検定: 条件戦3クラス間の勝率差
    cond_classes = ["1win", "2win", "3win"]
    contingency = []
    for cl in cond_classes:
        subset = fav1[fav1["class_label"] == cl]
        wins = subset["is_win"].sum()
        losses = len(subset) - wins
        contingency.append([wins, losses])

    contingency = np.array(contingency)
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nχ²検定（条件戦3クラス間の1番人気勝率差）:")
    print(f"  χ² = {chi2:.4f}, p = {p_val:.6f}, dof = {dof}")
    print(f"  → {'有意差あり (p < 0.05)' if p_val < 0.05 else '有意差なし'}")
    results["chi2_condition_classes"] = {"chi2": chi2, "p": p_val, "dof": dof}

    # ------------------------------------------------------------------
    # 1.2 条件戦 vs OP・重賞
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 1.2: 条件戦 vs OP・重賞（1番人気勝率）")
    print("=" * 70)

    cond_fav = fav1[fav1["class_label"].isin(cond_classes)]
    open_graded_fav = fav1[fav1["class_label"].isin(["open", "graded"])]

    n1, k1 = len(cond_fav), cond_fav["is_win"].sum()
    n2, k2 = len(open_graded_fav), open_graded_fav["is_win"].sum()
    p1 = k1 / n1 if n1 > 0 else 0
    p2 = k2 / n2 if n2 > 0 else 0

    # 2標本Z検定
    z_stat, z_p = _two_proportion_z_test(k1, n1, k2, n2)
    print(f"  条件戦 1番人気勝率: {p1:.4f} (N={n1})")
    print(f"  OP/重賞 1番人気勝率: {p2:.4f} (N={n2})")
    print(f"  Z = {z_stat:.4f}, p = {z_p:.6f}")
    print(f"  → {'有意差あり (p < 0.05)' if z_p < 0.05 else '有意差なし'}")
    results["condition_vs_open"] = {
        "cond_rate": p1, "cond_n": n1,
        "open_rate": p2, "open_n": n2,
        "z": z_stat, "p": z_p,
    }

    # ------------------------------------------------------------------
    # 1.3 人気帯別（1-3番人気）の詳細統計（条件戦）
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 1.3: 条件戦 人気帯別の成績")
    print("=" * 70)

    cond_df = df[df["class_label"].isin(cond_classes)].copy()
    cond_df["is_win"] = (cond_df["kakuteijyuni"] == 1).astype(int)
    cond_df["is_place"] = (cond_df["kakuteijyuni"] <= 3).astype(int)
    cond_df["is_2nd_3rd"] = (cond_df["kakuteijyuni"].isin([2, 3])).astype(int)

    rows = []
    for ninki in range(1, 6):
        subset = cond_df[cond_df["tanninki"] == ninki]
        if len(subset) == 0:
            continue
        n = len(subset)
        rows.append({
            "人気": ninki,
            "N": n,
            "勝率": subset["is_win"].mean(),
            "2-3着率": subset["is_2nd_3rd"].mean(),
            "複勝率": subset["is_place"].mean(),
            "平均オッズ": subset["tanodds"].mean(),
        })
    ninki_df = pd.DataFrame(rows)
    _print_table(ninki_df, "条件戦 人気帯別の成績")
    results["condition_by_ninki"] = ninki_df

    # ------------------------------------------------------------------
    # 1.4 2019年前後の1番人気勝率変化（条件戦）
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 1.4: 降級廃止前後の1番人気勝率変化（条件戦）")
    print("=" * 70)

    cond_fav1 = cond_df[cond_df["tanninki"] == 1].copy()
    cond_fav1["period"] = np.where(cond_fav1["year_int"] <= 2019, "2015-2019", "2020-2025")

    for period in ["2015-2019", "2020-2025"]:
        subset = cond_fav1[cond_fav1["period"] == period]
        n = len(subset)
        wr = subset["is_win"].mean() if n > 0 else 0
        pr = subset["is_place"].mean() if n > 0 else 0
        ci_lo, ci_hi = _proportion_ci(wr, n) if n > 0 else (0, 0)
        print(f"  {period}: 勝率={wr:.4f} [{ci_lo:.4f}, {ci_hi:.4f}], 複勝率={pr:.4f} (N={n})")

    # 期間間のZ検定
    pre = cond_fav1[cond_fav1["period"] == "2015-2019"]
    post = cond_fav1[cond_fav1["period"] == "2020-2025"]
    if len(pre) > 0 and len(post) > 0:
        z, p = _two_proportion_z_test(
            pre["is_win"].sum(), len(pre),
            post["is_win"].sum(), len(post),
        )
        print(f"  Z = {z:.4f}, p = {p:.6f}")
        results["pre_post_2019"] = {"z": z, "p": p}

    # ------------------------------------------------------------------
    # 1.5 昇格リスク有無の基本分布
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 1.5: 昇格リスクの基本分布（条件戦）")
    print("=" * 70)

    for cl in cond_classes:
        subset = cond_df[cond_df["class_label"] == cl]
        risk = subset["promotion_risk"]
        print(f"  {cl}: リスクあり={risk.sum()} ({100*risk.mean():.1f}%), "
              f"リスクなし={len(subset)-risk.sum()}")

    return results


def _proportion_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """二項比率のWald信頼区間."""
    if n == 0:
        return (0.0, 0.0)
    se = np.sqrt(p * (1 - p) / n)
    return (max(0, p - z * se), min(1, p + z * se))


def _two_proportion_z_test(
    k1: int, n1: int, k2: int, n2: int
) -> tuple[float, float]:
    """2標本比率のZ検定（両側）."""
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    p_val = 2 * stats.norm.sf(abs(z))
    return (z, p_val)


def _print_table(df: pd.DataFrame, title: str) -> None:
    """テーブルを見やすく表示."""
    print(f"\n{title}:")
    print("-" * 70)
    # float列を小数4桁に
    formatters = {}
    for col in df.columns:
        if df[col].dtype == float:
            formatters[col] = lambda x: f"{x:.4f}"
    print(df.to_string(index=False, formatters=formatters))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    results = run_step1()
