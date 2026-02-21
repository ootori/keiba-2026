"""Step 5: モデル予測との乖離分析.

既存LightGBMモデルの予測確率と実績の乖離を馬主別に分析する。
モデルが存在しない場合はオッズベースの期待勝率を使用する。
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


def run_step5(df: pd.DataFrame | None = None) -> dict:
    """Step 5 を実行する."""
    if df is None:
        df = load_base_dataset()

    results = {}

    # 条件戦 + 人気1-3番
    cond_classes = ["1win", "2win", "3win"]
    cond = df[
        (df["class_label"].isin(cond_classes))
        & (df["tanninki"].isin([1, 2, 3]))
    ].copy()

    cond["is_win"] = (cond["kakuteijyuni"] == 1).astype(int)
    cond = cond.dropna(subset=["tanodds"])

    # ------------------------------------------------------------------
    # 5.1 オッズベースの期待勝率と実績の乖離
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 5.1: オッズベースの期待勝率 vs 実績")
    print("=" * 70)

    # 単勝オッズから暗示確率を計算（控除率約25%を考慮）
    # 暗示確率 = 1/odds, 正規化後がmarket probability
    # ただしここでは単純に 1/odds をそのまま使い、
    # 実際の勝率との差（residual）を馬主別に分析する
    cond["implied_prob"] = 1.0 / cond["tanodds"].clip(lower=1.0)

    # レース内で正規化（合計1.0に）
    race_key = ["year", "monthday", "jyocd", "kaiji", "nichiji", "racenum"]
    # ただし条件戦人気上位のみの部分データなので、ここでは正規化しない

    # 全体のキャリブレーション
    overall_implied = cond["implied_prob"].mean()
    overall_actual = cond["is_win"].mean()
    print(f"  平均暗示確率: {overall_implied:.4f}")
    print(f"  実際の勝率: {overall_actual:.4f}")
    print(f"  差: {overall_actual - overall_implied:+.4f}")

    # 残差 = 実績 - 暗示確率
    cond["residual"] = cond["is_win"] - cond["implied_prob"]

    # ------------------------------------------------------------------
    # 5.2 馬主別の残差分析
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 5.2: 馬主別の残差分析（実績 - オッズ暗示確率）")
    print("=" * 70)

    owner_residuals = (
        cond.groupby(["banusicode", "banusiname"])
        .agg(
            N=("residual", "count"),
            mean_residual=("residual", "mean"),
            std_residual=("residual", "std"),
            mean_implied=("implied_prob", "mean"),
            actual_win_rate=("is_win", "mean"),
            promo_risk_count=("promotion_risk", "sum"),
        )
        .reset_index()
    )

    # 最小出走数フィルタ
    owner_residuals = owner_residuals[owner_residuals["N"] >= MIN_ENTRIES].copy()
    print(f"\n  対象馬主: {len(owner_residuals)} (N>={MIN_ENTRIES})")

    # 1標本t検定: 残差が有意に負か
    p_values = []
    t_stats = []
    for _, row in owner_residuals.iterrows():
        # 馬主の全出走の残差を取得
        owner_data = cond[cond["banusicode"] == row["banusicode"]]["residual"]
        t, p = stats.ttest_1samp(owner_data, 0, alternative="less")
        p_values.append(p)
        t_stats.append(t)

    owner_residuals["t_stat"] = t_stats
    owner_residuals["p_value"] = p_values

    # BH-FDR補正
    owner_residuals["p_bh"] = _benjamini_hochberg(
        owner_residuals["p_value"].values
    )

    # 有意な馬主
    sig = owner_residuals[owner_residuals["p_bh"] < 0.05].sort_values("mean_residual")

    if len(sig) > 0:
        print(f"\n  残差が有意に負の馬主: {len(sig)}名")
        print("-" * 100)
        print(f"  {'馬主名':<20s} {'N':<5s} {'勝率':<7s} {'暗示確率':<8s} "
              f"{'残差':<8s} {'t値':<8s} {'p_bh':<8s} {'リスク':<5s}")
        print("-" * 100)
        for _, row in sig.head(20).iterrows():
            print(f"  {row['banusiname']:<20s} "
                  f"{int(row['N']):<5d} "
                  f"{row['actual_win_rate']:.3f}   "
                  f"{row['mean_implied']:.4f}   "
                  f"{row['mean_residual']:+.4f}   "
                  f"{row['t_stat']:.3f}   "
                  f"{row['p_bh']:.4f}   "
                  f"{int(row['promo_risk_count']):<5d}")
    else:
        print("  有意に負の残差を示す馬主は検出されませんでした。")

    results["sig_negative_residual_owners"] = sig

    # ------------------------------------------------------------------
    # 5.3 昇格リスク有無での残差比較
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 5.3: 昇格リスク有無での残差比較")
    print("=" * 70)

    risk_yes = cond[cond["promotion_risk"]]
    risk_no = cond[~cond["promotion_risk"]]

    if len(risk_yes) > 0 and len(risk_no) > 0:
        res_yes = risk_yes["residual"].mean()
        res_no = risk_no["residual"].mean()
        t, p = stats.ttest_ind(
            risk_yes["residual"], risk_no["residual"], alternative="less"
        )
        print(f"  リスクあり: 平均残差={res_yes:+.4f} (N={len(risk_yes)})")
        print(f"  リスクなし: 平均残差={res_no:+.4f} (N={len(risk_no)})")
        print(f"  差: {res_yes - res_no:+.4f}")
        print(f"  t={t:.4f}, 片側p={p:.6f}")
        print(f"  → {'有意: リスクありは暗示確率に対して過少パフォーマンス' if p < 0.05 else '有意でない'}")

        results["risk_residual_test"] = {"t": t, "p": p}

    # ------------------------------------------------------------------
    # 5.4 全体の保存
    # ------------------------------------------------------------------
    out_path = OUTPUT_DIR / "owner_residuals.csv"
    owner_residuals.to_csv(out_path, index=False)
    print(f"\n  保存: {out_path}")

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
    run_step5()
