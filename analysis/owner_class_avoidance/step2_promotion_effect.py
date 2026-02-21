"""Step 2: 昇格閾値効果の検出（コア分析）.

「昇格リスクあり」の馬が人気に反して負ける頻度が高いかを、
Z検定・ロジスティック回帰・差分の差分（DID）で検証する。
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


def run_step2(df: pd.DataFrame | None = None) -> dict:
    """Step 2 を実行する."""
    if df is None:
        df = load_base_dataset()

    results = {}

    # 条件戦に限定
    cond_classes = ["1win", "2win", "3win"]
    cond = df[df["class_label"].isin(cond_classes)].copy()
    cond["is_win"] = (cond["kakuteijyuni"] == 1).astype(int)
    cond["is_place"] = (cond["kakuteijyuni"] <= 3).astype(int)
    cond["is_2nd_3rd"] = (cond["kakuteijyuni"].isin([2, 3])).astype(int)

    # ------------------------------------------------------------------
    # 2.1 昇格リスクあり vs なしの勝率比較（層別Z検定）
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 2.1: 昇格リスクあり vs なし（層別比較）")
    print("=" * 70)

    n_tests = 0
    all_results_21 = []

    for cl in cond_classes:
        for ninki in [1, 2, 3]:
            subset = cond[(cond["class_label"] == cl) & (cond["tanninki"] == ninki)]
            risk_yes = subset[subset["promotion_risk"]]
            risk_no = subset[~subset["promotion_risk"]]

            if len(risk_yes) < 30 or len(risk_no) < 30:
                continue

            n_tests += 1
            wr_yes = risk_yes["is_win"].mean()
            wr_no = risk_no["is_win"].mean()
            pr_yes = risk_yes["is_2nd_3rd"].mean()
            pr_no = risk_no["is_2nd_3rd"].mean()

            # 片側Z検定: risk_yesの勝率 < risk_noの勝率
            z, p_two = _two_proportion_z_test(
                risk_yes["is_win"].sum(), len(risk_yes),
                risk_no["is_win"].sum(), len(risk_no),
            )
            # 片側p値（risk_yesが低い方向）
            p_one = stats.norm.cdf(z) if z < 0 else 1 - stats.norm.cdf(z)
            # ← risk_yesのほうが低いことを検定するので z<0 なら p_one = P(Z<z)

            row = {
                "class": cl,
                "ninki": ninki,
                "N_risk": len(risk_yes),
                "N_safe": len(risk_no),
                "win_risk": wr_yes,
                "win_safe": wr_no,
                "diff": wr_yes - wr_no,
                "2nd3rd_risk": pr_yes,
                "2nd3rd_safe": pr_no,
                "z": z,
                "p_one": p_one,
            }
            all_results_21.append(row)

            star = "***" if p_one < 0.001 else "**" if p_one < 0.01 else "*" if p_one < 0.05 else ""
            print(f"  {cl} {ninki}番人気: "
                  f"リスクあり勝率={wr_yes:.4f}(N={len(risk_yes)}) "
                  f"リスクなし勝率={wr_no:.4f}(N={len(risk_no)}) "
                  f"差={wr_yes-wr_no:+.4f} z={z:.3f} p={p_one:.4f} {star}")

    # Bonferroni補正
    if all_results_21:
        alpha_bonf = 0.05 / n_tests
        print(f"\n  Bonferroni補正後の有意水準: α = {alpha_bonf:.6f} (検定数={n_tests})")
        sig_count = sum(1 for r in all_results_21 if r["p_one"] < alpha_bonf)
        print(f"  補正後も有意な検定: {sig_count}/{n_tests}")

    results_21_df = pd.DataFrame(all_results_21)
    results["stratified_z_test"] = results_21_df

    # ------------------------------------------------------------------
    # 2.1b 全条件戦を集約した比較（人気1-3番）
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Step 2.1b: 全条件戦・人気1-3番の集約比較")
    print("-" * 70)

    top3_ninki = cond[cond["tanninki"].isin([1, 2, 3])]
    risk_all_yes = top3_ninki[top3_ninki["promotion_risk"]]
    risk_all_no = top3_ninki[~top3_ninki["promotion_risk"]]

    if len(risk_all_yes) > 0 and len(risk_all_no) > 0:
        wr_y = risk_all_yes["is_win"].mean()
        wr_n = risk_all_no["is_win"].mean()
        z, _ = _two_proportion_z_test(
            risk_all_yes["is_win"].sum(), len(risk_all_yes),
            risk_all_no["is_win"].sum(), len(risk_all_no),
        )
        p_one = stats.norm.cdf(z)
        print(f"  リスクあり: 勝率={wr_y:.4f} (N={len(risk_all_yes)})")
        print(f"  リスクなし: 勝率={wr_n:.4f} (N={len(risk_all_no)})")
        print(f"  差: {wr_y-wr_n:+.4f}")
        print(f"  Z={z:.4f}, 片側p={p_one:.6f}")

        # 2-3着率の比較
        pr_y = risk_all_yes["is_2nd_3rd"].mean()
        pr_n = risk_all_no["is_2nd_3rd"].mean()
        print(f"  2-3着率: リスクあり={pr_y:.4f}, リスクなし={pr_n:.4f}, 差={pr_y-pr_n:+.4f}")

    # ------------------------------------------------------------------
    # 2.2 ロジスティック回帰
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 2.2: ロジスティック回帰（交絡制御）")
    print("=" * 70)

    try:
        import statsmodels.api as sm

        reg_df = cond[cond["tanninki"].isin([1, 2, 3])].copy()
        reg_df = reg_df.dropna(subset=["tanodds", "kakuteijyuni", "field_size", "kyori"])

        # 説明変数
        reg_df["log_odds"] = np.log(reg_df["tanodds"].clip(lower=1.0))
        reg_df["promo_risk"] = reg_df["promotion_risk"].astype(int)

        # クラスダミー（基準: 1win）
        reg_df["class_2win"] = (reg_df["class_label"] == "2win").astype(int)
        reg_df["class_3win"] = (reg_df["class_label"] == "3win").astype(int)

        # トラックタイプ（芝=1, ダート=0）
        reg_df["is_turf"] = reg_df["trackcd"].apply(
            lambda x: 1 if str(x).strip() in ("10", "11", "12", "13", "14",
                                                "17", "18", "19", "20", "21",
                                                "22", "23", "24", "25", "26") else 0
        )

        # 距離カテゴリ
        reg_df["dist_km"] = reg_df["kyori"] / 1000.0

        X_cols = ["promo_risk", "log_odds", "field_size", "class_2win", "class_3win",
                  "is_turf", "dist_km"]
        X = reg_df[X_cols].astype(float)
        X = sm.add_constant(X)
        y = reg_df["is_win"]

        model = sm.Logit(y, X).fit(disp=False)
        print(model.summary2().tables[1].to_string())

        # promotion_risk の係数
        coef = model.params["promo_risk"]
        pval = model.pvalues["promo_risk"]
        ci = model.conf_int().loc["promo_risk"]
        or_val = np.exp(coef)

        print(f"\n  promotion_risk 係数: {coef:.4f} (OR={or_val:.4f})")
        print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"  p値: {pval:.6f}")
        print(f"  → {'有意 (p < 0.05): 昇格リスクは勝率に影響' if pval < 0.05 else '有意でない'}")

        results["logistic"] = {
            "coef": coef, "or": or_val, "p": pval,
            "ci_low": ci[0], "ci_high": ci[1],
        }

    except ImportError:
        print("  statsmodels が未インストールのためスキップ")

    # ------------------------------------------------------------------
    # 2.3 DID分析: 2019年降級廃止の前後
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 2.3: 差分の差分（DID）分析")
    print("=" * 70)

    try:
        import statsmodels.api as sm

        did_df = cond[cond["tanninki"].isin([1, 2, 3])].copy()
        did_df = did_df.dropna(subset=["tanodds", "kakuteijyuni", "field_size"])

        did_df["post_2019"] = (did_df["year_int"] >= 2020).astype(int)
        did_df["promo_risk"] = did_df["promotion_risk"].astype(int)
        did_df["interaction"] = did_df["promo_risk"] * did_df["post_2019"]
        did_df["log_odds"] = np.log(did_df["tanodds"].clip(lower=1.0))

        X_cols = ["promo_risk", "post_2019", "interaction", "log_odds", "field_size"]
        X = did_df[X_cols].astype(float)
        X = sm.add_constant(X)
        y = did_df["is_win"]

        model_did = sm.Logit(y, X).fit(disp=False)
        print(model_did.summary2().tables[1].to_string())

        # 交互作用項
        coef_int = model_did.params["interaction"]
        pval_int = model_did.pvalues["interaction"]
        print(f"\n  DID交互作用項: {coef_int:.4f}, p={pval_int:.6f}")
        print(f"  → {'有意: 降級廃止後に昇格リスク効果が拡大' if pval_int < 0.05 else '有意でない'}")

        results["did"] = {"coef": coef_int, "p": pval_int}

        # 期間ごとの単純比較
        print("\n  期間別の昇格リスク効果:")
        for period, label in [(0, "2015-2019"), (1, "2020-2025")]:
            subset = did_df[did_df["post_2019"] == period]
            risk_y = subset[subset["promo_risk"] == 1]
            risk_n = subset[subset["promo_risk"] == 0]
            if len(risk_y) > 0 and len(risk_n) > 0:
                wr_y = risk_y["is_win"].mean()
                wr_n = risk_n["is_win"].mean()
                print(f"  {label}: リスクあり={wr_y:.4f}(N={len(risk_y)}), "
                      f"リスクなし={wr_n:.4f}(N={len(risk_n)}), "
                      f"差={wr_y-wr_n:+.4f}")

    except ImportError:
        print("  statsmodels が未インストールのためスキップ")

    return results


def _two_proportion_z_test(
    k1: int, n1: int, k2: int, n2: int
) -> tuple[float, float]:
    """2標本比率のZ検定."""
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    p_val = 2 * stats.norm.sf(abs(z))
    return (z, p_val)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run_step2()
