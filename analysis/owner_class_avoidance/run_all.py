"""全ステップ実行スクリプト.

Step 0-5 を順番に実行し、結果をまとめて表示する。

Usage:
    python -m analysis.owner_class_avoidance.run_all
    # or
    python analysis/owner_class_avoidance/run_all.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.owner_class_avoidance.step0_data_preparation import (
    build_base_dataset,
    load_base_dataset,
    OUTPUT_DIR,
)
from analysis.owner_class_avoidance.step1_baseline_stats import run_step1
from analysis.owner_class_avoidance.step2_promotion_effect import run_step2
from analysis.owner_class_avoidance.step3_owner_analysis import run_step3
from analysis.owner_class_avoidance.step4_pattern_analysis import run_step4
from analysis.owner_class_avoidance.step5_model_residuals import run_step5

logger = logging.getLogger(__name__)


def main() -> None:
    """全ステップ実行."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    start_time = time.time()

    print("=" * 70)
    print("馬主による意図的着順調整の統計的検証")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 0: データ準備
    # ------------------------------------------------------------------
    print("\n\n" + "#" * 70)
    print("# Step 0: データ準備")
    print("#" * 70)

    base_path = OUTPUT_DIR / "base_dataset.parquet"
    if base_path.exists():
        logger.info("既存データを使用: %s", base_path)
        df = load_base_dataset()
    else:
        df = build_base_dataset()

    # ------------------------------------------------------------------
    # Step 1: 基礎統計
    # ------------------------------------------------------------------
    print("\n\n" + "#" * 70)
    print("# Step 1: 基礎統計")
    print("#" * 70)
    results_1 = run_step1(df)

    # ------------------------------------------------------------------
    # Step 2: 昇格閾値効果
    # ------------------------------------------------------------------
    print("\n\n" + "#" * 70)
    print("# Step 2: 昇格閾値効果の検出")
    print("#" * 70)
    results_2 = run_step2(df)

    # ------------------------------------------------------------------
    # Step 3: 馬主別分析
    # ------------------------------------------------------------------
    print("\n\n" + "#" * 70)
    print("# Step 3: 馬主別分析")
    print("#" * 70)
    results_3 = run_step3(df)

    # ------------------------------------------------------------------
    # Step 4: パターン分析
    # ------------------------------------------------------------------
    print("\n\n" + "#" * 70)
    print("# Step 4: 連続着順パターン分析")
    print("#" * 70)
    results_4 = run_step4(df)

    # ------------------------------------------------------------------
    # Step 5: モデル予測乖離
    # ------------------------------------------------------------------
    print("\n\n" + "#" * 70)
    print("# Step 5: モデル予測との乖離分析")
    print("#" * 70)
    results_5 = run_step5(df)

    # ------------------------------------------------------------------
    # 総合レポート
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n\n" + "=" * 70)
    print("総合レポート")
    print("=" * 70)
    print(f"\n実行時間: {elapsed:.1f}秒")
    print(f"出力ディレクトリ: {OUTPUT_DIR}")

    # 主要な結論をまとめる
    print("\n--- 主要な発見 ---")

    # Step 2 の結果
    if "logistic" in results_2:
        lg = results_2["logistic"]
        sig = "有意" if lg["p"] < 0.05 else "有意でない"
        print(f"\n[昇格リスク効果]")
        print(f"  ロジスティック回帰: OR={lg['or']:.4f}, p={lg['p']:.6f} → {sig}")

    if "did" in results_2:
        did = results_2["did"]
        sig = "有意" if did["p"] < 0.05 else "有意でない"
        print(f"  DID（2019年前後）: coef={did['coef']:.4f}, p={did['p']:.6f} → {sig}")

    # Step 3 の結果
    if "sig_low_win_owners" in results_3:
        n_sig = len(results_3["sig_low_win_owners"])
        print(f"\n[馬主別分析]")
        print(f"  勝率が有意に低い馬主: {n_sig}名")

    if "sig_high_23_owners" in results_3:
        n_sig = len(results_3["sig_high_23_owners"])
        print(f"  2-3着率が有意に高い馬主: {n_sig}名")

    # Step 5 の結果
    if "risk_residual_test" in results_5:
        rt = results_5["risk_residual_test"]
        sig = "有意" if rt["p"] < 0.05 else "有意でない"
        print(f"\n[オッズ乖離分析]")
        print(f"  昇格リスクあり vs なし: t={rt['t']:.4f}, p={rt['p']:.6f} → {sig}")

    print("\n" + "=" * 70)
    print("分析完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
