"""Step 4: 連続着順パターン分析.

人気上位（1-3番人気）なのに2-3着を連続する馬の頻度が
偶然で説明できるかを検証する。
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


def run_step4(df: pd.DataFrame | None = None) -> dict:
    """Step 4 を実行する."""
    if df is None:
        df = load_base_dataset()

    results = {}

    # 条件戦 + 人気1-3番
    cond_classes = ["1win", "2win", "3win"]
    cond = df[
        (df["class_label"].isin(cond_classes))
        & (df["tanninki"].isin([1, 2, 3]))
    ].copy()
    cond = cond.sort_values(["kettonum", "year", "monthday"])

    cond["is_win"] = (cond["kakuteijyuni"] == 1).astype(int)
    cond["is_2nd_3rd"] = (cond["kakuteijyuni"].isin([2, 3])).astype(int)

    # ------------------------------------------------------------------
    # 4.1 連続2-3着ストリークの検出
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 4.1: 人気馬の連続2-3着ストリーク（条件戦）")
    print("=" * 70)

    # 馬ごとに条件戦での着順シーケンスを構築
    streaks = _find_streaks(cond, "is_2nd_3rd")

    # 全体の2-3着率（期待値計算用）
    p_2nd3rd = cond["is_2nd_3rd"].mean()
    print(f"\n  全体の2-3着率: {p_2nd3rd:.4f}")

    # ストリーク長ごとの観測 vs 期待
    print(f"\n  {'ストリーク長':<12s} {'観測数':<8s} {'期待数':<10s} {'比率':<8s}")
    print("-" * 45)

    # 各馬のシーケンス長から期待値を計算
    horse_seq_lengths = cond.groupby("kettonum").size()
    total_opportunities = {}
    for length in [2, 3, 4, 5]:
        # 各馬の連続した length レースの窓の数
        opportunities = sum(max(0, seq_len - length + 1)
                          for seq_len in horse_seq_lengths)
        expected = opportunities * (p_2nd3rd ** length)
        observed = sum(1 for s in streaks if s["length"] >= length)
        ratio = observed / expected if expected > 0 else float("inf")
        total_opportunities[length] = {
            "observed": observed,
            "expected": expected,
            "ratio": ratio,
        }
        print(f"  {length}連続以上    {observed:<8d} {expected:<10.1f} {ratio:<8.2f}")

    results["streak_vs_expected"] = total_opportunities

    # ------------------------------------------------------------------
    # 4.2 長期滞在パターン（同クラスに8走以上）
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 4.2: 同クラス長期滞在馬（8走以上で1着なし）")
    print("=" * 70)

    # 条件戦全馬（人気関係なし）
    cond_all = df[df["class_label"].isin(cond_classes)].copy()
    cond_all = cond_all.sort_values(["kettonum", "year", "monthday"])

    # 馬×クラスごとの出走数と成績を集計
    class_stays = (
        cond_all.groupby(["kettonum", "class_label"])
        .agg(
            n_races=("kakuteijyuni", "count"),
            n_wins=("kakuteijyuni", lambda x: (x == 1).sum()),
            n_2nd3rd=("kakuteijyuni", lambda x: x.isin([2, 3]).sum()),
            banusicode=("banusicode", "first"),
            banusiname=("banusiname", "first"),
            min_ninki=("tanninki", "min"),
            mean_ninki=("tanninki", "mean"),
        )
        .reset_index()
    )

    # 8走以上で0勝
    long_stays = class_stays[
        (class_stays["n_races"] >= 8) & (class_stays["n_wins"] == 0)
    ].copy()
    long_stays = long_stays.sort_values("n_races", ascending=False)

    print(f"\n  8走以上で0勝の馬×クラス: {len(long_stays)} ケース")

    # その中で人気上位が多い馬（平均人気3以内）
    long_favored = long_stays[long_stays["mean_ninki"] <= 3]
    print(f"  うち平均人気3以内: {len(long_favored)} ケース")

    if len(long_favored) > 0:
        print(f"\n  {'馬主名':<20s} {'クラス':<6s} {'出走':<4s} {'勝利':<4s} "
              f"{'2-3着':<5s} {'平均人気':<8s}")
        print("-" * 55)
        for _, row in long_favored.head(20).iterrows():
            print(f"  {row['banusiname']:<20s} {row['class_label']:<6s} "
                  f"{int(row['n_races']):<4d} {int(row['n_wins']):<4d} "
                  f"{int(row['n_2nd3rd']):<5d} {row['mean_ninki']:.1f}")

    results["long_stays"] = long_stays
    results["long_favored"] = long_favored

    # ------------------------------------------------------------------
    # 4.3 「遅い昇格」パターン
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 4.3: 遅い昇格パターン（多数2-3着→やっと1着）")
    print("=" * 70)

    # 条件戦で最終的に1着を取った馬×クラスについて、
    # そのクラスでの走行回数分布を確認
    eventual_winners = class_stays[class_stays["n_wins"] >= 1].copy()
    eventual_winners["races_to_win"] = eventual_winners["n_races"]  # 簡易: 総走行数

    print(f"\n  最終的に昇格した馬×クラス: {len(eventual_winners)} ケース")
    print(f"\n  クラス内走行数の分布:")
    print(eventual_winners["n_races"].describe())

    # 5走以上かかって昇格 + 2-3着が多い
    slow_winners = eventual_winners[
        (eventual_winners["n_races"] >= 5) & (eventual_winners["n_2nd3rd"] >= 3)
    ]
    print(f"\n  5走以上 + 2-3着3回以上で昇格: {len(slow_winners)} ケース")

    if len(slow_winners) > 0:
        # 馬主別に集計
        slow_by_owner = (
            slow_winners.groupby(["banusicode", "banusiname"])
            .agg(n_cases=("n_races", "count"))
            .reset_index()
            .sort_values("n_cases", ascending=False)
        )
        print(f"\n  上位馬主（遅い昇格パターン）:")
        for _, row in slow_by_owner.head(15).iterrows():
            print(f"    {row['banusiname']:<20s}: {int(row['n_cases'])} ケース")

    results["slow_winners"] = slow_winners

    # ------------------------------------------------------------------
    # 4.4 具体的な連続2-3着ストリーク馬のリスト
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 4.4: 注目すべき連続2-3着ストリーク（3連続以上）")
    print("=" * 70)

    notable_streaks = [s for s in streaks if s["length"] >= 3]
    notable_streaks.sort(key=lambda s: s["length"], reverse=True)

    if notable_streaks:
        print(f"\n  3連続以上: {len(notable_streaks)} ケース")

        # 馬主別に集計
        streak_owner_counts: dict[str, int] = {}
        for s in notable_streaks:
            name = s.get("banusiname", "不明")
            streak_owner_counts[name] = streak_owner_counts.get(name, 0) + 1

        sorted_owners = sorted(streak_owner_counts.items(), key=lambda x: -x[1])
        print(f"\n  上位馬主（3連続以上2-3着ストリーク）:")
        for name, count in sorted_owners[:15]:
            print(f"    {name:<20s}: {count} ケース")
    else:
        print("  3連続以上のストリークは検出されませんでした。")

    results["notable_streaks"] = notable_streaks

    return results


def _find_streaks(
    df: pd.DataFrame,
    flag_col: str,
    min_length: int = 2,
) -> list[dict]:
    """馬ごとにフラグの連続ストリークを検出する.

    Args:
        df: ソート済みデータ（kettonum, year, monthday順）
        flag_col: 0/1のフラグカラム
        min_length: 最小ストリーク長

    Returns:
        ストリーク情報のリスト
    """
    streaks = []

    for kettonum, group in df.groupby("kettonum"):
        flags = group[flag_col].values
        owner_name = group["banusiname"].iloc[0]
        owner_code = group["banusicode"].iloc[0]

        # 連続フラグを検出
        current_len = 0
        for i, f in enumerate(flags):
            if f == 1:
                current_len += 1
            else:
                if current_len >= min_length:
                    streaks.append({
                        "kettonum": kettonum,
                        "banusicode": owner_code,
                        "banusiname": owner_name,
                        "length": current_len,
                        "end_idx": i - 1,
                    })
                current_len = 0

        # 末尾のストリーク
        if current_len >= min_length:
            streaks.append({
                "kettonum": kettonum,
                "banusicode": owner_code,
                "banusiname": owner_name,
                "length": current_len,
                "end_idx": len(flags) - 1,
            })

    return streaks


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run_step4()
