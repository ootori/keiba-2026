#!/usr/bin/env python3
"""予想1位 vs 人気1位 の単勝回収率分析.

予想1位と人気1位の一致/不一致で分けた場合の回収率を比較する。
対象は2025年の検証データ。

使用モデル: ranking_win（LambdaRank + relevance-mode win + supplement bms_detail rating）
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RACE_KEY_COLS, CATEGORICAL_FEATURES
from src.features.pipeline import FeaturePipeline
from src.features.supplement import merge_supplements
from src.model.trainer import ModelTrainer
from src.model.evaluator import ModelEvaluator
from src.db import query_df

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    # --- 設定 ---
    model_name = "ranking_win"
    train_start = "2015"
    valid_year = "2025"
    supplement_names = ["bms_detail", "rating"]

    # --- 特徴量ロード ---
    logger.info("特徴量ロード中...")
    all_df = FeaturePipeline.load_years(
        train_start, valid_year,
        supplement_names=supplement_names,
    )
    valid_df = all_df[all_df["_key_year"] == valid_year].copy()
    logger.info("検証データ: %d行", len(valid_df))

    # --- モデルロード ---
    logger.info("モデルロード: %s", model_name)
    trainer = ModelTrainer()
    model = trainer.load_model(name=model_name)
    ranking = trainer.ranking
    feature_columns = trainer.feature_columns

    logger.info("ranking=%s, target_type=%s", ranking, trainer.target_type)

    # --- 予測 ---
    X_valid = valid_df[feature_columns].copy()
    for col in CATEGORICAL_FEATURES:
        if col in X_valid.columns:
            X_valid[col] = X_valid[col].astype("category")

    y_pred = model.predict(X_valid)
    valid_df["pred_score"] = y_pred

    # --- レースごとに予想1位 × 人気1位 を判定 ---
    key_cols = [f"_key_{c}" for c in RACE_KEY_COLS if f"_key_{c}" in valid_df.columns]
    evaluator = ModelEvaluator()

    # 馬単分析用の記録
    umatan_records = []

    # 払戻データ取得
    harai_data = evaluator._get_harai_data(valid_df)

    for group_key, group in valid_df.groupby(key_cols):
        if len(group) < 2:
            continue

        race_key_vals = dict(zip(RACE_KEY_COLS, group_key)) if isinstance(group_key, tuple) else {}

        # 予測上位2頭を特定
        top2 = group.nlargest(2, "pred_score")
        pred_top1 = top2.iloc[0]
        pred_top2 = top2.iloc[1]
        pred_top1_umaban = evaluator._format_umaban(pred_top1.get("post_umaban", ""))
        pred_top2_umaban = evaluator._format_umaban(pred_top2.get("post_umaban", ""))

        # オッズ取得 → 人気1位を判定
        odds_dict = evaluator._get_odds_from_db(race_key_vals)
        if not odds_dict:
            continue

        ninki_ranks = evaluator._derive_ninki_rank(odds_dict)
        pred_top1_ninki = ninki_ranks.get(pred_top1_umaban, 99)

        # 条件: 予想1位 ≠ 人気1位 かつ オッズ10.1倍以上
        if pred_top1_ninki == 1:
            continue
        pred_top1_odds = odds_dict.get(pred_top1_umaban, 0)
        if pred_top1_odds < 10.1:
            continue

        # 人気1位の馬番を特定
        ninki1_umaban = ""
        for uma, rank in ninki_ranks.items():
            if rank == 1:
                ninki1_umaban = uma
                break

        race_key_str = evaluator._race_key_str(race_key_vals)
        race_harai = harai_data.get(race_key_str, {})
        umatan_harai = race_harai.get("umatan", {})
        tansho_harai = race_harai.get("tansho", {})

        # 戦略A: 予想1位→予想2位 の馬単
        kumi_a = evaluator._make_kumi_umatan(pred_top1_umaban, pred_top2_umaban)
        payout_a = umatan_harai.get(kumi_a, 0)

        # 戦略B: 予想1位→人気1位 の馬単
        kumi_b = evaluator._make_kumi_umatan(pred_top1_umaban, ninki1_umaban)
        payout_b = umatan_harai.get(kumi_b, 0)

        # 単勝（参考）
        payout_tansho = tansho_harai.get(pred_top1_umaban, 0)

        kakutei = pred_top1.get("kakuteijyuni", "?")

        umatan_records.append({
            "race_key": race_key_str,
            "pred1_umaban": pred_top1_umaban,
            "pred2_umaban": pred_top2_umaban,
            "ninki1_umaban": ninki1_umaban,
            "odds": pred_top1_odds,
            "ninki": pred_top1_ninki,
            "kakuteijyuni": kakutei,
            "payout_tansho": payout_tansho,
            "kumi_a": kumi_a,
            "payout_a": payout_a,
            "kumi_b": kumi_b,
            "payout_b": payout_b,
        })

    # --- 結果表示 ---
    df = pd.DataFrame(umatan_records)
    n = len(df)

    print("\n" + "=" * 70)
    print("予想1位 ≠ 人気1位 かつ オッズ10.1倍以上 の馬単分析（2025年）")
    print("=" * 70)
    print(f"対象レース数: {n}")
    print()

    # 単勝（参考）
    tansho_wins = (df["payout_tansho"] > 0).sum()
    tansho_bet = n * 100
    tansho_ret = int(df["payout_tansho"].sum())
    print("--- 参考: 単勝 (予想1位) ---")
    print(f"  的中: {tansho_wins}/{n} ({tansho_wins/n*100:.1f}%)")
    print(f"  投資: {tansho_bet:,}円 → 回収: {tansho_ret:,}円")
    print(f"  回収率: {tansho_ret/tansho_bet*100:.1f}%  収支: {tansho_ret-tansho_bet:+,}円")

    # 戦略A: 予想1位→予想2位
    a_wins = (df["payout_a"] > 0).sum()
    a_bet = n * 100
    a_ret = int(df["payout_a"].sum())
    print()
    print("--- 戦略A: 馬単 予想1位→予想2位 ---")
    print(f"  的中: {a_wins}/{n} ({a_wins/n*100:.1f}%)")
    print(f"  投資: {a_bet:,}円 → 回収: {a_ret:,}円")
    print(f"  回収率: {a_ret/a_bet*100:.1f}%  収支: {a_ret-a_bet:+,}円")

    # 戦略B: 予想1位→人気1位
    b_wins = (df["payout_b"] > 0).sum()
    b_bet = n * 100
    b_ret = int(df["payout_b"].sum())
    print()
    print("--- 戦略B: 馬単 予想1位→人気1位 ---")
    print(f"  的中: {b_wins}/{n} ({b_wins/n*100:.1f}%)")
    print(f"  投資: {b_bet:,}円 → 回収: {b_ret:,}円")
    print(f"  回収率: {b_ret/b_bet*100:.1f}%  収支: {b_ret-b_bet:+,}円")

    # 戦略A+B 両方買い（2点買い）
    ab_bet = n * 200
    ab_ret = int(df["payout_a"].sum() + df["payout_b"].sum())
    ab_wins = ((df["payout_a"] > 0) | (df["payout_b"] > 0)).sum()
    print()
    print("--- 戦略A+B: 両方買い（2点買い） ---")
    print(f"  的中: {ab_wins}/{n} ({ab_wins/n*100:.1f}%)")
    print(f"  投資: {ab_bet:,}円 → 回収: {ab_ret:,}円")
    print(f"  回収率: {ab_ret/ab_bet*100:.1f}%  収支: {ab_ret-ab_bet:+,}円")

    # --- オッズ帯別 ---
    df["kakuteijyuni"] = pd.to_numeric(df["kakuteijyuni"], errors="coerce")
    df["odds_band"] = pd.cut(
        df["odds"],
        bins=[10, 20, 50, 200],
        labels=["10.1~20.0", "20.1~50.0", "50.1~"],
    )
    print()
    print("--- オッズ帯別成績 ---")
    print(f"{'オッズ帯':<14} {'件数':>4} | {'単勝':>10} | {'馬単A(→予2)':>12} | {'馬単B(→人1)':>12} | {'A+B両方':>10}")
    print("-" * 80)
    for band, grp in df.groupby("odds_band", observed=True):
        gn = len(grp)
        gb = gn * 100
        t_r = grp["payout_tansho"].sum()
        a_r = grp["payout_a"].sum()
        b_r = grp["payout_b"].sum()
        t_roi = t_r / gb * 100 if gb > 0 else 0
        a_roi = a_r / gb * 100 if gb > 0 else 0
        b_roi = b_r / gb * 100 if gb > 0 else 0
        ab_roi = (a_r + b_r) / (gb * 2) * 100 if gb > 0 else 0
        print(f"  {band:<12} {gn:>4} | {t_roi:>8.1f}% | {a_roi:>10.1f}% | {b_roi:>10.1f}% | {ab_roi:>8.1f}%")

    # --- 人気順別 ---
    print()
    print("--- 予想1位の人気順別成績 ---")
    print(f"{'人気':<8} {'件数':>4} | {'単勝':>10} | {'馬単A(→予2)':>12} | {'馬単B(→人1)':>12} | {'A+B両方':>10}")
    print("-" * 80)
    for ninki, grp in df.groupby("ninki"):
        gn = len(grp)
        gb = gn * 100
        t_r = grp["payout_tansho"].sum()
        a_r = grp["payout_a"].sum()
        b_r = grp["payout_b"].sum()
        t_roi = t_r / gb * 100 if gb > 0 else 0
        a_roi = a_r / gb * 100 if gb > 0 else 0
        b_roi = b_r / gb * 100 if gb > 0 else 0
        ab_roi = (a_r + b_r) / (gb * 2) * 100 if gb > 0 else 0
        print(f"  {int(ninki)}位   {gn:>4} | {t_roi:>8.1f}% | {a_roi:>10.1f}% | {b_roi:>10.1f}% | {ab_roi:>8.1f}%")


if __name__ == "__main__":
    main()
