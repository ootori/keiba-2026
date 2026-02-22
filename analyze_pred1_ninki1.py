"""予想1位かつ人気1位の馬のみの単勝回収率を算出する."""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import lightgbm as lgb
import pandas as pd
import numpy as np

from src.config import (
    MODEL_DIR, RACE_KEY_COLS, CATEGORICAL_FEATURES,
)
from src.features.pipeline import FeaturePipeline
from src.db import query_df

RACE_KEYS = ["year", "monthday", "jyocd", "kaiji", "nichiji", "racenum"]
KEY_COLS = [f"_key_{c}" for c in RACE_KEY_COLS]


def main() -> None:
    model_name = "ranking_supp_model"

    # モデル・特徴量リストをロード
    model = lgb.Booster(model_file=str(MODEL_DIR / f"{model_name}.txt"))
    with open(MODEL_DIR / f"{model_name}_features.txt") as f:
        feature_columns = [line.strip() for line in f if line.strip()]

    # 2025年の検証データをロード（サプリメント込み）
    pipeline = FeaturePipeline(include_odds=False)
    df = pipeline.load_years(2025, 2025, supplement_names=["bms_detail", "rating"])
    print(f"2025年データ: {len(df)}行")

    # 予測
    X = df[feature_columns].copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")
    df["pred_score"] = model.predict(X)

    # レースごとに予想順位を付与
    df["pred_rank"] = df.groupby(KEY_COLS)["pred_score"].rank(
        ascending=False, method="first"
    ).astype(int)

    # 馬番を2桁ゼロ埋め（DBと合わせる）
    df["umaban_02"] = df["post_umaban"].astype(int).apply(lambda x: f"{x:02d}")

    # ------------------------------------------------------------------
    # 人気順の取得（n_odds_tanpukuから単勝オッズで導出）
    # ------------------------------------------------------------------
    odds_df = query_df("""
        SELECT
            year, monthday, jyocd, kaiji, nichiji, racenum,
            umaban,
            CAST(tanodds AS numeric) AS tanodds
        FROM n_odds_tanpuku
        WHERE year = '2025'
          AND tanodds ~ '^[0-9]+'
          AND CAST(tanodds AS numeric) > 0
    """)
    odds_df["tanodds"] = odds_df["tanodds"].astype(float)
    print(f"オッズデータ: {len(odds_df)}行")

    # レース内で人気順（オッズ昇順）
    odds_df["ninki_rank"] = odds_df.groupby(RACE_KEYS)["tanodds"].rank(
        method="first"
    ).astype(int)

    # dfにマージ（キーを揃える）
    odds_merge = odds_df.rename(
        columns={c: f"_key_{c}" for c in RACE_KEY_COLS}
    ).rename(columns={"umaban": "umaban_02"})

    df = df.merge(
        odds_merge[KEY_COLS + ["umaban_02", "tanodds", "ninki_rank"]],
        on=KEY_COLS + ["umaban_02"],
        how="left",
    )
    n_matched = df["ninki_rank"].notna().sum()
    print(f"オッズマージ: {n_matched}/{len(df)} 行にオッズあり")

    # ------------------------------------------------------------------
    # 払戻データの取得
    # ------------------------------------------------------------------
    harai_df = query_df("""
        SELECT *
        FROM n_harai
        WHERE year = '2025'
          AND datakubun IN ('1', '2')
    """)
    print(f"払戻データ: {len(harai_df)}行")

    # 単勝払戻を縦持ちに変換
    tansho_records = []
    for i in range(1, 4):
        ub_col = f"paytansyoumaban{i}"
        pay_col = f"paytansyopay{i}"
        if ub_col in harai_df.columns and pay_col in harai_df.columns:
            sub = harai_df[RACE_KEYS + [ub_col, pay_col]].copy()
            sub = sub.rename(columns={ub_col: "win_umaban", pay_col: "win_pay"})
            sub = sub[sub["win_umaban"].notna() & (sub["win_umaban"].str.strip() != "")]
            tansho_records.append(sub)

    tansho_df = pd.concat(tansho_records, ignore_index=True)
    tansho_df["win_umaban"] = tansho_df["win_umaban"].str.strip()
    tansho_df["win_pay"] = pd.to_numeric(tansho_df["win_pay"], errors="coerce").fillna(0).astype(int)

    # 複勝払戻を縦持ちに変換
    fukusho_records = []
    for i in range(1, 6):
        ub_col = f"payfukusyoumaban{i}"
        pay_col = f"payfukusyopay{i}"
        if ub_col in harai_df.columns and pay_col in harai_df.columns:
            sub = harai_df[RACE_KEYS + [ub_col, pay_col]].copy()
            sub = sub.rename(columns={ub_col: "place_umaban", pay_col: "place_pay"})
            sub = sub[sub["place_umaban"].notna() & (sub["place_umaban"].str.strip() != "")]
            fukusho_records.append(sub)

    fukusho_df = pd.concat(fukusho_records, ignore_index=True)
    fukusho_df["place_umaban"] = fukusho_df["place_umaban"].str.strip()
    fukusho_df["place_pay"] = pd.to_numeric(fukusho_df["place_pay"], errors="coerce").fillna(0).astype(int)

    # 払戻をリネームしてマージ用に準備
    tansho_m = tansho_df.rename(columns={c: f"_key_{c}" for c in RACE_KEY_COLS})
    tansho_m = tansho_m.rename(columns={"win_umaban": "umaban_02"})

    fukusho_m = fukusho_df.rename(columns={c: f"_key_{c}" for c in RACE_KEY_COLS})
    fukusho_m = fukusho_m.rename(columns={"place_umaban": "umaban_02"})

    # ------------------------------------------------------------------
    # 予想1位 AND 人気1位の馬を抽出
    # ------------------------------------------------------------------
    mask = (df["pred_rank"] == 1) & (df["ninki_rank"] == 1)
    target = df[mask].copy()
    all_races = df.groupby(KEY_COLS).ngroups
    print(f"\n予想1位かつ人気1位: {len(target)}頭 / 全{all_races}レース ({len(target)/all_races*100:.1f}%)")

    # 単勝払戻をマージ
    target = target.merge(
        tansho_m[KEY_COLS + ["umaban_02", "win_pay"]],
        on=KEY_COLS + ["umaban_02"],
        how="left",
    )
    target["win_pay"] = target["win_pay"].fillna(0).astype(int)

    # 複勝払戻をマージ
    target = target.merge(
        fukusho_m[KEY_COLS + ["umaban_02", "place_pay"]],
        on=KEY_COLS + ["umaban_02"],
        how="left",
    )
    target["place_pay"] = target["place_pay"].fillna(0).astype(int)

    # ------------------------------------------------------------------
    # 結果集計
    # ------------------------------------------------------------------
    n = len(target)
    bet = n * 100

    tansho_return = target["win_pay"].sum()
    tansho_hit = (target["win_pay"] > 0).sum()
    roi_t = tansho_return / bet * 100 if bet > 0 else 0

    fukusho_return = target["place_pay"].sum()
    fukusho_hit = (target["place_pay"] > 0).sum()
    roi_f = fukusho_return / bet * 100 if bet > 0 else 0

    print("\n" + "=" * 60)
    print("【結果】予想1位 AND 人気1位 の馬だけ買った場合（2025年）")
    print("=" * 60)
    print(f"対象レース数: {n} / {all_races} ({n/all_races*100:.1f}%)")

    print(f"\n--- 単勝 ---")
    print(f"購入金額:  {bet:,}円")
    print(f"払戻金額:  {tansho_return:,}円")
    print(f"回収率:    {roi_t:.1f}%")
    print(f"的中率:    {tansho_hit}/{n} ({tansho_hit/n*100:.1f}%)")
    if tansho_hit > 0:
        print(f"平均配当:  {tansho_return/tansho_hit:.0f}円")

    print(f"\n--- 複勝 ---")
    print(f"購入金額:  {bet:,}円")
    print(f"払戻金額:  {fukusho_return:,}円")
    print(f"回収率:    {roi_f:.1f}%")
    print(f"的中率:    {fukusho_hit}/{n} ({fukusho_hit/n*100:.1f}%)")
    if fukusho_hit > 0:
        print(f"平均配当:  {fukusho_return/fukusho_hit:.0f}円")

    # 着順分布
    if "kakuteijyuni" in target.columns:
        finish = pd.to_numeric(target["kakuteijyuni"], errors="coerce").dropna().astype(int)
        print(f"\n--- 着順分布 ---")
        for j in range(1, 6):
            cnt = (finish == j).sum()
            print(f"  {j}着: {cnt}頭 ({cnt/n*100:.1f}%)")
        cnt_rest = (finish > 5).sum()
        print(f"  6着以下: {cnt_rest}頭 ({cnt_rest/n*100:.1f}%)")
        top3 = (finish <= 3).sum()
        print(f"  3着以内率: {top3}/{n} ({top3/n*100:.1f}%)")

    # ------------------------------------------------------------------
    # 比較: 予想1位のみ（人気不問）
    # ------------------------------------------------------------------
    pred1 = df[df["pred_rank"] == 1].copy()
    pred1 = pred1.merge(
        tansho_m[KEY_COLS + ["umaban_02", "win_pay"]],
        on=KEY_COLS + ["umaban_02"], how="left",
    )
    pred1["win_pay"] = pred1["win_pay"].fillna(0).astype(int)
    pred1 = pred1.merge(
        fukusho_m[KEY_COLS + ["umaban_02", "place_pay"]],
        on=KEY_COLS + ["umaban_02"], how="left",
    )
    pred1["place_pay"] = pred1["place_pay"].fillna(0).astype(int)

    n1 = len(pred1)
    print(f"\n--- 比較: 予想1位のみ（人気不問）---")
    print(f"対象: {n1}レース")
    roi1_t = pred1["win_pay"].sum() / (n1 * 100) * 100
    roi1_f = pred1["place_pay"].sum() / (n1 * 100) * 100
    hit1_t = (pred1["win_pay"] > 0).sum()
    hit1_f = (pred1["place_pay"] > 0).sum()
    print(f"単勝: 回収率={roi1_t:.1f}%, 的中率={hit1_t}/{n1} ({hit1_t/n1*100:.1f}%)")
    print(f"複勝: 回収率={roi1_f:.1f}%, 的中率={hit1_f}/{n1} ({hit1_f/n1*100:.1f}%)")

    # ------------------------------------------------------------------
    # 比較: 人気1位のみ（予想不問）
    # ------------------------------------------------------------------
    ninki1 = df[df["ninki_rank"] == 1].copy()
    ninki1 = ninki1.merge(
        tansho_m[KEY_COLS + ["umaban_02", "win_pay"]],
        on=KEY_COLS + ["umaban_02"], how="left",
    )
    ninki1["win_pay"] = ninki1["win_pay"].fillna(0).astype(int)
    ninki1 = ninki1.merge(
        fukusho_m[KEY_COLS + ["umaban_02", "place_pay"]],
        on=KEY_COLS + ["umaban_02"], how="left",
    )
    ninki1["place_pay"] = ninki1["place_pay"].fillna(0).astype(int)

    nn = len(ninki1)
    print(f"\n--- 比較: 人気1位のみ（予想不問）---")
    print(f"対象: {nn}レース")
    roi_n_t = ninki1["win_pay"].sum() / (nn * 100) * 100
    roi_n_f = ninki1["place_pay"].sum() / (nn * 100) * 100
    hit_n_t = (ninki1["win_pay"] > 0).sum()
    hit_n_f = (ninki1["place_pay"] > 0).sum()
    print(f"単勝: 回収率={roi_n_t:.1f}%, 的中率={hit_n_t}/{nn} ({hit_n_t/nn*100:.1f}%)")
    print(f"複勝: 回収率={roi_n_f:.1f}%, 的中率={hit_n_f}/{nn} ({hit_n_f/nn*100:.1f}%)")

    # ------------------------------------------------------------------
    # 比較: 予想1位だが人気1位ではない馬
    # ------------------------------------------------------------------
    pred1_not_ninki1 = df[(df["pred_rank"] == 1) & (df["ninki_rank"] != 1) & df["ninki_rank"].notna()].copy()
    pred1_not_ninki1 = pred1_not_ninki1.merge(
        tansho_m[KEY_COLS + ["umaban_02", "win_pay"]],
        on=KEY_COLS + ["umaban_02"], how="left",
    )
    pred1_not_ninki1["win_pay"] = pred1_not_ninki1["win_pay"].fillna(0).astype(int)
    pred1_not_ninki1 = pred1_not_ninki1.merge(
        fukusho_m[KEY_COLS + ["umaban_02", "place_pay"]],
        on=KEY_COLS + ["umaban_02"], how="left",
    )
    pred1_not_ninki1["place_pay"] = pred1_not_ninki1["place_pay"].fillna(0).astype(int)

    npn = len(pred1_not_ninki1)
    if npn > 0:
        print(f"\n--- 比較: 予想1位だが人気1位ではない馬 ---")
        print(f"対象: {npn}レース")
        roi_pn_t = pred1_not_ninki1["win_pay"].sum() / (npn * 100) * 100
        roi_pn_f = pred1_not_ninki1["place_pay"].sum() / (npn * 100) * 100
        hit_pn_t = (pred1_not_ninki1["win_pay"] > 0).sum()
        hit_pn_f = (pred1_not_ninki1["place_pay"] > 0).sum()
        print(f"単勝: 回収率={roi_pn_t:.1f}%, 的中率={hit_pn_t}/{npn} ({hit_pn_t/npn*100:.1f}%)")
        print(f"複勝: 回収率={roi_pn_f:.1f}%, 的中率={hit_pn_f}/{npn} ({hit_pn_f/npn*100:.1f}%)")

    # ------------------------------------------------------------------
    # 馬単検証: 予想1位→人気1位（予想1位≠人気1位のレースのみ）
    # ------------------------------------------------------------------

    # 馬単払戻を縦持ちに変換
    umatan_records = []
    for i in range(1, 7):
        kumi_col = f"payumatankumi{i}"
        pay_col = f"payumatanpay{i}"
        if kumi_col in harai_df.columns and pay_col in harai_df.columns:
            sub = harai_df[RACE_KEYS + [kumi_col, pay_col]].copy()
            sub = sub.rename(columns={kumi_col: "umatan_kumi", pay_col: "umatan_pay"})
            sub = sub[sub["umatan_kumi"].notna() & (sub["umatan_kumi"].str.strip() != "")]
            umatan_records.append(sub)

    umatan_df = pd.concat(umatan_records, ignore_index=True)
    umatan_df["umatan_kumi"] = umatan_df["umatan_kumi"].str.strip()
    umatan_df["umatan_pay"] = pd.to_numeric(
        umatan_df["umatan_pay"], errors="coerce"
    ).fillna(0).astype(int)

    umatan_m = umatan_df.rename(columns={c: f"_key_{c}" for c in RACE_KEY_COLS})

    # 予想1位≠人気1位のレースを抽出し、そのレースの人気1位馬を特定
    pred1_df = df[df["pred_rank"] == 1].copy()
    ninki1_df = df[df["ninki_rank"] == 1].copy()

    # 予想1位の馬番（1着側）
    pred1_for_umatan = pred1_df[KEY_COLS + ["umaban_02"]].rename(
        columns={"umaban_02": "pred1_umaban"}
    )
    # 人気1位の馬番（2着側）
    ninki1_for_umatan = ninki1_df[KEY_COLS + ["umaban_02"]].rename(
        columns={"umaban_02": "ninki1_umaban"}
    )

    # レース単位でマージ
    umatan_target = pred1_for_umatan.merge(
        ninki1_for_umatan, on=KEY_COLS, how="inner"
    )

    # 予想1位≠人気1位のレースのみ
    umatan_target = umatan_target[
        umatan_target["pred1_umaban"] != umatan_target["ninki1_umaban"]
    ].copy()

    # 馬単の組番を作成（予想1位→人気1位の順）
    umatan_target["umatan_kumi"] = (
        umatan_target["pred1_umaban"] + umatan_target["ninki1_umaban"]
    )

    # 払戻をマージ
    umatan_target = umatan_target.merge(
        umatan_m[KEY_COLS + ["umatan_kumi", "umatan_pay"]],
        on=KEY_COLS + ["umatan_kumi"],
        how="left",
    )
    umatan_target["umatan_pay"] = umatan_target["umatan_pay"].fillna(0).astype(int)

    n_ut = len(umatan_target)
    bet_ut = n_ut * 100
    return_ut = umatan_target["umatan_pay"].sum()
    hit_ut = (umatan_target["umatan_pay"] > 0).sum()
    roi_ut = return_ut / bet_ut * 100 if bet_ut > 0 else 0

    print(f"\n{'=' * 60}")
    print("【馬単】予想1位→人気1位（予想1位≠人気1位のレースのみ）")
    print("=" * 60)
    print(f"対象レース数: {n_ut} / {all_races} ({n_ut/all_races*100:.1f}%)")
    print(f"購入金額:  {bet_ut:,}円")
    print(f"払戻金額:  {return_ut:,}円")
    print(f"回収率:    {roi_ut:.1f}%")
    print(f"的中率:    {hit_ut}/{n_ut} ({hit_ut/n_ut*100:.1f}%)")
    if hit_ut > 0:
        print(f"平均配当:  {return_ut/hit_ut:.0f}円")

    # 的中レースの配当分布
    if hit_ut > 0:
        hits = umatan_target[umatan_target["umatan_pay"] > 0]["umatan_pay"]
        print(f"\n--- 的中時の配当分布 ---")
        print(f"  最小:   {hits.min():,}円")
        print(f"  中央値: {hits.median():,.0f}円")
        print(f"  平均:   {hits.mean():,.0f}円")
        print(f"  最大:   {hits.max():,}円")

    # 参考: 逆順（人気1位→予想1位）も計算
    umatan_target_rev = umatan_target.copy()
    umatan_target_rev["umatan_kumi_rev"] = (
        umatan_target_rev["ninki1_umaban"] + umatan_target_rev["pred1_umaban"]
    )
    umatan_target_rev = umatan_target_rev.merge(
        umatan_m[KEY_COLS + ["umatan_kumi", "umatan_pay"]].rename(
            columns={"umatan_kumi": "umatan_kumi_rev", "umatan_pay": "umatan_pay_rev"}
        ),
        on=KEY_COLS + ["umatan_kumi_rev"],
        how="left",
    )
    umatan_target_rev["umatan_pay_rev"] = umatan_target_rev["umatan_pay_rev"].fillna(0).astype(int)

    return_rev = umatan_target_rev["umatan_pay_rev"].sum()
    hit_rev = (umatan_target_rev["umatan_pay_rev"] > 0).sum()
    roi_rev = return_rev / bet_ut * 100 if bet_ut > 0 else 0

    print(f"\n--- 参考: 逆順（人気1位→予想1位）---")
    print(f"払戻金額:  {return_rev:,}円")
    print(f"回収率:    {roi_rev:.1f}%")
    print(f"的中率:    {hit_rev}/{n_ut} ({hit_rev/n_ut*100:.1f}%)")

    # 両方向合算（馬連相当）
    print(f"\n--- 参考: 両方向購入（200円/レース）---")
    total_both = return_ut + return_rev
    bet_both = bet_ut * 2
    roi_both = total_both / bet_both * 100 if bet_both > 0 else 0
    hit_both = hit_ut + hit_rev
    print(f"購入金額:  {bet_both:,}円")
    print(f"払戻金額:  {total_both:,}円")
    print(f"回収率:    {roi_both:.1f}%")
    print(f"的中数:    {hit_both}/{n_ut * 2}")

    # ------------------------------------------------------------------
    # 馬単検証2: 予想1位→予想2位
    #   条件: 予想1位≠人気1位 かつ 予想2位≠人気1位
    #   （予想Top2がどちらも1番人気ではないレース）
    # ------------------------------------------------------------------
    pred2_df = df[df["pred_rank"] == 2].copy()

    # 予想2位の馬番
    pred2_for_umatan = pred2_df[KEY_COLS + ["umaban_02", "ninki_rank"]].rename(
        columns={"umaban_02": "pred2_umaban", "ninki_rank": "pred2_ninki"}
    )

    # 予想1位 + 予想2位 + 人気1位をレース単位で結合
    umatan2_target = pred1_for_umatan.merge(
        pred2_for_umatan, on=KEY_COLS, how="inner"
    ).merge(
        ninki1_for_umatan, on=KEY_COLS, how="inner"
    )

    # 条件: 予想1位≠人気1位 かつ 予想2位≠人気1位
    umatan2_target = umatan2_target[
        (umatan2_target["pred1_umaban"] != umatan2_target["ninki1_umaban"])
        & (umatan2_target["pred2_umaban"] != umatan2_target["ninki1_umaban"])
    ].copy()

    # 馬単の組番（予想1位→予想2位）
    umatan2_target["umatan_kumi"] = (
        umatan2_target["pred1_umaban"] + umatan2_target["pred2_umaban"]
    )

    # 払戻をマージ
    umatan2_target = umatan2_target.merge(
        umatan_m[KEY_COLS + ["umatan_kumi", "umatan_pay"]],
        on=KEY_COLS + ["umatan_kumi"],
        how="left",
    )
    umatan2_target["umatan_pay"] = umatan2_target["umatan_pay"].fillna(0).astype(int)

    n_ut2 = len(umatan2_target)
    bet_ut2 = n_ut2 * 100
    return_ut2 = umatan2_target["umatan_pay"].sum()
    hit_ut2 = (umatan2_target["umatan_pay"] > 0).sum()
    roi_ut2 = return_ut2 / bet_ut2 * 100 if bet_ut2 > 0 else 0

    print(f"\n{'=' * 60}")
    print("【馬単】予想1位→予想2位")
    print("  条件: 予想1位≠人気1位 かつ 予想2位≠人気1位")
    print("=" * 60)
    print(f"対象レース数: {n_ut2} / {all_races} ({n_ut2/all_races*100:.1f}%)")
    print(f"購入金額:  {bet_ut2:,}円")
    print(f"払戻金額:  {return_ut2:,}円")
    print(f"回収率:    {roi_ut2:.1f}%")
    print(f"的中率:    {hit_ut2}/{n_ut2} ({hit_ut2/n_ut2*100:.1f}%)")
    if hit_ut2 > 0:
        print(f"平均配当:  {return_ut2/hit_ut2:.0f}円")

    if hit_ut2 > 0:
        hits2 = umatan2_target[umatan2_target["umatan_pay"] > 0]["umatan_pay"]
        print(f"\n--- 的中時の配当分布 ---")
        print(f"  最小:   {hits2.min():,}円")
        print(f"  中央値: {hits2.median():,.0f}円")
        print(f"  平均:   {hits2.mean():,.0f}円")
        print(f"  最大:   {hits2.max():,}円")

    # 参考: 予想1位≠人気1位だが予想2位=人気1位のケース（先程の検証と比較用）
    umatan2_with_ninki = pred1_for_umatan.merge(
        pred2_for_umatan, on=KEY_COLS, how="inner"
    ).merge(
        ninki1_for_umatan, on=KEY_COLS, how="inner"
    )
    umatan2_with_ninki = umatan2_with_ninki[
        (umatan2_with_ninki["pred1_umaban"] != umatan2_with_ninki["ninki1_umaban"])
        & (umatan2_with_ninki["pred2_umaban"] == umatan2_with_ninki["ninki1_umaban"])
    ].copy()

    n_wn = len(umatan2_with_ninki)
    print(f"\n--- 内訳: 予想1位≠人気1位のレース({n_ut}件)の予想2位 ---")
    print(f"  予想2位=人気1位: {n_wn}件 ({n_wn/n_ut*100:.1f}%)")
    print(f"  予想2位≠人気1位: {n_ut2}件 ({n_ut2/n_ut*100:.1f}%)")


if __name__ == "__main__":
    main()
