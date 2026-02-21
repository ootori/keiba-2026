"""初ブリンカーの回収率分析.

「初ブリンカーの馬は買い」という格言の検証。
初めてブリンカーを装着したレースの単勝・複勝回収率を、
それ以外のケースと比較する。
"""

from __future__ import annotations

import sys
sys.path.insert(0, "/Users/kotaniwa/src/keiba-2026")

import pandas as pd
import numpy as np
from src.db import query_df

RACE_KEY = ["year", "monthday", "jyocd", "kaiji", "nichiji", "racenum"]


def fetch_blinker_data(year_start: int = 2015, year_end: int = 2025) -> pd.DataFrame:
    """ブリンカー使用情報と着順を取得する."""
    sql = """
    SELECT
        ur.year, ur.monthday, ur.jyocd, ur.kaiji, ur.nichiji, ur.racenum,
        ur.kettonum, ur.umaban, ur.blinker,
        ur.kakuteijyuni,
        ur.datakubun,
        ur.ijyocd
    FROM n_uma_race ur
    WHERE ur.datakubun = '7'
      AND ur.ijyocd = '0'
      AND CAST(ur.year AS integer) BETWEEN %(year_start)s AND %(year_end)s
      AND ur.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
    ORDER BY ur.kettonum, ur.year, ur.monthday
    """
    return query_df(sql, {"year_start": year_start, "year_end": year_end})


def fetch_harai_data(year_start: int = 2015, year_end: int = 2025) -> tuple[pd.DataFrame, pd.DataFrame]:
    """払戻データを取得し、単勝・複勝を縦持ちに変換する."""
    # 単勝: paytansyoumaban1-3, paytansyopay1-3
    # 複勝: payfukusyoumaban1-5, payfukusyopay1-5
    sql = """
    SELECT year, monthday, jyocd, kaiji, nichiji, racenum,
           paytansyoumaban1, paytansyopay1,
           paytansyoumaban2, paytansyopay2,
           paytansyoumaban3, paytansyopay3,
           payfukusyoumaban1, payfukusyopay1,
           payfukusyoumaban2, payfukusyopay2,
           payfukusyoumaban3, payfukusyopay3,
           payfukusyoumaban4, payfukusyopay4,
           payfukusyoumaban5, payfukusyopay5
    FROM n_harai
    WHERE datakubun IN ('1', '2')
      AND CAST(year AS integer) BETWEEN %(year_start)s AND %(year_end)s
      AND jyocd IN ('01','02','03','04','05','06','07','08','09','10')
    """
    df = query_df(sql, {"year_start": year_start, "year_end": year_end})

    def _to_long(df: pd.DataFrame, prefix: str, pairs: list[tuple[str, str]]) -> pd.DataFrame:
        records = []
        for _, row in df.iterrows():
            rk = {k: row[k] for k in RACE_KEY}
            for uma_col, pay_col in pairs:
                uma = str(row[uma_col]).strip()
                pay = str(row[pay_col]).strip()
                if uma and pay:
                    try:
                        uma_int = int(uma)
                        pay_int = int(pay)
                        if uma_int > 0 and pay_int > 0:
                            records.append({
                                **rk,
                                "umaban_int": uma_int,
                                f"{prefix}_pay": pay_int,
                            })
                    except (ValueError, TypeError):
                        pass
        return pd.DataFrame(records)

    tansho_pairs = [
        ("paytansyoumaban1", "paytansyopay1"),
        ("paytansyoumaban2", "paytansyopay2"),
        ("paytansyoumaban3", "paytansyopay3"),
    ]
    fukusho_pairs = [
        ("payfukusyoumaban1", "payfukusyopay1"),
        ("payfukusyoumaban2", "payfukusyopay2"),
        ("payfukusyoumaban3", "payfukusyopay3"),
        ("payfukusyoumaban4", "payfukusyopay4"),
        ("payfukusyoumaban5", "payfukusyopay5"),
    ]

    tansho_df = _to_long(df, "tansho", tansho_pairs)
    fukusho_df = _to_long(df, "fukusho", fukusho_pairs)

    # 払戻金額は100円単位で格納されている（420 = 420円 for 100円購入）
    # そのまま使えばよい

    return tansho_df, fukusho_df


def identify_first_blinker(df: pd.DataFrame) -> pd.DataFrame:
    """各馬の初ブリンカーレースを特定する."""
    df = df.copy()
    df["blinker_int"] = df["blinker"].apply(lambda x: 1 if str(x).strip() == "1" else 0)
    df = df.sort_values(["kettonum", "year", "monthday"]).reset_index(drop=True)

    # 馬ごとにこれまでのブリンカー累計使用回数を計算
    df["prev_blinker_cumsum"] = df.groupby("kettonum")["blinker_int"].cumsum() - df["blinker_int"]

    # 初ブリンカー = 今回ブリンカー使用 AND これまでの累計使用回数が0
    df["is_first_blinker"] = (df["blinker_int"] == 1) & (df["prev_blinker_cumsum"] == 0)

    # カテゴリ分け
    conditions = [
        df["is_first_blinker"],
        (df["blinker_int"] == 1) & (~df["is_first_blinker"]),
        df["blinker_int"] == 0,
    ]
    labels = ["初ブリンカー", "2回目以降ブリンカー", "ブリンカーなし"]
    df["blinker_category"] = np.select(conditions, labels, default="不明")

    return df


def merge_harai(df: pd.DataFrame, tansho_df: pd.DataFrame, fukusho_df: pd.DataFrame) -> pd.DataFrame:
    """払戻データを結合する."""
    df = df.copy()
    df["jyuni"] = pd.to_numeric(df["kakuteijyuni"], errors="coerce")
    df["umaban_int"] = pd.to_numeric(df["umaban"], errors="coerce").astype("Int64")

    merge_keys = RACE_KEY + ["umaban_int"]

    if not tansho_df.empty:
        df = df.merge(tansho_df, on=merge_keys, how="left")
        df["tansho_pay"] = df["tansho_pay"].fillna(0)
    else:
        df["tansho_pay"] = 0

    if not fukusho_df.empty:
        df = df.merge(fukusho_df, on=merge_keys, how="left")
        df["fukusho_pay"] = df["fukusho_pay"].fillna(0)
    else:
        df["fukusho_pay"] = 0

    return df


def calc_stats(df: pd.DataFrame, group_col: str = "blinker_category") -> pd.DataFrame:
    """カテゴリ別の統計を計算する."""
    results = []
    for cat in ["初ブリンカー", "2回目以降ブリンカー", "ブリンカーなし"]:
        sub = df[df[group_col] == cat]
        n = len(sub)
        if n == 0:
            continue

        win_rate = (sub["jyuni"] == 1).sum() / n * 100
        place_rate = (sub["jyuni"] <= 3).sum() / n * 100
        tansho_roi = sub["tansho_pay"].sum() / n  # 100円あたりの回収（円）
        fukusho_roi = sub["fukusho_pay"].sum() / n

        results.append({
            "カテゴリ": cat,
            "出走数": n,
            "勝率(%)": round(win_rate, 2),
            "複勝率(%)": round(place_rate, 2),
            "単勝回収率(%)": round(tansho_roi, 1),
            "複勝回収率(%)": round(fukusho_roi, 1),
        })

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("初ブリンカー回収率分析")
    print("=" * 70)

    year_start, year_end = 2015, 2025

    print(f"\n[1] データ取得中 ({year_start}-{year_end})...")
    race_df = fetch_blinker_data(year_start, year_end)
    print(f"    出走データ: {len(race_df):,} 件")

    print("[2] 払戻データ取得中...")
    tansho_df, fukusho_df = fetch_harai_data(year_start, year_end)
    print(f"    単勝払戻: {len(tansho_df):,} 件, 複勝払戻: {len(fukusho_df):,} 件")
    # 払戻金額のサニティチェック
    if not tansho_df.empty:
        print(f"    単勝払戻金額: 平均{tansho_df['tansho_pay'].mean():.0f}円, "
              f"中央値{tansho_df['tansho_pay'].median():.0f}円, "
              f"最大{tansho_df['tansho_pay'].max():.0f}円")

    print("[3] 初ブリンカー判定中...")
    # 正確な判定のため2001年以降の全履歴で判定
    full_df = fetch_blinker_data(2001, year_end)
    full_df = identify_first_blinker(full_df)

    # 分析対象期間に絞る
    df = full_df[full_df["year"].apply(lambda x: int(x) >= year_start)].copy()

    for cat in ["初ブリンカー", "2回目以降ブリンカー", "ブリンカーなし"]:
        cnt = (df["blinker_category"] == cat).sum()
        print(f"    {cat}: {cnt:,} 件")

    print("[4] 払戻データ結合中...")
    df = merge_harai(df, tansho_df, fukusho_df)

    print(f"\n{'=' * 70}")
    print("■ 全期間集計 (2015-2025)")
    print("=" * 70)
    summary = calc_stats(df)
    print(summary.to_string(index=False))

    # 年度別集計
    print(f"\n{'=' * 70}")
    print("■ 年度別集計")
    print("=" * 70)

    yearly_results = []
    for year_val in sorted(df["year"].unique(), key=int):
        yr = int(year_val)
        if yr < year_start:
            continue
        sub = df[df["year"] == year_val]
        for cat in ["初ブリンカー", "2回目以降ブリンカー", "ブリンカーなし"]:
            csub = sub[sub["blinker_category"] == cat]
            n = len(csub)
            if n == 0:
                continue
            yearly_results.append({
                "年": yr,
                "カテゴリ": cat,
                "出走数": n,
                "勝率(%)": round((csub["jyuni"] == 1).sum() / n * 100, 2),
                "複勝率(%)": round((csub["jyuni"] <= 3).sum() / n * 100, 2),
                "単勝回収率(%)": round(csub["tansho_pay"].sum() / n, 1),
                "複勝回収率(%)": round(csub["fukusho_pay"].sum() / n, 1),
            })

    yearly_df = pd.DataFrame(yearly_results)

    # 単勝回収率ピボット
    print("\n単勝回収率(%):")
    pivot_t = yearly_df.pivot_table(index="年", columns="カテゴリ", values="単勝回収率(%)", aggfunc="first")
    pivot_t = pivot_t[["初ブリンカー", "2回目以降ブリンカー", "ブリンカーなし"]]
    print(pivot_t.to_string())

    # 複勝回収率ピボット
    print("\n複勝回収率(%):")
    pivot_f = yearly_df.pivot_table(index="年", columns="カテゴリ", values="複勝回収率(%)", aggfunc="first")
    pivot_f = pivot_f[["初ブリンカー", "2回目以降ブリンカー", "ブリンカーなし"]]
    print(pivot_f.to_string())

    # 勝率ピボット
    print("\n勝率(%):")
    pivot_w = yearly_df.pivot_table(index="年", columns="カテゴリ", values="勝率(%)", aggfunc="first")
    pivot_w = pivot_w[["初ブリンカー", "2回目以降ブリンカー", "ブリンカーなし"]]
    print(pivot_w.to_string())

    # 複勝率ピボット
    print("\n複勝率(%):")
    pivot_p = yearly_df.pivot_table(index="年", columns="カテゴリ", values="複勝率(%)", aggfunc="first")
    pivot_p = pivot_p[["初ブリンカー", "2回目以降ブリンカー", "ブリンカーなし"]]
    print(pivot_p.to_string())

    # 差分サマリ
    print(f"\n{'=' * 70}")
    print("■ 初ブリンカー vs ブリンカーなし（差分）")
    print("=" * 70)
    print(f"{'年':>6} {'単勝回収率差':>12} {'複勝回収率差':>12} {'勝率差':>8} {'複勝率差':>8}")
    for year_val in sorted(df["year"].unique(), key=int):
        yr = int(year_val)
        if yr < year_start:
            continue
        fb = yearly_df[(yearly_df["年"] == yr) & (yearly_df["カテゴリ"] == "初ブリンカー")]
        nb = yearly_df[(yearly_df["年"] == yr) & (yearly_df["カテゴリ"] == "ブリンカーなし")]
        if not fb.empty and not nb.empty:
            t_diff = fb.iloc[0]["単勝回収率(%)"] - nb.iloc[0]["単勝回収率(%)"]
            f_diff = fb.iloc[0]["複勝回収率(%)"] - nb.iloc[0]["複勝回収率(%)"]
            w_diff = fb.iloc[0]["勝率(%)"] - nb.iloc[0]["勝率(%)"]
            p_diff = fb.iloc[0]["複勝率(%)"] - nb.iloc[0]["複勝率(%)"]
            print(f"{yr:>6} {t_diff:>+12.1f}pt {f_diff:>+12.1f}pt {w_diff:>+8.2f}pt {p_diff:>+8.2f}pt")

    # 全期間の差分
    fb_all = summary[summary["カテゴリ"] == "初ブリンカー"]
    nb_all = summary[summary["カテゴリ"] == "ブリンカーなし"]
    if not fb_all.empty and not nb_all.empty:
        print(f"\n{'全期間':>6} "
              f"{fb_all.iloc[0]['単勝回収率(%)'] - nb_all.iloc[0]['単勝回収率(%)']:>+12.1f}pt "
              f"{fb_all.iloc[0]['複勝回収率(%)'] - nb_all.iloc[0]['複勝回収率(%)']:>+12.1f}pt "
              f"{fb_all.iloc[0]['勝率(%)'] - nb_all.iloc[0]['勝率(%)']:>+8.2f}pt "
              f"{fb_all.iloc[0]['複勝率(%)'] - nb_all.iloc[0]['複勝率(%)']:>+8.2f}pt")

    print(f"\n{'=' * 70}")
    print("分析完了")


if __name__ == "__main__":
    main()
