"""初ダート・初芝の回収率分析.

「初ダートの馬は買い」「初芝の馬は買い」という仮説の検証。
初めてダート（芝）を走るレースの単勝・複勝回収率を、
それ以外のケースと比較する。
"""

from __future__ import annotations

import sys
sys.path.insert(0, "/Users/kotaniwa/src/keiba-2026")

import argparse
import pandas as pd
import numpy as np
from src.db import query_df

RACE_KEY = ["year", "monthday", "jyocd", "kaiji", "nichiji", "racenum"]


def fetch_race_data(year_start: int = 2015, year_end: int = 2025) -> pd.DataFrame:
    """出走情報とトラック種別を取得する."""
    sql = """
    SELECT
        ur.year, ur.monthday, ur.jyocd, ur.kaiji, ur.nichiji, ur.racenum,
        ur.kettonum, ur.umaban,
        ur.kakuteijyuni,
        ur.datakubun,
        ur.ijyocd,
        r.trackcd
    FROM n_uma_race ur
    JOIN n_race r ON r.year = ur.year AND r.monthday = ur.monthday
        AND r.jyocd = ur.jyocd AND r.kaiji = ur.kaiji
        AND r.nichiji = ur.nichiji AND r.racenum = ur.racenum
        AND r.datakubun = '7'
    WHERE ur.datakubun = '7'
      AND ur.ijyocd = '0'
      AND CAST(ur.year AS integer) BETWEEN %(year_start)s AND %(year_end)s
      AND ur.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
    ORDER BY ur.kettonum, ur.year, ur.monthday
    """
    return query_df(sql, {"year_start": year_start, "year_end": year_end})


def fetch_harai_data(year_start: int = 2015, year_end: int = 2025) -> tuple[pd.DataFrame, pd.DataFrame]:
    """払戻データを取得し、単勝・複勝を縦持ちに変換する."""
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

    return tansho_df, fukusho_df


def classify_track_type(trackcd: str) -> str:
    """トラックコードから芝/ダート/障害を判定する."""
    try:
        cd = int(trackcd)
    except (ValueError, TypeError):
        return "unknown"
    if 10 <= cd <= 22:
        return "turf"
    elif 23 <= cd <= 29:
        return "dirt"
    elif 51 <= cd <= 59:
        return "jump"
    return "unknown"


def identify_first_surface(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """各馬の初ダートまたは初芝レースを特定する.

    Args:
        df: 出走データ（kettonum, trackcdを含む）
        target: "dirt" or "turf"
    """
    df = df.copy()
    df["track_type"] = df["trackcd"].apply(classify_track_type)

    # 障害レースを除外
    df = df[df["track_type"].isin(["turf", "dirt"])].copy()

    df = df.sort_values(["kettonum", "year", "monthday"]).reset_index(drop=True)

    # 当該馬が target のトラックを走ったかフラグ
    df["is_target"] = (df["track_type"] == target).astype(int)

    # 馬ごとにこれまでの target 走行累計回数を計算
    df["prev_target_cumsum"] = df.groupby("kettonum")["is_target"].cumsum() - df["is_target"]

    # 初 target = 今回 target AND これまでの累計が0
    df["is_first"] = (df["is_target"] == 1) & (df["prev_target_cumsum"] == 0)

    if target == "dirt":
        labels = {
            "first": "初ダート",
            "repeat": "2回目以降ダート",
            "other": "芝のみ",
        }
    else:
        labels = {
            "first": "初芝",
            "repeat": "2回目以降芝",
            "other": "ダートのみ",
        }

    conditions = [
        df["is_first"],
        (df["is_target"] == 1) & (~df["is_first"]),
        df["is_target"] == 0,
    ]
    choices = [labels["first"], labels["repeat"], labels["other"]]
    df["surface_category"] = np.select(conditions, choices, default="不明")

    return df, labels


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


def calc_stats(df: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    """カテゴリ別の統計を計算する."""
    order = [labels["first"], labels["repeat"], labels["other"]]
    results = []
    for cat in order:
        sub = df[df["surface_category"] == cat]
        n = len(sub)
        if n == 0:
            continue

        win_rate = (sub["jyuni"] == 1).sum() / n * 100
        place_rate = (sub["jyuni"] <= 3).sum() / n * 100
        tansho_roi = sub["tansho_pay"].sum() / n
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


def run_analysis(target: str, year_start: int = 2015, year_end: int = 2025):
    """指定したトラック種別の初挑戦分析を実行する."""
    target_label = "初ダート" if target == "dirt" else "初芝"

    print("=" * 70)
    print(f"{target_label}回収率分析")
    print("=" * 70)

    print(f"\n[1] データ取得中 ({year_start}-{year_end})...")
    race_df = fetch_race_data(year_start, year_end)
    print(f"    出走データ: {len(race_df):,} 件")

    print("[2] 払戻データ取得中...")
    tansho_df, fukusho_df = fetch_harai_data(year_start, year_end)
    print(f"    単勝払戻: {len(tansho_df):,} 件, 複勝払戻: {len(fukusho_df):,} 件")
    if not tansho_df.empty:
        print(f"    単勝払戻金額: 平均{tansho_df['tansho_pay'].mean():.0f}円, "
              f"中央値{tansho_df['tansho_pay'].median():.0f}円, "
              f"最大{tansho_df['tansho_pay'].max():.0f}円")

    print(f"[3] {target_label}判定中...")
    # 正確な判定のため2001年以降の全履歴で判定
    full_df = fetch_race_data(2001, year_end)
    full_df, labels = identify_first_surface(full_df, target)

    # 分析対象期間に絞る
    df = full_df[full_df["year"].apply(lambda x: int(x) >= year_start)].copy()

    for cat in [labels["first"], labels["repeat"], labels["other"]]:
        cnt = (df["surface_category"] == cat).sum()
        print(f"    {cat}: {cnt:,} 件")

    print("[4] 払戻データ結合中...")
    df = merge_harai(df, tansho_df, fukusho_df)

    print(f"\n{'=' * 70}")
    print(f"■ 全期間集計 ({year_start}-{year_end})")
    print("=" * 70)
    summary = calc_stats(df, labels)
    print(summary.to_string(index=False))

    # 年度別集計
    print(f"\n{'=' * 70}")
    print("■ 年度別集計")
    print("=" * 70)

    order = [labels["first"], labels["repeat"], labels["other"]]
    yearly_results = []
    for year_val in sorted(df["year"].unique(), key=int):
        yr = int(year_val)
        if yr < year_start:
            continue
        sub = df[df["year"] == year_val]
        for cat in order:
            csub = sub[sub["surface_category"] == cat]
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
    pivot_t = pivot_t[order]
    print(pivot_t.to_string())

    # 複勝回収率ピボット
    print("\n複勝回収率(%):")
    pivot_f = yearly_df.pivot_table(index="年", columns="カテゴリ", values="複勝回収率(%)", aggfunc="first")
    pivot_f = pivot_f[order]
    print(pivot_f.to_string())

    # 勝率ピボット
    print("\n勝率(%):")
    pivot_w = yearly_df.pivot_table(index="年", columns="カテゴリ", values="勝率(%)", aggfunc="first")
    pivot_w = pivot_w[order]
    print(pivot_w.to_string())

    # 複勝率ピボット
    print("\n複勝率(%):")
    pivot_p = yearly_df.pivot_table(index="年", columns="カテゴリ", values="複勝率(%)", aggfunc="first")
    pivot_p = pivot_p[order]
    print(pivot_p.to_string())

    # 差分サマリ
    first_label = labels["first"]
    other_label = labels["other"]
    print(f"\n{'=' * 70}")
    print(f"■ {first_label} vs {other_label}（差分）")
    print("=" * 70)
    print(f"{'年':>6} {'単勝回収率差':>12} {'複勝回収率差':>12} {'勝率差':>8} {'複勝率差':>8}")
    for year_val in sorted(df["year"].unique(), key=int):
        yr = int(year_val)
        if yr < year_start:
            continue
        fb = yearly_df[(yearly_df["年"] == yr) & (yearly_df["カテゴリ"] == first_label)]
        nb = yearly_df[(yearly_df["年"] == yr) & (yearly_df["カテゴリ"] == other_label)]
        if not fb.empty and not nb.empty:
            t_diff = fb.iloc[0]["単勝回収率(%)"] - nb.iloc[0]["単勝回収率(%)"]
            f_diff = fb.iloc[0]["複勝回収率(%)"] - nb.iloc[0]["複勝回収率(%)"]
            w_diff = fb.iloc[0]["勝率(%)"] - nb.iloc[0]["勝率(%)"]
            p_diff = fb.iloc[0]["複勝率(%)"] - nb.iloc[0]["複勝率(%)"]
            print(f"{yr:>6} {t_diff:>+12.1f}pt {f_diff:>+12.1f}pt {w_diff:>+8.2f}pt {p_diff:>+8.2f}pt")

    # 全期間の差分
    fb_all = summary[summary["カテゴリ"] == first_label]
    nb_all = summary[summary["カテゴリ"] == other_label]
    if not fb_all.empty and not nb_all.empty:
        print(f"\n{'全期間':>6} "
              f"{fb_all.iloc[0]['単勝回収率(%)'] - nb_all.iloc[0]['単勝回収率(%)']:>+12.1f}pt "
              f"{fb_all.iloc[0]['複勝回収率(%)'] - nb_all.iloc[0]['複勝回収率(%)']:>+12.1f}pt "
              f"{fb_all.iloc[0]['勝率(%)'] - nb_all.iloc[0]['勝率(%)']:>+8.2f}pt "
              f"{fb_all.iloc[0]['複勝率(%)'] - nb_all.iloc[0]['複勝率(%)']:>+8.2f}pt")

    print(f"\n{'=' * 70}")
    print("分析完了")

    return summary, yearly_df, labels


def main():
    parser = argparse.ArgumentParser(description="初ダート・初芝の回収率分析")
    parser.add_argument("--target", choices=["dirt", "turf"], required=True,
                        help="分析対象: dirt=初ダート, turf=初芝")
    args = parser.parse_args()

    run_analysis(args.target)


if __name__ == "__main__":
    main()
