"""種牡馬別回収率分析スクリプト.

洋芝（札幌・函館）/ 芝（それ以外）/ ダート × 競馬場別 × 距離別で
種牡馬ごとの単勝回収率を検証する。
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.db import query_df

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.1f}".format)


def load_data(year_start: int = 2019, year_end: int = 2024) -> pd.DataFrame:
    """種牡馬別回収率の基礎データを一括取得する."""
    sql = """
    SELECT
        ur.year,
        ur.monthday,
        ur.jyocd,
        ur.kaiji,
        ur.nichiji,
        ur.racenum,
        r.trackcd,
        CAST(r.kyori AS integer) AS kyori,
        sk.fnum AS father_id,
        hs.bamei AS father_name,
        ur.umaban,
        CAST(ur.kakuteijyuni AS integer) AS jyuni
    FROM n_uma_race ur
    JOIN n_race r
      ON ur.year = r.year
     AND ur.monthday = r.monthday
     AND ur.jyocd = r.jyocd
     AND ur.kaiji = r.kaiji
     AND ur.nichiji = r.nichiji
     AND ur.racenum = r.racenum
    JOIN n_sanku sk
      ON ur.kettonum = sk.kettonum
    JOIN n_hansyoku hs
      ON sk.fnum = hs.hansyokunum
    WHERE ur.datakubun = '7'
      AND ur.year >= %(y1)s
      AND ur.year <= %(y2)s
      AND ur.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
      AND ur.ijyocd = '0'
      AND ur.kakuteijyuni ~ '^[0-9]+$'
      AND r.trackcd ~ '^[0-9]+$'
    """
    df = query_df(sql, {"y1": str(year_start), "y2": str(year_end)})
    print(f"  出走データ: {len(df):,} 件")

    # 払戻データ
    harai_sql = """
    SELECT
        year, monthday, jyocd, kaiji, nichiji, racenum,
        paytansyoumaban1, paytansyopay1,
        paytansyoumaban2, paytansyopay2,
        paytansyoumaban3, paytansyopay3
    FROM n_harai
    WHERE datakubun IN ('1', '2')
      AND year >= %(y1)s
      AND year <= %(y2)s
      AND jyocd IN ('01','02','03','04','05','06','07','08','09','10')
    """
    harai = query_df(harai_sql, {"y1": str(year_start), "y2": str(year_end)})
    print(f"  払戻データ: {len(harai):,} 件")

    # race_key
    rk_cols = ["year", "monthday", "jyocd", "kaiji", "nichiji", "racenum"]
    df["rk"] = df[rk_cols].astype(str).agg("_".join, axis=1)
    harai["rk"] = harai[rk_cols].astype(str).agg("_".join, axis=1)

    # 払戻辞書（race_key -> {umaban_padded: pay}）
    pay_dict: dict[str, dict[str, int]] = {}
    for _, row in harai.iterrows():
        rk = row["rk"]
        d: dict[str, int] = {}
        for i in range(1, 4):
            ub = str(row[f"paytansyoumaban{i}"]).strip()
            py = str(row[f"paytansyopay{i}"]).strip()
            if ub and py.isdigit() and int(py) > 0:
                d[ub] = int(py)
        if d:
            pay_dict[rk] = d

    # 単勝払戻を付与
    def get_pay(row):
        if row["jyuni"] != 1:
            return 0
        rk = row["rk"]
        if rk not in pay_dict:
            return 0
        ub = str(row["umaban"]).strip().zfill(2)
        return pay_dict[rk].get(ub, 0)

    df["tansho_pay"] = df.apply(get_pay, axis=1)

    # サーフェス分類
    trackcd_int = df["trackcd"].astype(str).str.strip().astype(int)
    jyocd = df["jyocd"].astype(str).str.strip()
    df["surface"] = np.where(
        trackcd_int >= 23, "ダート",
        np.where(jyocd.isin(["01", "02"]), "洋芝", "芝")
    )

    # 距離帯
    df["dist_cat"] = pd.cut(
        df["kyori"],
        bins=[0, 1400, 1800, 2200, 9999],
        labels=["~1400", "1401-1800", "1801-2200", "2201~"]
    )

    # 競馬場名
    jyo_map = {
        "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
        "05": "東京", "06": "中山", "07": "中京", "08": "京都",
        "09": "阪神", "10": "小倉",
    }
    df["jyo_name"] = jyocd.map(jyo_map)

    return df


def sire_agg(df: pd.DataFrame, group_cols: list[str], min_n: int = 100) -> pd.DataFrame:
    """種牡馬 × group_cols で集計."""
    g = df.groupby(["father_id", "father_name"] + group_cols).agg(
        n=("jyuni", "size"),
        wins=("jyuni", lambda x: (x == 1).sum()),
        top3=("jyuni", lambda x: (x <= 3).sum()),
        pay_sum=("tansho_pay", "sum"),
    ).reset_index()
    g["win%"] = g["wins"] / g["n"] * 100
    g["top3%"] = g["top3"] / g["n"] * 100
    g["roi%"] = g["pay_sum"] / (g["n"] * 100) * 100
    return g[g["n"] >= min_n].sort_values("roi%", ascending=False)


def show_ranking(agg_df: pd.DataFrame, title: str, split_col: str | None = None, top_n: int = 15):
    """ランキング表示."""
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")

    if split_col:
        groups = sorted(agg_df[split_col].unique())
    else:
        groups = [None]

    for g in groups:
        if g is not None:
            sub = agg_df[agg_df[split_col] == g]
            print(f"\n--- {g} ({len(sub)}種牡馬) ---")
        else:
            sub = agg_df

        top = sub.head(top_n)
        bot = sub.tail(top_n).iloc[::-1]

        print(f"\n  【高回収率 Top{top_n}】")
        print(f"  {'種牡馬':<16s}  出走   勝率   複勝率  単回収率")
        for _, r in top.iterrows():
            print(f"  {r['father_name']:<16s} {r['n']:5.0f}  {r['win%']:5.1f}%  {r['top3%']:5.1f}%  {r['roi%']:6.1f}%")

        print(f"\n  【低回収率 Bottom{top_n}】")
        print(f"  {'種牡馬':<16s}  出走   勝率   複勝率  単回収率")
        for _, r in bot.iterrows():
            print(f"  {r['father_name']:<16s} {r['n']:5.0f}  {r['win%']:5.1f}%  {r['top3%']:5.1f}%  {r['roi%']:6.1f}%")


def show_cross(df: pd.DataFrame, top_n_sires: int = 30):
    """主要種牡馬のサーフェス×距離クロス集計."""
    print(f"\n{'='*100}")
    print(f"  主要種牡馬 サーフェス×距離帯 回収率クロス集計 (min 30出走)")
    print(f"{'='*100}")

    top_sires = (
        df.groupby(["father_id", "father_name"]).size()
        .reset_index(name="total_n")
        .nlargest(top_n_sires, "total_n")
    )

    cross = sire_agg(df, ["surface", "dist_cat"], min_n=30)
    dist_labels = ["~1400", "1401-1800", "1801-2200", "2201~"]

    for _, sire in top_sires.iterrows():
        sd = cross[cross["father_id"] == sire["father_id"]]
        if len(sd) == 0:
            continue
        print(f"\n  {sire['father_name']} (総出走: {sire['total_n']:,})")
        for surf in ["洋芝", "芝", "ダート"]:
            ss = sd[sd["surface"] == surf]
            vals = []
            for dc in dist_labels:
                row = ss[ss["dist_cat"] == dc]
                if len(row) > 0:
                    r = row.iloc[0]
                    vals.append(f"{dc}: {r['roi%']:5.1f}%({int(r['n'])})")
                else:
                    vals.append(f"{dc}:   --- ")

            print(f"    {surf:<4s} | {'  '.join(vals)}")


def show_summary(agg_df: pd.DataFrame, split_col: str):
    """回収率分布サマリ."""
    print(f"\n{'='*100}")
    print(f"  種牡馬別回収率の分布サマリ（{split_col}別）")
    print(f"{'='*100}")
    for g in sorted(agg_df[split_col].unique()):
        sub = agg_df[agg_df[split_col] == g]
        print(f"\n  {g}: {len(sub)}種牡馬")
        print(f"    回収率  平均: {sub['roi%'].mean():.1f}%  中央値: {sub['roi%'].median():.1f}%  "
              f"std: {sub['roi%'].std():.1f}%")
        print(f"    範囲: {sub['roi%'].min():.1f}% ~ {sub['roi%'].max():.1f}%")
        over100 = (sub['roi%'] > 100).sum()
        print(f"    回収率>100%: {over100}/{len(sub)} ({over100/len(sub)*100:.1f}%)")


def main():
    print("データ読み込み中...")
    df = load_data(2019, 2024)
    print(f"  分析対象: {len(df):,} 件  種牡馬数: {df['father_id'].nunique()}")

    # 全体の基準回収率
    baseline_roi = df["tansho_pay"].sum() / (len(df) * 100) * 100
    print(f"  全体単勝回収率(ベースライン): {baseline_roi:.1f}%")

    # 1. サーフェス別
    agg_surf = sire_agg(df, ["surface"], min_n=100)
    show_ranking(agg_surf, "種牡馬別回収率 - サーフェス別（洋芝/芝/ダート）", "surface", top_n=15)
    show_summary(agg_surf, "surface")

    # 2. 競馬場別
    agg_jyo = sire_agg(df, ["jyo_name"], min_n=50)
    show_ranking(agg_jyo, "種牡馬別回収率 - 競馬場別", "jyo_name", top_n=10)

    # 3. 距離帯別
    agg_dist = sire_agg(df, ["dist_cat"], min_n=100)
    show_ranking(agg_dist, "種牡馬別回収率 - 距離帯別", "dist_cat", top_n=15)
    show_summary(agg_dist, "dist_cat")

    # 4. クロス集計
    show_cross(df, top_n_sires=30)

    # 5. 回収率の差が大きいケース（歪みが大きい）
    print(f"\n{'='*100}")
    print(f"  サーフェス間で回収率に大きな差がある種牡馬（min 200出走 / サーフェス）")
    print(f"{'='*100}")
    agg_surf_wide = sire_agg(df, ["surface"], min_n=200)
    pivot = agg_surf_wide.pivot_table(index=["father_id", "father_name"], columns="surface", values="roi%")
    pivot = pivot.dropna(subset=["芝", "ダート"])  # 両方データがある種牡馬のみ
    pivot["芝ダ差"] = (pivot["芝"] - pivot["ダート"]).abs()
    pivot = pivot.sort_values("芝ダ差", ascending=False)

    print(f"\n  {'種牡馬':<16s}  {'洋芝':>8s}  {'芝':>8s}  {'ダート':>8s}  {'芝ダ差':>8s}")
    for (fid, fname), row in pivot.head(20).iterrows():
        yo = f"{row['洋芝']:.1f}%" if pd.notna(row.get('洋芝')) else "  ---"
        print(f"  {fname:<16s}  {yo:>8s}  {row['芝']:7.1f}%  {row['ダート']:7.1f}%  {row['芝ダ差']:7.1f}%")

    print("\n完了")


if __name__ == "__main__":
    main()
