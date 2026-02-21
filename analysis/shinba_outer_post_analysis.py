"""新馬戦での大外枠の回収率分析.

「新馬戦では大外枠が有利（ゲート入りに慣れていない馬が多く、
最後にゲートインする大外の馬が落ち着いてスタートできる）」
という格言の検証。

新馬戦の大外枠（各レースの最大馬番）の勝率・複勝率・回収率を、
それ以外の枠順と比較する。
"""

from __future__ import annotations

import sys
sys.path.insert(0, "/Users/kotaniwa/src/keiba-2026")

import pandas as pd
import numpy as np
from src.db import query_df

RACE_KEY = ["year", "monthday", "jyocd", "kaiji", "nichiji", "racenum"]


def fetch_shinba_data(year_start: int = 2015, year_end: int = 2025) -> pd.DataFrame:
    """新馬戦の出走データを取得する.

    新馬戦の判定: jyokencd5 = '701'（2歳新馬）or '702'（3歳新馬）
    ※ JRA-VAN仕様で jyokencd5 の '701' が2歳新馬、'702' が3歳新馬
    ただし実際のDB値は要確認。代替として jyokencd5 が '7' で始まるものを
    新馬とする方法もある。
    """
    sql = """
    SELECT
        ur.year, ur.monthday, ur.jyocd, ur.kaiji, ur.nichiji, ur.racenum,
        ur.kettonum, ur.umaban,
        ur.kakuteijyuni,
        r.trackcd,
        r.kyori
    FROM n_uma_race ur
    JOIN n_race r
        ON ur.year = r.year
        AND ur.monthday = r.monthday
        AND ur.jyocd = r.jyocd
        AND ur.kaiji = r.kaiji
        AND ur.nichiji = r.nichiji
        AND ur.racenum = r.racenum
    WHERE ur.datakubun = '7'
      AND ur.ijyocd = '0'
      AND r.datakubun = '7'
      AND CAST(ur.year AS integer) BETWEEN %(year_start)s AND %(year_end)s
      AND ur.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
      AND (r.jyokencd5 LIKE '7%%' OR r.jyokencd5 IN ('701', '702'))
    ORDER BY ur.year, ur.monthday, ur.jyocd, ur.racenum, ur.umaban
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


def classify_post_position(df: pd.DataFrame) -> pd.DataFrame:
    """各レースの大外枠を判定し、枠順カテゴリを付与する.

    カテゴリ:
    - 大外枠: レース内で最大馬番の馬
    - 外枠(大外以外): 馬番が頭数の上位25%（大外を除く）
    - 中枠: 馬番が頭数の中央50%
    - 内枠: 馬番が頭数の下位25%
    また、馬番グループ(1-3, 4-6, 7-9, 10-12, 13+)でも分類する。
    """
    df = df.copy()
    df["umaban_int"] = pd.to_numeric(df["umaban"], errors="coerce").astype("Int64")

    # レース内の頭数と最大馬番
    race_info = df.groupby(RACE_KEY)["umaban_int"].agg(
        tousu="count", max_umaban="max"
    ).reset_index()
    df = df.merge(race_info, on=RACE_KEY, how="left")

    # 大外枠判定
    df["is_outermost"] = df["umaban_int"] == df["max_umaban"]

    # 馬番位置（正規化: 0=最内, 1=最外）
    df["post_normalized"] = (df["umaban_int"] - 1) / (df["tousu"] - 1).clip(lower=1)

    # 4分割カテゴリ
    conditions = [
        df["is_outermost"],
        df["post_normalized"] >= 0.75,
        df["post_normalized"] >= 0.25,
        df["post_normalized"] < 0.25,
    ]
    labels = ["大外枠", "外枠(大外以外)", "中枠", "内枠"]
    df["post_category"] = np.select(conditions, labels, default="中枠")

    # 馬番グループ
    conditions_g = [
        df["umaban_int"] <= 3,
        df["umaban_int"] <= 6,
        df["umaban_int"] <= 9,
        df["umaban_int"] <= 12,
        df["umaban_int"] > 12,
    ]
    labels_g = ["1-3番", "4-6番", "7-9番", "10-12番", "13番以降"]
    df["post_group"] = np.select(conditions_g, labels_g, default="不明")

    return df


def merge_harai(df: pd.DataFrame, tansho_df: pd.DataFrame, fukusho_df: pd.DataFrame) -> pd.DataFrame:
    """払戻データを結合する."""
    df = df.copy()
    df["jyuni"] = pd.to_numeric(df["kakuteijyuni"], errors="coerce")

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


def calc_stats(df: pd.DataFrame, group_col: str, categories: list[str]) -> pd.DataFrame:
    """カテゴリ別の統計を計算する."""
    results = []
    for cat in categories:
        sub = df[df[group_col] == cat]
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


def calc_stats_by_field_size(df: pd.DataFrame) -> pd.DataFrame:
    """頭数帯別の大外枠 vs その他の統計."""
    results = []
    # 頭数帯: 少頭数(~10), 中頭数(11-14), 多頭数(15+)
    bins = [(0, 10, "~10頭"), (11, 14, "11-14頭"), (15, 99, "15頭以上")]
    for lo, hi, label in bins:
        sub = df[(df["tousu"] >= lo) & (df["tousu"] <= hi)]
        if len(sub) == 0:
            continue

        for is_outer, cat_name in [(True, f"大外枠({label})"), (False, f"その他({label})")]:
            s = sub[sub["is_outermost"] == is_outer]
            n = len(s)
            if n == 0:
                continue
            results.append({
                "カテゴリ": cat_name,
                "出走数": n,
                "勝率(%)": round((s["jyuni"] == 1).sum() / n * 100, 2),
                "複勝率(%)": round((s["jyuni"] <= 3).sum() / n * 100, 2),
                "単勝回収率(%)": round(s["tansho_pay"].sum() / n, 1),
                "複勝回収率(%)": round(s["fukusho_pay"].sum() / n, 1),
            })

    return pd.DataFrame(results)


def calc_stats_by_course(df: pd.DataFrame) -> pd.DataFrame:
    """芝/ダート別の大外枠 vs その他の統計."""
    df = df.copy()
    df["trackcd_int"] = pd.to_numeric(df["trackcd"], errors="coerce")
    # trackcd: 10=芝直線, 11-22=芝左/右, 23-29=ダート左/右, etc.
    # 簡易判定: 10-22=芝, 23以上=ダート
    df["course_type"] = np.where(df["trackcd_int"] <= 22, "芝", "ダート")

    results = []
    for course in ["芝", "ダート"]:
        sub = df[df["course_type"] == course]
        if len(sub) == 0:
            continue
        for is_outer, cat_name in [(True, f"大外枠({course})"), (False, f"その他({course})")]:
            s = sub[sub["is_outermost"] == is_outer]
            n = len(s)
            if n == 0:
                continue
            results.append({
                "カテゴリ": cat_name,
                "出走数": n,
                "勝率(%)": round((s["jyuni"] == 1).sum() / n * 100, 2),
                "複勝率(%)": round((s["jyuni"] <= 3).sum() / n * 100, 2),
                "単勝回収率(%)": round(s["tansho_pay"].sum() / n, 1),
                "複勝回収率(%)": round(s["fukusho_pay"].sum() / n, 1),
            })

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("新馬戦における大外枠の回収率分析")
    print("=" * 70)

    year_start, year_end = 2015, 2025

    print(f"\n[1] 新馬戦データ取得中 ({year_start}-{year_end})...")
    race_df = fetch_shinba_data(year_start, year_end)
    print(f"    出走データ: {len(race_df):,} 件")

    if len(race_df) == 0:
        # jyokencd5 の値を確認
        print("\n    ※ 新馬戦データが0件です。jyokencd5 の値を確認します...")
        check_sql = """
        SELECT DISTINCT r.jyokencd5, COUNT(*) as cnt
        FROM n_race r
        WHERE r.datakubun = '7'
          AND CAST(r.year AS integer) BETWEEN %(year_start)s AND %(year_end)s
          AND r.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
        GROUP BY r.jyokencd5
        ORDER BY cnt DESC
        LIMIT 30
        """
        check_df = query_df(check_sql, {"year_start": year_start, "year_end": year_end})
        print(check_df.to_string(index=False))
        return

    # レース数
    race_count = race_df.groupby(RACE_KEY).ngroups
    print(f"    レース数: {race_count:,}")

    print("[2] 払戻データ取得中...")
    tansho_df, fukusho_df = fetch_harai_data(year_start, year_end)
    print(f"    単勝払戻: {len(tansho_df):,} 件, 複勝払戻: {len(fukusho_df):,} 件")

    print("[3] 枠順カテゴリ分類中...")
    df = classify_post_position(race_df)

    # 頭数の分布
    tousu_dist = df.groupby(RACE_KEY)["tousu"].first()
    print(f"    頭数: 平均{tousu_dist.mean():.1f}, 中央値{tousu_dist.median():.0f}, "
          f"最小{tousu_dist.min()}, 最大{tousu_dist.max()}")

    for cat in ["大外枠", "外枠(大外以外)", "中枠", "内枠"]:
        cnt = (df["post_category"] == cat).sum()
        print(f"    {cat}: {cnt:,} 件")

    print("[4] 払戻データ結合中...")
    df = merge_harai(df, tansho_df, fukusho_df)

    # === 全期間集計 ===
    print(f"\n{'=' * 70}")
    print("■ 全期間集計 (2015-2025): 枠順カテゴリ別")
    print("=" * 70)
    post_cats = ["大外枠", "外枠(大外以外)", "中枠", "内枠"]
    summary = calc_stats(df, "post_category", post_cats)
    print(summary.to_string(index=False))

    # 馬番グループ別
    print(f"\n{'=' * 70}")
    print("■ 全期間集計: 馬番グループ別")
    print("=" * 70)
    group_cats = ["1-3番", "4-6番", "7-9番", "10-12番", "13番以降"]
    group_summary = calc_stats(df, "post_group", group_cats)
    print(group_summary.to_string(index=False))

    # === 大外枠 vs その他（シンプル比較） ===
    print(f"\n{'=' * 70}")
    print("■ 全期間集計: 大外枠 vs その他")
    print("=" * 70)
    df["simple_category"] = np.where(df["is_outermost"], "大外枠", "その他")
    simple_summary = calc_stats(df, "simple_category", ["大外枠", "その他"])
    print(simple_summary.to_string(index=False))

    # === 頭数帯別 ===
    print(f"\n{'=' * 70}")
    print("■ 頭数帯別: 大外枠 vs その他")
    print("=" * 70)
    field_summary = calc_stats_by_field_size(df)
    print(field_summary.to_string(index=False))

    # === 芝/ダート別 ===
    print(f"\n{'=' * 70}")
    print("■ 芝/ダート別: 大外枠 vs その他")
    print("=" * 70)
    course_summary = calc_stats_by_course(df)
    print(course_summary.to_string(index=False))

    # === 年度別推移 ===
    print(f"\n{'=' * 70}")
    print("■ 年度別推移: 大外枠 vs その他")
    print("=" * 70)

    yearly_results = []
    for year_val in sorted(df["year"].unique(), key=int):
        yr = int(year_val)
        if yr < year_start:
            continue
        sub = df[df["year"] == year_val]
        for cat in ["大外枠", "その他"]:
            csub = sub[sub["simple_category"] == cat]
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
    pivot_t = pivot_t[["大外枠", "その他"]]
    pivot_t["差分"] = pivot_t["大外枠"] - pivot_t["その他"]
    print(pivot_t.to_string())

    # 複勝回収率ピボット
    print("\n複勝回収率(%):")
    pivot_f = yearly_df.pivot_table(index="年", columns="カテゴリ", values="複勝回収率(%)", aggfunc="first")
    pivot_f = pivot_f[["大外枠", "その他"]]
    pivot_f["差分"] = pivot_f["大外枠"] - pivot_f["その他"]
    print(pivot_f.to_string())

    # 勝率・複勝率ピボット
    print("\n勝率(%):")
    pivot_w = yearly_df.pivot_table(index="年", columns="カテゴリ", values="勝率(%)", aggfunc="first")
    pivot_w = pivot_w[["大外枠", "その他"]]
    pivot_w["差分"] = pivot_w["大外枠"] - pivot_w["その他"]
    print(pivot_w.to_string())

    print("\n複勝率(%):")
    pivot_p = yearly_df.pivot_table(index="年", columns="カテゴリ", values="複勝率(%)", aggfunc="first")
    pivot_p = pivot_p[["大外枠", "その他"]]
    pivot_p["差分"] = pivot_p["大外枠"] - pivot_p["その他"]
    print(pivot_p.to_string())

    # === 差分サマリ ===
    print(f"\n{'=' * 70}")
    print("■ 大外枠 vs その他（差分サマリ）")
    print("=" * 70)
    print(f"{'年':>6} {'単勝回収率差':>12} {'複勝回収率差':>12} {'勝率差':>8} {'複勝率差':>8}")
    for year_val in sorted(df["year"].unique(), key=int):
        yr = int(year_val)
        if yr < year_start:
            continue
        outer = yearly_df[(yearly_df["年"] == yr) & (yearly_df["カテゴリ"] == "大外枠")]
        other = yearly_df[(yearly_df["年"] == yr) & (yearly_df["カテゴリ"] == "その他")]
        if not outer.empty and not other.empty:
            t_diff = outer.iloc[0]["単勝回収率(%)"] - other.iloc[0]["単勝回収率(%)"]
            f_diff = outer.iloc[0]["複勝回収率(%)"] - other.iloc[0]["複勝回収率(%)"]
            w_diff = outer.iloc[0]["勝率(%)"] - other.iloc[0]["勝率(%)"]
            p_diff = outer.iloc[0]["複勝率(%)"] - other.iloc[0]["複勝率(%)"]
            print(f"{yr:>6} {t_diff:>+12.1f}pt {f_diff:>+12.1f}pt {w_diff:>+8.2f}pt {p_diff:>+8.2f}pt")

    # 全期間の差分
    outer_all = simple_summary[simple_summary["カテゴリ"] == "大外枠"]
    other_all = simple_summary[simple_summary["カテゴリ"] == "その他"]
    if not outer_all.empty and not other_all.empty:
        print(f"\n{'全期間':>6} "
              f"{outer_all.iloc[0]['単勝回収率(%)'] - other_all.iloc[0]['単勝回収率(%)']:>+12.1f}pt "
              f"{outer_all.iloc[0]['複勝回収率(%)'] - other_all.iloc[0]['複勝回収率(%)']:>+12.1f}pt "
              f"{outer_all.iloc[0]['勝率(%)'] - other_all.iloc[0]['勝率(%)']:>+8.2f}pt "
              f"{outer_all.iloc[0]['複勝率(%)'] - other_all.iloc[0]['複勝率(%)']:>+8.2f}pt")

    print(f"\n{'=' * 70}")
    print("分析完了")


if __name__ == "__main__":
    main()
