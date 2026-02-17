"""馬基本属性・過去成績・条件別成績・馬体重・間隔 特徴量.

カテゴリ 1(horse_basic), 2(horse_perf), 3(horse_cond),
9(bw), 14(interval) を担当。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.features.base import FeatureExtractor
from src.db import query_df
from src.config import MISSING_NUMERIC, MISSING_RATE
from src.utils.code_master import (
    track_type,
    distance_category,
    baba_code_for_track,
    interval_category,
)


class HorseFeatureExtractor(FeatureExtractor):
    """馬の基本属性・成績・体重・間隔の特徴量を抽出する."""

    _FEATURES: list[str] = [
        # カテゴリ1: 馬基本属性
        "horse_sex",
        "horse_age",
        "horse_tozai",
        "horse_blinker",
        "horse_keiro",
        # カテゴリ2: 過去成績
        "horse_run_count",
        "horse_win_count",
        "horse_win_rate",
        "horse_rentai_rate",
        "horse_fukusho_rate",
        "horse_win_rate_last5",
        "horse_rentai_rate_last5",
        "horse_fukusho_rate_last5",
        "horse_avg_jyuni_last5",
        "horse_avg_jyuni_last3",
        "horse_last_jyuni",
        "horse_last2_jyuni",
        "horse_best_jyuni_last5",
        # カテゴリ3: 条件別成績
        "horse_turf_fukusho_rate",
        "horse_dirt_fukusho_rate",
        "horse_dist_short_rate",
        "horse_dist_mile_rate",
        "horse_dist_middle_rate",
        "horse_dist_long_rate",
        "horse_same_jyo_rate",
        "horse_same_dist_rate",
        "horse_same_track_rate",
        "horse_heavy_rate",
        "horse_good_rate",
        "horse_grade_rate",
        "horse_same_jyo_runs",
        "horse_same_dist_runs",
        # カテゴリ9: 馬体重
        "bw_weight",
        "bw_change",
        "bw_abs_change",
        "bw_weight_vs_avg",
        "bw_is_big_change",
        # カテゴリ14: 間隔
        "interval_days",
        "interval_category",
        "interval_days_prev2",
        "interval_is_rensho",
        "interval_is_kyuumei",
        # カテゴリ8: 負担重量（前走差）
        "weight_futan_diff",
    ]

    @property
    def feature_names(self) -> list[str]:
        return self._FEATURES

    def extract(
        self,
        race_key: dict[str, str],
        uma_race_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """馬関連特徴量を一括抽出する.

        Args:
            race_key: レースキー辞書
            uma_race_df: 当該レース出走馬情報

        Returns:
            特徴量 DataFrame（kettonum をインデックス）
        """
        race_date = race_key["year"] + race_key["monthday"]

        # 出走馬の基本情報を取得
        basic_df = self._get_basic_info(race_key)
        if basic_df.empty:
            return self._empty_result(uma_race_df)

        # 全出走馬の過去成績を一括取得（N+1回避）
        kettonums = basic_df["kettonum"].tolist()
        past_df = self._get_past_results(kettonums, race_date)

        # レース情報（距離・トラック・馬場・競馬場・グレード）
        race_info = self._get_race_condition(race_key)

        results: list[dict[str, Any]] = []
        for _, horse in basic_df.iterrows():
            kn = str(horse["kettonum"]).strip()
            feat = self._build_horse_features(
                horse, past_df, kn, race_date, race_info
            )
            results.append(feat)

        df = pd.DataFrame(results)
        df = df.set_index("kettonum")
        return df

    # ------------------------------------------------------------------
    # 内部メソッド: データ取得
    # ------------------------------------------------------------------

    def _get_basic_info(self, race_key: dict[str, str]) -> pd.DataFrame:
        """出走馬の基本情報を取得する."""
        sql = """
        SELECT kettonum, umaban, sexcd, barei, tozaicd, blinker,
               keirocd, futan, bataijyu, zogenfugo, zogensa,
               kisyucode
        FROM n_uma_race
        WHERE year = %(year)s AND monthday = %(monthday)s
          AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
          AND nichiji = %(nichiji)s AND racenum = %(racenum)s
          AND datakubun IN ('1','2','3','4','5','6','7')
          AND ijyocd = '0'
        ORDER BY CAST(umaban AS integer)
        """
        return query_df(sql, race_key)

    def _get_past_results(
        self,
        kettonums: list[str],
        race_date: str,
    ) -> pd.DataFrame:
        """全出走馬の過去成績を一括取得する（N+1回避）.

        直近30走分を取得し、Python側で集計する。
        """
        if not kettonums:
            return pd.DataFrame()

        sql = """
        SELECT
            ur.kettonum,
            ur.kakuteijyuni,
            ur.time,
            ur.harontimel3,
            ur.futan,
            ur.bataijyu,
            ur.kisyucode,
            ur.ijyocd,
            ur.year,
            ur.monthday,
            ur.jyocd,
            r.kyori,
            r.trackcd,
            r.sibababacd,
            r.dirtbabacd,
            r.gradecd
        FROM n_uma_race ur
        JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
        WHERE ur.kettonum IN %(kettonums)s
          AND ur.datakubun = '7'
          AND ur.ijyocd = '0'
          AND r.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
          AND (ur.year || ur.monthday) < %(race_date)s
        ORDER BY ur.kettonum, ur.year DESC, ur.monthday DESC
        """
        params: dict[str, Any] = {
            "kettonums": tuple(kettonums),
            "race_date": race_date,
        }
        return query_df(sql, params)

    def _get_race_condition(self, race_key: dict[str, str]) -> dict[str, str]:
        """当該レースの条件を取得する."""
        sql = """
        SELECT kyori, trackcd, sibababacd, dirtbabacd, jyocd, gradecd
        FROM n_race
        WHERE year = %(year)s AND monthday = %(monthday)s
          AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
          AND nichiji = %(nichiji)s AND racenum = %(racenum)s
        LIMIT 1
        """
        df = query_df(sql, race_key)
        if df.empty:
            return {}
        return df.iloc[0].to_dict()

    # ------------------------------------------------------------------
    # 内部メソッド: 特徴量構築
    # ------------------------------------------------------------------

    def _build_horse_features(
        self,
        horse: pd.Series,
        past_df: pd.DataFrame,
        kettonum: str,
        race_date: str,
        race_info: dict[str, str],
    ) -> dict[str, Any]:
        """1頭分の特徴量を構築する."""
        feat: dict[str, Any] = {"kettonum": kettonum}

        # --- カテゴリ1: 馬基本属性 ---
        feat["horse_sex"] = str(horse.get("sexcd", "")).strip()
        feat["horse_age"] = self._safe_int(horse.get("barei"))
        feat["horse_tozai"] = str(horse.get("tozaicd", "")).strip()
        feat["horse_blinker"] = self._safe_int(horse.get("blinker"), default=0)
        feat["horse_keiro"] = str(horse.get("keirocd", "")).strip()

        # この馬の過去成績を取り出す
        h_past = past_df[past_df["kettonum"] == kettonum].copy() if not past_df.empty else pd.DataFrame()

        # --- カテゴリ2: 過去成績 ---
        feat.update(self._calc_past_performance(h_past))

        # --- カテゴリ3: 条件別成績 ---
        feat.update(self._calc_condition_performance(h_past, race_info))

        # --- カテゴリ9: 馬体重 ---
        feat.update(self._calc_body_weight(horse, h_past))

        # --- カテゴリ14: 間隔 ---
        feat.update(self._calc_interval(h_past, race_date))

        # --- カテゴリ8: 負担重量（前走差） ---
        feat.update(self._calc_futan_diff(horse, h_past))

        return feat

    def _calc_past_performance(self, h_past: pd.DataFrame) -> dict[str, Any]:
        """カテゴリ2: 過去成績を集計する."""
        result: dict[str, Any] = {}
        total = len(h_past)

        if total == 0:
            result["horse_run_count"] = 0
            result["horse_win_count"] = 0
            result["horse_win_rate"] = MISSING_RATE
            result["horse_rentai_rate"] = MISSING_RATE
            result["horse_fukusho_rate"] = MISSING_RATE
            result["horse_win_rate_last5"] = MISSING_RATE
            result["horse_rentai_rate_last5"] = MISSING_RATE
            result["horse_fukusho_rate_last5"] = MISSING_RATE
            result["horse_avg_jyuni_last5"] = MISSING_NUMERIC
            result["horse_avg_jyuni_last3"] = MISSING_NUMERIC
            result["horse_last_jyuni"] = MISSING_NUMERIC
            result["horse_last2_jyuni"] = MISSING_NUMERIC
            result["horse_best_jyuni_last5"] = MISSING_NUMERIC
            return result

        # 着順を数値化
        jyuni = h_past["kakuteijyuni"].apply(
            lambda x: self._safe_int(x, default=99)
        )

        result["horse_run_count"] = total
        wins = (jyuni == 1).sum()
        result["horse_win_count"] = int(wins)
        result["horse_win_rate"] = self._safe_rate(int(wins), total)
        result["horse_rentai_rate"] = self._safe_rate(
            int((jyuni <= 2).sum()), total
        )
        result["horse_fukusho_rate"] = self._safe_rate(
            int((jyuni <= 3).sum()), total
        )

        # 直近N走
        last5 = jyuni.head(5)
        last3 = jyuni.head(3)
        n5 = len(last5)
        n3 = len(last3)

        result["horse_win_rate_last5"] = self._safe_rate(
            int((last5 == 1).sum()), n5
        )
        result["horse_rentai_rate_last5"] = self._safe_rate(
            int((last5 <= 2).sum()), n5
        )
        result["horse_fukusho_rate_last5"] = self._safe_rate(
            int((last5 <= 3).sum()), n5
        )
        result["horse_avg_jyuni_last5"] = float(last5[last5 < 99].mean()) if (last5 < 99).any() else MISSING_NUMERIC
        result["horse_avg_jyuni_last3"] = float(last3[last3 < 99].mean()) if (last3 < 99).any() else MISSING_NUMERIC
        result["horse_last_jyuni"] = int(jyuni.iloc[0]) if total > 0 else MISSING_NUMERIC
        result["horse_last2_jyuni"] = int(jyuni.iloc[1]) if total > 1 else MISSING_NUMERIC
        result["horse_best_jyuni_last5"] = int(last5.min()) if n5 > 0 else MISSING_NUMERIC

        return result

    def _calc_condition_performance(
        self,
        h_past: pd.DataFrame,
        race_info: dict[str, str],
    ) -> dict[str, Any]:
        """カテゴリ3: 条件別成績を集計する."""
        result: dict[str, Any] = {}

        if h_past.empty or not race_info:
            for key in [
                "horse_turf_fukusho_rate", "horse_dirt_fukusho_rate",
                "horse_dist_short_rate", "horse_dist_mile_rate",
                "horse_dist_middle_rate", "horse_dist_long_rate",
                "horse_same_jyo_rate", "horse_same_dist_rate",
                "horse_same_track_rate", "horse_heavy_rate",
                "horse_good_rate", "horse_grade_rate",
            ]:
                result[key] = MISSING_RATE
            result["horse_same_jyo_runs"] = 0
            result["horse_same_dist_runs"] = 0
            return result

        # 着順の数値化
        h = h_past.copy()
        h["_jyuni"] = h["kakuteijyuni"].apply(
            lambda x: self._safe_int(x, default=99)
        )
        h["_top3"] = (h["_jyuni"] <= 3).astype(int)
        h["_track_type"] = h["trackcd"].apply(
            lambda x: track_type(str(x).strip())
        )
        h["_kyori_int"] = h["kyori"].apply(
            lambda x: self._safe_int(x, default=0)
        )
        h["_dist_cat"] = h["_kyori_int"].apply(distance_category)
        h["_baba_cd"] = h.apply(
            lambda row: baba_code_for_track(
                str(row["trackcd"]).strip(),
                str(row.get("sibababacd", "")).strip(),
                str(row.get("dirtbabacd", "")).strip(),
            ),
            axis=1,
        )

        # 芝/ダート
        turf = h[h["_track_type"] == "turf"]
        dirt = h[h["_track_type"] == "dirt"]
        result["horse_turf_fukusho_rate"] = self._safe_rate(
            int(turf["_top3"].sum()), len(turf)
        )
        result["horse_dirt_fukusho_rate"] = self._safe_rate(
            int(dirt["_top3"].sum()), len(dirt)
        )

        # 距離帯別
        for cat in ["short", "mile", "middle", "long"]:
            subset = h[h["_dist_cat"] == cat]
            result[f"horse_dist_{cat}_rate"] = self._safe_rate(
                int(subset["_top3"].sum()), len(subset)
            )

        # 同一競馬場
        current_jyo = str(race_info.get("jyocd", "")).strip()
        same_jyo = h[h["jyocd"].str.strip() == current_jyo]
        result["horse_same_jyo_rate"] = self._safe_rate(
            int(same_jyo["_top3"].sum()), len(same_jyo)
        )
        result["horse_same_jyo_runs"] = len(same_jyo)

        # 同一距離（±100m）
        current_dist = self._safe_int(race_info.get("kyori"), default=0)
        same_dist = h[
            (h["_kyori_int"] >= current_dist - 100)
            & (h["_kyori_int"] <= current_dist + 100)
        ]
        result["horse_same_dist_rate"] = self._safe_rate(
            int(same_dist["_top3"].sum()), len(same_dist)
        )
        result["horse_same_dist_runs"] = len(same_dist)

        # 同一トラック種別
        current_tt = track_type(str(race_info.get("trackcd", "")).strip())
        same_track = h[h["_track_type"] == current_tt]
        result["horse_same_track_rate"] = self._safe_rate(
            int(same_track["_top3"].sum()), len(same_track)
        )

        # 馬場状態別
        heavy = h[h["_baba_cd"].isin(["3", "4"])]
        good = h[h["_baba_cd"] == "1"]
        result["horse_heavy_rate"] = self._safe_rate(
            int(heavy["_top3"].sum()), len(heavy)
        )
        result["horse_good_rate"] = self._safe_rate(
            int(good["_top3"].sum()), len(good)
        )

        # 重賞成績
        grade = h[h["gradecd"].str.strip().isin(["A", "B", "C", "D"])]
        result["horse_grade_rate"] = self._safe_rate(
            int(grade["_top3"].sum()), len(grade)
        )

        return result

    def _calc_body_weight(
        self,
        horse: pd.Series,
        h_past: pd.DataFrame,
    ) -> dict[str, Any]:
        """カテゴリ9: 馬体重の特徴量を計算する."""
        result: dict[str, Any] = {}

        bw = self._safe_int(horse.get("bataijyu"))
        result["bw_weight"] = bw

        # 増減
        zogen_fugo = str(horse.get("zogenfugo", "")).strip()
        zogen_sa = self._safe_int(horse.get("zogensa"), default=0)
        if zogen_fugo == "-":
            change = -zogen_sa
        elif zogen_fugo == "+":
            change = zogen_sa
        else:
            change = 0
        result["bw_change"] = change
        result["bw_abs_change"] = abs(change)
        result["bw_is_big_change"] = 1 if abs(change) >= 10 else 0

        # 過去平均体重との差
        if not h_past.empty:
            past_bw = h_past["bataijyu"].apply(
                lambda x: self._safe_float(x, default=np.nan)
            )
            avg_bw = past_bw.mean()
            result["bw_weight_vs_avg"] = (
                float(bw - avg_bw) if bw > 0 and pd.notna(avg_bw) else 0.0
            )
        else:
            result["bw_weight_vs_avg"] = 0.0

        return result

    def _calc_interval(
        self,
        h_past: pd.DataFrame,
        race_date: str,
    ) -> dict[str, Any]:
        """カテゴリ14: 間隔・ローテーション特徴量を計算する."""
        result: dict[str, Any] = {}

        if h_past.empty:
            result["interval_days"] = MISSING_NUMERIC
            result["interval_category"] = "unknown"
            result["interval_days_prev2"] = MISSING_NUMERIC
            result["interval_is_rensho"] = 0
            result["interval_is_kyuumei"] = 0
            return result

        try:
            current_dt = datetime.strptime(race_date[:8], "%Y%m%d")
        except ValueError:
            result["interval_days"] = MISSING_NUMERIC
            result["interval_category"] = "unknown"
            result["interval_days_prev2"] = MISSING_NUMERIC
            result["interval_is_rensho"] = 0
            result["interval_is_kyuumei"] = 0
            return result

        # 前走日付
        prev1_date_str = str(h_past.iloc[0]["year"]).strip() + str(h_past.iloc[0]["monthday"]).strip()
        try:
            prev1_dt = datetime.strptime(prev1_date_str[:8], "%Y%m%d")
            days1 = (current_dt - prev1_dt).days
        except ValueError:
            days1 = -1

        result["interval_days"] = days1
        result["interval_category"] = interval_category(days1)
        result["interval_is_rensho"] = 1 if 0 <= days1 <= 7 else 0
        result["interval_is_kyuumei"] = 1 if days1 >= 90 else 0

        # 前々走
        if len(h_past) >= 2:
            prev2_date_str = str(h_past.iloc[1]["year"]).strip() + str(h_past.iloc[1]["monthday"]).strip()
            try:
                prev2_dt = datetime.strptime(prev2_date_str[:8], "%Y%m%d")
                days2 = (current_dt - prev2_dt).days
            except ValueError:
                days2 = -1
            result["interval_days_prev2"] = days2
        else:
            result["interval_days_prev2"] = MISSING_NUMERIC

        return result

    def _calc_futan_diff(
        self,
        horse: pd.Series,
        h_past: pd.DataFrame,
    ) -> dict[str, Any]:
        """カテゴリ8: 負担重量の前走差."""
        result: dict[str, Any] = {}
        current_futan = self._safe_float(horse.get("futan"), default=-1.0)
        current_futan_kg = current_futan / 10.0 if current_futan > 0 else -1.0

        if not h_past.empty and current_futan_kg > 0:
            prev_futan = self._safe_float(h_past.iloc[0].get("futan"), default=-1.0)
            prev_futan_kg = prev_futan / 10.0 if prev_futan > 0 else -1.0
            if prev_futan_kg > 0:
                result["weight_futan_diff"] = current_futan_kg - prev_futan_kg
            else:
                result["weight_futan_diff"] = 0.0
        else:
            result["weight_futan_diff"] = 0.0

        return result

    def _empty_result(self, uma_race_df: pd.DataFrame) -> pd.DataFrame:
        """空の結果を返す."""
        idx = uma_race_df["kettonum"].tolist() if "kettonum" in uma_race_df.columns else []
        return pd.DataFrame(index=idx, columns=self._FEATURES, dtype=object)
