"""レース条件 + 枠順・馬番 + 負担重量 特徴量（カテゴリ6, 7, 8）.

当該レースの n_race, n_uma_race から直接取得する特徴量。
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.features.base import FeatureExtractor
from src.db import query_df
from src.utils.code_master import (
    track_type,
    course_direction,
    baba_code_for_track,
)


class RaceFeatureExtractor(FeatureExtractor):
    """レース条件・枠順・負担重量の特徴量を抽出する."""

    _FEATURES: list[str] = [
        # カテゴリ6: レース条件
        "race_jyo_cd",
        "race_distance",
        "race_track_cd",
        "race_track_type",
        "race_course_dir",
        "race_baba_cd",
        "race_tenko_cd",
        "race_grade_cd",
        "race_syubetu_cd",
        "race_jyuryo_cd",
        "race_jyoken_cd",
        "race_tosu",
        "race_month",
        "race_is_tokubetsu",
        # カテゴリ7: 枠順・馬番
        "post_wakuban",
        "post_umaban",
        "post_umaban_norm",
        "post_is_inner",
        "post_is_outer",
        # カテゴリ8: 負担重量
        "weight_futan",
        "weight_futan_vs_avg",
    ]

    @property
    def feature_names(self) -> list[str]:
        return self._FEATURES

    def extract(
        self,
        race_key: dict[str, str],
        uma_race_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """レース条件・枠順・負担重量の特徴量を抽出する.

        Args:
            race_key: レースキー辞書
            uma_race_df: 当該レース出走馬情報

        Returns:
            特徴量 DataFrame（kettonum をインデックス）
        """
        # レース情報を取得
        race_info = self._get_race_info(race_key)
        if race_info.empty:
            return self._empty_result(uma_race_df)

        race = race_info.iloc[0]

        # 出走馬情報を取得
        horses = self._get_horse_info(race_key)
        if horses.empty:
            return self._empty_result(uma_race_df)

        # 出走頭数
        tosu = len(horses)

        # 負担重量の平均
        horses["_futan_raw"] = horses["futan"].apply(
            lambda x: self._safe_float(x, default=np.nan)
        )
        horses["_futan_kg"] = horses["_futan_raw"] / 10.0
        avg_futan = horses["_futan_kg"].mean()

        # トラック種別
        trackcd = str(race.get("trackcd", "")).strip()
        tt = track_type(trackcd)
        cd = course_direction(trackcd)
        baba_cd = baba_code_for_track(
            trackcd,
            str(race.get("sibababacd", "")).strip(),
            str(race.get("dirtbabacd", "")).strip(),
        )

        # MonthDay → 月
        monthday = str(race.get("monthday", "0000")).strip()
        month = self._safe_int(monthday[:2], default=0)

        # 特別戦フラグ
        tokunum = str(race.get("tokunum", "0000")).strip()
        is_tokubetsu = 0 if tokunum == "0000" or not tokunum else 1

        results: list[dict] = []
        for _, h in horses.iterrows():
            kettonum = str(h["kettonum"]).strip()
            wakuban = self._safe_int(h.get("wakuban"))
            umaban = self._safe_int(h.get("umaban"))
            futan_kg = h["_futan_kg"] if pd.notna(h["_futan_kg"]) else -1.0

            results.append(
                {
                    "kettonum": kettonum,
                    # レース条件
                    "race_jyo_cd": str(race.get("jyocd", "")).strip(),
                    "race_distance": self._safe_int(race.get("kyori")),
                    "race_track_cd": trackcd,
                    "race_track_type": tt,
                    "race_course_dir": cd,
                    "race_baba_cd": self._safe_int(baba_cd, default=0),
                    "race_tenko_cd": self._safe_int(race.get("tenkocd"), default=0),
                    "race_grade_cd": str(race.get("gradecd", "")).strip(),
                    "race_syubetu_cd": str(race.get("syubetucd", "")).strip(),
                    "race_jyuryo_cd": str(race.get("jyuryocd", "")).strip(),
                    "race_jyoken_cd": str(race.get("jyokencd5", "")).strip(),
                    "race_tosu": tosu,
                    "race_month": month,
                    "race_is_tokubetsu": is_tokubetsu,
                    # 枠順・馬番
                    "post_wakuban": wakuban,
                    "post_umaban": umaban,
                    "post_umaban_norm": (
                        umaban / tosu if tosu > 0 and umaban > 0 else -1.0
                    ),
                    "post_is_inner": 1 if 1 <= wakuban <= 3 else 0,
                    "post_is_outer": 1 if 6 <= wakuban <= 8 else 0,
                    # 負担重量
                    "weight_futan": futan_kg,
                    "weight_futan_vs_avg": (
                        futan_kg - avg_futan
                        if futan_kg > 0 and pd.notna(avg_futan)
                        else 0.0
                    ),
                }
            )

        df = pd.DataFrame(results)
        df = df.set_index("kettonum")
        return df

    def _get_race_info(self, race_key: dict[str, str]) -> pd.DataFrame:
        """n_race からレース情報を取得する."""
        sql = """
        SELECT jyocd, kyori, trackcd, sibababacd, dirtbabacd, tenkocd,
               gradecd, syubetucd, jyuryocd, jyokencd5, monthday,
               tokunum, syussotosu
        FROM n_race
        WHERE year = %(year)s AND monthday = %(monthday)s
          AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
          AND nichiji = %(nichiji)s AND racenum = %(racenum)s
          AND datakubun IN ('1','2','3','4','5','6','7')
        ORDER BY datakubun DESC
        LIMIT 1
        """
        return query_df(sql, race_key)

    def _get_horse_info(self, race_key: dict[str, str]) -> pd.DataFrame:
        """n_uma_race から出走馬情報を取得する."""
        sql = """
        SELECT kettonum, umaban, wakuban, futan
        FROM n_uma_race
        WHERE year = %(year)s AND monthday = %(monthday)s
          AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
          AND nichiji = %(nichiji)s AND racenum = %(racenum)s
          AND datakubun IN ('1','2','3','4','5','6','7')
          AND ijyocd = '0'
        ORDER BY CAST(umaban AS integer)
        """
        return query_df(sql, race_key)

    def _empty_result(self, uma_race_df: pd.DataFrame) -> pd.DataFrame:
        """空の結果を返す."""
        idx = uma_race_df["kettonum"].tolist() if "kettonum" in uma_race_df.columns else []
        return pd.DataFrame(
            index=idx,
            columns=self._FEATURES,
            dtype=object,
        )
