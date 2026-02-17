"""血統 特徴量（カテゴリ13）."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.features.base import FeatureExtractor
from src.db import query_df
from src.config import MISSING_NUMERIC, MISSING_RATE, MISSING_CATEGORY
from src.utils.code_master import track_type, distance_category


class BloodlineFeatureExtractor(FeatureExtractor):
    """血統の特徴量を抽出する."""

    _FEATURES: list[str] = [
        "blood_father_id",
        "blood_bms_id",
        "blood_father_keito",
        "blood_bms_keito",
        "blood_father_turf_rate",
        "blood_father_dirt_rate",
        "blood_father_dist_rate",
        "blood_bms_turf_rate",
        "blood_bms_dirt_rate",
        "blood_inbreed_flag",
    ]

    @property
    def feature_names(self) -> list[str]:
        return self._FEATURES

    def extract(
        self,
        race_key: dict[str, str],
        uma_race_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """血統特徴量を抽出する."""
        race_date = race_key["year"] + race_key["monthday"]
        kettonums = (
            uma_race_df["kettonum"].tolist()
            if "kettonum" in uma_race_df.columns
            else []
        )
        if not kettonums:
            return self._empty_result(uma_race_df)

        # レース距離の取得（距離帯別成績用）
        race_info = self._get_race_distance(race_key)
        current_dist = self._safe_int(race_info.get("kyori"), default=0)
        current_dist_cat = distance_category(current_dist) if current_dist > 0 else "middle"

        # 血統情報を一括取得
        blood_info = self._get_blood_info(kettonums)

        # 父馬・母父の繁殖登録番号を収集
        father_nums: set[str] = set()
        bms_nums: set[str] = set()
        for kn in kettonums:
            kn_str = str(kn).strip()
            info = blood_info.get(kn_str, {})
            f_num = str(info.get("father_num", "")).strip()
            m_num = str(info.get("bms_num", "")).strip()
            if f_num:
                father_nums.add(f_num)
            if m_num:
                bms_nums.add(m_num)

        # 系統情報を取得
        all_nums = father_nums | bms_nums
        keito_map = self._get_keito_map(list(all_nums)) if all_nums else {}

        # 父産駒成績 / 母父産駒成績を集計
        father_stats = self._get_sire_stats(list(father_nums), race_date, current_dist_cat) if father_nums else {}
        bms_stats = self._get_sire_stats(list(bms_nums), race_date, current_dist_cat) if bms_nums else {}

        results: list[dict[str, Any]] = []
        for kn in kettonums:
            kn_str = str(kn).strip()
            info = blood_info.get(kn_str, {})
            feat: dict[str, Any] = {"kettonum": kn_str}

            f_num = str(info.get("father_num", "")).strip()
            m_num = str(info.get("bms_num", "")).strip()

            feat["blood_father_id"] = f_num if f_num else MISSING_CATEGORY
            feat["blood_bms_id"] = m_num if m_num else MISSING_CATEGORY

            # 系統
            feat["blood_father_keito"] = keito_map.get(f_num, MISSING_CATEGORY)
            feat["blood_bms_keito"] = keito_map.get(m_num, MISSING_CATEGORY)

            # 父産駒成績
            fs = father_stats.get(f_num, {})
            feat["blood_father_turf_rate"] = fs.get("turf_rate", MISSING_RATE)
            feat["blood_father_dirt_rate"] = fs.get("dirt_rate", MISSING_RATE)
            feat["blood_father_dist_rate"] = fs.get("dist_rate", MISSING_RATE)

            # 母父産駒成績
            bs = bms_stats.get(m_num, {})
            feat["blood_bms_turf_rate"] = bs.get("turf_rate", MISSING_RATE)
            feat["blood_bms_dirt_rate"] = bs.get("dirt_rate", MISSING_RATE)

            # 近親交配フラグ（3代以内に同一祖先がいるか）
            feat["blood_inbreed_flag"] = info.get("inbreed_flag", 0)

            results.append(feat)

        return pd.DataFrame(results).set_index("kettonum")

    # ------------------------------------------------------------------
    # データ取得
    # ------------------------------------------------------------------

    def _get_race_distance(self, race_key: dict[str, str]) -> dict[str, str]:
        sql = """
        SELECT kyori FROM n_race
        WHERE year = %(year)s AND monthday = %(monthday)s
          AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
          AND nichiji = %(nichiji)s AND racenum = %(racenum)s
        LIMIT 1
        """
        df = query_df(sql, race_key)
        return df.iloc[0].to_dict() if not df.empty else {}

    def _get_blood_info(
        self,
        kettonums: list[str],
    ) -> dict[str, dict[str, Any]]:
        """UMAマスタから血統情報（父・母父の繁殖番号）を取得する.

        n_uma テーブルの ketto3info* カラムから父と母父の繁殖登録番号を取得。
        カラム名はDB依存だが、一般的には:
        - ketto3infohannum1 = 父の繁殖番号
        - ketto3infohannum3 = 母父の繁殖番号
        (1=父, 2=母, 3=父父, 4=父母, 5=母父, 6=母母)
        ※ EveryDB2の仕様では 1:父, 2:母 なので母父は sanku テーブルから取得
        """
        if not kettonums:
            return {}

        # n_sanku から4代祖先情報を取得
        sql = """
        SELECT kettonum, fnum, mnum, mfnum,
               ffnum, fmnum, mfnum AS mf_num2, mmnum,
               fffnum, ffmnum, fmfnum, fmmnum,
               mffnum, mfmnum, mmfnum, mmmnum
        FROM n_sanku
        WHERE kettonum IN %(kettonums)s
        """
        try:
            df = query_df(sql, {"kettonums": tuple(kettonums)})
        except Exception:
            # n_sanku にカラムがない場合のフォールバック
            return self._get_blood_info_fallback(kettonums)

        result: dict[str, dict[str, Any]] = {}
        for _, row in df.iterrows():
            kn = str(row["kettonum"]).strip()

            father_num = str(row.get("fnum", "")).strip()
            bms_num = str(row.get("mfnum", "")).strip()

            # 近親交配チェック（3代以内の繁殖番号で重複があるか）
            ancestors = [
                str(row.get(col, "")).strip()
                for col in [
                    "fnum", "mnum",
                    "ffnum", "fmnum", "mfnum", "mmnum",
                    "fffnum", "ffmnum", "fmfnum", "fmmnum",
                    "mffnum", "mfmnum", "mmfnum", "mmmnum",
                ]
            ]
            # 空文字を除外して重複チェック
            valid_ancestors = [a for a in ancestors if a and a != "0000000000"]
            inbreed = 1 if len(valid_ancestors) != len(set(valid_ancestors)) else 0

            result[kn] = {
                "father_num": father_num,
                "bms_num": bms_num,
                "inbreed_flag": inbreed,
            }

        return result

    def _get_blood_info_fallback(
        self,
        kettonums: list[str],
    ) -> dict[str, dict[str, Any]]:
        """n_uma テーブルの ketto3info カラムから血統情報を取得する（フォールバック）."""
        sql = """
        SELECT kettonum,
               ketto3infohannum1 AS father_num,
               ketto3infohannum3 AS bms_num
        FROM n_uma
        WHERE kettonum IN %(kettonums)s
        """
        try:
            df = query_df(sql, {"kettonums": tuple(kettonums)})
        except Exception:
            return {}

        result: dict[str, dict[str, Any]] = {}
        for _, row in df.iterrows():
            kn = str(row["kettonum"]).strip()
            result[kn] = {
                "father_num": str(row.get("father_num", "")).strip(),
                "bms_num": str(row.get("bms_num", "")).strip(),
                "inbreed_flag": 0,
            }
        return result

    def _get_keito_map(
        self,
        hansyoku_nums: list[str],
    ) -> dict[str, str]:
        """繁殖登録番号→系統名の辞書を取得する."""
        if not hansyoku_nums:
            return {}
        sql = """
        SELECT hansyokunum, keitoname
        FROM n_keito
        WHERE hansyokunum IN %(nums)s
        """
        df = query_df(sql, {"nums": tuple(hansyoku_nums)})
        return {
            str(row["hansyokunum"]).strip(): str(row["keitoname"]).strip()
            for _, row in df.iterrows()
        }

    def _get_sire_stats(
        self,
        sire_nums: list[str],
        race_date: str,
        current_dist_cat: str,
    ) -> dict[str, dict[str, float]]:
        """種牡馬（父/母父）の産駒成績を集計する（過去3年）."""
        if not sire_nums:
            return {}

        try:
            year_start = str(int(race_date[:4]) - 3)
        except ValueError:
            return {}

        sql = """
        SELECT
            s.fnum AS sire_num,
            COUNT(*) AS total,
            SUM(CASE WHEN CAST(ur.kakuteijyuni AS integer) <= 3 THEN 1 ELSE 0 END) AS top3,
            SUM(CASE WHEN CAST(r.trackcd AS int) BETWEEN 10 AND 22 THEN 1 ELSE 0 END) AS turf_runs,
            SUM(CASE WHEN CAST(r.trackcd AS int) BETWEEN 10 AND 22
                      AND CAST(ur.kakuteijyuni AS int) <= 3 THEN 1 ELSE 0 END) AS turf_top3,
            SUM(CASE WHEN CAST(r.trackcd AS int) BETWEEN 23 AND 29 THEN 1 ELSE 0 END) AS dirt_runs,
            SUM(CASE WHEN CAST(r.trackcd AS int) BETWEEN 23 AND 29
                      AND CAST(ur.kakuteijyuni AS int) <= 3 THEN 1 ELSE 0 END) AS dirt_top3
        FROM n_sanku s
        JOIN n_uma_race ur ON s.kettonum = ur.kettonum
        JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
        WHERE s.fnum IN %(sire_nums)s
          AND ur.datakubun = '7'
          AND ur.ijyocd = '0'
          AND r.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
          AND r.year >= %(year_start)s
          AND (r.year || r.monthday) < %(race_date)s
        GROUP BY s.fnum
        """
        df = query_df(sql, {
            "sire_nums": tuple(sire_nums),
            "year_start": year_start,
            "race_date": race_date,
        })

        # 距離帯別も取得
        dist_stats = self._get_sire_dist_stats(
            sire_nums, race_date, year_start, current_dist_cat
        )

        result: dict[str, dict[str, float]] = {}
        for _, row in df.iterrows():
            sn = str(row["sire_num"]).strip()
            turf_runs = self._safe_int(row.get("turf_runs"))
            turf_top3 = self._safe_int(row.get("turf_top3"))
            dirt_runs = self._safe_int(row.get("dirt_runs"))
            dirt_top3 = self._safe_int(row.get("dirt_top3"))

            result[sn] = {
                "turf_rate": self._safe_rate(turf_top3, turf_runs),
                "dirt_rate": self._safe_rate(dirt_top3, dirt_runs),
                "dist_rate": dist_stats.get(sn, MISSING_RATE),
            }
        return result

    def _get_sire_dist_stats(
        self,
        sire_nums: list[str],
        race_date: str,
        year_start: str,
        dist_cat: str,
    ) -> dict[str, float]:
        """種牡馬の距離帯別成績を取得する."""
        # 距離帯に応じた条件
        dist_conditions = {
            "short": "CAST(r.kyori AS int) <= 1400",
            "mile": "CAST(r.kyori AS int) BETWEEN 1401 AND 1800",
            "middle": "CAST(r.kyori AS int) BETWEEN 1801 AND 2200",
            "long": "CAST(r.kyori AS int) >= 2201",
        }
        dist_cond = dist_conditions.get(dist_cat, "1=1")

        sql = f"""
        SELECT
            s.fnum AS sire_num,
            COUNT(*) AS total,
            SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) <= 3 THEN 1 ELSE 0 END) AS top3
        FROM n_sanku s
        JOIN n_uma_race ur ON s.kettonum = ur.kettonum
        JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
        WHERE s.fnum IN %(sire_nums)s
          AND ur.datakubun = '7'
          AND ur.ijyocd = '0'
          AND r.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
          AND r.year >= %(year_start)s
          AND (r.year || r.monthday) < %(race_date)s
          AND {dist_cond}
        GROUP BY s.fnum
        """
        df = query_df(sql, {
            "sire_nums": tuple(sire_nums),
            "year_start": year_start,
            "race_date": race_date,
        })

        result: dict[str, float] = {}
        for _, row in df.iterrows():
            sn = str(row["sire_num"]).strip()
            result[sn] = self._safe_rate(
                self._safe_int(row.get("top3")),
                self._safe_int(row.get("total")),
            )
        return result

    def _empty_result(self, uma_race_df: pd.DataFrame) -> pd.DataFrame:
        idx = (
            uma_race_df["kettonum"].tolist()
            if "kettonum" in uma_race_df.columns
            else []
        )
        return pd.DataFrame(index=idx, columns=self._FEATURES, dtype=object)
