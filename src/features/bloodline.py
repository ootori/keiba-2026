"""血統 特徴量（カテゴリ13）.

既存10特徴量 + 新規8特徴量 = 合計18特徴量。
新規特徴量:
  - blood_mother_id: 母馬繁殖登録番号
  - blood_mother_keito: 母系統名
  - blood_nicks_rate: 父×母父コンビの産駒複勝率
  - blood_nicks_runs: 父×母父コンビの産駒出走数
  - blood_father_baba_rate: 父産駒の「今回の馬場状態」での複勝率
  - blood_father_jyo_rate: 父産駒の「今回の競馬場」での複勝率
  - blood_inbreed_generation: 近親交配が発生した最も近い世代（0=なし, 2=2代, 3=3代）
  - blood_mother_produce_rate: 母の産駒成績（兄弟姉妹の複勝率）
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.features.base import FeatureExtractor
from src.db import query_df
from src.config import MISSING_NUMERIC, MISSING_RATE, MISSING_CATEGORY
from src.utils.code_master import track_type, distance_category, baba_code_for_track

logger = logging.getLogger(__name__)


class BloodlineFeatureExtractor(FeatureExtractor):
    """血統の特徴量を抽出する."""

    _FEATURES: list[str] = [
        # 既存特徴量
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
        # 新規特徴量
        "blood_mother_id",
        "blood_mother_keito",
        "blood_nicks_rate",
        "blood_nicks_runs",
        "blood_father_baba_rate",
        "blood_father_jyo_rate",
        "blood_inbreed_generation",
        "blood_mother_produce_rate",
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

        # レース情報の取得（距離帯別・馬場状態別・競馬場別成績用）
        race_info = self._get_race_info(race_key)
        current_dist = self._safe_int(race_info.get("kyori"), default=0)
        current_dist_cat = (
            distance_category(current_dist) if current_dist > 0 else "middle"
        )
        current_jyocd = race_key.get("jyocd", "")
        current_trackcd = str(race_info.get("trackcd", "")).strip()
        current_siba_baba = str(race_info.get("sibababacd", "")).strip()
        current_dirt_baba = str(race_info.get("dirtbabacd", "")).strip()
        current_baba_cd = baba_code_for_track(
            current_trackcd, current_siba_baba, current_dirt_baba
        )

        # 血統情報を一括取得
        blood_info = self._get_blood_info(kettonums)

        # 父馬・母父・母の繁殖登録番号を収集
        father_nums: set[str] = set()
        bms_nums: set[str] = set()
        mother_nums: set[str] = set()
        for kn in kettonums:
            kn_str = str(kn).strip()
            info = blood_info.get(kn_str, {})
            f_num = str(info.get("father_num", "")).strip()
            m_num = str(info.get("bms_num", "")).strip()
            mo_num = str(info.get("mother_num", "")).strip()
            if f_num:
                father_nums.add(f_num)
            if m_num:
                bms_nums.add(m_num)
            if mo_num:
                mother_nums.add(mo_num)

        # 系統情報を取得（父・母父・母をすべて含む）
        all_keito_nums = father_nums | bms_nums | mother_nums
        keito_map = (
            self._get_keito_map(list(all_keito_nums)) if all_keito_nums else {}
        )

        # 父産駒成績 / 母父産駒成績を集計
        try:
            year_start = str(int(race_date[:4]) - 3)
        except ValueError:
            year_start = "2012"

        father_stats = (
            self._get_sire_stats(
                list(father_nums), race_date, current_dist_cat
            )
            if father_nums
            else {}
        )
        bms_stats = (
            self._get_sire_stats(
                list(bms_nums), race_date, current_dist_cat
            )
            if bms_nums
            else {}
        )

        # 父産駒の馬場状態別成績
        father_baba_stats = (
            self._get_sire_baba_stats(
                list(father_nums), race_date, year_start, current_baba_cd,
                current_trackcd,
            )
            if father_nums and current_baba_cd
            else {}
        )

        # 父産駒の競馬場別成績
        father_jyo_stats = (
            self._get_sire_jyo_stats(
                list(father_nums), race_date, year_start, current_jyocd,
            )
            if father_nums and current_jyocd
            else {}
        )

        # ニックス（父×母父）成績を集計
        nicks_pairs: list[tuple[str, str]] = []
        for kn in kettonums:
            kn_str = str(kn).strip()
            info = blood_info.get(kn_str, {})
            f_num = str(info.get("father_num", "")).strip()
            m_num = str(info.get("bms_num", "")).strip()
            if f_num and m_num:
                nicks_pairs.append((f_num, m_num))
        nicks_stats = (
            self._get_nicks_stats(nicks_pairs, race_date)
            if nicks_pairs
            else {}
        )

        # 母の産駒成績を集計
        mother_produce_stats = (
            self._get_mother_produce_stats(
                list(mother_nums), kettonums, race_date,
            )
            if mother_nums
            else {}
        )

        results: list[dict[str, Any]] = []
        for kn in kettonums:
            kn_str = str(kn).strip()
            info = blood_info.get(kn_str, {})
            feat: dict[str, Any] = {"kettonum": kn_str}

            f_num = str(info.get("father_num", "")).strip()
            m_num = str(info.get("bms_num", "")).strip()
            mo_num = str(info.get("mother_num", "")).strip()

            # --- 既存特徴量 ---
            feat["blood_father_id"] = f_num if f_num else MISSING_CATEGORY
            feat["blood_bms_id"] = m_num if m_num else MISSING_CATEGORY

            feat["blood_father_keito"] = keito_map.get(
                f_num, MISSING_CATEGORY
            )
            feat["blood_bms_keito"] = keito_map.get(m_num, MISSING_CATEGORY)

            fs = father_stats.get(f_num, {})
            feat["blood_father_turf_rate"] = fs.get("turf_rate", MISSING_RATE)
            feat["blood_father_dirt_rate"] = fs.get("dirt_rate", MISSING_RATE)
            feat["blood_father_dist_rate"] = fs.get("dist_rate", MISSING_RATE)

            bs = bms_stats.get(m_num, {})
            feat["blood_bms_turf_rate"] = bs.get("turf_rate", MISSING_RATE)
            feat["blood_bms_dirt_rate"] = bs.get("dirt_rate", MISSING_RATE)

            feat["blood_inbreed_flag"] = info.get("inbreed_flag", 0)

            # --- 新規特徴量 ---

            # 母馬ID・系統
            feat["blood_mother_id"] = mo_num if mo_num else MISSING_CATEGORY
            feat["blood_mother_keito"] = keito_map.get(
                mo_num, MISSING_CATEGORY
            )

            # ニックス（父×母父）
            nicks_key = f"{f_num}_{m_num}" if f_num and m_num else ""
            ns = nicks_stats.get(nicks_key, {})
            feat["blood_nicks_rate"] = ns.get("rate", MISSING_RATE)
            feat["blood_nicks_runs"] = ns.get("runs", 0)

            # 父産駒の馬場状態別成績
            feat["blood_father_baba_rate"] = father_baba_stats.get(
                f_num, MISSING_RATE
            )

            # 父産駒の競馬場別成績
            feat["blood_father_jyo_rate"] = father_jyo_stats.get(
                f_num, MISSING_RATE
            )

            # 近親交配世代
            feat["blood_inbreed_generation"] = info.get(
                "inbreed_generation", 0
            )

            # 母の産駒成績
            feat["blood_mother_produce_rate"] = mother_produce_stats.get(
                mo_num, MISSING_RATE
            )

            results.append(feat)

        return pd.DataFrame(results).set_index("kettonum")

    # ------------------------------------------------------------------
    # データ取得
    # ------------------------------------------------------------------

    def _get_race_info(self, race_key: dict[str, str]) -> dict[str, str]:
        """レースの距離・トラック・馬場状態を取得する."""
        sql = """
        SELECT kyori, trackcd, sibababacd, dirtbabacd
        FROM n_race
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
        """n_sankuから血統情報（父・母・母父の繁殖番号）を取得する.

        Returns:
            kettonum → {father_num, mother_num, bms_num,
                        inbreed_flag, inbreed_generation} の辞書
        """
        if not kettonums:
            return {}

        sql = """
        SELECT kettonum, fnum, mnum, mfnum,
               ffnum, fmnum, mmnum,
               fffnum, ffmnum, fmfnum, fmmnum,
               mffnum, mfmnum, mmfnum, mmmnum
        FROM n_sanku
        WHERE kettonum IN %(kettonums)s
        """
        try:
            df = query_df(sql, {"kettonums": tuple(kettonums)})
        except Exception:
            return self._get_blood_info_fallback(kettonums)

        result: dict[str, dict[str, Any]] = {}
        for _, row in df.iterrows():
            kn = str(row["kettonum"]).strip()

            father_num = str(row.get("fnum", "")).strip()
            mother_num = str(row.get("mnum", "")).strip()
            bms_num = str(row.get("mfnum", "")).strip()

            # 近親交配チェック（世代情報付き）
            inbreed_flag, inbreed_gen = self._check_inbreeding(row)

            result[kn] = {
                "father_num": father_num,
                "mother_num": mother_num,
                "bms_num": bms_num,
                "inbreed_flag": inbreed_flag,
                "inbreed_generation": inbreed_gen,
            }

        return result

    @staticmethod
    def _check_inbreeding(row: pd.Series) -> tuple[int, int]:
        """近親交配をチェックし、フラグと最も近い重複世代を返す.

        Args:
            row: n_sanku の1行

        Returns:
            (inbreed_flag, inbreed_generation)
            inbreed_flag: 0=なし, 1=あり
            inbreed_generation: 0=なし, 2=2代で重複, 3=3代で重複
        """
        def _clean(val: Any) -> str:
            s = str(val).strip() if val is not None else ""
            return s if s and s != "0000000000" else ""

        # 世代別の祖先番号
        gen1 = [_clean(row.get("fnum")), _clean(row.get("mnum"))]
        gen2 = [
            _clean(row.get("ffnum")), _clean(row.get("fmnum")),
            _clean(row.get("mfnum")), _clean(row.get("mmnum")),
        ]
        gen3 = [
            _clean(row.get("fffnum")), _clean(row.get("ffmnum")),
            _clean(row.get("fmfnum")), _clean(row.get("fmmnum")),
            _clean(row.get("mffnum")), _clean(row.get("mfmnum")),
            _clean(row.get("mmfnum")), _clean(row.get("mmmnum")),
        ]

        # 全世代の有効な祖先
        all_ancestors = [a for a in gen1 + gen2 + gen3 if a]
        if len(all_ancestors) == len(set(all_ancestors)):
            return 0, 0  # 重複なし

        # 世代別に重複をチェック（近い世代から）
        # 2代チェック: gen1 と gen2 の中で重複がないか
        seen_gen2: set[str] = set()
        for a in gen1 + gen2:
            if a:
                if a in seen_gen2:
                    return 1, 2
                seen_gen2.add(a)

        # 3代チェック: gen3 まで含めて重複
        return 1, 3

    def _get_blood_info_fallback(
        self,
        kettonums: list[str],
    ) -> dict[str, dict[str, Any]]:
        """n_uma テーブルから血統情報を取得する（フォールバック）."""
        sql = """
        SELECT kettonum,
               ketto3infohannum1 AS father_num,
               ketto3infohannum2 AS mother_num,
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
                "mother_num": str(row.get("mother_num", "")).strip(),
                "bms_num": str(row.get("bms_num", "")).strip(),
                "inbreed_flag": 0,
                "inbreed_generation": 0,
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
            SUM(CASE WHEN CAST(ur.kakuteijyuni AS integer) <= 3
                THEN 1 ELSE 0 END) AS top3,
            SUM(CASE WHEN CAST(r.trackcd AS int) BETWEEN 10 AND 22
                THEN 1 ELSE 0 END) AS turf_runs,
            SUM(CASE WHEN CAST(r.trackcd AS int) BETWEEN 10 AND 22
                      AND CAST(ur.kakuteijyuni AS int) <= 3
                THEN 1 ELSE 0 END) AS turf_top3,
            SUM(CASE WHEN CAST(r.trackcd AS int) BETWEEN 23 AND 29
                THEN 1 ELSE 0 END) AS dirt_runs,
            SUM(CASE WHEN CAST(r.trackcd AS int) BETWEEN 23 AND 29
                      AND CAST(ur.kakuteijyuni AS int) <= 3
                THEN 1 ELSE 0 END) AS dirt_top3
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
            SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) <= 3
                THEN 1 ELSE 0 END) AS top3
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

    def _get_sire_baba_stats(
        self,
        sire_nums: list[str],
        race_date: str,
        year_start: str,
        baba_cd: str,
        trackcd: str,
    ) -> dict[str, float]:
        """父産駒の馬場状態別複勝率を取得する（過去3年）.

        Args:
            sire_nums: 父繁殖登録番号リスト
            race_date: レース日付（YYYYMMDD）
            year_start: 集計開始年
            baba_cd: 今回の馬場状態コード（1=良, 2=稍重, 3=重, 4=不良）
            trackcd: 今回のトラックコード（芝/ダートの判定用）
        """
        if not sire_nums or not baba_cd:
            return {}

        # トラック種別に応じた馬場状態カラムを選択
        try:
            tcd = int(trackcd)
        except (ValueError, TypeError):
            return {}

        if 10 <= tcd <= 22:
            baba_col = "r.sibababacd"
        elif 23 <= tcd <= 29:
            baba_col = "r.dirtbabacd"
        else:
            return {}

        sql = f"""
        SELECT
            s.fnum AS sire_num,
            COUNT(*) AS total,
            SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) <= 3
                THEN 1 ELSE 0 END) AS top3
        FROM n_sanku s
        JOIN n_uma_race ur ON s.kettonum = ur.kettonum
        JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
        WHERE s.fnum IN %(sire_nums)s
          AND ur.datakubun = '7'
          AND ur.ijyocd = '0'
          AND r.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
          AND r.year >= %(year_start)s
          AND (r.year || r.monthday) < %(race_date)s
          AND {baba_col} = %(baba_cd)s
        GROUP BY s.fnum
        """
        df = query_df(sql, {
            "sire_nums": tuple(sire_nums),
            "year_start": year_start,
            "race_date": race_date,
            "baba_cd": baba_cd,
        })

        result: dict[str, float] = {}
        for _, row in df.iterrows():
            sn = str(row["sire_num"]).strip()
            result[sn] = self._safe_rate(
                self._safe_int(row.get("top3")),
                self._safe_int(row.get("total")),
            )
        return result

    def _get_sire_jyo_stats(
        self,
        sire_nums: list[str],
        race_date: str,
        year_start: str,
        jyocd: str,
    ) -> dict[str, float]:
        """父産駒の競馬場別複勝率を取得する（過去3年）."""
        if not sire_nums or not jyocd:
            return {}

        sql = """
        SELECT
            s.fnum AS sire_num,
            COUNT(*) AS total,
            SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) <= 3
                THEN 1 ELSE 0 END) AS top3
        FROM n_sanku s
        JOIN n_uma_race ur ON s.kettonum = ur.kettonum
        JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
        WHERE s.fnum IN %(sire_nums)s
          AND ur.datakubun = '7'
          AND ur.ijyocd = '0'
          AND r.jyocd = %(jyocd)s
          AND r.year >= %(year_start)s
          AND (r.year || r.monthday) < %(race_date)s
        GROUP BY s.fnum
        """
        df = query_df(sql, {
            "sire_nums": tuple(sire_nums),
            "year_start": year_start,
            "race_date": race_date,
            "jyocd": jyocd,
        })

        result: dict[str, float] = {}
        for _, row in df.iterrows():
            sn = str(row["sire_num"]).strip()
            result[sn] = self._safe_rate(
                self._safe_int(row.get("top3")),
                self._safe_int(row.get("total")),
            )
        return result

    def _get_nicks_stats(
        self,
        pairs: list[tuple[str, str]],
        race_date: str,
    ) -> dict[str, dict[str, Any]]:
        """父×母父のニックス（相性）成績を集計する（過去5年）.

        同じ父×母父の組み合わせを持つ全産駒の成績を集計し、
        その組み合わせの複勝率と出走数を返す。

        Args:
            pairs: (father_num, bms_num) のリスト
            race_date: レース日付（YYYYMMDD）

        Returns:
            "father_bms" → {"rate": float, "runs": int} の辞書
        """
        if not pairs:
            return {}

        try:
            year_start = str(int(race_date[:4]) - 5)
        except ValueError:
            return {}

        # ユニークなペアのみ処理
        unique_pairs = list(set(pairs))

        result: dict[str, dict[str, Any]] = {}
        for f_num, bms_num in unique_pairs:
            nicks_key = f"{f_num}_{bms_num}"
            sql = """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) <= 3
                    THEN 1 ELSE 0 END) AS top3
            FROM n_sanku s
            JOIN n_uma_race ur ON s.kettonum = ur.kettonum
            JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji,
                                 racenum)
            WHERE s.fnum = %(father_num)s
              AND s.mfnum = %(bms_num)s
              AND ur.datakubun = '7'
              AND ur.ijyocd = '0'
              AND r.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
              AND r.year >= %(year_start)s
              AND (r.year || r.monthday) < %(race_date)s
            """
            try:
                df = query_df(sql, {
                    "father_num": f_num,
                    "bms_num": bms_num,
                    "year_start": year_start,
                    "race_date": race_date,
                })
            except Exception as e:
                logger.warning("ニックス集計エラー (%s×%s): %s", f_num, bms_num, e)
                continue

            if df.empty:
                continue

            total = self._safe_int(df.iloc[0].get("total"), default=0)
            top3 = self._safe_int(df.iloc[0].get("top3"), default=0)
            result[nicks_key] = {
                "rate": self._safe_rate(top3, total),
                "runs": total,
            }

        return result

    def _get_mother_produce_stats(
        self,
        mother_nums: list[str],
        current_kettonums: list[str],
        race_date: str,
    ) -> dict[str, float]:
        """母の産駒（兄弟姉妹）の複勝率を取得する.

        当該馬自身は除外して計算する（兄弟姉妹のみ）。

        Args:
            mother_nums: 母馬の繁殖登録番号リスト
            current_kettonums: 現在のレースの出走馬のkettonum（自身除外用）
            race_date: レース日付（YYYYMMDD）

        Returns:
            mother_num → 複勝率の辞書
        """
        if not mother_nums:
            return {}

        try:
            year_start = str(int(race_date[:4]) - 10)
        except ValueError:
            return {}

        sql = """
        SELECT
            s.mnum AS mother_num,
            COUNT(*) AS total,
            SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) <= 3
                THEN 1 ELSE 0 END) AS top3
        FROM n_sanku s
        JOIN n_uma_race ur ON s.kettonum = ur.kettonum
        JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
        WHERE s.mnum IN %(mother_nums)s
          AND s.kettonum NOT IN %(exclude_kettonums)s
          AND ur.datakubun = '7'
          AND ur.ijyocd = '0'
          AND r.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
          AND r.year >= %(year_start)s
          AND (r.year || r.monthday) < %(race_date)s
        GROUP BY s.mnum
        """
        try:
            df = query_df(sql, {
                "mother_nums": tuple(mother_nums),
                "exclude_kettonums": tuple(current_kettonums),
                "year_start": year_start,
                "race_date": race_date,
            })
        except Exception as e:
            logger.warning("母産駒成績取得エラー: %s", e)
            return {}

        result: dict[str, float] = {}
        for _, row in df.iterrows():
            mn = str(row["mother_num"]).strip()
            result[mn] = self._safe_rate(
                self._safe_int(row.get("top3"), default=0),
                self._safe_int(row.get("total"), default=0),
            )
        return result

    def _empty_result(self, uma_race_df: pd.DataFrame) -> pd.DataFrame:
        idx = (
            uma_race_df["kettonum"].tolist()
            if "kettonum" in uma_race_df.columns
            else []
        )
        return pd.DataFrame(index=idx, columns=self._FEATURES, dtype=object)
