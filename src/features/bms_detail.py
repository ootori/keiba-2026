"""BMS（母父）条件別パフォーマンス 特徴量（v2提案B: サプリメント）.

blood_bms_id（重要度2位）が持つ情報を条件別に展開する。
father側の6条件別特徴量に対し、BMS側は2つのみという非対称を解消する。

特徴量一覧:
    - blood_bms_dist_rate:       母父産駒の今回距離帯での複勝率（過去3年）
    - blood_bms_baba_rate:       母父産駒の今回馬場状態での複勝率（過去3年）
    - blood_bms_jyo_rate:        母父産駒の今回競馬場での複勝率（過去3年）
    - blood_father_age_rate:     父産駒の同馬齢での複勝率（過去5年）
    - blood_nicks_track_rate:    父×母父ニックスの芝/ダート別複勝率（過去5年）
    - blood_father_class_rate:   父産駒のクラス別成績（過去3年）

ノイズ抑制策:
    - 最小サンプル数閾値（MIN_SAMPLES / MIN_SAMPLES_NICKS）を設け、
      サンプル不足の率は NaN（LightGBMネイティブ欠損）として返す
    - MISSING_RATE=0.0 ではなく NaN を使い「データなし」と「0%」を区別
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from src.features.base import FeatureExtractor
from src.db import query_df
from src.utils.code_master import (
    track_type,
    distance_category,
    baba_code_for_track,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# ノイズ抑制パラメータ
# --------------------------------------------------------------------------
# サンプル数がこの閾値未満の場合はNaN（LightGBMが欠損として扱う）
MIN_SAMPLES: int = 20
# ニックス（父×母父）はさらに掛け合わせが細かいため高めの閾値
MIN_SAMPLES_NICKS: int = 30
# 欠損値: LightGBMはNaNをネイティブに処理可能
_NAN = float("nan")


def _classify_class(gradecd: str, jyokencd5: str) -> str:
    """GradeCD と JyokenCD5 からクラスを分類する.

    Args:
        gradecd: GradeCD値（A/B/C/D=重賞、E=特別、空白=一般）
        jyokencd5: JyokenCD5値

    Returns:
        "grade"（重賞）, "open"（オープン）, "jouken"（条件戦）
    """
    g = str(gradecd).strip()
    if g in ("A", "B", "C", "D"):
        return "grade"

    cd5 = str(jyokencd5).strip()
    if not cd5:
        return "jouken"
    try:
        val = int(cd5)
    except (ValueError, TypeError):
        return "jouken"

    # OP・L（リステッド）は open、それ以外は条件戦
    if val >= 900:
        return "open"
    return "jouken"


def _safe_rate_with_threshold(
    top3: int, total: int, min_samples: int = MIN_SAMPLES,
) -> float:
    """最小サンプル数を考慮した安全な率計算.

    サンプル数が min_samples 未満の場合は NaN を返す。
    これにより LightGBM が「データなし」として扱い、
    少数サンプルからのノイジーな率で誤学習することを防ぐ。

    Args:
        top3: 3着以内の件数
        total: 全件数
        min_samples: 最小サンプル数の閾値

    Returns:
        複勝率（float）。サンプル不足の場合は NaN
    """
    if total < min_samples:
        return _NAN
    return top3 / total


class BMSDetailFeatureExtractor(FeatureExtractor):
    """BMS（母父）条件別パフォーマンスの特徴量を抽出する（サプリメント）."""

    _FEATURES: list[str] = [
        "blood_bms_dist_rate",
        "blood_bms_baba_rate",
        "blood_bms_jyo_rate",
        "blood_father_age_rate",
        "blood_nicks_track_rate",
        "blood_father_class_rate",
    ]

    @property
    def feature_names(self) -> list[str]:
        return self._FEATURES

    def extract(
        self,
        race_key: dict[str, str],
        uma_race_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """BMS条件別特徴量を抽出する."""
        kettonums = (
            uma_race_df["kettonum"].tolist()
            if "kettonum" in uma_race_df.columns
            else []
        )
        if not kettonums:
            return pd.DataFrame(columns=self._FEATURES)

        race_date = race_key["year"] + race_key["monthday"]

        # --- レース情報の取得 ---
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
        current_track_type = track_type(current_trackcd)
        current_gradecd = str(race_info.get("gradecd", "")).strip()
        current_jyokencd5 = str(race_info.get("jyokencd5", "")).strip()
        current_class = _classify_class(current_gradecd, current_jyokencd5)

        # --- 出走馬の馬齢取得 ---
        horse_barei = self._get_horse_barei(race_key, kettonums)

        # --- 血統情報を一括取得（父・母父の繁殖登録番号） ---
        blood_info = self._get_blood_info(kettonums)

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

        # 集計期間
        try:
            year_start_3y = str(int(race_date[:4]) - 3)
            year_start_5y = str(int(race_date[:4]) - 5)
        except ValueError:
            year_start_3y = "2012"
            year_start_5y = "2010"

        # --- B1: BMS距離帯別成績 ---
        bms_dist_stats = (
            self._get_bms_dist_stats(
                list(bms_nums), race_date, year_start_3y, current_dist_cat
            )
            if bms_nums
            else {}
        )

        # --- B2: BMS馬場状態別成績 ---
        bms_baba_stats = (
            self._get_bms_baba_stats(
                list(bms_nums), race_date, year_start_3y,
                current_baba_cd, current_trackcd,
            )
            if bms_nums and current_baba_cd and current_baba_cd != "0"
            else {}
        )

        # --- B3: BMS競馬場別成績 ---
        bms_jyo_stats = (
            self._get_bms_jyo_stats(
                list(bms_nums), race_date, year_start_3y, current_jyocd,
            )
            if bms_nums and current_jyocd
            else {}
        )

        # --- B4: 父産駒の馬齢別成績 ---
        barei_father_map: dict[int, set[str]] = {}
        for kn in kettonums:
            kn_str = str(kn).strip()
            info = blood_info.get(kn_str, {})
            f_num = str(info.get("father_num", "")).strip()
            barei = horse_barei.get(kn_str, -1)
            if f_num and barei > 0:
                barei_father_map.setdefault(barei, set()).add(f_num)

        father_age_stats: dict[str, dict[int, float]] = {}
        for barei_val, fnums in barei_father_map.items():
            stats = self._get_sire_age_stats(
                list(fnums), race_date, year_start_5y, barei_val,
            )
            for fn, rate in stats.items():
                father_age_stats.setdefault(fn, {})[barei_val] = rate

        # --- B5: ニックスのトラック種別別成績 ---
        nicks_pairs: list[tuple[str, str]] = []
        for kn in kettonums:
            kn_str = str(kn).strip()
            info = blood_info.get(kn_str, {})
            f_num = str(info.get("father_num", "")).strip()
            m_num = str(info.get("bms_num", "")).strip()
            if f_num and m_num:
                nicks_pairs.append((f_num, m_num))

        nicks_track_stats = (
            self._get_nicks_track_stats(
                nicks_pairs, race_date, year_start_5y, current_track_type,
            )
            if nicks_pairs and current_track_type in ("turf", "dirt")
            else {}
        )

        # --- B6: 父産駒のクラス別成績 ---
        father_class_stats = (
            self._get_sire_class_stats(
                list(father_nums), race_date, year_start_3y, current_class,
            )
            if father_nums
            else {}
        )

        # --- 特徴量組み立て ---
        results: list[dict[str, Any]] = []
        for kn in kettonums:
            kn_str = str(kn).strip()
            info = blood_info.get(kn_str, {})
            feat: dict[str, Any] = {"kettonum": kn_str}

            f_num = str(info.get("father_num", "")).strip()
            m_num = str(info.get("bms_num", "")).strip()
            barei = horse_barei.get(kn_str, -1)

            # B1: BMS距離帯別
            feat["blood_bms_dist_rate"] = bms_dist_stats.get(m_num, _NAN)

            # B2: BMS馬場状態別
            feat["blood_bms_baba_rate"] = bms_baba_stats.get(m_num, _NAN)

            # B3: BMS競馬場別
            feat["blood_bms_jyo_rate"] = bms_jyo_stats.get(m_num, _NAN)

            # B4: 父産駒の馬齢別
            age_map = father_age_stats.get(f_num, {})
            feat["blood_father_age_rate"] = age_map.get(barei, _NAN)

            # B5: ニックストラック種別別
            nicks_key = f"{f_num}_{m_num}" if f_num and m_num else ""
            feat["blood_nicks_track_rate"] = nicks_track_stats.get(
                nicks_key, _NAN
            )

            # B6: 父産駒クラス別
            feat["blood_father_class_rate"] = father_class_stats.get(
                f_num, _NAN
            )

            results.append(feat)

        return pd.DataFrame(results).set_index("kettonum")

    # ------------------------------------------------------------------
    # データ取得
    # ------------------------------------------------------------------

    def _get_race_info(self, race_key: dict[str, str]) -> dict[str, str]:
        """レースの距離・トラック・馬場状態・グレード等を取得する."""
        sql = """
        SELECT kyori, trackcd, sibababacd, dirtbabacd, gradecd, jyokencd5
        FROM n_race
        WHERE year = %(year)s AND monthday = %(monthday)s
          AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
          AND nichiji = %(nichiji)s AND racenum = %(racenum)s
        LIMIT 1
        """
        df = query_df(sql, race_key)
        return df.iloc[0].to_dict() if not df.empty else {}

    def _get_horse_barei(
        self,
        race_key: dict[str, str],
        kettonums: list[str],
    ) -> dict[str, int]:
        """出走馬の馬齢を取得する."""
        sql = """
        SELECT kettonum, barei
        FROM n_uma_race
        WHERE year = %(year)s AND monthday = %(monthday)s
          AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
          AND nichiji = %(nichiji)s AND racenum = %(racenum)s
          AND kettonum IN %(kettonums)s
          AND datakubun IN ('1','2','3','4','5','6','7')
        """
        params = dict(race_key)
        params["kettonums"] = tuple(kettonums)
        df = query_df(sql, params)
        result: dict[str, int] = {}
        for _, row in df.iterrows():
            kn = str(row["kettonum"]).strip()
            result[kn] = self._safe_int(row.get("barei"), default=-1)
        return result

    def _get_blood_info(
        self,
        kettonums: list[str],
    ) -> dict[str, dict[str, str]]:
        """n_sankuから父・母父の繁殖登録番号を取得する."""
        if not kettonums:
            return {}

        sql = """
        SELECT kettonum, fnum, mfnum
        FROM n_sanku
        WHERE kettonum IN %(kettonums)s
        """
        try:
            df = query_df(sql, {"kettonums": tuple(kettonums)})
        except Exception:
            return self._get_blood_info_fallback(kettonums)

        result: dict[str, dict[str, str]] = {}
        for _, row in df.iterrows():
            kn = str(row["kettonum"]).strip()
            result[kn] = {
                "father_num": str(row.get("fnum", "")).strip(),
                "bms_num": str(row.get("mfnum", "")).strip(),
            }
        return result

    def _get_blood_info_fallback(
        self,
        kettonums: list[str],
    ) -> dict[str, dict[str, str]]:
        """n_uma テーブルから血統情報を取得する（フォールバック）."""
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

        result: dict[str, dict[str, str]] = {}
        for _, row in df.iterrows():
            kn = str(row["kettonum"]).strip()
            result[kn] = {
                "father_num": str(row.get("father_num", "")).strip(),
                "bms_num": str(row.get("bms_num", "")).strip(),
            }
        return result

    # ------------------------------------------------------------------
    # B1: BMS距離帯別成績
    # ------------------------------------------------------------------

    def _get_bms_dist_stats(
        self,
        bms_nums: list[str],
        race_date: str,
        year_start: str,
        dist_cat: str,
    ) -> dict[str, float]:
        """母父産駒の距離帯別複勝率を取得する（過去3年）."""
        if not bms_nums:
            return {}

        dist_conditions = {
            "short": "CAST(r.kyori AS int) <= 1400",
            "mile": "CAST(r.kyori AS int) BETWEEN 1401 AND 1800",
            "middle": "CAST(r.kyori AS int) BETWEEN 1801 AND 2200",
            "long": "CAST(r.kyori AS int) >= 2201",
        }
        dist_cond = dist_conditions.get(dist_cat, "1=1")

        sql = f"""
        SELECT
            s.mfnum AS bms_num,
            COUNT(*) AS total,
            SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) <= 3
                THEN 1 ELSE 0 END) AS top3
        FROM n_sanku s
        JOIN n_uma_race ur ON s.kettonum = ur.kettonum
        JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
        WHERE s.mfnum IN %(bms_nums)s
          AND ur.datakubun = '7'
          AND ur.ijyocd = '0'
          AND r.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
          AND r.year >= %(year_start)s
          AND (r.year || r.monthday) < %(race_date)s
          AND {dist_cond}
        GROUP BY s.mfnum
        """
        try:
            df = query_df(sql, {
                "bms_nums": tuple(bms_nums),
                "year_start": year_start,
                "race_date": race_date,
            })
        except Exception as e:
            logger.warning("BMS距離帯別成績取得エラー: %s", e)
            return {}

        result: dict[str, float] = {}
        for _, row in df.iterrows():
            bn = str(row["bms_num"]).strip()
            total = self._safe_int(row.get("total"), default=0)
            top3 = self._safe_int(row.get("top3"), default=0)
            rate = _safe_rate_with_threshold(top3, total, MIN_SAMPLES)
            if not math.isnan(rate):
                result[bn] = rate
        return result

    # ------------------------------------------------------------------
    # B2: BMS馬場状態別成績
    # ------------------------------------------------------------------

    def _get_bms_baba_stats(
        self,
        bms_nums: list[str],
        race_date: str,
        year_start: str,
        baba_cd: str,
        trackcd: str,
    ) -> dict[str, float]:
        """母父産駒の馬場状態別複勝率を取得する（過去3年）."""
        if not bms_nums or not baba_cd:
            return {}

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
            s.mfnum AS bms_num,
            COUNT(*) AS total,
            SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) <= 3
                THEN 1 ELSE 0 END) AS top3
        FROM n_sanku s
        JOIN n_uma_race ur ON s.kettonum = ur.kettonum
        JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
        WHERE s.mfnum IN %(bms_nums)s
          AND ur.datakubun = '7'
          AND ur.ijyocd = '0'
          AND r.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
          AND r.year >= %(year_start)s
          AND (r.year || r.monthday) < %(race_date)s
          AND {baba_col} = %(baba_cd)s
        GROUP BY s.mfnum
        """
        try:
            df = query_df(sql, {
                "bms_nums": tuple(bms_nums),
                "year_start": year_start,
                "race_date": race_date,
                "baba_cd": baba_cd,
            })
        except Exception as e:
            logger.warning("BMS馬場状態別成績取得エラー: %s", e)
            return {}

        result: dict[str, float] = {}
        for _, row in df.iterrows():
            bn = str(row["bms_num"]).strip()
            total = self._safe_int(row.get("total"), default=0)
            top3 = self._safe_int(row.get("top3"), default=0)
            rate = _safe_rate_with_threshold(top3, total, MIN_SAMPLES)
            if not math.isnan(rate):
                result[bn] = rate
        return result

    # ------------------------------------------------------------------
    # B3: BMS競馬場別成績
    # ------------------------------------------------------------------

    def _get_bms_jyo_stats(
        self,
        bms_nums: list[str],
        race_date: str,
        year_start: str,
        jyocd: str,
    ) -> dict[str, float]:
        """母父産駒の競馬場別複勝率を取得する（過去3年）."""
        if not bms_nums or not jyocd:
            return {}

        sql = """
        SELECT
            s.mfnum AS bms_num,
            COUNT(*) AS total,
            SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) <= 3
                THEN 1 ELSE 0 END) AS top3
        FROM n_sanku s
        JOIN n_uma_race ur ON s.kettonum = ur.kettonum
        JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
        WHERE s.mfnum IN %(bms_nums)s
          AND ur.datakubun = '7'
          AND ur.ijyocd = '0'
          AND r.jyocd = %(jyocd)s
          AND r.year >= %(year_start)s
          AND (r.year || r.monthday) < %(race_date)s
        GROUP BY s.mfnum
        """
        try:
            df = query_df(sql, {
                "bms_nums": tuple(bms_nums),
                "year_start": year_start,
                "race_date": race_date,
                "jyocd": jyocd,
            })
        except Exception as e:
            logger.warning("BMS競馬場別成績取得エラー: %s", e)
            return {}

        result: dict[str, float] = {}
        for _, row in df.iterrows():
            bn = str(row["bms_num"]).strip()
            total = self._safe_int(row.get("total"), default=0)
            top3 = self._safe_int(row.get("top3"), default=0)
            rate = _safe_rate_with_threshold(top3, total, MIN_SAMPLES)
            if not math.isnan(rate):
                result[bn] = rate
        return result

    # ------------------------------------------------------------------
    # B4: 父産駒の馬齢別成績
    # ------------------------------------------------------------------

    def _get_sire_age_stats(
        self,
        sire_nums: list[str],
        race_date: str,
        year_start: str,
        barei: int,
    ) -> dict[str, float]:
        """父産駒の同馬齢での複勝率を取得する（過去5年）."""
        if not sire_nums or barei <= 0:
            return {}

        sql = """
        SELECT
            s.fnum AS sire_num,
            COUNT(*) AS total,
            SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) <= 3
                THEN 1 ELSE 0 END) AS top3
        FROM n_sanku s
        JOIN n_uma_race ur ON s.kettonum = ur.kettonum
        WHERE s.fnum IN %(sire_nums)s
          AND ur.datakubun = '7'
          AND ur.ijyocd = '0'
          AND CAST(ur.barei AS int) = %(barei)s
          AND ur.year >= %(year_start)s
          AND (ur.year || ur.monthday) < %(race_date)s
        GROUP BY s.fnum
        """
        try:
            df = query_df(sql, {
                "sire_nums": tuple(sire_nums),
                "year_start": year_start,
                "race_date": race_date,
                "barei": barei,
            })
        except Exception as e:
            logger.warning("父産駒馬齢別成績取得エラー: %s", e)
            return {}

        result: dict[str, float] = {}
        for _, row in df.iterrows():
            sn = str(row["sire_num"]).strip()
            total = self._safe_int(row.get("total"), default=0)
            top3 = self._safe_int(row.get("top3"), default=0)
            rate = _safe_rate_with_threshold(top3, total, MIN_SAMPLES)
            if not math.isnan(rate):
                result[sn] = rate
        return result

    # ------------------------------------------------------------------
    # B5: ニックスのトラック種別別成績
    # ------------------------------------------------------------------

    def _get_nicks_track_stats(
        self,
        pairs: list[tuple[str, str]],
        race_date: str,
        year_start: str,
        current_track_type: str,
    ) -> dict[str, float]:
        """父×母父ニックスの芝/ダート別複勝率を取得する（過去5年）.

        Args:
            pairs: (father_num, bms_num) のリスト
            race_date: レース日付（YYYYMMDD）
            year_start: 集計開始年
            current_track_type: "turf" or "dirt"

        Returns:
            "father_bms" → 複勝率の辞書
        """
        if not pairs or current_track_type not in ("turf", "dirt"):
            return {}

        unique_pairs = list(set(pairs))
        father_nums = list({p[0] for p in unique_pairs})
        bms_nums = list({p[1] for p in unique_pairs})

        # トラック種別フィルタ
        if current_track_type == "turf":
            track_cond = "CAST(r.trackcd AS int) BETWEEN 10 AND 22"
        else:
            track_cond = "CAST(r.trackcd AS int) BETWEEN 23 AND 29"

        sql = f"""
        SELECT
            s.fnum AS father_num,
            s.mfnum AS bms_num,
            COUNT(*) AS total,
            SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) <= 3
                THEN 1 ELSE 0 END) AS top3
        FROM n_sanku s
        JOIN n_uma_race ur ON s.kettonum = ur.kettonum
        JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
        WHERE s.fnum IN %(father_nums)s
          AND s.mfnum IN %(bms_nums)s
          AND ur.datakubun = '7'
          AND ur.ijyocd = '0'
          AND r.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
          AND {track_cond}
          AND r.year >= %(year_start)s
          AND (r.year || r.monthday) < %(race_date)s
        GROUP BY s.fnum, s.mfnum
        """
        try:
            df = query_df(sql, {
                "father_nums": tuple(father_nums),
                "bms_nums": tuple(bms_nums),
                "year_start": year_start,
                "race_date": race_date,
            })
        except Exception as e:
            logger.warning("ニックストラック種別別成績取得エラー: %s", e)
            return {}

        target_pairs = {f"{f}_{b}" for f, b in unique_pairs}
        result: dict[str, float] = {}
        for _, row in df.iterrows():
            fn = str(row["father_num"]).strip()
            bn = str(row["bms_num"]).strip()
            nicks_key = f"{fn}_{bn}"
            if nicks_key not in target_pairs:
                continue
            total = self._safe_int(row.get("total"), default=0)
            top3 = self._safe_int(row.get("top3"), default=0)
            rate = _safe_rate_with_threshold(
                top3, total, MIN_SAMPLES_NICKS,
            )
            if not math.isnan(rate):
                result[nicks_key] = rate
        return result

    # ------------------------------------------------------------------
    # B6: 父産駒のクラス別成績
    # ------------------------------------------------------------------

    def _get_sire_class_stats(
        self,
        sire_nums: list[str],
        race_date: str,
        year_start: str,
        current_class: str,
    ) -> dict[str, float]:
        """父産駒のクラス別（重賞/OP/条件戦）複勝率を取得する（過去3年）.

        Args:
            sire_nums: 父繁殖登録番号リスト
            race_date: レース日付（YYYYMMDD）
            year_start: 集計開始年
            current_class: "grade", "open", "jouken"

        Returns:
            sire_num → 複勝率の辞書
        """
        if not sire_nums:
            return {}

        # クラスに応じたWHERE条件を構築
        if current_class == "grade":
            class_cond = "r.gradecd IN ('A','B','C','D')"
        elif current_class == "open":
            class_cond = (
                "(r.gradecd NOT IN ('A','B','C','D') OR r.gradecd IS NULL"
                " OR TRIM(r.gradecd) = '')"
                " AND CAST(COALESCE(NULLIF(TRIM(r.jyokencd5), ''), '0')"
                " AS int) >= 900"
            )
        else:  # jouken
            class_cond = (
                "(r.gradecd NOT IN ('A','B','C','D') OR r.gradecd IS NULL"
                " OR TRIM(r.gradecd) = '')"
                " AND CAST(COALESCE(NULLIF(TRIM(r.jyokencd5), ''), '0')"
                " AS int) < 900"
            )

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
          AND {class_cond}
        GROUP BY s.fnum
        """
        try:
            df = query_df(sql, {
                "sire_nums": tuple(sire_nums),
                "year_start": year_start,
                "race_date": race_date,
            })
        except Exception as e:
            logger.warning("父産駒クラス別成績取得エラー: %s", e)
            return {}

        result: dict[str, float] = {}
        for _, row in df.iterrows():
            sn = str(row["sire_num"]).strip()
            total = self._safe_int(row.get("total"), default=0)
            top3 = self._safe_int(row.get("top3"), default=0)
            rate = _safe_rate_with_threshold(top3, total, MIN_SAMPLES)
            if not math.isnan(rate):
                result[sn] = rate
        return result
