"""スピード指数 + 脚質 特徴量（カテゴリ4, 5）."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from src.features.base import FeatureExtractor
from src.db import query_df
from src.config import MISSING_NUMERIC, MISSING_RATE
from src.utils.code_master import time_to_sec, haron_time_to_sec, track_type, baba_code_for_track
from src.utils.base_time import calc_speed_index, get_or_build_base_time


class SpeedStyleFeatureExtractor(FeatureExtractor):
    """スピード指数と脚質の特徴量を抽出する."""

    _FEATURES: list[str] = [
        # カテゴリ4: スピード指数
        "speed_time_last",
        "speed_time_avg_last3",
        "speed_l3f_last",
        "speed_l3f_avg_last3",
        "speed_l3f_best_last5",
        "speed_l3f_rank_last",
        "speed_timediff_last",
        "speed_timediff_avg_last3",
        "speed_index_last",
        "speed_index_avg_last3",
        "speed_index_max_last5",
        "speed_l3f_time_ratio",
        # カテゴリ5: 脚質
        "style_type_last",
        "style_type_mode_last5",
        "style_avg_pos_1c_last3",
        "style_avg_pos_3c_last3",
        "style_avg_pos_4c_last3",
        "style_pos_change_last",
        "style_front_ratio_last5",
    ]

    def __init__(self) -> None:
        self._base_time_dict: dict[tuple[str, str, str], float] | None = None

    @property
    def feature_names(self) -> list[str]:
        return self._FEATURES

    def _ensure_base_time(self) -> dict[tuple[str, str, str], float]:
        """基準タイムテーブルをロード/構築する."""
        if self._base_time_dict is None:
            self._base_time_dict = get_or_build_base_time()
        return self._base_time_dict

    def extract(
        self,
        race_key: dict[str, str],
        uma_race_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """スピード・脚質の特徴量を抽出する."""
        race_date = race_key["year"] + race_key["monthday"]
        base_time_dict = self._ensure_base_time()

        # 出走馬のkettonumリスト
        kettonums = uma_race_df["kettonum"].tolist() if "kettonum" in uma_race_df.columns else []
        if not kettonums:
            return self._empty_result(uma_race_df)

        # 過去成績を一括取得
        past_df = self._get_past_results_with_style(kettonums, race_date)

        # 前走のレース全体情報（上がり3F順位算出用）
        last_race_l3f = self._get_last_race_l3f(past_df)

        results: list[dict[str, Any]] = []
        for kn in kettonums:
            kn_str = str(kn).strip()
            h_past = past_df[past_df["kettonum"] == kn_str] if not past_df.empty else pd.DataFrame()
            feat = self._build_features(kn_str, h_past, base_time_dict, last_race_l3f)
            results.append(feat)

        df = pd.DataFrame(results).set_index("kettonum")
        return df

    # ------------------------------------------------------------------
    # データ取得
    # ------------------------------------------------------------------

    def _get_past_results_with_style(
        self,
        kettonums: list[str],
        race_date: str,
    ) -> pd.DataFrame:
        """過去成績（タイム・脚質・コーナー順位を含む）を一括取得する."""
        if not kettonums:
            return pd.DataFrame()

        sql = """
        SELECT
            ur.kettonum,
            ur.kakuteijyuni,
            ur.time,
            ur.harontimel3,
            ur.timediff,
            ur.kyakusitukubun,
            ur.jyuni1c, ur.jyuni2c, ur.jyuni3c, ur.jyuni4c,
            ur.year, ur.monthday, ur.jyocd, ur.kaiji, ur.nichiji, ur.racenum,
            r.kyori, r.trackcd, r.sibababacd, r.dirtbabacd
        FROM n_uma_race ur
        JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
        WHERE ur.kettonum IN %(kettonums)s
          AND ur.datakubun = '7'
          AND ur.ijyocd = '0'
          AND r.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
          AND (ur.year || ur.monthday) < %(race_date)s
        ORDER BY ur.kettonum, ur.year DESC, ur.monthday DESC
        """
        return query_df(sql, {"kettonums": tuple(kettonums), "race_date": race_date})

    def _get_last_race_l3f(
        self,
        past_df: pd.DataFrame,
    ) -> dict[str, dict[str, Any]]:
        """各馬の前走における同レース出走馬の上がり3F情報を取得する.

        Returns:
            kettonum → {'race_key': ..., 'rank': rank_in_race} の辞書
        """
        if past_df.empty:
            return {}

        # 各馬の前走（最新走）を取得
        first_rows = past_df.groupby("kettonum").first().reset_index()
        result: dict[str, dict[str, Any]] = {}

        for _, row in first_rows.iterrows():
            kn = str(row["kettonum"]).strip()
            l3f = haron_time_to_sec(str(row.get("harontimel3", "")).strip())
            if l3f is None:
                result[kn] = {"rank": MISSING_NUMERIC}
                continue

            # 前走の全出走馬の上がり3Fを取得して順位算出
            race_l3f_sql = """
            SELECT harontimel3
            FROM n_uma_race
            WHERE year = %(year)s AND monthday = %(monthday)s
              AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
              AND nichiji = %(nichiji)s AND racenum = %(racenum)s
              AND datakubun = '7' AND ijyocd = '0'
              AND LENGTH(TRIM(harontimel3)) >= 3
            """
            race_params = {
                "year": str(row["year"]).strip(),
                "monthday": str(row["monthday"]).strip(),
                "jyocd": str(row["jyocd"]).strip(),
                "kaiji": str(row["kaiji"]).strip(),
                "nichiji": str(row["nichiji"]).strip(),
                "racenum": str(row["racenum"]).strip(),
            }
            try:
                race_l3f_df = query_df(race_l3f_sql, race_params)
                all_l3f = race_l3f_df["harontimel3"].apply(
                    lambda x: haron_time_to_sec(str(x).strip())
                ).dropna().sort_values()
                rank = int((all_l3f <= l3f).sum())
                result[kn] = {"rank": rank}
            except Exception:
                result[kn] = {"rank": MISSING_NUMERIC}

        return result

    # ------------------------------------------------------------------
    # 特徴量構築
    # ------------------------------------------------------------------

    def _build_features(
        self,
        kettonum: str,
        h_past: pd.DataFrame,
        base_time_dict: dict[tuple[str, str, str], float],
        last_race_l3f: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """1頭分の特徴量を構築する."""
        feat: dict[str, Any] = {"kettonum": kettonum}

        if h_past.empty:
            for f in self._FEATURES:
                if f.startswith("style_type"):
                    feat[f] = "0"
                else:
                    feat[f] = MISSING_NUMERIC
            return feat

        # --- スピード系の準備 ---
        times_sec = h_past["time"].apply(lambda x: time_to_sec(str(x).strip()))
        l3f_sec = h_past["harontimel3"].apply(lambda x: haron_time_to_sec(str(x).strip()))
        timediff = h_past["timediff"].apply(lambda x: self._safe_float(x, default=np.nan))

        # スピード指数の計算
        speed_indices: list[float] = []
        for _, row in h_past.head(5).iterrows():
            t = time_to_sec(str(row.get("time", "")).strip())
            if t is None or t <= 0:
                speed_indices.append(0.0)
                continue
            kyori = str(row.get("kyori", "")).strip()
            tt = track_type(str(row.get("trackcd", "")).strip())
            baba = baba_code_for_track(
                str(row.get("trackcd", "")).strip(),
                str(row.get("sibababacd", "")).strip(),
                str(row.get("dirtbabacd", "")).strip(),
            )
            si = calc_speed_index(t, kyori, tt, baba, base_time_dict)
            speed_indices.append(si)

        # --- カテゴリ4: スピード指数 ---
        t_last = times_sec.iloc[0] if len(times_sec) > 0 else np.nan
        feat["speed_time_last"] = float(t_last) if pd.notna(t_last) else MISSING_NUMERIC

        last3_times = times_sec.head(3).dropna()
        feat["speed_time_avg_last3"] = float(last3_times.mean()) if len(last3_times) > 0 else MISSING_NUMERIC

        l3f_last = l3f_sec.iloc[0] if len(l3f_sec) > 0 else np.nan
        feat["speed_l3f_last"] = float(l3f_last) if pd.notna(l3f_last) else MISSING_NUMERIC

        last3_l3f = l3f_sec.head(3).dropna()
        feat["speed_l3f_avg_last3"] = float(last3_l3f.mean()) if len(last3_l3f) > 0 else MISSING_NUMERIC

        last5_l3f = l3f_sec.head(5).dropna()
        feat["speed_l3f_best_last5"] = float(last5_l3f.min()) if len(last5_l3f) > 0 else MISSING_NUMERIC

        # 上がり3F順位
        l3f_info = last_race_l3f.get(kettonum, {})
        feat["speed_l3f_rank_last"] = l3f_info.get("rank", MISSING_NUMERIC)

        # タイム差
        td_last = timediff.iloc[0] if len(timediff) > 0 else np.nan
        feat["speed_timediff_last"] = (
            float(td_last) / 10.0 if pd.notna(td_last) else MISSING_NUMERIC
        )

        last3_td = timediff.head(3).dropna()
        feat["speed_timediff_avg_last3"] = (
            float(last3_td.mean()) / 10.0 if len(last3_td) > 0 else MISSING_NUMERIC
        )

        # スピード指数
        feat["speed_index_last"] = speed_indices[0] if speed_indices else MISSING_NUMERIC
        feat["speed_index_avg_last3"] = (
            float(np.mean(speed_indices[:3])) if len(speed_indices) >= 1 else MISSING_NUMERIC
        )
        feat["speed_index_max_last5"] = (
            float(np.max(speed_indices[:5])) if speed_indices else MISSING_NUMERIC
        )

        # 上がり3F / 走破タイム比
        if pd.notna(l3f_last) and pd.notna(t_last) and t_last > 0:
            feat["speed_l3f_time_ratio"] = float(l3f_last / t_last)
        else:
            feat["speed_l3f_time_ratio"] = MISSING_NUMERIC

        # --- カテゴリ5: 脚質 ---
        kyakusitu = h_past["kyakusitukubun"].apply(
            lambda x: str(x).strip() if x else "0"
        )

        feat["style_type_last"] = kyakusitu.iloc[0] if len(kyakusitu) > 0 else "0"

        # 直近5走の最頻脚質
        last5_style = kyakusitu.head(5).tolist()
        valid_styles = [s for s in last5_style if s in ("1", "2", "3", "4")]
        if valid_styles:
            feat["style_type_mode_last5"] = Counter(valid_styles).most_common(1)[0][0]
        else:
            feat["style_type_mode_last5"] = "0"

        # コーナー通過順位
        for col, fname in [
            ("jyuni1c", "style_avg_pos_1c_last3"),
            ("jyuni3c", "style_avg_pos_3c_last3"),
            ("jyuni4c", "style_avg_pos_4c_last3"),
        ]:
            vals = h_past[col].head(3).apply(
                lambda x: self._safe_float(x, default=np.nan)
            ).dropna()
            feat[fname] = float(vals.mean()) if len(vals) > 0 else MISSING_NUMERIC

        # 前走の4角→最終着順の変動
        if len(h_past) > 0:
            pos4c = self._safe_float(h_past.iloc[0].get("jyuni4c"), default=np.nan)
            jyuni = self._safe_float(h_past.iloc[0].get("kakuteijyuni"), default=np.nan)
            if pd.notna(pos4c) and pd.notna(jyuni) and pos4c > 0 and jyuni > 0:
                feat["style_pos_change_last"] = float(pos4c - jyuni)
            else:
                feat["style_pos_change_last"] = MISSING_NUMERIC
        else:
            feat["style_pos_change_last"] = MISSING_NUMERIC

        # 直近5走で3角3番手以内だった割合
        pos3c_last5 = h_past["jyuni3c"].head(5).apply(
            lambda x: self._safe_int(str(x).strip()) if x else 0
        )
        valid_pos3c = pos3c_last5[pos3c_last5 > 0]
        if len(valid_pos3c) > 0:
            feat["style_front_ratio_last5"] = float(
                (valid_pos3c <= 3).sum() / len(valid_pos3c)
            )
        else:
            feat["style_front_ratio_last5"] = MISSING_RATE

        return feat

    def _empty_result(self, uma_race_df: pd.DataFrame) -> pd.DataFrame:
        idx = uma_race_df["kettonum"].tolist() if "kettonum" in uma_race_df.columns else []
        return pd.DataFrame(index=idx, columns=self._FEATURES, dtype=object)
