"""調教データ 特徴量（カテゴリ12）."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.features.base import FeatureExtractor
from src.db import query_df
from src.config import MISSING_NUMERIC
from src.utils.code_master import haron_time_to_sec


class TrainingFeatureExtractor(FeatureExtractor):
    """調教データの特徴量を抽出する."""

    _FEATURES: list[str] = [
        "training_hanro_time4",
        "training_hanro_time3",
        "training_hanro_lap1",
        "training_hanro_accel",
        "training_wc_time_best",
        "training_days_from_last",
        "training_count_2weeks",
    ]

    @property
    def feature_names(self) -> list[str]:
        return self._FEATURES

    def extract(
        self,
        race_key: dict[str, str],
        uma_race_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """調教データ特徴量を抽出する."""
        race_date = race_key["year"] + race_key["monthday"]
        kettonums = (
            uma_race_df["kettonum"].tolist()
            if "kettonum" in uma_race_df.columns
            else []
        )
        if not kettonums:
            return self._empty_result(uma_race_df)

        # 2週前の日付を計算
        try:
            race_dt = datetime.strptime(race_date[:8], "%Y%m%d")
            from datetime import timedelta
            two_weeks_ago = (race_dt - timedelta(days=14)).strftime("%Y%m%d")
        except ValueError:
            two_weeks_ago = race_date

        # 坂路の最終追切を一括取得
        hanro_df = self._get_last_hanro(kettonums, race_date)

        # ウッドチップの最速タイムを一括取得
        wc_df = self._get_best_woodchip(kettonums, two_weeks_ago, race_date)

        # 直近2週間の調教本数
        train_count = self._get_training_count(kettonums, two_weeks_ago, race_date)

        results: list[dict[str, Any]] = []
        for kn in kettonums:
            kn_str = str(kn).strip()
            feat: dict[str, Any] = {"kettonum": kn_str}

            # --- 坂路 ---
            if kn_str in hanro_df:
                hr = hanro_df[kn_str]
                t4 = haron_time_to_sec(str(hr.get("harontime4", "")).strip())
                t3 = haron_time_to_sec(str(hr.get("harontime3", "")).strip())
                lap1 = haron_time_to_sec(str(hr.get("laptime1", "")).strip())
                lap4 = haron_time_to_sec(str(hr.get("laptime4", "")).strip())

                feat["training_hanro_time4"] = t4 if t4 is not None else MISSING_NUMERIC
                feat["training_hanro_time3"] = t3 if t3 is not None else MISSING_NUMERIC
                feat["training_hanro_lap1"] = lap1 if lap1 is not None else MISSING_NUMERIC

                # 加速度 = LapTime4(最初) - LapTime1(最後)
                if lap4 is not None and lap1 is not None:
                    feat["training_hanro_accel"] = lap4 - lap1
                else:
                    feat["training_hanro_accel"] = MISSING_NUMERIC

                # 最終追切からの日数
                cdate = str(hr.get("chokyodate", "")).strip()
                try:
                    c_dt = datetime.strptime(cdate[:8], "%Y%m%d")
                    feat["training_days_from_last"] = (race_dt - c_dt).days
                except ValueError:
                    feat["training_days_from_last"] = MISSING_NUMERIC
            else:
                feat["training_hanro_time4"] = MISSING_NUMERIC
                feat["training_hanro_time3"] = MISSING_NUMERIC
                feat["training_hanro_lap1"] = MISSING_NUMERIC
                feat["training_hanro_accel"] = MISSING_NUMERIC
                feat["training_days_from_last"] = MISSING_NUMERIC

            # --- ウッドチップ ---
            feat["training_wc_time_best"] = wc_df.get(kn_str, MISSING_NUMERIC)

            # --- 調教本数 ---
            feat["training_count_2weeks"] = train_count.get(kn_str, 0)

            # training_days_from_last: ウッドチップからも更新
            if feat["training_days_from_last"] == MISSING_NUMERIC and kn_str in wc_df:
                # ウッドチップのデータがある場合はそちらの日付も考慮
                pass  # 最終追切日はhanroから取る方針

            results.append(feat)

        return pd.DataFrame(results).set_index("kettonum")

    # ------------------------------------------------------------------
    # データ取得
    # ------------------------------------------------------------------

    def _get_last_hanro(
        self,
        kettonums: list[str],
        race_date: str,
    ) -> dict[str, dict[str, Any]]:
        """坂路調教の最終追切を一括取得する."""
        if not kettonums:
            return {}
        sql = """
        SELECT DISTINCT ON (kettonum)
            kettonum, chokyodate,
            harontime4, harontime3,
            laptime4, laptime3, laptime2, laptime1
        FROM n_hanro
        WHERE kettonum IN %(kettonums)s
          AND chokyodate <= %(race_date)s
        ORDER BY kettonum, chokyodate DESC, chokyotime DESC
        """
        df = query_df(sql, {"kettonums": tuple(kettonums), "race_date": race_date})
        return {
            str(row["kettonum"]).strip(): row.to_dict()
            for _, row in df.iterrows()
        }

    def _get_best_woodchip(
        self,
        kettonums: list[str],
        date_from: str,
        date_to: str,
    ) -> dict[str, float]:
        """直近2週間のウッドチップ最速5Fタイムを取得する.

        ※ n_wood_chip テーブルにはharontime5がある想定。
        なければ harontime6 等で代替。
        """
        if not kettonums:
            return {}
        sql = """
        SELECT kettonum, MIN(harontime5) AS best_time
        FROM n_wood_chip
        WHERE kettonum IN %(kettonums)s
          AND chokyodate >= %(date_from)s
          AND chokyodate <= %(date_to)s
          AND LENGTH(TRIM(harontime5)) >= 3
        GROUP BY kettonum
        """
        try:
            df = query_df(sql, {
                "kettonums": tuple(kettonums),
                "date_from": date_from,
                "date_to": date_to,
            })
        except Exception:
            # harontime5 カラムがない場合のフォールバック
            return {}

        result: dict[str, float] = {}
        for _, row in df.iterrows():
            kn = str(row["kettonum"]).strip()
            t = haron_time_to_sec(str(row.get("best_time", "")).strip())
            if t is not None:
                result[kn] = t
        return result

    def _get_training_count(
        self,
        kettonums: list[str],
        date_from: str,
        date_to: str,
    ) -> dict[str, int]:
        """直近2週間の調教本数（坂路+ウッドチップ合計）を取得する."""
        if not kettonums:
            return {}

        # 坂路
        sql_hanro = """
        SELECT kettonum, COUNT(*) AS cnt
        FROM n_hanro
        WHERE kettonum IN %(kettonums)s
          AND chokyodate >= %(date_from)s
          AND chokyodate <= %(date_to)s
        GROUP BY kettonum
        """
        # ウッドチップ
        sql_wc = """
        SELECT kettonum, COUNT(*) AS cnt
        FROM n_wood_chip
        WHERE kettonum IN %(kettonums)s
          AND chokyodate >= %(date_from)s
          AND chokyodate <= %(date_to)s
        GROUP BY kettonum
        """
        params = {
            "kettonums": tuple(kettonums),
            "date_from": date_from,
            "date_to": date_to,
        }

        result: dict[str, int] = {}
        for sql in [sql_hanro, sql_wc]:
            try:
                df = query_df(sql, params)
                for _, row in df.iterrows():
                    kn = str(row["kettonum"]).strip()
                    result[kn] = result.get(kn, 0) + self._safe_int(row.get("cnt"), default=0)
            except Exception:
                pass

        return result

    def _empty_result(self, uma_race_df: pd.DataFrame) -> pd.DataFrame:
        idx = (
            uma_race_df["kettonum"].tolist()
            if "kettonum" in uma_race_df.columns
            else []
        )
        return pd.DataFrame(index=idx, columns=self._FEATURES, dtype=object)
