"""オッズ・人気 特徴量（カテゴリ15）."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.features.base import FeatureExtractor
from src.db import query_df
from src.config import MISSING_NUMERIC


class OddsFeatureExtractor(FeatureExtractor):
    """オッズ・人気の特徴量を抽出する."""

    _FEATURES: list[str] = [
        "odds_tan",
        "odds_ninki",
        "odds_log",
        "odds_fuku_low",
        "odds_fuku_high",
        "odds_is_favorite",
        "odds_is_top3_ninki",
    ]

    @property
    def feature_names(self) -> list[str]:
        return self._FEATURES

    def extract(
        self,
        race_key: dict[str, str],
        uma_race_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """オッズ特徴量を抽出する.

        n_odds_tanpuku から最新のオッズデータを取得する。
        datakubun: 4=確定, 3=最終, 2=前日売最終, 1=中間
        の優先順位で取得する。
        """
        odds_df = self._get_odds(race_key)

        # 出走馬のkettonumと馬番の対応
        horse_umaban = self._get_horse_umaban(race_key)

        results: list[dict[str, Any]] = []
        kettonums = (
            uma_race_df["kettonum"].tolist()
            if "kettonum" in uma_race_df.columns
            else []
        )

        for kn in kettonums:
            kn_str = str(kn).strip()
            feat: dict[str, Any] = {"kettonum": kn_str}

            # kettonum → umaban のマッピング
            umaban = horse_umaban.get(kn_str, "")

            if not odds_df.empty and umaban:
                horse_odds = odds_df[odds_df["umaban"].str.strip() == umaban]
                if not horse_odds.empty:
                    row = horse_odds.iloc[0]
                    tan_odds = self._parse_odds(str(row.get("tanodds", "")))
                    fuku_low = self._parse_odds(str(row.get("fukuoddslow", "")))
                    fuku_high = self._parse_odds(str(row.get("fukuoddshigh", "")))
                    ninki = self._safe_int(row.get("tanninki"), default=0)

                    feat["odds_tan"] = tan_odds
                    feat["odds_ninki"] = ninki
                    feat["odds_log"] = (
                        math.log(tan_odds) if tan_odds > 0 else MISSING_NUMERIC
                    )
                    feat["odds_fuku_low"] = fuku_low
                    feat["odds_fuku_high"] = fuku_high
                    feat["odds_is_favorite"] = 1 if ninki == 1 else 0
                    feat["odds_is_top3_ninki"] = 1 if 1 <= ninki <= 3 else 0
                else:
                    self._fill_missing(feat)
            else:
                self._fill_missing(feat)

            results.append(feat)

        return pd.DataFrame(results).set_index("kettonum")

    def _get_odds(self, race_key: dict[str, str]) -> pd.DataFrame:
        """n_odds_tanpuku からオッズを取得する.

        n_odds_tanpuku（明細テーブル）には DataKubun カラムがなく、
        レースキー + 馬番 で一意。EveryDB2側で最新データに上書きされるため、
        そのまま取得すれば最新のオッズが得られる。
        """
        sql = """
        SELECT umaban, tanodds, tanninki, fukuoddslow, fukuoddshigh
        FROM n_odds_tanpuku
        WHERE year = %(year)s AND monthday = %(monthday)s
          AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
          AND nichiji = %(nichiji)s AND racenum = %(racenum)s
        """
        return query_df(sql, race_key)

    def _get_horse_umaban(self, race_key: dict[str, str]) -> dict[str, str]:
        """kettonum → umaban のマッピングを取得する."""
        sql = """
        SELECT kettonum, umaban
        FROM n_uma_race
        WHERE year = %(year)s AND monthday = %(monthday)s
          AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
          AND nichiji = %(nichiji)s AND racenum = %(racenum)s
          AND datakubun IN ('1','2','3','4','5','6','7')
          AND ijyocd = '0'
        """
        df = query_df(sql, race_key)
        return {
            str(row["kettonum"]).strip(): str(row["umaban"]).strip()
            for _, row in df.iterrows()
        }

    def _parse_odds(self, odds_str: str) -> float:
        """オッズ文字列を数値に変換する.

        '9999' → 999.9 以上
        '0000' → 無投票
        """
        s = odds_str.strip()
        if not s or s == "0000":
            return MISSING_NUMERIC
        try:
            val = int(s) / 10.0
            return val
        except ValueError:
            return MISSING_NUMERIC

    def _fill_missing(self, feat: dict[str, Any]) -> None:
        """欠損値で埋める."""
        feat["odds_tan"] = MISSING_NUMERIC
        feat["odds_ninki"] = MISSING_NUMERIC
        feat["odds_log"] = MISSING_NUMERIC
        feat["odds_fuku_low"] = MISSING_NUMERIC
        feat["odds_fuku_high"] = MISSING_NUMERIC
        feat["odds_is_favorite"] = 0
        feat["odds_is_top3_ninki"] = 0

    def _empty_result(self, uma_race_df: pd.DataFrame) -> pd.DataFrame:
        idx = (
            uma_race_df["kettonum"].tolist()
            if "kettonum" in uma_race_df.columns
            else []
        )
        return pd.DataFrame(index=idx, columns=self._FEATURES, dtype=object)
