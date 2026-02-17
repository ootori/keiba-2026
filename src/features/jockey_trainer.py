"""騎手・調教師 特徴量（カテゴリ10, 11）."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.base import FeatureExtractor
from src.db import query_df
from src.config import MISSING_NUMERIC, MISSING_RATE


class JockeyTrainerFeatureExtractor(FeatureExtractor):
    """騎手・調教師の特徴量を抽出する."""

    _FEATURES: list[str] = [
        # カテゴリ10: 騎手
        "jockey_code",
        "jockey_win_rate_year",
        "jockey_fukusho_rate_year",
        "jockey_minarai",
        "jockey_win_rate_jyo",
        "jockey_same_horse_rate",
        "jockey_change_flag",
        "jockey_avg_ninki_diff",
        # カテゴリ11: 調教師
        "trainer_code",
        "trainer_win_rate_year",
        "trainer_fukusho_rate_year",
        "trainer_win_rate_jyo",
        "trainer_tozai",
        "trainer_jockey_combo_rate",
        "trainer_jockey_combo_runs",
    ]

    @property
    def feature_names(self) -> list[str]:
        return self._FEATURES

    def extract(
        self,
        race_key: dict[str, str],
        uma_race_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """騎手・調教師の特徴量を抽出する."""
        race_date = race_key["year"] + race_key["monthday"]
        race_year = race_key["year"]
        jyocd = race_key["jyocd"]

        # 出走馬情報
        horses = self._get_horse_info(race_key)
        if horses.empty:
            return self._empty_result(uma_race_df)

        # 騎手コード・調教師コードの一覧
        kisyu_codes = horses["kisyucode"].dropna().unique().tolist()
        chokyo_codes = horses["chokyosicode"].dropna().unique().tolist()
        kettonums = horses["kettonum"].tolist()

        # 騎手年間成績
        jockey_year_stats = self._get_jockey_year_stats(kisyu_codes, race_year)

        # 騎手の競馬場別成績（過去2年）
        jockey_jyo_stats = self._get_jockey_jyo_stats(
            kisyu_codes, jyocd, race_date
        )

        # 調教師年間成績
        trainer_year_stats = self._get_trainer_year_stats(chokyo_codes, race_year)

        # 調教師の競馬場別成績
        trainer_jyo_stats = self._get_trainer_jyo_stats(
            chokyo_codes, jyocd, race_date
        )

        # 騎手マスタ（見習区分）
        jockey_master = self._get_jockey_master(kisyu_codes)

        # 調教師マスタ（東西所属）
        trainer_master = self._get_trainer_master(chokyo_codes)

        # 馬ごとの前走騎手 + 同馬×同騎手成績 + 騎手の人気差
        past_info = self._get_past_jockey_info(kettonums, race_date)

        # 騎手×調教師コンビ成績
        combo_stats = self._get_jockey_trainer_combo(
            horses, race_date
        )

        results: list[dict[str, Any]] = []
        for _, h in horses.iterrows():
            kn = str(h["kettonum"]).strip()
            kc = str(h.get("kisyucode", "")).strip()
            cc = str(h.get("chokyosicode", "")).strip()

            feat: dict[str, Any] = {"kettonum": kn}

            # --- 騎手 ---
            feat["jockey_code"] = kc

            # 年間成績
            jy = jockey_year_stats.get(kc, {})
            feat["jockey_win_rate_year"] = jy.get("win_rate", MISSING_RATE)
            feat["jockey_fukusho_rate_year"] = jy.get("fukusho_rate", MISSING_RATE)

            # 見習
            jm = jockey_master.get(kc, {})
            feat["jockey_minarai"] = self._safe_int(jm.get("minaraicd"), default=0)

            # 競馬場別
            jj = jockey_jyo_stats.get(kc, {})
            feat["jockey_win_rate_jyo"] = jj.get("win_rate", MISSING_RATE)

            # 同馬騎乗成績
            pi = past_info.get(kn, {})
            feat["jockey_same_horse_rate"] = pi.get(
                "same_horse_rate", MISSING_RATE
            )
            feat["jockey_change_flag"] = pi.get("change_flag", 0)
            feat["jockey_avg_ninki_diff"] = pi.get(
                "avg_ninki_diff", MISSING_NUMERIC
            )

            # --- 調教師 ---
            feat["trainer_code"] = cc

            ty = trainer_year_stats.get(cc, {})
            feat["trainer_win_rate_year"] = ty.get("win_rate", MISSING_RATE)
            feat["trainer_fukusho_rate_year"] = ty.get("fukusho_rate", MISSING_RATE)

            tj = trainer_jyo_stats.get(cc, {})
            feat["trainer_win_rate_jyo"] = tj.get("win_rate", MISSING_RATE)

            tm = trainer_master.get(cc, {})
            feat["trainer_tozai"] = str(tm.get("tozaicd", "")).strip()

            # コンビ成績
            combo_key = f"{kc}_{cc}"
            cs = combo_stats.get(combo_key, {})
            feat["trainer_jockey_combo_rate"] = cs.get(
                "fukusho_rate", MISSING_RATE
            )
            feat["trainer_jockey_combo_runs"] = cs.get("runs", 0)

            results.append(feat)

        return pd.DataFrame(results).set_index("kettonum")

    # ------------------------------------------------------------------
    # データ取得
    # ------------------------------------------------------------------

    def _get_horse_info(self, race_key: dict[str, str]) -> pd.DataFrame:
        sql = """
        SELECT kettonum, umaban, kisyucode, chokyosicode
        FROM n_uma_race
        WHERE year = %(year)s AND monthday = %(monthday)s
          AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
          AND nichiji = %(nichiji)s AND racenum = %(racenum)s
          AND datakubun IN ('1','2','3','4','5','6','7')
          AND ijyocd = '0'
        """
        return query_df(sql, race_key)

    def _get_jockey_year_stats(
        self, codes: list[str], year: str
    ) -> dict[str, dict[str, float]]:
        """騎手の年間成績を取得する."""
        if not codes:
            return {}
        sql = """
        SELECT kisyucode,
            CAST(heichichakukaisu1 AS integer) AS w1,
            CAST(heichichakukaisu2 AS integer) AS w2,
            CAST(heichichakukaisu3 AS integer) AS w3,
            CAST(heichichakukaisu4 AS integer) AS w4,
            CAST(heichichakukaisu5 AS integer) AS w5,
            CAST(heichichakukaisu6 AS integer) AS w6
        FROM n_kisyu_seiseki
        WHERE kisyucode IN %(codes)s
          AND setyear = %(year)s
        """
        df = query_df(sql, {"codes": tuple(codes), "year": year})
        result: dict[str, dict[str, float]] = {}
        for _, row in df.iterrows():
            code = str(row["kisyucode"]).strip()
            total = sum(self._safe_int(row.get(f"w{i}")) for i in range(1, 7))
            total = max(total, 0)
            wins = self._safe_int(row.get("w1"), default=0)
            top3 = wins + self._safe_int(row.get("w2")) + self._safe_int(row.get("w3"))
            result[code] = {
                "win_rate": self._safe_rate(wins, total),
                "fukusho_rate": self._safe_rate(top3, total),
            }
        return result

    def _get_trainer_year_stats(
        self, codes: list[str], year: str
    ) -> dict[str, dict[str, float]]:
        """調教師の年間成績を取得する."""
        if not codes:
            return {}
        sql = """
        SELECT chokyosicode,
            CAST(heichichakukaisu1 AS integer) AS w1,
            CAST(heichichakukaisu2 AS integer) AS w2,
            CAST(heichichakukaisu3 AS integer) AS w3,
            CAST(heichichakukaisu4 AS integer) AS w4,
            CAST(heichichakukaisu5 AS integer) AS w5,
            CAST(heichichakukaisu6 AS integer) AS w6
        FROM n_chokyo_seiseki
        WHERE chokyosicode IN %(codes)s
          AND setyear = %(year)s
        """
        df = query_df(sql, {"codes": tuple(codes), "year": year})
        result: dict[str, dict[str, float]] = {}
        for _, row in df.iterrows():
            code = str(row["chokyosicode"]).strip()
            total = sum(self._safe_int(row.get(f"w{i}")) for i in range(1, 7))
            total = max(total, 0)
            wins = self._safe_int(row.get("w1"), default=0)
            top3 = wins + self._safe_int(row.get("w2")) + self._safe_int(row.get("w3"))
            result[code] = {
                "win_rate": self._safe_rate(wins, total),
                "fukusho_rate": self._safe_rate(top3, total),
            }
        return result

    def _get_jockey_jyo_stats(
        self, codes: list[str], jyocd: str, race_date: str
    ) -> dict[str, dict[str, float]]:
        """騎手の競馬場別成績（過去2年）を集計する."""
        if not codes:
            return {}
        try:
            year_start = str(int(race_date[:4]) - 2)
        except ValueError:
            return {}

        sql = """
        SELECT ur.kisyucode,
            COUNT(*) AS total,
            SUM(CASE WHEN CAST(ur.kakuteijyuni AS integer) = 1 THEN 1 ELSE 0 END) AS wins
        FROM n_uma_race ur
        WHERE ur.kisyucode IN %(codes)s
          AND ur.jyocd = %(jyocd)s
          AND ur.datakubun = '7'
          AND ur.ijyocd = '0'
          AND ur.year >= %(year_start)s
          AND (ur.year || ur.monthday) < %(race_date)s
        GROUP BY ur.kisyucode
        """
        df = query_df(sql, {
            "codes": tuple(codes),
            "jyocd": jyocd,
            "year_start": year_start,
            "race_date": race_date,
        })
        result: dict[str, dict[str, float]] = {}
        for _, row in df.iterrows():
            code = str(row["kisyucode"]).strip()
            result[code] = {
                "win_rate": self._safe_rate(
                    self._safe_int(row.get("wins")),
                    self._safe_int(row.get("total")),
                )
            }
        return result

    def _get_trainer_jyo_stats(
        self, codes: list[str], jyocd: str, race_date: str
    ) -> dict[str, dict[str, float]]:
        """調教師の競馬場別成績（過去2年）を集計する."""
        if not codes:
            return {}
        try:
            year_start = str(int(race_date[:4]) - 2)
        except ValueError:
            return {}

        sql = """
        SELECT ur.chokyosicode,
            COUNT(*) AS total,
            SUM(CASE WHEN CAST(ur.kakuteijyuni AS integer) = 1 THEN 1 ELSE 0 END) AS wins
        FROM n_uma_race ur
        WHERE ur.chokyosicode IN %(codes)s
          AND ur.jyocd = %(jyocd)s
          AND ur.datakubun = '7'
          AND ur.ijyocd = '0'
          AND ur.year >= %(year_start)s
          AND (ur.year || ur.monthday) < %(race_date)s
        GROUP BY ur.chokyosicode
        """
        df = query_df(sql, {
            "codes": tuple(codes),
            "jyocd": jyocd,
            "year_start": year_start,
            "race_date": race_date,
        })
        result: dict[str, dict[str, float]] = {}
        for _, row in df.iterrows():
            code = str(row["chokyosicode"]).strip()
            result[code] = {
                "win_rate": self._safe_rate(
                    self._safe_int(row.get("wins")),
                    self._safe_int(row.get("total")),
                )
            }
        return result

    def _get_jockey_master(
        self, codes: list[str]
    ) -> dict[str, dict[str, str]]:
        """騎手マスタ情報を取得する."""
        if not codes:
            return {}
        sql = """
        SELECT kisyucode, minaraicd
        FROM n_kisyu
        WHERE kisyucode IN %(codes)s
        """
        df = query_df(sql, {"codes": tuple(codes)})
        return {
            str(row["kisyucode"]).strip(): row.to_dict()
            for _, row in df.iterrows()
        }

    def _get_trainer_master(
        self, codes: list[str]
    ) -> dict[str, dict[str, str]]:
        """調教師マスタ情報を取得する."""
        if not codes:
            return {}
        sql = """
        SELECT chokyosicode, tozaicd
        FROM n_chokyo
        WHERE chokyosicode IN %(codes)s
        """
        df = query_df(sql, {"codes": tuple(codes)})
        return {
            str(row["chokyosicode"]).strip(): row.to_dict()
            for _, row in df.iterrows()
        }

    def _get_past_jockey_info(
        self,
        kettonums: list[str],
        race_date: str,
    ) -> dict[str, dict[str, Any]]:
        """馬ごとの前走騎手・同馬×同騎手成績・人気差を取得する."""
        if not kettonums:
            return {}
        sql = """
        SELECT kettonum, kisyucode, kakuteijyuni, ninki,
               year, monthday
        FROM n_uma_race
        WHERE kettonum IN %(kettonums)s
          AND datakubun = '7'
          AND ijyocd = '0'
          AND (year || monthday) < %(race_date)s
        ORDER BY kettonum, year DESC, monthday DESC
        """
        df = query_df(sql, {"kettonums": tuple(kettonums), "race_date": race_date})
        if df.empty:
            return {}

        result: dict[str, dict[str, Any]] = {}
        for kn in kettonums:
            kn_str = str(kn).strip()
            h_past = df[df["kettonum"] == kn_str]
            if h_past.empty:
                result[kn_str] = {}
                continue

            prev_kisyu = str(h_past.iloc[0].get("kisyucode", "")).strip()

            # 人気差（着順 - 人気）の平均
            ninki_diff_vals = []
            for _, row in h_past.head(20).iterrows():
                j = self._safe_int(row.get("kakuteijyuni"))
                n = self._safe_int(row.get("ninki"))
                if j > 0 and n > 0:
                    ninki_diff_vals.append(j - n)

            # 同馬×同騎手成績
            same_jockey = h_past[h_past["kisyucode"].str.strip() == prev_kisyu]
            same_total = len(same_jockey)
            same_top3 = sum(
                1 for _, r in same_jockey.iterrows()
                if self._safe_int(r.get("kakuteijyuni")) <= 3
            )

            result[kn_str] = {
                "change_flag": 0,  # 後でパイプラインで判定
                "prev_kisyu": prev_kisyu,
                "same_horse_rate": self._safe_rate(same_top3, same_total),
                "avg_ninki_diff": (
                    float(np.mean(ninki_diff_vals))
                    if ninki_diff_vals
                    else MISSING_NUMERIC
                ),
            }

        return result

    def _get_jockey_trainer_combo(
        self,
        horses: pd.DataFrame,
        race_date: str,
    ) -> dict[str, dict[str, Any]]:
        """騎手×調教師コンビの成績を取得する."""
        combos = set()
        for _, h in horses.iterrows():
            kc = str(h.get("kisyucode", "")).strip()
            cc = str(h.get("chokyosicode", "")).strip()
            if kc and cc:
                combos.add((kc, cc))

        if not combos:
            return {}

        try:
            year_start = str(int(race_date[:4]) - 2)
        except ValueError:
            return {}

        # 全騎手・調教師のペア成績を一括取得
        all_kisyu = tuple(set(k for k, _ in combos))
        all_chokyo = tuple(set(c for _, c in combos))

        sql = """
        SELECT kisyucode, chokyosicode,
            COUNT(*) AS total,
            SUM(CASE WHEN CAST(kakuteijyuni AS integer) <= 3 THEN 1 ELSE 0 END) AS top3
        FROM n_uma_race
        WHERE kisyucode IN %(kisyu_codes)s
          AND chokyosicode IN %(chokyo_codes)s
          AND datakubun = '7'
          AND ijyocd = '0'
          AND year >= %(year_start)s
          AND (year || monthday) < %(race_date)s
        GROUP BY kisyucode, chokyosicode
        """
        df = query_df(sql, {
            "kisyu_codes": all_kisyu,
            "chokyo_codes": all_chokyo,
            "year_start": year_start,
            "race_date": race_date,
        })

        result: dict[str, dict[str, Any]] = {}
        for _, row in df.iterrows():
            kc = str(row["kisyucode"]).strip()
            cc = str(row["chokyosicode"]).strip()
            key = f"{kc}_{cc}"
            total = self._safe_int(row.get("total"))
            top3 = self._safe_int(row.get("top3"))
            result[key] = {
                "fukusho_rate": self._safe_rate(top3, total),
                "runs": total,
            }
        return result

    def _empty_result(self, uma_race_df: pd.DataFrame) -> pd.DataFrame:
        idx = uma_race_df["kettonum"].tolist() if "kettonum" in uma_race_df.columns else []
        return pd.DataFrame(index=idx, columns=self._FEATURES, dtype=object)
