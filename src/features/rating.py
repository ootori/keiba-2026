"""Glickoレーティング特徴量（サプリメント）.

n_uma_race_rating（馬別レーティング）と n_kisyu_race_rating（騎手別レーティング）から
レーティング特徴量を抽出する。

特徴量一覧:
    馬レーティング（8個）:
        - rating_horse_all:           馬の総合Glickoレーティング
        - rating_horse_all_rd:        総合RD（偏差）
        - rating_horse_surface:       芝/ダート別レーティング（フォールバック: all）
        - rating_horse_surface_rd:    サーフェス別RD
        - rating_horse_best:          max(all, surface)
        - rating_horse_surface_exists: サーフェス別レーティングの有無（1/0）
        - rating_horse_races_rated:   レーティング済みレース数
        - rating_horse_rd_ratio:      surface_rd / all_rd

    騎手レーティング（6個）:
        - rating_jockey_all:          騎手の総合レーティング
        - rating_jockey_all_rd:       騎手の総合RD
        - rating_jockey_surface:      騎手の芝/ダート別レーティング
        - rating_jockey_surface_rd:   騎手のサーフェス別RD
        - rating_jockey_jyo:          騎手の当該競馬場レーティング
        - rating_jockey_jyo_rd:       騎手の当該競馬場RD

    合算（2個）:
        - rating_combined:            horse_surface + jockey_surface
        - rating_combined_rd:         sqrt(horse_surface_rd^2 + jockey_surface_rd^2)

    レース内相対 馬単体（6個）:
        - rating_horse_diff_top1〜top5: 自馬 - レース内N位のsurface_rating
        - rating_horse_diff_strongest2: all_rating 1位と2位の差（全馬共通値）

    レース内相対 合算（6個）:
        - rating_combined_diff_top1〜top5: 自馬 - レース内N位のcombined
        - rating_combined_diff_strongest2: combined 1位と2位の差（全馬共通値）

データリーク防止:
    - 馬レーティング: 当該レースより前のレースのレーティングのみ使用
    - 騎手レーティング: 当該レース日より前の日のレーティングのみ使用
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from src.features.base import FeatureExtractor
from src.db import query_df
from src.utils.code_master import track_type

logger = logging.getLogger(__name__)

_NAN = float("nan")


class RatingFeatureExtractor(FeatureExtractor):
    """Glickoレーティング特徴量を抽出する（サプリメント）."""

    _FEATURES: list[str] = [
        # 馬レーティング
        "rating_horse_all",
        "rating_horse_all_rd",
        "rating_horse_surface",
        "rating_horse_surface_rd",
        "rating_horse_best",
        "rating_horse_surface_exists",
        "rating_horse_races_rated",
        "rating_horse_rd_ratio",
        # 騎手レーティング
        "rating_jockey_all",
        "rating_jockey_all_rd",
        "rating_jockey_surface",
        "rating_jockey_surface_rd",
        "rating_jockey_jyo",
        "rating_jockey_jyo_rd",
        # 合算
        "rating_combined",
        "rating_combined_rd",
        # レース内相対（馬単体）
        "rating_horse_diff_top1",
        "rating_horse_diff_top2",
        "rating_horse_diff_top3",
        "rating_horse_diff_top4",
        "rating_horse_diff_top5",
        "rating_horse_diff_strongest2",
        # レース内相対（合算）
        "rating_combined_diff_top1",
        "rating_combined_diff_top2",
        "rating_combined_diff_top3",
        "rating_combined_diff_top4",
        "rating_combined_diff_top5",
        "rating_combined_diff_strongest2",
    ]

    @property
    def feature_names(self) -> list[str]:
        return self._FEATURES

    def extract(
        self,
        race_key: dict[str, str],
        uma_race_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Glickoレーティング特徴量を抽出する."""
        kettonums = (
            uma_race_df["kettonum"].tolist()
            if "kettonum" in uma_race_df.columns
            else []
        )
        if not kettonums:
            return pd.DataFrame(columns=self._FEATURES)

        race_date = race_key["year"] + race_key["monthday"]

        # --- レース情報取得（トラック種別） ---
        current_track_type = self._get_track_type(race_key)
        current_jyocd = race_key.get("jyocd", "")

        # --- 出走馬の騎手コード取得 ---
        horse_info = self._get_horse_jockey_info(race_key, kettonums)

        kisyu_codes: list[str] = []
        kn_to_kisyu: dict[str, str] = {}
        for kn_str, info in horse_info.items():
            kc = info.get("kisyucode", "")
            if kc:
                kisyu_codes.append(kc)
                kn_to_kisyu[kn_str] = kc

        # --- 馬レーティング取得 ---
        horse_ratings = self._get_horse_ratings(kettonums, race_date)

        # --- 馬レーティング数取得 ---
        horse_rated_counts = self._get_horse_rating_counts(
            kettonums, race_date,
        )

        # --- 騎手レーティング取得 ---
        jockey_ratings = (
            self._get_jockey_ratings(
                list(set(kisyu_codes)), race_date,
            )
            if kisyu_codes
            else {}
        )

        # --- 馬ごとに特徴量組み立て ---
        results: list[dict[str, Any]] = []
        for kn in kettonums:
            kn_str = str(kn).strip()
            feat: dict[str, Any] = {"kettonum": kn_str}

            hr = horse_ratings.get(kn_str, {})
            kc = kn_to_kisyu.get(kn_str, "")
            jr = jockey_ratings.get(kc, {}) if kc else {}

            # 馬レーティング
            h_all = self._safe_rating(hr.get("all_rating"))
            h_all_rd = self._safe_rating(hr.get("all_rd"))
            h_surface, h_surface_rd, h_surface_exists = (
                self._select_surface_rating(hr, current_track_type)
            )

            feat["rating_horse_all"] = h_all
            feat["rating_horse_all_rd"] = h_all_rd
            feat["rating_horse_surface"] = h_surface
            feat["rating_horse_surface_rd"] = h_surface_rd
            feat["rating_horse_surface_exists"] = h_surface_exists

            # best = max(all, surface)
            if not _is_nan(h_all) and not _is_nan(h_surface):
                feat["rating_horse_best"] = max(h_all, h_surface)
            elif not _is_nan(h_surface):
                feat["rating_horse_best"] = h_surface
            elif not _is_nan(h_all):
                feat["rating_horse_best"] = h_all
            else:
                feat["rating_horse_best"] = _NAN

            # races_rated
            feat["rating_horse_races_rated"] = horse_rated_counts.get(
                kn_str, 0,
            )

            # rd_ratio
            if (
                not _is_nan(h_surface_rd)
                and not _is_nan(h_all_rd)
                and h_all_rd > 0
            ):
                feat["rating_horse_rd_ratio"] = h_surface_rd / h_all_rd
            else:
                feat["rating_horse_rd_ratio"] = _NAN

            # 騎手レーティング
            j_all = self._safe_rating(jr.get("all_rating"))
            j_all_rd = self._safe_rating(jr.get("all_rd"))
            j_surface, j_surface_rd, _ = self._select_surface_rating(
                jr, current_track_type,
            )
            j_jyo, j_jyo_rd = self._select_jyo_rating(
                jr, current_jyocd,
            )

            feat["rating_jockey_all"] = j_all
            feat["rating_jockey_all_rd"] = j_all_rd
            feat["rating_jockey_surface"] = j_surface
            feat["rating_jockey_surface_rd"] = j_surface_rd
            feat["rating_jockey_jyo"] = j_jyo
            feat["rating_jockey_jyo_rd"] = j_jyo_rd

            # 合算
            if not _is_nan(h_surface) and not _is_nan(j_surface):
                feat["rating_combined"] = h_surface + j_surface
            elif not _is_nan(h_surface):
                feat["rating_combined"] = h_surface
            else:
                feat["rating_combined"] = _NAN

            if not _is_nan(h_surface_rd) and not _is_nan(j_surface_rd):
                feat["rating_combined_rd"] = math.sqrt(
                    h_surface_rd ** 2 + j_surface_rd ** 2,
                )
            elif not _is_nan(h_surface_rd):
                feat["rating_combined_rd"] = h_surface_rd
            else:
                feat["rating_combined_rd"] = _NAN

            results.append(feat)

        if not results:
            return pd.DataFrame(columns=self._FEATURES)

        df = pd.DataFrame(results).set_index("kettonum")

        # --- レース内相対特徴量（馬単体ベース） ---
        self._add_race_relative_features(
            df,
            rating_col="rating_horse_surface",
            all_rating_col="rating_horse_all",
            prefix="rating_horse",
        )

        # --- レース内相対特徴量（合算ベース） ---
        self._add_race_relative_features(
            df,
            rating_col="rating_combined",
            all_rating_col="rating_combined",
            prefix="rating_combined",
        )

        return df

    # ------------------------------------------------------------------
    # レース内相対特徴量
    # ------------------------------------------------------------------

    @staticmethod
    def _add_race_relative_features(
        df: pd.DataFrame,
        rating_col: str,
        all_rating_col: str,
        prefix: str,
    ) -> None:
        """レース内の相対的なレーティング差を計算してdfに追加する.

        Args:
            df: 馬ごとの特徴量DataFrame（インプレース変更）
            rating_col: diff_top1〜top5 の計算に使うカラム名
            all_rating_col: strongest2 の計算に使うカラム名
            prefix: 出力カラム名のプレフィクス
        """
        # diff_top1〜top5
        if rating_col in df.columns:
            valid_ratings = df[rating_col].dropna().sort_values(
                ascending=False,
            )
            for n in range(1, 6):
                col_name = f"{prefix}_diff_top{n}"
                if len(valid_ratings) >= n:
                    top_n_val = valid_ratings.iloc[n - 1]
                    df[col_name] = df[rating_col] - top_n_val
                else:
                    df[col_name] = _NAN
        else:
            for n in range(1, 6):
                df[f"{prefix}_diff_top{n}"] = _NAN

        # strongest2: 1位と2位の差（全馬共通値）
        col_name_s2 = f"{prefix}_diff_strongest2"
        if all_rating_col in df.columns:
            valid_all = df[all_rating_col].dropna().sort_values(
                ascending=False,
            )
            if len(valid_all) >= 2:
                df[col_name_s2] = valid_all.iloc[0] - valid_all.iloc[1]
            else:
                df[col_name_s2] = _NAN
        else:
            df[col_name_s2] = _NAN

    # ------------------------------------------------------------------
    # サーフェス / 競馬場 レーティング選択
    # ------------------------------------------------------------------

    def _select_surface_rating(
        self,
        rating_row: dict[str, Any],
        current_track_type: str,
    ) -> tuple[float, float, int]:
        """サーフェスに合わせたレーティングを選択する.

        Args:
            rating_row: レーティング辞書
            current_track_type: "turf" or "dirt"

        Returns:
            (rating, rd, surface_exists)
        """
        if current_track_type == "turf":
            rating = self._safe_rating(rating_row.get("shiba_rating"))
            rd = self._safe_rating(rating_row.get("shiba_rd"))
        elif current_track_type == "dirt":
            rating = self._safe_rating(rating_row.get("dirt_rating"))
            rd = self._safe_rating(rating_row.get("dirt_rd"))
        else:
            rating = _NAN
            rd = _NAN

        if _is_nan(rating):
            # フォールバック: all_rating
            rating = self._safe_rating(rating_row.get("all_rating"))
            rd = self._safe_rating(rating_row.get("all_rd"))
            surface_exists = 0
        else:
            surface_exists = 1

        return rating, rd, surface_exists

    @staticmethod
    def _select_jyo_rating(
        jockey_rating: dict[str, Any],
        jyocd: str,
    ) -> tuple[float, float]:
        """騎手の競馬場別レーティングを選択する.

        Args:
            jockey_rating: 騎手レーティング辞書
            jyocd: 競馬場コード（"01"〜"10"）

        Returns:
            (rating, rd)
        """
        jyocd_stripped = str(jyocd).strip()
        if not jyocd_stripped:
            return _NAN, _NAN

        rating_key = f"jyocd_{jyocd_stripped}_rating"
        rd_key = f"jyocd_{jyocd_stripped}_rd"

        rating = jockey_rating.get(rating_key)
        rd = jockey_rating.get(rd_key)

        if rating is None or (isinstance(rating, float) and math.isnan(rating)):
            return _NAN, _NAN
        if rd is None or (isinstance(rd, float) and math.isnan(rd)):
            return float(rating), _NAN

        return float(rating), float(rd)

    # ------------------------------------------------------------------
    # データ取得
    # ------------------------------------------------------------------

    def _get_track_type(self, race_key: dict[str, str]) -> str:
        """レースのトラック種別を取得する."""
        sql = """
        SELECT trackcd
        FROM n_race
        WHERE year = %(year)s AND monthday = %(monthday)s
          AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
          AND nichiji = %(nichiji)s AND racenum = %(racenum)s
        LIMIT 1
        """
        try:
            df = query_df(sql, race_key)
        except Exception as e:
            logger.warning("トラック種別取得エラー: %s", e)
            return ""
        if df.empty:
            return ""
        trackcd = str(df.iloc[0]["trackcd"]).strip()
        return track_type(trackcd)

    def _get_horse_jockey_info(
        self,
        race_key: dict[str, str],
        kettonums: list[str],
    ) -> dict[str, dict[str, str]]:
        """出走馬のkisyucodeを取得する."""
        sql = """
        SELECT kettonum, umaban, kisyucode
        FROM n_uma_race
        WHERE year = %(year)s AND monthday = %(monthday)s
          AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
          AND nichiji = %(nichiji)s AND racenum = %(racenum)s
          AND kettonum IN %(kettonums)s
          AND datakubun IN ('1','2','3','4','5','6','7')
          AND ijyocd = '0'
        """
        params = dict(race_key)
        params["kettonums"] = tuple(kettonums)
        try:
            df = query_df(sql, params)
        except Exception as e:
            logger.warning("馬情報取得エラー: %s", e)
            return {}

        result: dict[str, dict[str, str]] = {}
        for _, row in df.iterrows():
            kn = str(row["kettonum"]).strip()
            result[kn] = {
                "umaban": str(row.get("umaban", "")).strip(),
                "kisyucode": str(row.get("kisyucode", "")).strip(),
            }
        return result

    def _get_horse_ratings(
        self,
        kettonums: list[str],
        race_date: str,
    ) -> dict[str, dict[str, float]]:
        """馬のレーティングを取得する（直前レース時点）.

        Args:
            kettonums: 血統登録番号リスト
            race_date: レース日付（YYYYMMDD）

        Returns:
            kettonum → {all_rating, all_rd, shiba_rating, ...} の辞書
        """
        if not kettonums:
            return {}

        sql = """
        SELECT DISTINCT ON (rr.kettonum)
            rr.kettonum,
            rr.all_rating,
            rr.all_rd,
            rr.shiba_rating,
            rr.shiba_rd,
            rr.dirt_rating,
            rr.dirt_rd
        FROM n_uma_race_rating rr
        WHERE rr.kettonum IN %(kettonums)s
          AND (rr.year || rr.monthday) < %(race_date)s
        ORDER BY rr.kettonum, rr.year DESC, rr.monthday DESC
        """
        try:
            df = query_df(sql, {
                "kettonums": tuple(kettonums),
                "race_date": race_date,
            })
        except Exception as e:
            logger.warning("馬レーティング取得エラー: %s", e)
            return {}

        result: dict[str, dict[str, float]] = {}
        for _, row in df.iterrows():
            kn = str(row["kettonum"]).strip()
            result[kn] = {
                "all_rating": self._safe_rating(row.get("all_rating")),
                "all_rd": self._safe_rating(row.get("all_rd")),
                "shiba_rating": self._safe_rating(row.get("shiba_rating")),
                "shiba_rd": self._safe_rating(row.get("shiba_rd")),
                "dirt_rating": self._safe_rating(row.get("dirt_rating")),
                "dirt_rd": self._safe_rating(row.get("dirt_rd")),
            }
        return result

    def _get_horse_rating_counts(
        self,
        kettonums: list[str],
        race_date: str,
    ) -> dict[str, int]:
        """馬のレーティング済みレース数を取得する.

        Args:
            kettonums: 血統登録番号リスト
            race_date: レース日付（YYYYMMDD）

        Returns:
            kettonum → レーティング数の辞書
        """
        if not kettonums:
            return {}

        sql = """
        SELECT kettonum, COUNT(*) AS rated_count
        FROM n_uma_race_rating
        WHERE kettonum IN %(kettonums)s
          AND (year || monthday) < %(race_date)s
        GROUP BY kettonum
        """
        try:
            df = query_df(sql, {
                "kettonums": tuple(kettonums),
                "race_date": race_date,
            })
        except Exception as e:
            logger.warning("馬レーティング数取得エラー: %s", e)
            return {}

        result: dict[str, int] = {}
        for _, row in df.iterrows():
            kn = str(row["kettonum"]).strip()
            result[kn] = int(row["rated_count"])
        return result

    def _get_jockey_ratings(
        self,
        kisyu_codes: list[str],
        race_date: str,
    ) -> dict[str, dict[str, float]]:
        """騎手のレーティングを取得する（前日時点）.

        Args:
            kisyu_codes: 騎手コードリスト
            race_date: レース日付（YYYYMMDD）

        Returns:
            kisyucode → {all_rating, all_rd, shiba_rating, ..., jyocd_01_rating, ...}
        """
        if not kisyu_codes:
            return {}

        # jyocd_XX カラムを動的に列挙
        jyo_cols = []
        for i in range(1, 11):
            jyo_cd = f"{i:02d}"
            jyo_cols.append(f"rr.jyocd_{jyo_cd}_rating")
            jyo_cols.append(f"rr.jyocd_{jyo_cd}_rd")
        jyo_cols_str = ", ".join(jyo_cols)

        sql = f"""
        SELECT DISTINCT ON (rr.kisyucode)
            rr.kisyucode,
            rr.all_rating,
            rr.all_rd,
            rr.shiba_rating,
            rr.shiba_rd,
            rr.dirt_rating,
            rr.dirt_rd,
            {jyo_cols_str}
        FROM n_kisyu_race_rating rr
        WHERE rr.kisyucode IN %(kisyu_codes)s
          AND (rr.year || rr.monthday) < %(race_date)s
        ORDER BY rr.kisyucode, rr.year DESC, rr.monthday DESC
        """
        try:
            df = query_df(sql, {
                "kisyu_codes": tuple(kisyu_codes),
                "race_date": race_date,
            })
        except Exception as e:
            logger.warning("騎手レーティング取得エラー: %s", e)
            return {}

        result: dict[str, dict[str, float]] = {}
        for _, row in df.iterrows():
            kc = str(row["kisyucode"]).strip()
            entry: dict[str, float] = {
                "all_rating": self._safe_rating(row.get("all_rating")),
                "all_rd": self._safe_rating(row.get("all_rd")),
                "shiba_rating": self._safe_rating(row.get("shiba_rating")),
                "shiba_rd": self._safe_rating(row.get("shiba_rd")),
                "dirt_rating": self._safe_rating(row.get("dirt_rating")),
                "dirt_rd": self._safe_rating(row.get("dirt_rd")),
            }
            # 競馬場別
            for i in range(1, 11):
                jyo_cd = f"{i:02d}"
                entry[f"jyocd_{jyo_cd}_rating"] = self._safe_rating(
                    row.get(f"jyocd_{jyo_cd}_rating"),
                )
                entry[f"jyocd_{jyo_cd}_rd"] = self._safe_rating(
                    row.get(f"jyocd_{jyo_cd}_rd"),
                )
            result[kc] = entry
        return result

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_rating(val: Any) -> float:
        """レーティング値を安全にfloatに変換する.

        None や非数値は NaN を返す。
        """
        if val is None:
            return _NAN
        try:
            v = float(val)
            if math.isnan(v):
                return _NAN
            return v
        except (ValueError, TypeError):
            return _NAN


def _is_nan(val: float) -> bool:
    """NaN判定ヘルパー."""
    try:
        return math.isnan(val)
    except (TypeError, ValueError):
        return True
