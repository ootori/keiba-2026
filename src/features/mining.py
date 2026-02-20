"""データマイニング予想 特徴量（カテゴリ18: JRA-VANマイニング）.

JRA-VANが提供するデータマイニング予想（DM予想）と
対戦型データマイニング予想（TM予想）を特徴量として抽出する。

データソース:
    - n_uma_race: DMTime, DMJyuni, DMGosaP, DMGosaM, DMKubun
    - n_mining: レース単位のDM予想（馬番×予想タイム/誤差）
    - n_taisengata_mining: 対戦型マイニングスコア（馬番×スコア）

注意:
    - DMKubun: 1=前日, 2=当日, 3=直前
    - オッズと同様にデータリーク防止の観点から、
      学習時は前日データ(DMKubun='1')を使用するのが安全
    - n_uma_raceのDMカラムは最新で上書きされる可能性があるため、
      n_miningテーブルからの取得も併用する
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.features.base import FeatureExtractor
from src.db import query_df
from src.config import MISSING_NUMERIC

logger = logging.getLogger(__name__)


class MiningFeatureExtractor(FeatureExtractor):
    """JRA-VANデータマイニング予想の特徴量を抽出する."""

    _FEATURES: list[str] = [
        "mining_dm_time",           # DM予想走破タイム（秒換算）
        "mining_dm_jyuni",          # DM予想順位
        "mining_dm_gosa_range",     # DM予想誤差幅（信頼度の指標: GosaP - GosaM）
        "mining_dm_gosa_p",         # DM予想誤差（プラス側）
        "mining_dm_gosa_m",         # DM予想誤差（マイナス側）
        "mining_dm_kubun",          # DM区分（1=前日,2=当日,3=直前）
        "mining_tm_score",          # 対戦型マイニングスコア（0.0〜100.0）
    ]

    @property
    def feature_names(self) -> list[str]:
        return self._FEATURES

    def extract(
        self,
        race_key: dict[str, str],
        uma_race_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """マイニング予想特徴量を抽出する.

        まず n_uma_race の DM カラムから取得を試み、
        欠損がある場合は n_mining テーブルで補完する。
        対戦型スコアは n_taisengata_mining から取得する。

        Args:
            race_key: レースキー辞書
            uma_race_df: 当該レースの出走馬情報 DataFrame

        Returns:
            kettonum をインデックスとする特徴量 DataFrame
        """
        kettonums = (
            uma_race_df["kettonum"].tolist()
            if "kettonum" in uma_race_df.columns
            else []
        )
        if not kettonums:
            return pd.DataFrame(columns=self._FEATURES)

        # kettonum → umaban マッピング
        horse_umaban = self._get_horse_umaban(race_key)

        # n_uma_race から DM データ取得
        dm_df = self._get_dm_from_uma_race(race_key)

        # n_mining から DM データ取得（補完用）
        mining_df = self._get_mining_data(race_key)

        # n_taisengata_mining から TM スコア取得
        tm_df = self._get_tm_data(race_key)

        results: list[dict[str, Any]] = []
        for kn in kettonums:
            kn_str = str(kn).strip()
            feat: dict[str, Any] = {"kettonum": kn_str}
            umaban = horse_umaban.get(kn_str, "")

            # --- DM予想（n_uma_race優先、n_miningで補完） ---
            dm_found = False
            if not dm_df.empty:
                horse_dm = dm_df[dm_df["kettonum"].str.strip() == kn_str]
                if not horse_dm.empty:
                    row = horse_dm.iloc[0]
                    dm_time = self._parse_dm_time(
                        str(row.get("dmtime", "")).strip()
                    )
                    dm_jyuni = self._safe_int(
                        row.get("dmjyuni"), default=-1
                    )
                    dm_gosa_p = self._parse_dm_time(
                        str(row.get("dmgosap", "")).strip()
                    )
                    dm_gosa_m = self._parse_dm_time(
                        str(row.get("dmgosam", "")).strip()
                    )
                    dm_kubun = self._safe_int(
                        row.get("dmkubun"), default=-1
                    )

                    if dm_time > 0:
                        feat["mining_dm_time"] = dm_time
                        feat["mining_dm_jyuni"] = (
                            dm_jyuni if dm_jyuni > 0 else MISSING_NUMERIC
                        )
                        feat["mining_dm_gosa_p"] = (
                            dm_gosa_p if dm_gosa_p >= 0 else MISSING_NUMERIC
                        )
                        feat["mining_dm_gosa_m"] = (
                            dm_gosa_m if dm_gosa_m >= 0 else MISSING_NUMERIC
                        )
                        feat["mining_dm_gosa_range"] = (
                            dm_gosa_p + dm_gosa_m
                            if dm_gosa_p >= 0 and dm_gosa_m >= 0
                            else MISSING_NUMERIC
                        )
                        feat["mining_dm_kubun"] = (
                            dm_kubun if dm_kubun > 0 else MISSING_NUMERIC
                        )
                        dm_found = True

            # n_mining テーブルで補完
            if not dm_found and not mining_df.empty and umaban:
                umaban_padded = umaban.zfill(2)
                horse_mining = mining_df[
                    mining_df["umaban"].str.strip().str.zfill(2) == umaban_padded
                ]
                if not horse_mining.empty:
                    row = horse_mining.iloc[0]
                    dm_time = self._parse_dm_time(
                        str(row.get("dmtime", "")).strip()
                    )
                    dm_gosa_p = self._parse_dm_time(
                        str(row.get("dmgosap", "")).strip()
                    )
                    dm_gosa_m = self._parse_dm_time(
                        str(row.get("dmgosam", "")).strip()
                    )

                    if dm_time > 0:
                        feat["mining_dm_time"] = dm_time
                        feat["mining_dm_jyuni"] = MISSING_NUMERIC  # n_miningに順位なし
                        feat["mining_dm_gosa_p"] = (
                            dm_gosa_p if dm_gosa_p >= 0 else MISSING_NUMERIC
                        )
                        feat["mining_dm_gosa_m"] = (
                            dm_gosa_m if dm_gosa_m >= 0 else MISSING_NUMERIC
                        )
                        feat["mining_dm_gosa_range"] = (
                            dm_gosa_p + dm_gosa_m
                            if dm_gosa_p >= 0 and dm_gosa_m >= 0
                            else MISSING_NUMERIC
                        )
                        feat["mining_dm_kubun"] = MISSING_NUMERIC
                        dm_found = True

            if not dm_found:
                feat["mining_dm_time"] = MISSING_NUMERIC
                feat["mining_dm_jyuni"] = MISSING_NUMERIC
                feat["mining_dm_gosa_p"] = MISSING_NUMERIC
                feat["mining_dm_gosa_m"] = MISSING_NUMERIC
                feat["mining_dm_gosa_range"] = MISSING_NUMERIC
                feat["mining_dm_kubun"] = MISSING_NUMERIC

            # --- 対戦型マイニングスコア ---
            if not tm_df.empty and umaban:
                umaban_padded = umaban.zfill(2)
                horse_tm = tm_df[
                    tm_df["umaban"].str.strip().str.zfill(2) == umaban_padded
                ]
                if not horse_tm.empty:
                    tm_score = self._safe_float(
                        horse_tm.iloc[0].get("tmscore"), default=-1.0
                    )
                    feat["mining_tm_score"] = (
                        tm_score if tm_score >= 0 else MISSING_NUMERIC
                    )
                else:
                    feat["mining_tm_score"] = MISSING_NUMERIC
            else:
                feat["mining_tm_score"] = MISSING_NUMERIC

            results.append(feat)

        return pd.DataFrame(results).set_index("kettonum")

    # ------------------------------------------------------------------
    # DB取得メソッド
    # ------------------------------------------------------------------

    def _get_dm_from_uma_race(
        self, race_key: dict[str, str],
    ) -> pd.DataFrame:
        """n_uma_race から DM 予想データを取得する."""
        sql = """
        SELECT kettonum, dmtime, dmjyuni, dmgosap, dmgosam, dmkubun
        FROM n_uma_race
        WHERE year = %(year)s AND monthday = %(monthday)s
          AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
          AND nichiji = %(nichiji)s AND racenum = %(racenum)s
          AND datakubun IN ('1','2','3','4','5','6','7')
          AND ijyocd = '0'
        """
        try:
            return query_df(sql, race_key)
        except Exception as e:
            logger.debug("n_uma_race DM取得エラー: %s", e)
            return pd.DataFrame()

    def _get_mining_data(
        self, race_key: dict[str, str],
    ) -> pd.DataFrame:
        """n_mining テーブルから DM 予想データを取得する.

        n_mining は馬番1〜9の情報が横持ちで格納されているため、
        縦持ちに変換して返す。テーブルが存在しない場合は空DataFrameを返す。
        """
        # まずテーブルの存在を確認
        try:
            check_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'n_mining'
            LIMIT 1
            """
            check_df = query_df(check_sql)
            if check_df.empty:
                return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

        # n_mining のカラム構造を取得して動的に対応
        try:
            schema_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'n_mining'
            ORDER BY ordinal_position
            """
            schema_df = query_df(schema_sql)
            col_names = schema_df["column_name"].tolist()

            # umaban/dmtime/dmgosap/dmgosam のパターンを検出
            umaban_cols = [c for c in col_names if "umaban" in c.lower()]
            dmtime_cols = [c for c in col_names if "dmtime" in c.lower()]
            dmgosap_cols = [c for c in col_names if "dmgosap" in c.lower()]
            dmgosam_cols = [c for c in col_names if "dmgosam" in c.lower()]

            if not umaban_cols or not dmtime_cols:
                logger.debug("n_mining: 期待するカラムが見つかりません")
                return pd.DataFrame()

            # 横持ち→縦持ち変換SQLを動的構築
            union_parts = []
            for i in range(min(len(umaban_cols), len(dmtime_cols))):
                ub = umaban_cols[i]
                dt = dmtime_cols[i] if i < len(dmtime_cols) else "NULL"
                gp = dmgosap_cols[i] if i < len(dmgosap_cols) else "NULL"
                gm = dmgosam_cols[i] if i < len(dmgosam_cols) else "NULL"
                part = f"SELECT {ub} AS umaban, {dt} AS dmtime, {gp} AS dmgosap, {gm} AS dmgosam"
                part += " FROM n_mining"
                part += " WHERE year = %(year)s AND monthday = %(monthday)s"
                part += " AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s"
                part += " AND nichiji = %(nichiji)s AND racenum = %(racenum)s"
                part += f" AND {ub} IS NOT NULL AND TRIM({ub}) != ''"
                union_parts.append(part)

            if not union_parts:
                return pd.DataFrame()

            sql = " UNION ALL ".join(union_parts)
            return query_df(sql, race_key)

        except Exception as e:
            logger.debug("n_mining 取得エラー: %s", e)
            return pd.DataFrame()

    def _get_tm_data(
        self, race_key: dict[str, str],
    ) -> pd.DataFrame:
        """n_taisengata_mining テーブルから対戦型スコアを取得する.

        n_taisengata_mining は馬番1〜18のスコアが横持ちのため、
        縦持ちに変換して返す。テーブルが存在しない場合は空DataFrameを返す。
        """
        try:
            check_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'n_taisengata_mining'
            LIMIT 1
            """
            check_df = query_df(check_sql)
            if check_df.empty:
                return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

        try:
            schema_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'n_taisengata_mining'
            ORDER BY ordinal_position
            """
            schema_df = query_df(schema_sql)
            col_names = schema_df["column_name"].tolist()

            umaban_cols = [c for c in col_names if "umaban" in c.lower()]
            tmscore_cols = [c for c in col_names if "tmscore" in c.lower()]

            if not umaban_cols or not tmscore_cols:
                logger.debug("n_taisengata_mining: 期待するカラムが見つかりません")
                return pd.DataFrame()

            union_parts = []
            for i in range(min(len(umaban_cols), len(tmscore_cols))):
                ub = umaban_cols[i]
                ts = tmscore_cols[i]
                part = f"SELECT {ub} AS umaban, {ts} AS tmscore"
                part += " FROM n_taisengata_mining"
                part += " WHERE year = %(year)s AND monthday = %(monthday)s"
                part += " AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s"
                part += " AND nichiji = %(nichiji)s AND racenum = %(racenum)s"
                part += f" AND {ub} IS NOT NULL AND TRIM({ub}) != ''"
                union_parts.append(part)

            if not union_parts:
                return pd.DataFrame()

            sql = " UNION ALL ".join(union_parts)
            return query_df(sql, race_key)

        except Exception as e:
            logger.debug("n_taisengata_mining 取得エラー: %s", e)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

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

    def _parse_dm_time(self, val: str) -> float:
        """DM予想タイム文字列を秒に変換する.

        EveryDB2のDMTimeは「分秒ミリ秒」形式（例: "1234" → 1分23秒4）。
        実際の格納形式はDB依存のため、複数パターンに対応する。

        Args:
            val: タイム文字列

        Returns:
            秒数（float）。変換不能なら -1.0
        """
        if not val or val.strip() == "" or val.strip() == "0":
            return -1.0
        s = val.strip()
        try:
            # 整数値の場合: 1/10秒単位の走破タイムとして解釈
            # 例: "1234" → 123.4秒
            num = int(s)
            if num <= 0:
                return -1.0
            return num / 10.0
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            return -1.0
