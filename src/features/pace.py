"""レースラップ・ペース特徴量（カテゴリ18: 新規）.

n_raceテーブルのHaronTimeS3（前3F）、HaronTimeL3（後3F）、LapTime1〜20を使い、
過去走のレースペース情報を各馬の特徴量として構築する。

ハイペース→差し有利、スローペース→逃げ有利という競馬の基本原則を
特徴量として反映する。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.features.base import FeatureExtractor
from src.db import query_df
from src.config import MISSING_NUMERIC, MISSING_RATE, RACE_KEY_COLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ペースタイプ分類
# ---------------------------------------------------------------------------

PACE_HIGH = 1    # 前傾ラップ（ハイペース）
PACE_MIDDLE = 2  # 平均ペース
PACE_SLOW = 3    # 後傾ラップ（スローペース）
PACE_UNKNOWN = 0


def classify_pace(s3f: float, l3f: float) -> int:
    """前3F/後3F比でペースを分類する.

    Args:
        s3f: 前3Fタイム（秒）
        l3f: 後3Fタイム（秒）

    Returns:
        ペースタイプ（PACE_HIGH / PACE_MIDDLE / PACE_SLOW / PACE_UNKNOWN）
    """
    if s3f <= 0 or l3f <= 0:
        return PACE_UNKNOWN
    ratio = s3f / l3f
    if ratio < 0.97:   # 前傾ラップ → ハイペース
        return PACE_HIGH
    elif ratio > 1.03:  # 後傾ラップ → スローペース
        return PACE_SLOW
    else:
        return PACE_MIDDLE


# ---------------------------------------------------------------------------
# 脚質×ペース適性マトリクス
# ---------------------------------------------------------------------------
# 脚質: 1=逃げ, 2=先行, 3=差し, 4=追込
# ペース: PACE_HIGH=1, PACE_MIDDLE=2, PACE_SLOW=3

_STYLE_PACE_MATRIX: dict[tuple[str, int], float] = {
    # 逃げ馬
    ("1", PACE_SLOW): 1.0,     # 逃げ×スロー → 有利
    ("1", PACE_MIDDLE): 0.3,   # 逃げ×ミドル → やや有利
    ("1", PACE_HIGH): -1.0,    # 逃げ×ハイ → 不利
    # 先行馬
    ("2", PACE_SLOW): 0.5,     # 先行×スロー → やや有利
    ("2", PACE_MIDDLE): 0.2,   # 先行×ミドル → 中立寄り
    ("2", PACE_HIGH): -0.3,    # 先行×ハイ → やや不利
    # 差し馬
    ("3", PACE_SLOW): -0.5,    # 差し×スロー → やや不利
    ("3", PACE_MIDDLE): 0.0,   # 差し×ミドル → 中立
    ("3", PACE_HIGH): 0.5,     # 差し×ハイ → やや有利
    # 追込馬
    ("4", PACE_SLOW): -1.0,    # 追込×スロー → 不利
    ("4", PACE_MIDDLE): -0.3,  # 追込×ミドル → やや不利
    ("4", PACE_HIGH): 1.0,     # 追込×ハイ → 有利
}


def _haron3_to_sec(val: str | None) -> float:
    """3ハロンタイム文字列（3桁 '999'）を秒に変換する.

    Args:
        val: 3桁ハロンタイム文字列（例: '345' → 34.5秒）

    Returns:
        秒数。変換不可の場合は 0.0
    """
    if val is None:
        return 0.0
    t = str(val).strip()
    if len(t) < 3:
        return 0.0
    try:
        return int(t[:2]) + int(t[2]) * 0.1
    except (ValueError, IndexError):
        return 0.0


class PaceFeatureExtractor(FeatureExtractor):
    """レースペース特徴量を抽出する.

    各馬の過去走のレースペース情報と、その馬の脚質との相性を
    特徴量として構築する。

    特徴量一覧:
        pace_s3f_last:              前走レースの前3F（秒）
        pace_l3f_last:              前走レースの後3F（秒）
        pace_s3f_l3f_ratio_last:    前走の前3F/後3F比（>1ならスロー）
        pace_type_last:             前走ペースタイプ（1=H/2=M/3=S/0=不明）
        pace_horse_style_pace_match: 脚質×ペース適性スコア（直近5走平均）
        pace_avg_front_ratio:       直近5走でハイペースレースでの好走率
        pace_avg_slow_ratio:        直近5走でスローペースレースでの好走率
    """

    _FEATURES: list[str] = [
        "pace_s3f_last",
        "pace_l3f_last",
        "pace_s3f_l3f_ratio_last",
        "pace_type_last",
        "pace_horse_style_pace_match",
        "pace_avg_front_ratio",
        "pace_avg_slow_ratio",
    ]

    @property
    def feature_names(self) -> list[str]:
        return self._FEATURES

    def extract(
        self,
        race_key: dict[str, str],
        uma_race_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """ペース特徴量を抽出する."""
        kettonums = (
            uma_race_df["kettonum"].tolist()
            if "kettonum" in uma_race_df.columns
            else []
        )
        if not kettonums:
            return self._empty_result(uma_race_df)

        race_date = race_key["year"] + race_key["monthday"]

        # 過去走の成績 + そのレースのペース情報を一括取得
        past_df = self._get_past_with_pace(kettonums, race_date)

        results: list[dict[str, Any]] = []
        for kn in kettonums:
            kn_str = str(kn).strip()
            h_past = (
                past_df[past_df["kettonum"] == kn_str]
                if not past_df.empty
                else pd.DataFrame()
            )
            feat = self._build_features(kn_str, h_past)
            results.append(feat)

        df = pd.DataFrame(results).set_index("kettonum")
        return df

    # ------------------------------------------------------------------
    # データ取得
    # ------------------------------------------------------------------

    def _get_past_with_pace(
        self,
        kettonums: list[str],
        race_date: str,
    ) -> pd.DataFrame:
        """過去成績＋レースペース情報を一括取得する.

        n_uma_race と n_race を結合し、各過去走のレースペース
        （前3F・後3F）と馬の脚質・着順を取得する。
        """
        if not kettonums:
            return pd.DataFrame()

        sql = """
        SELECT
            ur.kettonum,
            ur.kakuteijyuni,
            ur.kyakusitukubun,
            ur.year, ur.monthday, ur.jyocd, ur.kaiji, ur.nichiji, ur.racenum,
            r.harontimes3,
            r.harontimel3
        FROM n_uma_race ur
        JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
        WHERE ur.kettonum IN %(kettonums)s
          AND ur.datakubun = '7'
          AND ur.ijyocd = '0'
          AND r.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
          AND (ur.year || ur.monthday) < %(race_date)s
        ORDER BY ur.kettonum, ur.year DESC, ur.monthday DESC
        """
        return query_df(
            sql, {"kettonums": tuple(kettonums), "race_date": race_date}
        )

    # ------------------------------------------------------------------
    # 特徴量構築
    # ------------------------------------------------------------------

    def _build_features(
        self,
        kettonum: str,
        h_past: pd.DataFrame,
    ) -> dict[str, Any]:
        """1頭分のペース特徴量を構築する."""
        feat: dict[str, Any] = {"kettonum": kettonum}

        if h_past.empty:
            for f in self._FEATURES:
                if f == "pace_type_last":
                    feat[f] = PACE_UNKNOWN
                elif f in ("pace_avg_front_ratio", "pace_avg_slow_ratio"):
                    feat[f] = MISSING_RATE
                else:
                    feat[f] = MISSING_NUMERIC
            return feat

        # --- 各過去走のペース情報を計算 ---
        s3f_list: list[float] = []
        l3f_list: list[float] = []
        pace_type_list: list[int] = []
        style_list: list[str] = []
        jyuni_list: list[int] = []

        for _, row in h_past.head(5).iterrows():
            s3f = _haron3_to_sec(row.get("harontimes3"))
            l3f = _haron3_to_sec(row.get("harontimel3"))
            s3f_list.append(s3f)
            l3f_list.append(l3f)
            pace_type_list.append(classify_pace(s3f, l3f))
            style_list.append(str(row.get("kyakusitukubun", "0")).strip())
            jyuni = self._safe_int(row.get("kakuteijyuni"), default=99)
            jyuni_list.append(jyuni)

        # --- 前走ペース情報 ---
        s3f_last = s3f_list[0] if s3f_list else 0.0
        l3f_last = l3f_list[0] if l3f_list else 0.0

        feat["pace_s3f_last"] = s3f_last if s3f_last > 0 else MISSING_NUMERIC
        feat["pace_l3f_last"] = l3f_last if l3f_last > 0 else MISSING_NUMERIC

        # 前3F/後3F比
        if s3f_last > 0 and l3f_last > 0:
            feat["pace_s3f_l3f_ratio_last"] = s3f_last / l3f_last
        else:
            feat["pace_s3f_l3f_ratio_last"] = MISSING_NUMERIC

        # 前走ペースタイプ
        feat["pace_type_last"] = (
            pace_type_list[0] if pace_type_list else PACE_UNKNOWN
        )

        # --- 脚質×ペース適性スコア（直近5走の平均） ---
        match_scores: list[float] = []
        for i in range(len(pace_type_list)):
            pt = pace_type_list[i]
            st = style_list[i]
            if pt == PACE_UNKNOWN or st not in ("1", "2", "3", "4"):
                continue
            score = _STYLE_PACE_MATRIX.get((st, pt), 0.0)
            match_scores.append(score)

        if match_scores:
            feat["pace_horse_style_pace_match"] = float(np.mean(match_scores))
        else:
            feat["pace_horse_style_pace_match"] = MISSING_NUMERIC

        # --- ペース別好走率（直近5走） ---
        high_count = 0
        high_hit = 0
        slow_count = 0
        slow_hit = 0

        for i in range(len(pace_type_list)):
            pt = pace_type_list[i]
            jyuni = jyuni_list[i]
            if pt == PACE_HIGH:
                high_count += 1
                if jyuni <= 3:
                    high_hit += 1
            elif pt == PACE_SLOW:
                slow_count += 1
                if jyuni <= 3:
                    slow_hit += 1

        feat["pace_avg_front_ratio"] = (
            high_hit / high_count if high_count > 0 else MISSING_RATE
        )
        feat["pace_avg_slow_ratio"] = (
            slow_hit / slow_count if slow_count > 0 else MISSING_RATE
        )

        return feat

    def _empty_result(self, uma_race_df: pd.DataFrame) -> pd.DataFrame:
        idx = (
            uma_race_df["kettonum"].tolist()
            if "kettonum" in uma_race_df.columns
            else []
        )
        return pd.DataFrame(index=idx, columns=self._FEATURES, dtype=object)
