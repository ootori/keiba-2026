"""BMS条件別特徴量（bms_detail サプリメント）のテスト.

DB接続不要のユニットテスト。

実行方法:
    pytest tests/test_bms_detail_supplement.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# BMSDetailFeatureExtractor のユニットテスト
# ============================================================


class TestBMSDetailFeatureExtractor:
    """BMSDetailFeatureExtractor のテスト（DB不要）."""

    def test_feature_names(self) -> None:
        from src.features.bms_detail import BMSDetailFeatureExtractor

        ext = BMSDetailFeatureExtractor()
        names = ext.feature_names
        assert "blood_bms_dist_rate" in names
        assert "blood_bms_baba_rate" in names
        assert "blood_bms_jyo_rate" in names
        assert "blood_father_age_rate" in names
        assert "blood_nicks_track_rate" in names
        assert "blood_father_class_rate" in names
        assert len(names) == 6

    def test_extract_all_features(self) -> None:
        """全特徴量が正しく抽出されるテスト."""
        from src.features.bms_detail import BMSDetailFeatureExtractor

        ext = BMSDetailFeatureExtractor()
        race_key = {
            "year": "2024", "monthday": "0601",
            "jyocd": "05", "kaiji": "01",
            "nichiji": "01", "racenum": "01",
        }
        uma_race_df = pd.DataFrame({
            "kettonum": ["0000000001", "0000000002"],
            "umaban": ["1", "2"],
        })

        # モック: レース情報
        race_info = pd.DataFrame([{
            "kyori": "1600", "trackcd": "11",
            "sibababacd": "1", "dirtbabacd": "1",
            "gradecd": "A", "jyokencd5": "999",
        }])

        # モック: 馬齢
        barei_df = pd.DataFrame({
            "kettonum": ["0000000001", "0000000002"],
            "barei": ["3", "4"],
        })

        # モック: 血統情報
        blood_df = pd.DataFrame({
            "kettonum": ["0000000001", "0000000002"],
            "fnum": ["F001", "F002"],
            "mfnum": ["M001", "M002"],
        })

        # BMS距離帯別
        bms_dist_df = pd.DataFrame({
            "bms_num": ["M001", "M002"],
            "total": [100, 50],
            "top3": [25, 10],
        })

        # BMS馬場状態別
        bms_baba_df = pd.DataFrame({
            "bms_num": ["M001"],
            "total": [80],
            "top3": [20],
        })

        # BMS競馬場別
        bms_jyo_df = pd.DataFrame({
            "bms_num": ["M001", "M002"],
            "total": [60, 30],
            "top3": [18, 6],
        })

        # 父産駒馬齢別
        sire_age_df_3 = pd.DataFrame({
            "sire_num": ["F001"],
            "total": [200],
            "top3": [70],
        })
        sire_age_df_4 = pd.DataFrame({
            "sire_num": ["F002"],
            "total": [150],
            "top3": [45],
        })

        # ニックストラック種別別
        nicks_track_df = pd.DataFrame({
            "father_num": ["F001"],
            "bms_num": ["M001"],
            "total": [40],
            "top3": [12],
        })

        # 父産駒クラス別
        sire_class_df = pd.DataFrame({
            "sire_num": ["F001", "F002"],
            "total": [30, 20],
            "top3": [9, 4],
        })

        def mock_query_df(sql, params=None):
            """SQLの内容に応じてモックデータを返す."""
            sql_lower = sql.lower().strip()
            # レース情報
            if "kyori" in sql_lower and "gradecd" in sql_lower and "limit" in sql_lower:
                return race_info
            # 馬齢取得（kettonum IN + datakubun IN でフィルタ）
            if "barei" in sql_lower and "datakubun in" in sql_lower:
                return barei_df
            # BMS距離帯別（s.mfnum AS bms_num + kyori条件）
            if "s.mfnum as bms_num" in sql_lower and "kyori" in sql_lower:
                return bms_dist_df
            # BMS馬場状態別
            if "s.mfnum as bms_num" in sql_lower and "babacd" in sql_lower:
                return bms_baba_df
            # BMS競馬場別（r.jyocd = で判定）
            if "s.mfnum as bms_num" in sql_lower and "r.jyocd = " in sql_lower:
                return bms_jyo_df
            # ニックストラック種別別（fnum + mfnum + trackcd）
            if "s.fnum as father_num" in sql_lower and "s.mfnum as bms_num" in sql_lower:
                return nicks_track_df
            # 父産駒馬齢別
            if "s.fnum as sire_num" in sql_lower and "barei" in sql_lower:
                barei_val = params.get("barei", 0)
                if barei_val == 3:
                    return sire_age_df_3
                elif barei_val == 4:
                    return sire_age_df_4
                return pd.DataFrame(columns=["sire_num", "total", "top3"])
            # 父産駒クラス別
            if "s.fnum as sire_num" in sql_lower and ("gradecd" in sql_lower or "jyokencd5" in sql_lower):
                return sire_class_df
            # 血統情報（n_sanku基本）
            if "n_sanku" in sql_lower and "fnum" in sql_lower:
                return blood_df
            return pd.DataFrame()

        with patch("src.features.bms_detail.query_df", side_effect=mock_query_df):
            result = ext.extract(race_key, uma_race_df)

        assert len(result) == 2
        assert result.index.name == "kettonum"

        # B1: BMS距離帯別
        assert result.at["0000000001", "blood_bms_dist_rate"] == pytest.approx(0.25)
        assert result.at["0000000002", "blood_bms_dist_rate"] == pytest.approx(0.20)

        # B2: BMS馬場状態別
        assert result.at["0000000001", "blood_bms_baba_rate"] == pytest.approx(0.25)

        # B3: BMS競馬場別
        assert result.at["0000000001", "blood_bms_jyo_rate"] == pytest.approx(0.30)
        assert result.at["0000000002", "blood_bms_jyo_rate"] == pytest.approx(0.20)

        # B4: 父産駒馬齢別
        assert result.at["0000000001", "blood_father_age_rate"] == pytest.approx(0.35)
        assert result.at["0000000002", "blood_father_age_rate"] == pytest.approx(0.30)

        # B5: ニックストラック種別別
        assert result.at["0000000001", "blood_nicks_track_rate"] == pytest.approx(0.30)

        # B6: 父産駒クラス別
        assert result.at["0000000001", "blood_father_class_rate"] == pytest.approx(0.30)
        assert result.at["0000000002", "blood_father_class_rate"] == pytest.approx(0.20)

    def test_extract_empty_horses(self) -> None:
        """出走馬が空の場合."""
        from src.features.bms_detail import BMSDetailFeatureExtractor

        ext = BMSDetailFeatureExtractor()
        race_key = {
            "year": "2024", "monthday": "0601",
            "jyocd": "05", "kaiji": "01",
            "nichiji": "01", "racenum": "01",
        }
        uma_race_df = pd.DataFrame(columns=["kettonum", "umaban"])
        result = ext.extract(race_key, uma_race_df)
        assert result.empty

    def test_extract_missing_blood_info(self) -> None:
        """血統情報が取得できない場合は NaN が設定される."""
        from src.features.bms_detail import BMSDetailFeatureExtractor

        ext = BMSDetailFeatureExtractor()
        race_key = {
            "year": "2024", "monthday": "0601",
            "jyocd": "05", "kaiji": "01",
            "nichiji": "01", "racenum": "01",
        }
        uma_race_df = pd.DataFrame({
            "kettonum": ["0000000001"],
            "umaban": ["1"],
        })

        race_info = pd.DataFrame([{
            "kyori": "1600", "trackcd": "11",
            "sibababacd": "1", "dirtbabacd": "1",
            "gradecd": " ", "jyokencd5": "500",
        }])

        barei_df = pd.DataFrame({
            "kettonum": ["0000000001"],
            "barei": ["3"],
        })

        def mock_query_df(sql, params=None):
            sql_lower = sql.lower().strip()
            if "kyori" in sql_lower and "gradecd" in sql_lower:
                return race_info
            if "barei" in sql_lower and "datakubun" in sql_lower:
                return barei_df
            if "n_sanku" in sql_lower:
                return pd.DataFrame(columns=["kettonum", "fnum", "mfnum"])
            return pd.DataFrame()

        with patch("src.features.bms_detail.query_df", side_effect=mock_query_df):
            result = ext.extract(race_key, uma_race_df)

        # 血統情報なし → 全特徴量が NaN（LightGBMネイティブ欠損）
        import math
        assert math.isnan(result.at["0000000001", "blood_bms_dist_rate"])
        assert math.isnan(result.at["0000000001", "blood_bms_baba_rate"])
        assert math.isnan(result.at["0000000001", "blood_bms_jyo_rate"])
        assert math.isnan(result.at["0000000001", "blood_father_age_rate"])
        assert math.isnan(result.at["0000000001", "blood_nicks_track_rate"])
        assert math.isnan(result.at["0000000001", "blood_father_class_rate"])

    def test_min_samples_threshold(self) -> None:
        """最小サンプル数閾値のテスト."""
        from src.features.bms_detail import (
            _safe_rate_with_threshold,
            MIN_SAMPLES,
            MIN_SAMPLES_NICKS,
        )
        import math

        # サンプル数が閾値未満 → NaN
        assert math.isnan(_safe_rate_with_threshold(5, 10, MIN_SAMPLES))
        assert math.isnan(_safe_rate_with_threshold(3, 19, MIN_SAMPLES))

        # サンプル数が閾値以上 → 正しい率
        assert _safe_rate_with_threshold(6, 20, MIN_SAMPLES) == pytest.approx(0.3)
        assert _safe_rate_with_threshold(10, 100, MIN_SAMPLES) == pytest.approx(0.1)

        # ニックス用の高い閾値
        assert math.isnan(_safe_rate_with_threshold(5, 29, MIN_SAMPLES_NICKS))
        assert _safe_rate_with_threshold(9, 30, MIN_SAMPLES_NICKS) == pytest.approx(0.3)


# ============================================================
# _classify_class のテスト
# ============================================================


class TestClassifyClass:
    """_classify_class 関数のテスト."""

    def test_grade_race(self) -> None:
        from src.features.bms_detail import _classify_class

        assert _classify_class("A", "999") == "grade"
        assert _classify_class("B", "999") == "grade"
        assert _classify_class("C", "500") == "grade"
        assert _classify_class("D", "300") == "grade"

    def test_open_race(self) -> None:
        from src.features.bms_detail import _classify_class

        assert _classify_class(" ", "999") == "open"
        assert _classify_class("", "900") == "open"
        assert _classify_class("E", "950") == "open"

    def test_jouken_race(self) -> None:
        from src.features.bms_detail import _classify_class

        assert _classify_class(" ", "500") == "jouken"
        assert _classify_class("", "300") == "jouken"
        assert _classify_class("E", "700") == "jouken"
        assert _classify_class("", "") == "jouken"


# ============================================================
# サプリメント登録確認テスト
# ============================================================


class TestBMSDetailRegistration:
    """bms_detail がサプリメント登録簿に登録されていることの確認."""

    def test_registered_in_supplement(self) -> None:
        from src.features.supplement import list_available_supplements

        available = list_available_supplements()
        assert "bms_detail" in available

    def test_supplement_parquet_path(self) -> None:
        from src.features.supplement import supplement_parquet_path

        path = supplement_parquet_path("bms_detail", "2024")
        assert path.name == "bms_detail_2024.parquet"
        assert "supplements" in str(path)
