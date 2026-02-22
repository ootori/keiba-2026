"""Glickoレーティング特徴量（rating サプリメント）のテスト.

DB接続不要のユニットテスト。

実行方法:
    pytest tests/test_rating_supplement.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# RatingFeatureExtractor のユニットテスト
# ============================================================


class TestRatingFeatureExtractor:
    """RatingFeatureExtractor のテスト（DB不要）."""

    def test_feature_names(self) -> None:
        from src.features.rating import RatingFeatureExtractor

        ext = RatingFeatureExtractor()
        names = ext.feature_names
        assert len(names) == 28
        # 馬レーティング
        assert "rating_horse_all" in names
        assert "rating_horse_all_rd" in names
        assert "rating_horse_surface" in names
        assert "rating_horse_surface_rd" in names
        assert "rating_horse_best" in names
        assert "rating_horse_surface_exists" in names
        assert "rating_horse_races_rated" in names
        assert "rating_horse_rd_ratio" in names
        # 騎手レーティング
        assert "rating_jockey_all" in names
        assert "rating_jockey_all_rd" in names
        assert "rating_jockey_surface" in names
        assert "rating_jockey_surface_rd" in names
        assert "rating_jockey_jyo" in names
        assert "rating_jockey_jyo_rd" in names
        # 合算
        assert "rating_combined" in names
        assert "rating_combined_rd" in names
        # レース内相対（馬単体）
        for n in range(1, 6):
            assert f"rating_horse_diff_top{n}" in names
        assert "rating_horse_diff_strongest2" in names
        # レース内相対（合算）
        for n in range(1, 6):
            assert f"rating_combined_diff_top{n}" in names
        assert "rating_combined_diff_strongest2" in names

    def test_extract_all_features(self) -> None:
        """全特徴量が正しく抽出されるテスト（芝レース）."""
        from src.features.rating import RatingFeatureExtractor

        ext = RatingFeatureExtractor()
        race_key = {
            "year": "2024", "monthday": "0601",
            "jyocd": "05", "kaiji": "01",
            "nichiji": "01", "racenum": "01",
        }
        uma_race_df = pd.DataFrame({
            "kettonum": ["0000000001", "0000000002", "0000000003"],
            "umaban": ["1", "2", "3"],
        })

        # モック: トラック情報（芝）
        track_df = pd.DataFrame([{"trackcd": "11"}])

        # モック: 馬の騎手コード
        horse_info_df = pd.DataFrame({
            "kettonum": ["0000000001", "0000000002", "0000000003"],
            "umaban": ["1", "2", "3"],
            "kisyucode": ["K0001", "K0002", "K0003"],
        })

        # モック: 馬レーティング
        horse_rating_df = pd.DataFrame({
            "kettonum": ["0000000001", "0000000002", "0000000003"],
            "all_rating": [1600.0, 1500.0, 1400.0],
            "all_rd": [50.0, 60.0, 70.0],
            "shiba_rating": [1650.0, 1480.0, None],
            "shiba_rd": [55.0, 65.0, None],
            "dirt_rating": [1550.0, 1520.0, 1380.0],
            "dirt_rd": [60.0, 55.0, 75.0],
        })

        # モック: 馬レーティング数
        horse_count_df = pd.DataFrame({
            "kettonum": ["0000000001", "0000000002", "0000000003"],
            "rated_count": [20, 15, 3],
        })

        # モック: 騎手レーティング
        jockey_rating_df = pd.DataFrame({
            "kisyucode": ["K0001", "K0002", "K0003"],
            "all_rating": [1700.0, 1600.0, 1500.0],
            "all_rd": [40.0, 45.0, 50.0],
            "shiba_rating": [1720.0, 1580.0, 1510.0],
            "shiba_rd": [42.0, 48.0, 52.0],
            "dirt_rating": [1680.0, 1620.0, 1490.0],
            "dirt_rd": [45.0, 43.0, 55.0],
            "jyocd_01_rating": [1700.0, 1600.0, 1500.0],
            "jyocd_01_rd": [50.0, 55.0, 60.0],
            "jyocd_02_rating": [1710.0, 1590.0, None],
            "jyocd_02_rd": [48.0, 53.0, None],
            "jyocd_03_rating": [None, None, None],
            "jyocd_03_rd": [None, None, None],
            "jyocd_04_rating": [None, None, None],
            "jyocd_04_rd": [None, None, None],
            "jyocd_05_rating": [1730.0, 1610.0, 1520.0],
            "jyocd_05_rd": [44.0, 47.0, 54.0],
            "jyocd_06_rating": [None, None, None],
            "jyocd_06_rd": [None, None, None],
            "jyocd_07_rating": [None, None, None],
            "jyocd_07_rd": [None, None, None],
            "jyocd_08_rating": [None, None, None],
            "jyocd_08_rd": [None, None, None],
            "jyocd_09_rating": [None, None, None],
            "jyocd_09_rd": [None, None, None],
            "jyocd_10_rating": [None, None, None],
            "jyocd_10_rd": [None, None, None],
        })

        def mock_query_df(sql, params=None):
            sql_lower = sql.lower().strip()
            # トラック情報
            if "trackcd" in sql_lower and "n_race" in sql_lower and "limit" in sql_lower:
                return track_df
            # 馬の騎手コード
            if "kisyucode" in sql_lower and "n_uma_race" in sql_lower and "datakubun" in sql_lower:
                return horse_info_df
            # 馬レーティング数
            if "count(*)" in sql_lower and "n_uma_race_rating" in sql_lower:
                return horse_count_df
            # 馬レーティング
            if "n_uma_race_rating" in sql_lower and "distinct on" in sql_lower:
                return horse_rating_df
            # 騎手レーティング
            if "n_kisyu_race_rating" in sql_lower and "distinct on" in sql_lower:
                return jockey_rating_df
            return pd.DataFrame()

        with patch("src.features.rating.query_df", side_effect=mock_query_df):
            result = ext.extract(race_key, uma_race_df)

        assert len(result) == 3
        assert result.index.name == "kettonum"

        # 馬レーティング
        assert result.at["0000000001", "rating_horse_all"] == pytest.approx(1600.0)
        assert result.at["0000000001", "rating_horse_all_rd"] == pytest.approx(50.0)
        # 芝レースなのでshiba_ratingが使われる
        assert result.at["0000000001", "rating_horse_surface"] == pytest.approx(1650.0)
        assert result.at["0000000001", "rating_horse_surface_rd"] == pytest.approx(55.0)
        assert result.at["0000000001", "rating_horse_surface_exists"] == 1

        # 馬3はshiba_ratingがNoneなのでall_ratingにフォールバック
        assert result.at["0000000003", "rating_horse_surface"] == pytest.approx(1400.0)
        assert result.at["0000000003", "rating_horse_surface_exists"] == 0

        # best
        assert result.at["0000000001", "rating_horse_best"] == pytest.approx(1650.0)

        # races_rated
        assert result.at["0000000001", "rating_horse_races_rated"] == 20
        assert result.at["0000000003", "rating_horse_races_rated"] == 3

        # rd_ratio
        assert result.at["0000000001", "rating_horse_rd_ratio"] == pytest.approx(55.0 / 50.0)

        # 騎手レーティング
        assert result.at["0000000001", "rating_jockey_all"] == pytest.approx(1700.0)
        assert result.at["0000000001", "rating_jockey_surface"] == pytest.approx(1720.0)
        # jyocd=05なのでjyocd_05_ratingが使われる
        assert result.at["0000000001", "rating_jockey_jyo"] == pytest.approx(1730.0)
        assert result.at["0000000001", "rating_jockey_jyo_rd"] == pytest.approx(44.0)

        # 合算: horse_surface + jockey_surface
        assert result.at["0000000001", "rating_combined"] == pytest.approx(
            1650.0 + 1720.0,
        )
        expected_rd = (55.0 ** 2 + 42.0 ** 2) ** 0.5
        assert result.at["0000000001", "rating_combined_rd"] == pytest.approx(
            expected_rd,
        )

        # レース内相対（馬単体: surface_rating降順 → 1650, 1480, 1400）
        # 馬1(1650): diff_top1 = 0, diff_top2 = 170, diff_top3 = 250
        assert result.at["0000000001", "rating_horse_diff_top1"] == pytest.approx(0.0)
        assert result.at["0000000001", "rating_horse_diff_top2"] == pytest.approx(170.0)
        # 馬2(1480): diff_top1 = -170
        assert result.at["0000000002", "rating_horse_diff_top1"] == pytest.approx(-170.0)

        # strongest2: all_rating 1位(1600) - 2位(1500) = 100
        assert result.at["0000000001", "rating_horse_diff_strongest2"] == pytest.approx(100.0)

    def test_extract_empty_horses(self) -> None:
        """出走馬が空の場合."""
        from src.features.rating import RatingFeatureExtractor

        ext = RatingFeatureExtractor()
        race_key = {
            "year": "2024", "monthday": "0601",
            "jyocd": "05", "kaiji": "01",
            "nichiji": "01", "racenum": "01",
        }
        uma_race_df = pd.DataFrame(columns=["kettonum", "umaban"])
        result = ext.extract(race_key, uma_race_df)
        assert result.empty

    def test_extract_no_ratings(self) -> None:
        """レーティングが一切ない場合は全てNaN."""
        from src.features.rating import RatingFeatureExtractor

        ext = RatingFeatureExtractor()
        race_key = {
            "year": "2024", "monthday": "0601",
            "jyocd": "05", "kaiji": "01",
            "nichiji": "01", "racenum": "01",
        }
        uma_race_df = pd.DataFrame({
            "kettonum": ["0000000001"],
            "umaban": ["1"],
        })

        track_df = pd.DataFrame([{"trackcd": "11"}])
        horse_info_df = pd.DataFrame({
            "kettonum": ["0000000001"],
            "umaban": ["1"],
            "kisyucode": ["K0001"],
        })
        empty_horse_rating = pd.DataFrame(
            columns=["kettonum", "all_rating", "all_rd",
                     "shiba_rating", "shiba_rd", "dirt_rating", "dirt_rd"],
        )
        empty_count = pd.DataFrame(columns=["kettonum", "rated_count"])

        # 騎手レーティングも空
        jockey_cols = ["kisyucode", "all_rating", "all_rd",
                       "shiba_rating", "shiba_rd", "dirt_rating", "dirt_rd"]
        for i in range(1, 11):
            jyo_cd = f"{i:02d}"
            jockey_cols.extend([f"jyocd_{jyo_cd}_rating", f"jyocd_{jyo_cd}_rd"])
        empty_jockey_rating = pd.DataFrame(columns=jockey_cols)

        def mock_query_df(sql, params=None):
            sql_lower = sql.lower().strip()
            if "trackcd" in sql_lower and "limit" in sql_lower:
                return track_df
            if "kisyucode" in sql_lower and "datakubun" in sql_lower:
                return horse_info_df
            if "count(*)" in sql_lower and "n_uma_race_rating" in sql_lower:
                return empty_count
            if "n_uma_race_rating" in sql_lower:
                return empty_horse_rating
            if "n_kisyu_race_rating" in sql_lower:
                return empty_jockey_rating
            return pd.DataFrame()

        with patch("src.features.rating.query_df", side_effect=mock_query_df):
            result = ext.extract(race_key, uma_race_df)

        assert len(result) == 1
        # 全レーティング特徴量がNaN
        assert math.isnan(result.at["0000000001", "rating_horse_all"])
        assert math.isnan(result.at["0000000001", "rating_horse_surface"])
        assert math.isnan(result.at["0000000001", "rating_jockey_all"])
        assert math.isnan(result.at["0000000001", "rating_combined"])
        assert result.at["0000000001", "rating_horse_races_rated"] == 0
        assert result.at["0000000001", "rating_horse_surface_exists"] == 0

    def test_dirt_surface_selection(self) -> None:
        """ダートレースではdirt_ratingが選択される."""
        from src.features.rating import RatingFeatureExtractor

        ext = RatingFeatureExtractor()
        race_key = {
            "year": "2024", "monthday": "0601",
            "jyocd": "05", "kaiji": "01",
            "nichiji": "01", "racenum": "01",
        }
        uma_race_df = pd.DataFrame({
            "kettonum": ["0000000001"],
            "umaban": ["1"],
        })

        # ダートトラック
        track_df = pd.DataFrame([{"trackcd": "23"}])
        horse_info_df = pd.DataFrame({
            "kettonum": ["0000000001"],
            "umaban": ["1"],
            "kisyucode": ["K0001"],
        })
        horse_rating_df = pd.DataFrame({
            "kettonum": ["0000000001"],
            "all_rating": [1500.0],
            "all_rd": [50.0],
            "shiba_rating": [1600.0],
            "shiba_rd": [55.0],
            "dirt_rating": [1550.0],
            "dirt_rd": [60.0],
        })
        horse_count_df = pd.DataFrame({
            "kettonum": ["0000000001"],
            "rated_count": [10],
        })

        jockey_cols = ["kisyucode", "all_rating", "all_rd",
                       "shiba_rating", "shiba_rd", "dirt_rating", "dirt_rd"]
        for i in range(1, 11):
            jyo_cd = f"{i:02d}"
            jockey_cols.extend([f"jyocd_{jyo_cd}_rating", f"jyocd_{jyo_cd}_rd"])
        empty_jockey = pd.DataFrame(columns=jockey_cols)

        def mock_query_df(sql, params=None):
            sql_lower = sql.lower().strip()
            if "trackcd" in sql_lower and "limit" in sql_lower:
                return track_df
            if "kisyucode" in sql_lower and "datakubun" in sql_lower:
                return horse_info_df
            if "count(*)" in sql_lower and "n_uma_race_rating" in sql_lower:
                return horse_count_df
            if "n_uma_race_rating" in sql_lower:
                return horse_rating_df
            if "n_kisyu_race_rating" in sql_lower:
                return empty_jockey
            return pd.DataFrame()

        with patch("src.features.rating.query_df", side_effect=mock_query_df):
            result = ext.extract(race_key, uma_race_df)

        # ダートなのでdirt_ratingが選択される
        assert result.at["0000000001", "rating_horse_surface"] == pytest.approx(1550.0)
        assert result.at["0000000001", "rating_horse_surface_rd"] == pytest.approx(60.0)
        assert result.at["0000000001", "rating_horse_surface_exists"] == 1


# ============================================================
# 内部ヘルパーのテスト
# ============================================================


class TestSurfaceSelection:
    """_select_surface_rating のテスト."""

    def test_turf_available(self) -> None:
        from src.features.rating import RatingFeatureExtractor

        ext = RatingFeatureExtractor()
        row = {
            "all_rating": 1500.0, "all_rd": 50.0,
            "shiba_rating": 1600.0, "shiba_rd": 55.0,
            "dirt_rating": 1400.0, "dirt_rd": 60.0,
        }
        rating, rd, exists = ext._select_surface_rating(row, "turf")
        assert rating == pytest.approx(1600.0)
        assert rd == pytest.approx(55.0)
        assert exists == 1

    def test_turf_fallback(self) -> None:
        from src.features.rating import RatingFeatureExtractor

        ext = RatingFeatureExtractor()
        row = {
            "all_rating": 1500.0, "all_rd": 50.0,
            "shiba_rating": None, "shiba_rd": None,
            "dirt_rating": 1400.0, "dirt_rd": 60.0,
        }
        rating, rd, exists = ext._select_surface_rating(row, "turf")
        assert rating == pytest.approx(1500.0)
        assert rd == pytest.approx(50.0)
        assert exists == 0

    def test_unknown_track(self) -> None:
        from src.features.rating import RatingFeatureExtractor

        ext = RatingFeatureExtractor()
        row = {
            "all_rating": 1500.0, "all_rd": 50.0,
            "shiba_rating": 1600.0, "shiba_rd": 55.0,
        }
        rating, rd, exists = ext._select_surface_rating(row, "")
        # 不明トラック → all_ratingにフォールバック
        assert rating == pytest.approx(1500.0)
        assert exists == 0


class TestJyoRatingSelection:
    """_select_jyo_rating のテスト."""

    def test_jyo_available(self) -> None:
        from src.features.rating import RatingFeatureExtractor

        jockey = {
            "jyocd_05_rating": 1700.0,
            "jyocd_05_rd": 45.0,
        }
        rating, rd = RatingFeatureExtractor._select_jyo_rating(jockey, "05")
        assert rating == pytest.approx(1700.0)
        assert rd == pytest.approx(45.0)

    def test_jyo_missing(self) -> None:
        from src.features.rating import RatingFeatureExtractor

        jockey = {}
        rating, rd = RatingFeatureExtractor._select_jyo_rating(jockey, "05")
        assert math.isnan(rating)
        assert math.isnan(rd)

    def test_empty_jyocd(self) -> None:
        from src.features.rating import RatingFeatureExtractor

        jockey = {"jyocd_05_rating": 1700.0, "jyocd_05_rd": 45.0}
        rating, rd = RatingFeatureExtractor._select_jyo_rating(jockey, "")
        assert math.isnan(rating)
        assert math.isnan(rd)


# ============================================================
# レース内相対特徴量のテスト
# ============================================================


class TestRaceRelativeFeatures:
    """_add_race_relative_features のテスト."""

    def test_diff_top_computation(self) -> None:
        from src.features.rating import RatingFeatureExtractor

        df = pd.DataFrame(
            {
                "rating_test": [1600.0, 1500.0, 1400.0, 1300.0, 1200.0],
                "rating_test_all": [1600.0, 1500.0, 1400.0, 1300.0, 1200.0],
            },
            index=pd.Index(["A", "B", "C", "D", "E"], name="kettonum"),
        )

        RatingFeatureExtractor._add_race_relative_features(
            df,
            rating_col="rating_test",
            all_rating_col="rating_test_all",
            prefix="test",
        )

        # 馬A(1600): diff_top1=0, diff_top2=100, diff_top3=200
        assert df.at["A", "test_diff_top1"] == pytest.approx(0.0)
        assert df.at["A", "test_diff_top2"] == pytest.approx(100.0)
        assert df.at["A", "test_diff_top5"] == pytest.approx(400.0)

        # 馬C(1400): diff_top1=-200, diff_top2=-100, diff_top3=0
        assert df.at["C", "test_diff_top1"] == pytest.approx(-200.0)
        assert df.at["C", "test_diff_top2"] == pytest.approx(-100.0)
        assert df.at["C", "test_diff_top3"] == pytest.approx(0.0)

        # strongest2: 1600 - 1500 = 100
        assert df.at["A", "test_diff_strongest2"] == pytest.approx(100.0)
        # 全馬共通値
        assert df.at["C", "test_diff_strongest2"] == pytest.approx(100.0)

    def test_fewer_than_5_horses(self) -> None:
        from src.features.rating import RatingFeatureExtractor

        df = pd.DataFrame(
            {"rating_test": [1600.0, 1500.0], "rating_test_all": [1600.0, 1500.0]},
            index=pd.Index(["A", "B"], name="kettonum"),
        )

        RatingFeatureExtractor._add_race_relative_features(
            df,
            rating_col="rating_test",
            all_rating_col="rating_test_all",
            prefix="test",
        )

        assert df.at["A", "test_diff_top1"] == pytest.approx(0.0)
        assert df.at["A", "test_diff_top2"] == pytest.approx(100.0)
        # top3〜top5はNaN
        assert math.isnan(df.at["A", "test_diff_top3"])
        assert math.isnan(df.at["A", "test_diff_top4"])
        assert math.isnan(df.at["A", "test_diff_top5"])

    def test_with_nan_ratings(self) -> None:
        from src.features.rating import RatingFeatureExtractor

        df = pd.DataFrame(
            {
                "rating_test": [1600.0, float("nan"), 1400.0],
                "rating_test_all": [1600.0, float("nan"), 1400.0],
            },
            index=pd.Index(["A", "B", "C"], name="kettonum"),
        )

        RatingFeatureExtractor._add_race_relative_features(
            df,
            rating_col="rating_test",
            all_rating_col="rating_test_all",
            prefix="test",
        )

        # NaNの馬Bは diff_top* もNaN
        assert math.isnan(df.at["B", "test_diff_top1"])
        # 馬A(1600): top1=1600, top2=1400 → diff_top2=200
        assert df.at["A", "test_diff_top2"] == pytest.approx(200.0)
        # strongest2: 1600 - 1400 = 200（NaN除外して計算）
        assert df.at["A", "test_diff_strongest2"] == pytest.approx(200.0)


# ============================================================
# _safe_rating / _is_nan のテスト
# ============================================================


class TestSafeRating:
    """_safe_rating のテスト."""

    def test_normal_value(self) -> None:
        from src.features.rating import RatingFeatureExtractor

        assert RatingFeatureExtractor._safe_rating(1500.0) == pytest.approx(1500.0)
        assert RatingFeatureExtractor._safe_rating(0.0) == pytest.approx(0.0)

    def test_none(self) -> None:
        from src.features.rating import RatingFeatureExtractor

        assert math.isnan(RatingFeatureExtractor._safe_rating(None))

    def test_nan(self) -> None:
        from src.features.rating import RatingFeatureExtractor

        assert math.isnan(RatingFeatureExtractor._safe_rating(float("nan")))

    def test_string(self) -> None:
        from src.features.rating import RatingFeatureExtractor

        assert math.isnan(RatingFeatureExtractor._safe_rating("abc"))

    def test_is_nan(self) -> None:
        from src.features.rating import _is_nan

        assert _is_nan(float("nan")) is True
        assert _is_nan(1500.0) is False
        assert _is_nan(None) is True


# ============================================================
# サプリメント登録確認テスト
# ============================================================


class TestRatingRegistration:
    """rating がサプリメント登録簿に登録されていることの確認."""

    def test_registered_in_supplement(self) -> None:
        from src.features.supplement import list_available_supplements

        available = list_available_supplements()
        assert "rating" in available

    def test_supplement_parquet_path(self) -> None:
        from src.features.supplement import supplement_parquet_path

        path = supplement_parquet_path("rating", "2024")
        assert path.name == "rating_2024.parquet"
        assert "supplements" in str(path)
