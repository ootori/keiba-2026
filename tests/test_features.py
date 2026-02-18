"""特徴量抽出のテスト.

DB接続が必要なテストは @pytest.mark.db でマークし、
DB未接続時はスキップする。

実行方法:
    # 全テスト（DBなしでも動くもの）
    pytest tests/test_features.py -v

    # DB接続テストを含む
    pytest tests/test_features.py -v -m db
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# ユーティリティ関数のテスト（DB不要）
# ============================================================


class TestCodeMaster:
    """コード表変換ユーティリティのテスト."""

    def test_track_type_turf(self) -> None:
        from src.utils.code_master import track_type

        assert track_type("11") == "turf"
        assert track_type("17") == "turf"
        assert track_type("10") == "turf"
        assert track_type("22") == "turf"

    def test_track_type_dirt(self) -> None:
        from src.utils.code_master import track_type

        assert track_type("23") == "dirt"
        assert track_type("24") == "dirt"
        assert track_type("29") == "dirt"

    def test_track_type_jump(self) -> None:
        from src.utils.code_master import track_type

        assert track_type("51") == "jump"
        assert track_type("59") == "jump"

    def test_track_type_unknown(self) -> None:
        from src.utils.code_master import track_type

        assert track_type("99") == "unknown"
        assert track_type("") == "unknown"
        assert track_type("abc") == "unknown"

    def test_course_direction(self) -> None:
        from src.utils.code_master import course_direction

        assert course_direction("11") == "left"
        assert course_direction("17") == "right"
        assert course_direction("10") == "straight"
        assert course_direction("29") == "straight"
        assert course_direction("23") == "left"
        assert course_direction("24") == "right"

    def test_distance_category(self) -> None:
        from src.utils.code_master import distance_category

        assert distance_category(1000) == "short"
        assert distance_category(1400) == "short"
        assert distance_category(1600) == "mile"
        assert distance_category(1800) == "mile"
        assert distance_category(2000) == "middle"
        assert distance_category(2200) == "middle"
        assert distance_category(2400) == "long"
        assert distance_category(3600) == "long"

    def test_time_to_sec(self) -> None:
        from src.utils.code_master import time_to_sec

        assert time_to_sec("1234") == pytest.approx(83.4)
        assert time_to_sec("2003") == pytest.approx(120.3)
        assert time_to_sec("0590") == pytest.approx(59.0)
        assert time_to_sec("") is None
        assert time_to_sec("   ") is None
        assert time_to_sec(None) is None

    def test_haron_time_to_sec(self) -> None:
        from src.utils.code_master import haron_time_to_sec

        assert haron_time_to_sec("345") == pytest.approx(34.5)
        assert haron_time_to_sec("1234") == pytest.approx(123.4)
        assert haron_time_to_sec("") is None
        assert haron_time_to_sec(None) is None

    def test_interval_category(self) -> None:
        from src.utils.code_master import interval_category

        assert interval_category(5) == "rento"
        assert interval_category(7) == "1_2weeks"
        assert interval_category(14) == "1_2weeks"
        assert interval_category(21) == "3_4weeks"
        assert interval_category(42) == "5_8weeks"
        assert interval_category(70) == "9plus_weeks"
        assert interval_category(180) == "kyuumei"

    def test_baba_code_for_track(self) -> None:
        from src.utils.code_master import baba_code_for_track

        # 芝ならSibaBabaCD
        assert baba_code_for_track("11", "1", "3") == "1"
        # ダートならDirtBabaCD
        assert baba_code_for_track("23", "1", "3") == "3"


class TestBaseTimeCalc:
    """スピード指数算出のテスト."""

    def test_calc_speed_index(self) -> None:
        from src.utils.base_time import calc_speed_index

        base_dict = {
            ("1600", "turf", "1"): 96.0,
        }
        # 基準より速い → プラス
        si = calc_speed_index(95.0, "1600", "turf", "1", base_dict)
        assert si > 0

        # 基準より遅い → マイナス
        si = calc_speed_index(97.0, "1600", "turf", "1", base_dict)
        assert si < 0

        # 基準と同じ → 0
        si = calc_speed_index(96.0, "1600", "turf", "1", base_dict)
        assert si == pytest.approx(0.0)

    def test_calc_speed_index_missing_key(self) -> None:
        from src.utils.base_time import calc_speed_index

        base_dict: dict = {}
        si = calc_speed_index(95.0, "1600", "turf", "1", base_dict)
        assert si == 0.0


class TestFeatureExtractorBase:
    """基底クラスのユーティリティメソッドのテスト."""

    def test_safe_int(self) -> None:
        from src.features.base import FeatureExtractor

        class DummyExtractor(FeatureExtractor):
            def extract(self, race_key, uma_race_df):
                return pd.DataFrame()

            @property
            def feature_names(self):
                return []

        ext = DummyExtractor()
        assert ext._safe_int("123") == 123
        assert ext._safe_int("  456  ") == 456
        assert ext._safe_int("abc") == -1
        assert ext._safe_int(None) == -1
        assert ext._safe_int("", default=0) == 0

    def test_safe_float(self) -> None:
        from src.features.base import FeatureExtractor

        class DummyExtractor(FeatureExtractor):
            def extract(self, race_key, uma_race_df):
                return pd.DataFrame()

            @property
            def feature_names(self):
                return []

        ext = DummyExtractor()
        assert ext._safe_float("12.3") == pytest.approx(12.3)
        assert ext._safe_float("abc") == -1.0
        assert ext._safe_float(None) == -1.0

    def test_safe_rate(self) -> None:
        from src.features.base import FeatureExtractor

        class DummyExtractor(FeatureExtractor):
            def extract(self, race_key, uma_race_df):
                return pd.DataFrame()

            @property
            def feature_names(self):
                return []

        ext = DummyExtractor()
        assert ext._safe_rate(3, 10) == pytest.approx(0.3)
        assert ext._safe_rate(0, 10) == pytest.approx(0.0)
        assert ext._safe_rate(3, 0) == 0.0


# ============================================================
# DB接続が必要なテスト
# ============================================================


@pytest.fixture
def mock_query_df():
    """query_df をモックするフィクスチャ."""
    with patch("src.features.race.query_df") as mock:
        yield mock


class TestRaceFeatureExtractor:
    """レース条件特徴量のテスト（モック使用）."""

    def test_feature_names_count(self) -> None:
        from src.features.race import RaceFeatureExtractor

        ext = RaceFeatureExtractor()
        # レース条件14 + 枠順5 + 負担重量2 = 21
        assert len(ext.feature_names) == 21

    def test_extract_with_mock(self, mock_query_df) -> None:
        from src.features.race import RaceFeatureExtractor

        # レース情報のモック
        race_df = pd.DataFrame(
            [
                {
                    "jyocd": "09",
                    "kyori": "2200",
                    "trackcd": "17",
                    "sibababacd": "1",
                    "dirtbabacd": "",
                    "tenkocd": "1",
                    "gradecd": "A",
                    "syubetucd": "11",
                    "jyuryocd": "4",
                    "jyokencd5": "999",
                    "monthday": "0622",
                    "tokunum": "0123",
                    "syussotosu": "16",
                }
            ]
        )

        # 出走馬情報のモック
        horse_df = pd.DataFrame(
            [
                {"kettonum": "2019100001", "umaban": "01", "wakuban": "1", "futan": "570"},
                {"kettonum": "2019100002", "umaban": "02", "wakuban": "1", "futan": "560"},
            ]
        )

        mock_query_df.side_effect = [race_df, horse_df]

        ext = RaceFeatureExtractor()
        race_key = {
            "year": "2024",
            "monthday": "0622",
            "jyocd": "09",
            "kaiji": "03",
            "nichiji": "08",
            "racenum": "11",
        }
        uma_race_df = pd.DataFrame(
            {"kettonum": ["2019100001", "2019100002"]}
        )

        result = ext.extract(race_key, uma_race_df)

        assert len(result) == 2
        assert result.loc["2019100001", "race_distance"] == 2200
        assert result.loc["2019100001", "race_track_type"] == "turf"
        assert result.loc["2019100001", "race_course_dir"] == "right"
        assert result.loc["2019100001", "race_grade_cd"] == "A"
        assert result.loc["2019100001", "race_is_tokubetsu"] == 1


class TestHorseFeatureExtractor:
    """馬関連特徴量のテスト."""

    def test_feature_names_count(self) -> None:
        from src.features.horse import HorseFeatureExtractor

        ext = HorseFeatureExtractor()
        # 馬基本5 + 過去成績13 + 条件別14 + 馬体重5 + 間隔5 + 負担重量差1 = 43
        assert len(ext.feature_names) == 43


class TestSpeedStyleFeatureExtractor:
    """スピード・脚質特徴量のテスト."""

    def test_feature_names_count(self) -> None:
        from src.features.speed import SpeedStyleFeatureExtractor

        ext = SpeedStyleFeatureExtractor()
        # スピード12 + 脚質7 = 19
        assert len(ext.feature_names) == 19


class TestOddsFeatureExtractor:
    """オッズ特徴量のテスト."""

    def test_feature_names_count(self) -> None:
        from src.features.odds import OddsFeatureExtractor

        ext = OddsFeatureExtractor()
        assert len(ext.feature_names) == 7

    def test_parse_odds(self) -> None:
        from src.features.odds import OddsFeatureExtractor

        ext = OddsFeatureExtractor()
        assert ext._parse_odds("0320") == pytest.approx(32.0)
        assert ext._parse_odds("9999") == pytest.approx(999.9)
        assert ext._parse_odds("0000") == -1.0
        assert ext._parse_odds("") == -1.0


class TestFeaturePipeline:
    """パイプラインのテスト."""

    def test_cross_feature_names(self) -> None:
        from src.features.pipeline import FeaturePipeline

        names = FeaturePipeline._cross_feature_names()
        assert len(names) == 8
        assert "cross_dist_change" in names
        assert "cross_track_change" in names

    def test_relative_feature_names(self) -> None:
        from src.features.pipeline import FeaturePipeline

        names = FeaturePipeline._relative_feature_names()
        # 14ターゲット × 2（zscore + rank） = 28
        assert len(names) == 28
        assert "rel_speed_index_avg_last3_zscore" in names
        assert "rel_speed_index_avg_last3_rank" in names
        assert "rel_horse_fukusho_rate_zscore" in names
        assert "rel_jockey_win_rate_year_rank" in names

    def test_add_relative_features_zscore(self) -> None:
        """Zスコアが正しく計算されることを確認する."""
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(include_odds=False)

        # 3頭のテストデータ（スピード指数: 100, 90, 80 → 平均90, 標準偏差10）
        df = pd.DataFrame(
            {
                "speed_index_avg_last3": [100.0, 90.0, 80.0],
                "horse_fukusho_rate": [0.5, 0.3, 0.1],
            },
            index=["horse_a", "horse_b", "horse_c"],
        )

        result = pipeline._add_relative_features(df)

        # Zスコアの検証（speed_index_avg_last3: mean=90, std=10）
        assert "rel_speed_index_avg_last3_zscore" in result.columns
        assert result.loc["horse_a", "rel_speed_index_avg_last3_zscore"] == pytest.approx(1.0, abs=0.01)
        assert result.loc["horse_b", "rel_speed_index_avg_last3_zscore"] == pytest.approx(0.0, abs=0.01)
        assert result.loc["horse_c", "rel_speed_index_avg_last3_zscore"] == pytest.approx(-1.0, abs=0.01)

    def test_add_relative_features_rank(self) -> None:
        """ランクが正しく計算されることを確認する."""
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(include_odds=False)

        df = pd.DataFrame(
            {
                "speed_index_avg_last3": [100.0, 90.0, 80.0],
                "horse_avg_jyuni_last3": [3.0, 5.0, 2.0],  # 着順は小さい方が良い
            },
            index=["horse_a", "horse_b", "horse_c"],
        )

        result = pipeline._add_relative_features(df)

        # speed_index: ascending=False → 100が1位
        assert result.loc["horse_a", "rel_speed_index_avg_last3_rank"] == 1.0
        assert result.loc["horse_c", "rel_speed_index_avg_last3_rank"] == 3.0

        # avg_jyuni: ascending=True → 2.0が1位
        assert result.loc["horse_c", "rel_horse_avg_jyuni_last3_rank"] == 1.0
        assert result.loc["horse_b", "rel_horse_avg_jyuni_last3_rank"] == 3.0

    def test_add_relative_features_missing_column(self) -> None:
        """存在しない特徴量カラムはデフォルト値で埋められることを確認する."""
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(include_odds=False)

        # speed_index_avg_last3 がない DataFrame
        df = pd.DataFrame(
            {"horse_fukusho_rate": [0.5, 0.3]},
            index=["horse_a", "horse_b"],
        )

        result = pipeline._add_relative_features(df)
        assert result.loc["horse_a", "rel_speed_index_avg_last3_zscore"] == 0.0
        assert result.loc["horse_a", "rel_speed_index_avg_last3_rank"] == 0.0

    def test_add_relative_features_with_missing_values(self) -> None:
        """MISSING_NUMERIC値が正しくNaNとして扱われることを確認する."""
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(include_odds=False)

        df = pd.DataFrame(
            {
                "speed_index_avg_last3": [100.0, -1.0, 80.0],  # -1.0 = MISSING_NUMERIC
            },
            index=["horse_a", "horse_b", "horse_c"],
        )

        result = pipeline._add_relative_features(df)

        # MISSING_NUMERIC の馬はZスコアが0に埋められる
        assert result.loc["horse_b", "rel_speed_index_avg_last3_zscore"] == 0.0
        # 有効な2頭（100, 80）で mean=90, std=~14.14
        assert result.loc["horse_a", "rel_speed_index_avg_last3_zscore"] > 0
        assert result.loc["horse_c", "rel_speed_index_avg_last3_zscore"] < 0

    def test_add_relative_features_rate_zero_is_valid(self) -> None:
        """率系特徴量で0.0がNaN扱いされないことを確認する.

        horse_fukusho_rate=0.0は「複勝率ゼロ」という正当な値。
        MISSING_RATEと混同してNaN化してはいけない。
        """
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(include_odds=False)

        df = pd.DataFrame(
            {
                "horse_fukusho_rate": [0.5, 0.3, 0.0],  # 0.0 = 正当な値
                "horse_win_rate": [0.2, 0.0, 0.1],      # 0.0 = 正当な値
            },
            index=["horse_a", "horse_b", "horse_c"],
        )

        result = pipeline._add_relative_features(df)

        # 0.0は有効値として計算に含まれる（NaN扱いされない）
        # horse_fukusho_rate: [0.5, 0.3, 0.0] → mean≈0.267, 0.0は最下位
        zscore_c = result.loc["horse_c", "rel_horse_fukusho_rate_zscore"]
        assert zscore_c < 0, f"0.0の馬のZスコアは負であるべき: {zscore_c}"
        assert zscore_c != 0.0, "0.0の馬のZスコアが0.0 → NaN扱いされている"

        # horse_win_rate: 0.0の馬bもZスコアは0ではない
        zscore_b = result.loc["horse_b", "rel_horse_win_rate_zscore"]
        assert zscore_b < 0, f"0.0の馬のZスコアは負であるべき: {zscore_b}"

    def test_add_relative_features_blood_zero_is_missing(self) -> None:
        """血統率系特徴量では0.0が欠損扱いされることを確認する.

        blood_father_turf_rate=0.0は産駒データ不足を意味するため欠損扱い。
        """
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(include_odds=False)

        df = pd.DataFrame(
            {
                "blood_father_turf_rate": [0.3, 0.0, 0.2],  # 0.0 = データなし
            },
            index=["horse_a", "horse_b", "horse_c"],
        )

        result = pipeline._add_relative_features(df)

        # blood系の0.0はNaN → Zスコアは0.0（欠損埋め）になる
        assert result.loc["horse_b", "rel_blood_father_turf_rate_zscore"] == 0.0


# ============================================================
# 合計特徴量数の確認
# ============================================================


class TestTotalFeatureCount:
    """全特徴量の合計数を確認する."""

    def test_total_features(self) -> None:
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(include_odds=True)
        names = pipeline.feature_names

        # 基本約119 + クロス8 + 相対28 = 約155特徴量
        assert len(names) >= 140, f"特徴量が少なすぎます: {len(names)}"
        assert len(names) <= 180, f"特徴量が多すぎます: {len(names)}"

        # 重複がないこと
        assert len(names) == len(set(names)), (
            f"重複特徴量あり: {[n for n in names if names.count(n) > 1]}"
        )
