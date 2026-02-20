"""マイニング特徴量抽出器とサプリメントシステムのテスト.

DB接続不要のユニットテスト。
DB接続が必要なテストは @pytest.mark.db でマーク。

実行方法:
    pytest tests/test_mining_supplement.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# MiningFeatureExtractor のユニットテスト
# ============================================================


class TestMiningFeatureExtractor:
    """MiningFeatureExtractor のテスト（DB不要）."""

    def test_feature_names(self) -> None:
        from src.features.mining import MiningFeatureExtractor

        ext = MiningFeatureExtractor()
        names = ext.feature_names
        assert "mining_dm_time" in names
        assert "mining_dm_jyuni" in names
        assert "mining_dm_gosa_range" in names
        assert "mining_dm_gosa_p" in names
        assert "mining_dm_gosa_m" in names
        assert "mining_dm_kubun" in names
        assert "mining_tm_score" in names
        assert len(names) == 7

    def test_parse_dm_time_normal(self) -> None:
        from src.features.mining import MiningFeatureExtractor

        ext = MiningFeatureExtractor()
        # "1234" → 123.4秒
        assert ext._parse_dm_time("1234") == 123.4
        # "600" → 60.0秒
        assert ext._parse_dm_time("600") == 60.0

    def test_parse_dm_time_empty(self) -> None:
        from src.features.mining import MiningFeatureExtractor

        ext = MiningFeatureExtractor()
        assert ext._parse_dm_time("") == -1.0
        assert ext._parse_dm_time("0") == -1.0
        assert ext._parse_dm_time("  ") == -1.0

    def test_parse_dm_time_float(self) -> None:
        from src.features.mining import MiningFeatureExtractor

        ext = MiningFeatureExtractor()
        assert ext._parse_dm_time("123.4") == 123.4

    def test_extract_with_dm_data(self) -> None:
        """n_uma_race の DM データがある場合のテスト."""
        from src.features.mining import MiningFeatureExtractor

        ext = MiningFeatureExtractor()
        race_key = {
            "year": "2024", "monthday": "0101",
            "jyocd": "05", "kaiji": "01",
            "nichiji": "01", "racenum": "01",
        }
        uma_race_df = pd.DataFrame({
            "kettonum": ["0000000001", "0000000002"],
            "umaban": ["1", "2"],
        })

        # モック: n_uma_race DMデータ
        dm_df = pd.DataFrame({
            "kettonum": ["0000000001", "0000000002"],
            "dmtime": ["1234", "1240"],
            "dmjyuni": ["1", "3"],
            "dmgosap": ["5", "10"],
            "dmgosam": ["3", "8"],
            "dmkubun": ["1", "1"],
        })

        # モック: horse_umaban
        umaban_map = {"0000000001": "1", "0000000002": "2"}

        with (
            patch.object(ext, "_get_dm_from_uma_race", return_value=dm_df),
            patch.object(ext, "_get_mining_data", return_value=pd.DataFrame()),
            patch.object(ext, "_get_tm_data", return_value=pd.DataFrame()),
            patch.object(ext, "_get_horse_umaban", return_value=umaban_map),
        ):
            result = ext.extract(race_key, uma_race_df)

        assert len(result) == 2
        assert result.at["0000000001", "mining_dm_time"] == 123.4
        assert result.at["0000000001", "mining_dm_jyuni"] == 1
        assert result.at["0000000001", "mining_dm_gosa_p"] == 0.5
        assert result.at["0000000001", "mining_dm_gosa_m"] == 0.3
        assert result.at["0000000001", "mining_dm_gosa_range"] == pytest.approx(0.8)
        assert result.at["0000000001", "mining_dm_kubun"] == 1

        assert result.at["0000000002", "mining_dm_time"] == 124.0
        assert result.at["0000000002", "mining_dm_jyuni"] == 3

    def test_extract_missing_dm(self) -> None:
        """DMデータが欠損している場合のテスト."""
        from src.features.mining import MiningFeatureExtractor
        from src.config import MISSING_NUMERIC

        ext = MiningFeatureExtractor()
        race_key = {
            "year": "2024", "monthday": "0101",
            "jyocd": "05", "kaiji": "01",
            "nichiji": "01", "racenum": "01",
        }
        uma_race_df = pd.DataFrame({
            "kettonum": ["0000000001"],
            "umaban": ["1"],
        })

        with (
            patch.object(ext, "_get_dm_from_uma_race", return_value=pd.DataFrame()),
            patch.object(ext, "_get_mining_data", return_value=pd.DataFrame()),
            patch.object(ext, "_get_tm_data", return_value=pd.DataFrame()),
            patch.object(ext, "_get_horse_umaban", return_value={"0000000001": "1"}),
        ):
            result = ext.extract(race_key, uma_race_df)

        assert result.at["0000000001", "mining_dm_time"] == MISSING_NUMERIC
        assert result.at["0000000001", "mining_dm_jyuni"] == MISSING_NUMERIC
        assert result.at["0000000001", "mining_tm_score"] == MISSING_NUMERIC

    def test_extract_with_tm_data(self) -> None:
        """対戦型マイニングスコアがある場合のテスト."""
        from src.features.mining import MiningFeatureExtractor

        ext = MiningFeatureExtractor()
        race_key = {
            "year": "2024", "monthday": "0101",
            "jyocd": "05", "kaiji": "01",
            "nichiji": "01", "racenum": "01",
        }
        uma_race_df = pd.DataFrame({
            "kettonum": ["0000000001"],
            "umaban": ["1"],
        })

        tm_df = pd.DataFrame({
            "umaban": ["01"],
            "tmscore": ["75.5"],
        })

        with (
            patch.object(ext, "_get_dm_from_uma_race", return_value=pd.DataFrame()),
            patch.object(ext, "_get_mining_data", return_value=pd.DataFrame()),
            patch.object(ext, "_get_tm_data", return_value=tm_df),
            patch.object(ext, "_get_horse_umaban", return_value={"0000000001": "1"}),
        ):
            result = ext.extract(race_key, uma_race_df)

        assert result.at["0000000001", "mining_tm_score"] == 75.5

    def test_extract_empty_uma_race(self) -> None:
        """出走馬が空の場合."""
        from src.features.mining import MiningFeatureExtractor

        ext = MiningFeatureExtractor()
        race_key = {
            "year": "2024", "monthday": "0101",
            "jyocd": "05", "kaiji": "01",
            "nichiji": "01", "racenum": "01",
        }
        uma_race_df = pd.DataFrame(columns=["kettonum", "umaban"])
        result = ext.extract(race_key, uma_race_df)
        assert result.empty


# ============================================================
# サプリメントシステムのテスト
# ============================================================


class TestSupplementSystem:
    """サプリメント（差分特徴量）システムのテスト."""

    def test_list_available_supplements(self) -> None:
        from src.features.supplement import list_available_supplements

        available = list_available_supplements()
        assert "mining" in available

    def test_supplement_parquet_path(self) -> None:
        from src.features.supplement import supplement_parquet_path

        path = supplement_parquet_path("mining", "2024")
        assert path.name == "mining_2024.parquet"
        assert "supplements" in str(path)

    def test_merge_supplements_basic(self) -> None:
        """基本的なマージテスト."""
        from src.features.supplement import merge_supplements

        main_df = pd.DataFrame({
            "_key_year": ["2024", "2024"],
            "_key_monthday": ["0101", "0101"],
            "_key_jyocd": ["05", "05"],
            "_key_kaiji": ["01", "01"],
            "_key_nichiji": ["01", "01"],
            "_key_racenum": ["01", "01"],
            "kettonum": ["0000000001", "0000000002"],
            "horse_win_rate": [0.5, 0.3],
        })

        supp_df = pd.DataFrame({
            "_key_year": ["2024", "2024"],
            "_key_monthday": ["0101", "0101"],
            "_key_jyocd": ["05", "05"],
            "_key_kaiji": ["01", "01"],
            "_key_nichiji": ["01", "01"],
            "_key_racenum": ["01", "01"],
            "kettonum": ["0000000001", "0000000002"],
            "mining_dm_time": [123.4, 124.0],
            "mining_dm_jyuni": [1, 3],
        })

        with patch(
            "src.features.supplement.load_supplement_years",
            return_value=supp_df,
        ):
            result = merge_supplements(
                main_df, ["mining"], "2024", "2024"
            )

        assert "mining_dm_time" in result.columns
        assert "mining_dm_jyuni" in result.columns
        assert "horse_win_rate" in result.columns
        assert len(result) == 2
        assert result["mining_dm_time"].iloc[0] == 123.4

    def test_merge_supplements_overlap_columns(self) -> None:
        """重複カラムがある場合のマージテスト（サプリメント側で上書き）."""
        from src.features.supplement import merge_supplements

        main_df = pd.DataFrame({
            "_key_year": ["2024"],
            "_key_monthday": ["0101"],
            "_key_jyocd": ["05"],
            "_key_kaiji": ["01"],
            "_key_nichiji": ["01"],
            "_key_racenum": ["01"],
            "kettonum": ["0000000001"],
            "mining_dm_time": [100.0],  # 古い値
        })

        supp_df = pd.DataFrame({
            "_key_year": ["2024"],
            "_key_monthday": ["0101"],
            "_key_jyocd": ["05"],
            "_key_kaiji": ["01"],
            "_key_nichiji": ["01"],
            "_key_racenum": ["01"],
            "kettonum": ["0000000001"],
            "mining_dm_time": [123.4],  # 新しい値
        })

        with patch(
            "src.features.supplement.load_supplement_years",
            return_value=supp_df,
        ):
            result = merge_supplements(
                main_df, ["mining"], "2024", "2024"
            )

        # サプリメント側の値で上書きされる
        assert result["mining_dm_time"].iloc[0] == 123.4

    def test_merge_supplements_missing_keys(self) -> None:
        """マージキーがない場合はスキップ."""
        from src.features.supplement import merge_supplements

        main_df = pd.DataFrame({
            "horse_win_rate": [0.5],
        })

        result = merge_supplements(main_df, ["mining"], "2024", "2024")
        # マージキーがないのでそのまま返される
        assert "mining_dm_time" not in result.columns

    def test_merge_supplements_empty_list(self) -> None:
        """空のサプリメントリストの場合はそのまま返す."""
        from src.features.supplement import merge_supplements

        main_df = pd.DataFrame({
            "kettonum": ["0000000001"],
            "horse_win_rate": [0.5],
        })

        result = merge_supplements(main_df, [], "2024", "2024")
        assert result.equals(main_df)

    def test_load_supplement_years_file_not_found(self) -> None:
        """parquet が見つからない場合."""
        from src.features.supplement import load_supplement_years

        with pytest.raises(FileNotFoundError, match="サプリメント"):
            load_supplement_years("mining", "9999", "9999")


# ============================================================
# パイプライン統合テスト
# ============================================================


class TestPipelineSupplementIntegration:
    """pipeline.py の load_years + supplement マージ統合テスト."""

    def test_load_years_with_supplements(self) -> None:
        """load_years に supplement_names を渡す統合テスト."""
        from src.features.pipeline import FeaturePipeline

        main_df = pd.DataFrame({
            "_key_year": ["2024"],
            "_key_monthday": ["0101"],
            "_key_jyocd": ["05"],
            "_key_kaiji": ["01"],
            "_key_nichiji": ["01"],
            "_key_racenum": ["01"],
            "kettonum": ["0000000001"],
            "horse_win_rate": [0.5],
        })

        supp_df = pd.DataFrame({
            "_key_year": ["2024"],
            "_key_monthday": ["0101"],
            "_key_jyocd": ["05"],
            "_key_kaiji": ["01"],
            "_key_nichiji": ["01"],
            "_key_racenum": ["01"],
            "kettonum": ["0000000001"],
            "mining_dm_time": [123.4],
        })

        # load_years 内部の pd.read_parquet と year_parquet_path をモック
        mock_path = MagicMock()
        mock_path.exists.return_value = True

        with (
            patch(
                "src.features.pipeline.year_parquet_path",
                return_value=mock_path,
            ),
            patch("pandas.read_parquet", return_value=main_df),
            patch(
                "src.features.supplement.load_supplement_years",
                return_value=supp_df,
            ),
        ):
            result = FeaturePipeline.load_years(
                "2024", "2024", supplement_names=["mining"]
            )

        assert "mining_dm_time" in result.columns
        assert "horse_win_rate" in result.columns
