"""オッズ歪み補正のテスト."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 依存パッケージが未インストールの場合はモックで代替
for _mod in (
    "lightgbm", "psycopg2", "psycopg2.extras",
    "sklearn", "sklearn.metrics",
):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ModuleNotFoundError:
            sys.modules[_mod] = MagicMock()

from src.config import DEFAULT_ODDS_CORRECTION_CONFIG
from src.model.evaluator import ModelEvaluator
from src.odds_correction_stats import (
    load_odds_correction_stats,
    save_odds_correction_stats,
)
from src.utils.code_master import class_level


class TestDeriveNinkiRank:
    """_derive_ninki_rank のテスト."""

    def test_basic_ranking(self) -> None:
        """オッズが低い順にランク1, 2, 3になる."""
        odds = {"01": 2.5, "02": 10.0, "03": 5.0}
        result = ModelEvaluator._derive_ninki_rank(odds)
        assert result == {"01": 1, "03": 2, "02": 3}

    def test_single_horse(self) -> None:
        """1頭のみの場合ランク1."""
        odds = {"05": 3.0}
        result = ModelEvaluator._derive_ninki_rank(odds)
        assert result == {"05": 1}

    def test_empty_dict(self) -> None:
        """空辞書は空辞書を返す."""
        result = ModelEvaluator._derive_ninki_rank({})
        assert result == {}

    def test_same_odds(self) -> None:
        """同一オッズでもランクは一意（順序は安定ソートに依存）."""
        odds = {"01": 5.0, "02": 5.0, "03": 5.0}
        result = ModelEvaluator._derive_ninki_rank(odds)
        assert len(result) == 3
        assert set(result.values()) == {1, 2, 3}

    def test_many_horses(self) -> None:
        """多頭数でも正しくランク付けされる."""
        odds = {f"{i:02d}": float(i * 2) for i in range(1, 19)}
        result = ModelEvaluator._derive_ninki_rank(odds)
        assert result["01"] == 1
        assert result["18"] == 18


class TestApplyOddsCorrection:
    """_apply_odds_correction のテスト."""

    def setup_method(self) -> None:
        self.evaluator = ModelEvaluator()
        self.enabled_config: dict = {
            "enabled": True,
            "rules": DEFAULT_ODDS_CORRECTION_CONFIG["rules"],
        }

    def _make_row(self, **kwargs) -> pd.Series:
        """テスト用の特徴量行を作成."""
        defaults = {
            "jockey_win_rate_year": 0.0,
            "horse_last_jyuni": 99,
            "post_umaban": 1,
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_disabled_config_returns_original(self) -> None:
        """enabled=False のとき補正なし."""
        config = {"enabled": False, "rules": {}}
        row = self._make_row()
        result = self.evaluator._apply_odds_correction(10.0, row, 1, config)
        assert result == 10.0

    def test_none_config_returns_original(self) -> None:
        """config が空辞書のとき補正なし."""
        row = self._make_row()
        result = self.evaluator._apply_odds_correction(10.0, row, 1, {})
        assert result == 10.0

    def test_jockey_popular_discount_applied(self) -> None:
        """人気騎手×人気馬で割引が適用される."""
        row = self._make_row(
            jockey_win_rate_year=0.20,  # >= 0.15
            post_umaban=2,  # 偶数 → even_gate_boost も適用
        )
        ninki_rank = 2  # <= 3
        result = self.evaluator._apply_odds_correction(
            10.0, row, ninki_rank, self.enabled_config,
        )
        # jockey_popular_discount(0.90) × even_gate_boost(1.03) = 0.927
        expected = 10.0 * 0.90 * 1.03
        assert abs(result - expected) < 0.01

    def test_jockey_not_popular_enough(self) -> None:
        """騎手勝率が閾値未満なら騎手ルール不適用."""
        row = self._make_row(
            jockey_win_rate_year=0.10,  # < 0.15
            post_umaban=2,
        )
        ninki_rank = 1
        result = self.evaluator._apply_odds_correction(
            10.0, row, ninki_rank, self.enabled_config,
        )
        # 騎手割引なし、偶数ゲートのみ
        expected = 10.0 * 1.03
        assert abs(result - expected) < 0.01

    def test_jockey_not_ninki_enough(self) -> None:
        """人気順が閾値超なら騎手ルール不適用."""
        row = self._make_row(
            jockey_win_rate_year=0.20,
            post_umaban=2,
        )
        ninki_rank = 5  # > 3
        result = self.evaluator._apply_odds_correction(
            10.0, row, ninki_rank, self.enabled_config,
        )
        # 騎手割引なし、偶数ゲートのみ
        expected = 10.0 * 1.03
        assert abs(result - expected) < 0.01

    def test_form_popular_discount_applied(self) -> None:
        """前走好走×人気馬で割引が適用される."""
        row = self._make_row(
            horse_last_jyuni=2,  # <= 3
            post_umaban=2,
        )
        ninki_rank = 1  # <= 3
        result = self.evaluator._apply_odds_correction(
            10.0, row, ninki_rank, self.enabled_config,
        )
        # form_popular_discount(0.92) × even_gate_boost(1.03) = 0.9476
        expected = 10.0 * 0.92 * 1.03
        assert abs(result - expected) < 0.01

    def test_form_not_recent_winner(self) -> None:
        """前走好走でなければフォーム割引なし."""
        row = self._make_row(
            horse_last_jyuni=5,  # > 3
            post_umaban=2,
        )
        ninki_rank = 1
        result = self.evaluator._apply_odds_correction(
            10.0, row, ninki_rank, self.enabled_config,
        )
        expected = 10.0 * 1.03  # 偶数ゲートのみ
        assert abs(result - expected) < 0.01

    def test_odd_gate_discount(self) -> None:
        """奇数ゲートで割引が適用される."""
        row = self._make_row(post_umaban=3)
        ninki_rank = 10  # 人気なし
        result = self.evaluator._apply_odds_correction(
            10.0, row, ninki_rank, self.enabled_config,
        )
        expected = 10.0 * 0.97
        assert abs(result - expected) < 0.01

    def test_even_gate_boost(self) -> None:
        """偶数ゲートで上乗せが適用される."""
        row = self._make_row(post_umaban=4)
        ninki_rank = 10
        result = self.evaluator._apply_odds_correction(
            10.0, row, ninki_rank, self.enabled_config,
        )
        expected = 10.0 * 1.03
        assert abs(result - expected) < 0.01

    def test_multiple_rules_compound(self) -> None:
        """複数ルールが重複適用される（乗算）."""
        row = self._make_row(
            jockey_win_rate_year=0.20,  # 騎手ルール
            horse_last_jyuni=1,         # フォームルール
            post_umaban=3,              # 奇数ゲート
        )
        ninki_rank = 1  # 1番人気
        result = self.evaluator._apply_odds_correction(
            10.0, row, ninki_rank, self.enabled_config,
        )
        # 0.90 × 0.92 × 0.97 = 0.80316
        expected = 10.0 * 0.90 * 0.92 * 0.97
        assert abs(result - expected) < 0.01

    def test_no_rules_matching(self) -> None:
        """どのルールにも該当しない場合は補正なし."""
        row = self._make_row(
            jockey_win_rate_year=0.05,
            horse_last_jyuni=10,
            post_umaban=0,  # 無効値
        )
        ninki_rank = 15
        result = self.evaluator._apply_odds_correction(
            10.0, row, ninki_rank, self.enabled_config,
        )
        assert result == 10.0

    def test_custom_factor_values(self) -> None:
        """カスタム factor 値でも正しく動作する."""
        config: dict = {
            "enabled": True,
            "rules": {
                "odd_gate_discount": {"factor": 0.85},
            },
        }
        row = self._make_row(post_umaban=1)
        result = self.evaluator._apply_odds_correction(
            10.0, row, 5, config,
        )
        expected = 10.0 * 0.85
        assert abs(result - expected) < 0.01

    def test_empty_rules(self) -> None:
        """ルール辞書が空のとき補正なし."""
        config: dict = {"enabled": True, "rules": {}}
        row = self._make_row(post_umaban=1)
        result = self.evaluator._apply_odds_correction(10.0, row, 1, config)
        assert result == 10.0


class TestDefaultOddsCorrectionConfig:
    """DEFAULT_ODDS_CORRECTION_CONFIG の構造テスト."""

    def test_disabled_by_default(self) -> None:
        """デフォルトで無効."""
        assert DEFAULT_ODDS_CORRECTION_CONFIG["enabled"] is False

    def test_has_all_expected_rules(self) -> None:
        """想定するルールが全て定義されている."""
        rules = DEFAULT_ODDS_CORRECTION_CONFIG["rules"]
        assert "jockey_popular_discount" in rules
        assert "form_popular_discount" in rules
        assert "odd_gate_discount" in rules
        assert "even_gate_boost" in rules
        assert "class_upgrade" in rules
        assert "class_downgrade" in rules
        assert "filly_to_mixed" in rules
        assert "mixed_to_filly" in rules

    def test_factors_are_reasonable(self) -> None:
        """factor 値が妥当な範囲内."""
        rules = DEFAULT_ODDS_CORRECTION_CONFIG["rules"]
        for name, rule in rules.items():
            factor = rule["factor"]
            assert 0.5 <= factor <= 1.5, f"{name}: factor {factor} out of range"

    def test_discount_factors_below_one(self) -> None:
        """割引ルールの factor は 1.0 未満."""
        rules = DEFAULT_ODDS_CORRECTION_CONFIG["rules"]
        assert rules["jockey_popular_discount"]["factor"] < 1.0
        assert rules["form_popular_discount"]["factor"] < 1.0
        assert rules["odd_gate_discount"]["factor"] < 1.0

    def test_boost_factors_above_one(self) -> None:
        """上乗せルールの factor は 1.0 超."""
        rules = DEFAULT_ODDS_CORRECTION_CONFIG["rules"]
        assert rules["even_gate_boost"]["factor"] > 1.0


class TestNinkiTableCorrection:
    """ninki_table による人気順別補正のテスト."""

    def setup_method(self) -> None:
        self.evaluator = ModelEvaluator()

    def _make_row(self, **kwargs) -> pd.Series:
        defaults = {
            "jockey_win_rate_year": 0.0,
            "horse_last_jyuni": 99,
            "post_umaban": 0,
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_ninki_table_applied(self) -> None:
        """ninki_table の factor が適用される."""
        config: dict = {
            "enabled": True,
            "ninki_table": {1: 0.85, 2: 0.90, 3: 1.10},
            "rules": {},
        }
        row = self._make_row()
        result = self.evaluator._apply_odds_correction(10.0, row, 1, config)
        assert abs(result - 10.0 * 0.85) < 0.01

    def test_ninki_table_unpopular_boost(self) -> None:
        """人気薄の factor > 1.0 で上乗せされる."""
        config: dict = {
            "enabled": True,
            "ninki_table": {10: 1.25, 15: 1.40},
            "rules": {},
        }
        row = self._make_row()
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert abs(result - 10.0 * 1.25) < 0.01

    def test_ninki_table_missing_rank(self) -> None:
        """テーブルにない人気順は factor=1.0（補正なし）."""
        config: dict = {
            "enabled": True,
            "ninki_table": {1: 0.85},
            "rules": {},
        }
        row = self._make_row()
        result = self.evaluator._apply_odds_correction(10.0, row, 5, config)
        assert result == 10.0

    def test_ninki_table_empty(self) -> None:
        """ninki_table が空でも動作する."""
        config: dict = {
            "enabled": True,
            "ninki_table": {},
            "rules": {},
        }
        row = self._make_row()
        result = self.evaluator._apply_odds_correction(10.0, row, 1, config)
        assert result == 10.0

    def test_ninki_table_combined_with_rules(self) -> None:
        """ninki_table と個別ルールが乗算される."""
        config: dict = {
            "enabled": True,
            "ninki_table": {1: 0.90},
            "rules": {
                "odd_gate_discount": {"factor": 0.95},
            },
        }
        row = self._make_row(post_umaban=3)  # 奇数ゲート
        result = self.evaluator._apply_odds_correction(10.0, row, 1, config)
        expected = 10.0 * 0.90 * 0.95
        assert abs(result - expected) < 0.01

    def test_ninki_table_no_ninki_table_key(self) -> None:
        """config に ninki_table キーがなくても動作する."""
        config: dict = {
            "enabled": True,
            "rules": {"odd_gate_discount": {"factor": 0.97}},
        }
        row = self._make_row(post_umaban=1)
        result = self.evaluator._apply_odds_correction(10.0, row, 1, config)
        expected = 10.0 * 0.97
        assert abs(result - expected) < 0.01


class TestSaveLoadOddsCorrectionStats:
    """save / load のテスト."""

    def _make_sample_stats(self) -> dict:
        return {
            "generated_at": "2026-02-21T12:00:00",
            "period": {"start": "2022", "end": "2024"},
            "baseline_roi": 0.775,
            "baseline_samples": 100000,
            "min_samples": 1000,
            "ninki_table": {
                "1": {"factor": 0.85, "samples": 5000, "roi": 0.659},
                "2": {"factor": 0.90, "samples": 5000, "roi": 0.698},
                "10": {"factor": 1.20, "samples": 3000, "roi": 0.930},
            },
            "rules": {
                "jockey_popular_discount": {
                    "jockey_win_rate_threshold": 0.15,
                    "ninki_threshold": 3,
                    "factor": 0.87,
                    "samples": 2000,
                    "roi": 0.674,
                },
                "form_popular_discount": {
                    "last_jyuni_threshold": 3,
                    "ninki_threshold": 3,
                    "factor": 0.91,
                    "samples": 3000,
                    "roi": 0.705,
                },
                "odd_gate_discount": {
                    "factor": 0.97,
                    "samples": 50000,
                    "roi": 0.752,
                },
                "even_gate_boost": {
                    "factor": 1.03,
                    "samples": 50000,
                    "roi": 0.798,
                },
            },
        }

    def test_save_and_load_roundtrip(self) -> None:
        """save → load のラウンドトリップが正しい."""
        stats = self._make_sample_stats()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_odds_correction_stats(stats, path)
            config = load_odds_correction_stats(path)

            assert config["enabled"] is True
            assert 1 in config["ninki_table"]
            assert 2 in config["ninki_table"]
            assert 10 in config["ninki_table"]
            assert abs(config["ninki_table"][1] - 0.85) < 0.001
            assert abs(config["ninki_table"][10] - 1.20) < 0.001
            assert "jockey_popular_discount" in config["rules"]
            assert abs(config["rules"]["jockey_popular_discount"]["factor"] - 0.87) < 0.001
        finally:
            path.unlink(missing_ok=True)

    def test_load_nonexistent_raises(self) -> None:
        """存在しないファイルで FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_odds_correction_stats(Path("/nonexistent/path.json"))

    def test_load_ninki_table_int_keys(self) -> None:
        """JSON文字列キーが int に変換される."""
        stats = self._make_sample_stats()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_odds_correction_stats(stats, path)
            config = load_odds_correction_stats(path)

            for key in config["ninki_table"]:
                assert isinstance(key, int), f"key {key} should be int"
        finally:
            path.unlink(missing_ok=True)

    def test_load_config_structure(self) -> None:
        """ロードした config が evaluator の期待形式に合致する."""
        stats = self._make_sample_stats()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_odds_correction_stats(stats, path)
            config = load_odds_correction_stats(path)

            # 必須キー
            assert "enabled" in config
            assert "ninki_table" in config
            assert "rules" in config
            # evaluator が使う形式
            assert isinstance(config["ninki_table"], dict)
            assert isinstance(config["rules"], dict)
        finally:
            path.unlink(missing_ok=True)

    def test_saved_json_readable(self) -> None:
        """保存されたJSONが正しいフォーマット."""
        stats = self._make_sample_stats()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_odds_correction_stats(stats, path)
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)

            assert loaded["baseline_roi"] == 0.775
            assert loaded["period"]["start"] == "2022"
            assert "1" in loaded["ninki_table"]
        finally:
            path.unlink(missing_ok=True)

    def test_style_table_roundtrip(self) -> None:
        """style_table が正しく save/load される."""
        stats = self._make_sample_stats()
        stats["style_table"] = {
            "1": {"factor": 1.05, "samples": 8000, "roi": 0.81},
            "3": {"factor": 0.95, "samples": 50000, "roi": 0.74},
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_odds_correction_stats(stats, path)
            config = load_odds_correction_stats(path)

            assert "style_table" in config
            assert abs(config["style_table"]["1"] - 1.05) < 0.001
            assert abs(config["style_table"]["3"] - 0.95) < 0.001
        finally:
            path.unlink(missing_ok=True)

    def test_post_course_table_roundtrip(self) -> None:
        """post_course_table が正しく save/load される."""
        stats = self._make_sample_stats()
        stats["post_course_table"] = {
            "inner_turf_left": {"factor": 1.08, "samples": 12000, "roi": 0.84},
            "outer_niigata_straight": {"factor": 1.15, "samples": 1500, "roi": 0.89},
            "inner": {"factor": 1.06, "samples": 45000, "roi": 0.82},
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_odds_correction_stats(stats, path)
            config = load_odds_correction_stats(path)

            assert "post_course_table" in config
            assert abs(config["post_course_table"]["inner_turf_left"] - 1.08) < 0.001
            assert abs(config["post_course_table"]["inner"] - 1.06) < 0.001
        finally:
            path.unlink(missing_ok=True)

    def test_new_rules_roundtrip(self) -> None:
        """v2 ルールが正しく save/load される."""
        stats = self._make_sample_stats()
        stats["rules"]["class_upgrade"] = {
            "factor": 0.92, "samples": 4500, "roi": 0.71,
        }
        stats["rules"]["filly_to_mixed"] = {
            "factor": 0.90, "samples": 3200, "roi": 0.70,
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_odds_correction_stats(stats, path)
            config = load_odds_correction_stats(path)

            assert abs(config["rules"]["class_upgrade"]["factor"] - 0.92) < 0.001
            assert abs(config["rules"]["filly_to_mixed"]["factor"] - 0.90) < 0.001
        finally:
            path.unlink(missing_ok=True)

    def test_load_without_v2_tables(self) -> None:
        """v2テーブルがないJSONでもエラーなくロードできる（後方互換）."""
        stats = self._make_sample_stats()
        # style_table, post_course_table がない旧形式
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_odds_correction_stats(stats, path)
            config = load_odds_correction_stats(path)

            assert config["style_table"] == {}
            assert config["post_course_table"] == {}
        finally:
            path.unlink(missing_ok=True)


# =====================================================================
# v2: 脚質テーブル補正テスト
# =====================================================================


class TestStyleTableCorrection:
    """style_table による前走脚質別補正のテスト."""

    def setup_method(self) -> None:
        self.evaluator = ModelEvaluator()

    def _make_row(self, **kwargs) -> pd.Series:
        defaults = {
            "jockey_win_rate_year": 0.0,
            "horse_last_jyuni": 99,
            "post_umaban": 0,
            "style_type_last": "0",
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_style_table_applied(self) -> None:
        """style_table の factor が適用される."""
        config: dict = {
            "enabled": True,
            "style_table": {"3": 0.95},
            "rules": {},
        }
        row = self._make_row(style_type_last="3")
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert abs(result - 10.0 * 0.95) < 0.01

    def test_style_table_unknown_style(self) -> None:
        """脚質コード '0'（不明）は補正なし."""
        config: dict = {
            "enabled": True,
            "style_table": {"1": 1.05, "2": 1.02, "3": 0.98, "4": 0.93},
            "rules": {},
        }
        row = self._make_row(style_type_last="0")
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert result == 10.0

    def test_style_table_missing_style(self) -> None:
        """style_table にないコードは補正なし."""
        config: dict = {
            "enabled": True,
            "style_table": {"1": 1.05},
            "rules": {},
        }
        row = self._make_row(style_type_last="4")
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert result == 10.0

    def test_style_table_empty(self) -> None:
        """style_table が空でも動作する."""
        config: dict = {
            "enabled": True,
            "style_table": {},
            "rules": {},
        }
        row = self._make_row(style_type_last="1")
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert result == 10.0

    def test_style_table_combined_with_ninki(self) -> None:
        """ninki_table と style_table が乗算される."""
        config: dict = {
            "enabled": True,
            "ninki_table": {1: 0.90},
            "style_table": {"2": 1.05},
            "rules": {},
        }
        row = self._make_row(style_type_last="2")
        result = self.evaluator._apply_odds_correction(10.0, row, 1, config)
        expected = 10.0 * 0.90 * 1.05
        assert abs(result - expected) < 0.01


# =====================================================================
# v2: 馬番×コーステーブル補正テスト
# =====================================================================


class TestPostCourseTableCorrection:
    """post_course_table による馬番×コース別補正のテスト."""

    def setup_method(self) -> None:
        self.evaluator = ModelEvaluator()

    def _make_row(self, **kwargs) -> pd.Series:
        defaults = {
            "jockey_win_rate_year": 0.0,
            "horse_last_jyuni": 99,
            "post_umaban": 5,
            "race_jyo_cd": "05",
            "race_track_cd": "11",
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_post_course_exact_match(self) -> None:
        """詳細キー (mid_inner_turf_left) がマッチする."""
        config: dict = {
            "enabled": True,
            "post_course_table": {
                "mid_inner_turf_left": 1.08,
                "mid_inner": 1.01,
            },
            "rules": {},
        }
        row = self._make_row(post_umaban=5, race_jyo_cd="05", race_track_cd="11")
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert abs(result - 10.0 * 1.08) < 0.01

    def test_post_course_fallback_to_group(self) -> None:
        """詳細キーがない場合 post_group のみにフォールバック."""
        config: dict = {
            "enabled": True,
            "post_course_table": {
                "mid_inner": 1.03,
            },
            "rules": {},
        }
        row = self._make_row(post_umaban=5, race_jyo_cd="05", race_track_cd="11")
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert abs(result - 10.0 * 1.03) < 0.01

    def test_post_course_niigata_straight(self) -> None:
        """新潟直線コースが正しくカテゴリ化される."""
        config: dict = {
            "enabled": True,
            "post_course_table": {
                "outer_niigata_straight": 1.15,
                "outer": 0.95,
            },
            "rules": {},
        }
        row = self._make_row(
            post_umaban=12, race_jyo_cd="04", race_track_cd="10",
        )
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert abs(result - 10.0 * 1.15) < 0.01

    def test_post_course_replaces_gate_parity(self) -> None:
        """post_course_table がある場合 gate parity は適用されない."""
        config: dict = {
            "enabled": True,
            "post_course_table": {"inner_turf_left": 1.10},
            "rules": {
                "odd_gate_discount": {"factor": 0.80},
            },
        }
        # 馬番1 = inner, 奇数ゲート → gate parity ではなく post_course が適用される
        row = self._make_row(post_umaban=1, race_jyo_cd="05", race_track_cd="11")
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert abs(result - 10.0 * 1.10) < 0.01

    def test_legacy_gate_parity_fallback(self) -> None:
        """post_course_table がない場合 gate parity ルールにフォールバック."""
        config: dict = {
            "enabled": True,
            "rules": {
                "odd_gate_discount": {"factor": 0.97},
            },
        }
        row = self._make_row(post_umaban=3)
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert abs(result - 10.0 * 0.97) < 0.01

    def test_post_course_invalid_umaban(self) -> None:
        """馬番が無効値のとき補正なし."""
        config: dict = {
            "enabled": True,
            "post_course_table": {"inner_turf_left": 1.10},
            "rules": {},
        }
        row = self._make_row(post_umaban=0)
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert result == 10.0

    def test_post_group_classification(self) -> None:
        """_post_group が正しく分類する."""
        assert ModelEvaluator._post_group(1) == "inner"
        assert ModelEvaluator._post_group(3) == "inner"
        assert ModelEvaluator._post_group(4) == "mid_inner"
        assert ModelEvaluator._post_group(6) == "mid_inner"
        assert ModelEvaluator._post_group(7) == "mid_outer"
        assert ModelEvaluator._post_group(9) == "mid_outer"
        assert ModelEvaluator._post_group(10) == "outer"
        assert ModelEvaluator._post_group(18) == "outer"

    def test_course_category_classification(self) -> None:
        """_course_category が正しく分類する."""
        assert ModelEvaluator._course_category("04", "10") == "niigata_straight"
        assert ModelEvaluator._course_category("05", "10") == "turf_left"
        assert ModelEvaluator._course_category("05", "11") == "turf_left"
        assert ModelEvaluator._course_category("05", "12") == "turf_left"
        assert ModelEvaluator._course_category("06", "17") == "turf_right"
        assert ModelEvaluator._course_category("06", "18") == "turf_right"
        assert ModelEvaluator._course_category("05", "23") == "dirt_left"
        assert ModelEvaluator._course_category("06", "24") == "dirt_right"
        assert ModelEvaluator._course_category("05", "51") == "other"
        assert ModelEvaluator._course_category("05", "") == "other"


# =====================================================================
# v2: クラス変更補正テスト
# =====================================================================


class TestClassChangeCorrection:
    """クラス変更補正のテスト."""

    def setup_method(self) -> None:
        self.evaluator = ModelEvaluator()

    def _make_row(self, **kwargs) -> pd.Series:
        defaults = {
            "jockey_win_rate_year": 0.0,
            "horse_last_jyuni": 99,
            "post_umaban": 0,
            "cross_class_change": 0,
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_class_upgrade_discount(self) -> None:
        """cross_class_change > 0 で upgrade factor が適用される."""
        config: dict = {
            "enabled": True,
            "rules": {"class_upgrade": {"factor": 0.92}},
        }
        row = self._make_row(cross_class_change=1)
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert abs(result - 10.0 * 0.92) < 0.01

    def test_class_downgrade_boost(self) -> None:
        """cross_class_change < 0 で downgrade factor が適用される."""
        config: dict = {
            "enabled": True,
            "rules": {"class_downgrade": {"factor": 1.08}},
        }
        row = self._make_row(cross_class_change=-1)
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert abs(result - 10.0 * 1.08) < 0.01

    def test_class_same_no_correction(self) -> None:
        """cross_class_change == 0 は補正なし."""
        config: dict = {
            "enabled": True,
            "rules": {
                "class_upgrade": {"factor": 0.92},
                "class_downgrade": {"factor": 1.08},
            },
        }
        row = self._make_row(cross_class_change=0)
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert result == 10.0

    def test_class_change_missing_value(self) -> None:
        """cross_class_change が欠損のとき補正なし."""
        config: dict = {
            "enabled": True,
            "rules": {"class_upgrade": {"factor": 0.92}},
        }
        row = self._make_row()
        del row["cross_class_change"]
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert result == 10.0


# =====================================================================
# v2: 牝馬限定⇔混合遷移補正テスト
# =====================================================================


class TestFillyTransitionCorrection:
    """牝馬限定⇔混合の遷移補正のテスト."""

    def setup_method(self) -> None:
        self.evaluator = ModelEvaluator()

    def _make_row(self, **kwargs) -> pd.Series:
        defaults = {
            "jockey_win_rate_year": 0.0,
            "horse_last_jyuni": 99,
            "post_umaban": 0,
            "horse_sex": "2",
            "cross_prev_filly_only": 0,
            "cross_current_filly_only": 0,
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_filly_to_mixed_discount(self) -> None:
        """牝馬が限定→混合に移った場合 factor が適用される."""
        config: dict = {
            "enabled": True,
            "rules": {"filly_to_mixed": {"factor": 0.90}},
        }
        row = self._make_row(
            horse_sex="2",
            cross_prev_filly_only=1,
            cross_current_filly_only=0,
        )
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert abs(result - 10.0 * 0.90) < 0.01

    def test_mixed_to_filly_boost(self) -> None:
        """牝馬が混合→限定に移った場合 factor が適用される."""
        config: dict = {
            "enabled": True,
            "rules": {"mixed_to_filly": {"factor": 1.06}},
        }
        row = self._make_row(
            horse_sex="2",
            cross_prev_filly_only=0,
            cross_current_filly_only=1,
        )
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert abs(result - 10.0 * 1.06) < 0.01

    def test_non_filly_no_correction(self) -> None:
        """牡馬は牝馬遷移ルール不適用."""
        config: dict = {
            "enabled": True,
            "rules": {"filly_to_mixed": {"factor": 0.90}},
        }
        row = self._make_row(
            horse_sex="1",  # 牡馬
            cross_prev_filly_only=1,
            cross_current_filly_only=0,
        )
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert result == 10.0

    def test_filly_same_type_no_correction(self) -> None:
        """同タイプ間（限定→限定）は補正なし."""
        config: dict = {
            "enabled": True,
            "rules": {
                "filly_to_mixed": {"factor": 0.90},
                "mixed_to_filly": {"factor": 1.06},
            },
        }
        row = self._make_row(
            horse_sex="2",
            cross_prev_filly_only=1,
            cross_current_filly_only=1,
        )
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert result == 10.0


# =====================================================================
# class_level ヘルパーテスト
# =====================================================================


class TestClassLevel:
    """class_level ヘルパー関数のテスト."""

    def test_grade_race(self) -> None:
        """重賞は 1000."""
        assert class_level("999", "A") == 1000
        assert class_level("001", "B") == 1000
        assert class_level("703", "C") == 1000
        assert class_level("", "D") == 1000

    def test_open(self) -> None:
        """オープンは 900."""
        assert class_level("999", "") == 900
        assert class_level("999", " ") == 900

    def test_maiden(self) -> None:
        """新馬/未勝利は 100."""
        assert class_level("701", "") == 100
        assert class_level("702", "") == 100
        assert class_level("703", "") == 100

    def test_condition_race(self) -> None:
        """条件戦は jyokencd5 + 100."""
        assert class_level("001", "") == 101
        assert class_level("050", "") == 150
        assert class_level("100", "") == 200

    def test_invalid(self) -> None:
        """無効な値は -1."""
        assert class_level("", "") == -1
        assert class_level("abc", "") == -1
        assert class_level("500", "") == -1  # 101-200 の範囲外


# =====================================================================
# v3: 父系統×サーフェス別テーブル補正テスト
# =====================================================================


class TestSireSurfaceTableCorrection:
    """sire_surface_table による父系統×サーフェス別補正のテスト."""

    def setup_method(self) -> None:
        self.evaluator = ModelEvaluator()

    def _make_row(self, **kwargs) -> pd.Series:
        defaults = {
            "jockey_win_rate_year": 0.0,
            "horse_last_jyuni": 99,
            "post_umaban": 0,
            "blood_father_keito": "サンデーサイレンス",
            "race_track_cd": "11",
            "race_jyo_cd": "05",
            "race_distance": 1600,
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_sire_surface_factor_applied(self) -> None:
        """父系統×サーフェスのfactorが正しく乗算されること."""
        config: dict = {
            "enabled": True,
            "sire_surface_table": {"サンデーサイレンス_siba": 1.05},
            "rules": {},
        }
        row = self._make_row(
            blood_father_keito="サンデーサイレンス",
            race_track_cd="11",
            race_jyo_cd="05",
        )
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert abs(result - 10.0 * 1.05) < 0.01

    def test_sire_surface_missing_key_no_effect(self) -> None:
        """テーブルにキーがない場合 factor=1.0 であること."""
        config: dict = {
            "enabled": True,
            "sire_surface_table": {"キングカメハメハ_dirt": 0.88},
            "rules": {},
        }
        row = self._make_row(
            blood_father_keito="サンデーサイレンス",
            race_track_cd="11",
            race_jyo_cd="05",
        )
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert result == 10.0

    def test_sire_surface_empty_table_no_effect(self) -> None:
        """sire_surface_table が空辞書の場合、補正なしであること."""
        config: dict = {
            "enabled": True,
            "sire_surface_table": {},
            "rules": {},
        }
        row = self._make_row()
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert result == 10.0

    def test_sire_surface_yousiba_classification(self) -> None:
        """札幌(01)・函館(02)の芝コースが yousiba に分類されること."""
        config: dict = {
            "enabled": True,
            "sire_surface_table": {"サンデーサイレンス_yousiba": 1.12},
            "rules": {},
        }
        # 札幌 (01) + 芝コース (trackcd < 23)
        row = self._make_row(
            blood_father_keito="サンデーサイレンス",
            race_track_cd="11",
            race_jyo_cd="01",
        )
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert abs(result - 10.0 * 1.12) < 0.01

        # 函館 (02) + 芝コース
        row2 = self._make_row(
            blood_father_keito="サンデーサイレンス",
            race_track_cd="10",
            race_jyo_cd="02",
        )
        result2 = self.evaluator._apply_odds_correction(10.0, row2, 10, config)
        assert abs(result2 - 10.0 * 1.12) < 0.01

    def test_sire_surface_dirt_classification(self) -> None:
        """trackcd >= 23 が dirt に分類されること."""
        config: dict = {
            "enabled": True,
            "sire_surface_table": {"サンデーサイレンス_dirt": 0.88},
            "rules": {},
        }
        row = self._make_row(
            blood_father_keito="サンデーサイレンス",
            race_track_cd="23",
            race_jyo_cd="05",
        )
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert abs(result - 10.0 * 0.88) < 0.01

    def test_sire_surface_no_table_key_in_config(self) -> None:
        """config に sire_surface_table キーがなくても動作する."""
        config: dict = {
            "enabled": True,
            "rules": {},
        }
        row = self._make_row()
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert result == 10.0

    def test_sire_surface_combined_with_ninki(self) -> None:
        """ninki_table と sire_surface_table が乗算される."""
        config: dict = {
            "enabled": True,
            "ninki_table": {1: 0.90},
            "sire_surface_table": {"サンデーサイレンス_siba": 1.05},
            "rules": {},
        }
        row = self._make_row()
        result = self.evaluator._apply_odds_correction(10.0, row, 1, config)
        expected = 10.0 * 0.90 * 1.05
        assert abs(result - expected) < 0.01


# =====================================================================
# v3: 父系統×距離帯別テーブル補正テスト
# =====================================================================


class TestSireDistanceTableCorrection:
    """sire_distance_table による父系統×距離帯別補正のテスト."""

    def setup_method(self) -> None:
        self.evaluator = ModelEvaluator()

    def _make_row(self, **kwargs) -> pd.Series:
        defaults = {
            "jockey_win_rate_year": 0.0,
            "horse_last_jyuni": 99,
            "post_umaban": 0,
            "blood_father_keito": "サンデーサイレンス",
            "race_track_cd": "11",
            "race_jyo_cd": "05",
            "race_distance": 1600,
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_sire_distance_factor_applied(self) -> None:
        """父系統×距離帯のfactorが正しく乗算されること."""
        config: dict = {
            "enabled": True,
            "sire_distance_table": {"サンデーサイレンス_mile": 1.03},
            "rules": {},
        }
        row = self._make_row(race_distance=1600)
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert abs(result - 10.0 * 1.03) < 0.01

    def test_sire_distance_boundary_1400(self) -> None:
        """距離1400mが sprint に分類されること."""
        config: dict = {
            "enabled": True,
            "sire_distance_table": {"サンデーサイレンス_sprint": 0.95},
            "rules": {},
        }
        row = self._make_row(race_distance=1400)
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert abs(result - 10.0 * 0.95) < 0.01

    def test_sire_distance_boundary_1401(self) -> None:
        """距離1401mが mile に分類されること."""
        config: dict = {
            "enabled": True,
            "sire_distance_table": {"サンデーサイレンス_mile": 1.03},
            "rules": {},
        }
        row = self._make_row(race_distance=1401)
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert abs(result - 10.0 * 1.03) < 0.01

    def test_sire_distance_boundary_2200(self) -> None:
        """距離2200mが middle に分類されること."""
        config: dict = {
            "enabled": True,
            "sire_distance_table": {"サンデーサイレンス_middle": 1.08},
            "rules": {},
        }
        row = self._make_row(race_distance=2200)
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert abs(result - 10.0 * 1.08) < 0.01

    def test_sire_distance_boundary_2201(self) -> None:
        """距離2201mが long に分類されること."""
        config: dict = {
            "enabled": True,
            "sire_distance_table": {"サンデーサイレンス_long": 1.15},
            "rules": {},
        }
        row = self._make_row(race_distance=2201)
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert abs(result - 10.0 * 1.15) < 0.01

    def test_sire_distance_missing_key_no_effect(self) -> None:
        """テーブルにキーがない場合 factor=1.0 であること."""
        config: dict = {
            "enabled": True,
            "sire_distance_table": {"キングカメハメハ_sprint": 0.90},
            "rules": {},
        }
        row = self._make_row(
            blood_father_keito="サンデーサイレンス",
            race_distance=1200,
        )
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert result == 10.0

    def test_sire_distance_empty_table_no_effect(self) -> None:
        """sire_distance_table が空辞書の場合、補正なしであること."""
        config: dict = {
            "enabled": True,
            "sire_distance_table": {},
            "rules": {},
        }
        row = self._make_row()
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert result == 10.0

    def test_sire_distance_no_table_key_in_config(self) -> None:
        """config に sire_distance_table キーがなくても動作する."""
        config: dict = {
            "enabled": True,
            "rules": {},
        }
        row = self._make_row()
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert result == 10.0

    def test_sire_surface_and_distance_combined(self) -> None:
        """sire_surface_table と sire_distance_table が両方乗算される."""
        config: dict = {
            "enabled": True,
            "sire_surface_table": {"サンデーサイレンス_siba": 1.05},
            "sire_distance_table": {"サンデーサイレンス_mile": 1.03},
            "rules": {},
        }
        row = self._make_row(
            blood_father_keito="サンデーサイレンス",
            race_track_cd="11",
            race_jyo_cd="05",
            race_distance=1600,
        )
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        expected = 10.0 * 1.05 * 1.03
        assert abs(result - expected) < 0.01

    def test_sire_distance_zero_distance(self) -> None:
        """距離が0の場合 sprint に分類される（エッジケース）."""
        config: dict = {
            "enabled": True,
            "sire_distance_table": {"サンデーサイレンス_sprint": 0.95},
            "rules": {},
        }
        row = self._make_row(race_distance=0)
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        assert abs(result - 10.0 * 0.95) < 0.01


# =====================================================================
# v3: save/load ラウンドトリップテスト（新テーブル）
# =====================================================================


class TestSireTableRoundtrip:
    """sire_surface_table / sire_distance_table の save/load テスト."""

    def _make_sample_stats(self) -> dict:
        return {
            "generated_at": "2026-02-21T12:00:00",
            "period": {"start": "2022", "end": "2024"},
            "baseline_roi": 0.775,
            "baseline_samples": 100000,
            "min_samples": 1000,
            "ninki_table": {
                "1": {"factor": 0.85, "samples": 5000, "roi": 0.659},
            },
            "style_table": {},
            "post_course_table": {},
            "sire_surface_table": {
                "サンデーサイレンス_siba": {
                    "factor": 1.05, "samples": 12000, "roi": 0.814,
                },
                "サンデーサイレンス_dirt": {
                    "factor": 0.88, "samples": 8000, "roi": 0.682,
                },
            },
            "sire_distance_table": {
                "サンデーサイレンス_sprint": {
                    "factor": 0.95, "samples": 10000, "roi": 0.736,
                },
                "ディープインパクト_long": {
                    "factor": 1.15, "samples": 3000, "roi": 0.891,
                },
            },
            "rules": {},
        }

    def test_sire_surface_table_roundtrip(self) -> None:
        """sire_surface_table が正しく save/load される."""
        stats = self._make_sample_stats()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_odds_correction_stats(stats, path)
            config = load_odds_correction_stats(path)

            assert "sire_surface_table" in config
            assert abs(config["sire_surface_table"]["サンデーサイレンス_siba"] - 1.05) < 0.001
            assert abs(config["sire_surface_table"]["サンデーサイレンス_dirt"] - 0.88) < 0.001
        finally:
            path.unlink(missing_ok=True)

    def test_sire_distance_table_roundtrip(self) -> None:
        """sire_distance_table が正しく save/load される."""
        stats = self._make_sample_stats()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_odds_correction_stats(stats, path)
            config = load_odds_correction_stats(path)

            assert "sire_distance_table" in config
            assert abs(config["sire_distance_table"]["サンデーサイレンス_sprint"] - 0.95) < 0.001
            assert abs(config["sire_distance_table"]["ディープインパクト_long"] - 1.15) < 0.001
        finally:
            path.unlink(missing_ok=True)

    def test_load_without_sire_tables(self) -> None:
        """sire テーブルがないJSONでもエラーなくロードできる（後方互換）."""
        stats = self._make_sample_stats()
        del stats["sire_surface_table"]
        del stats["sire_distance_table"]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_odds_correction_stats(stats, path)
            config = load_odds_correction_stats(path)

            assert config["sire_surface_table"] == {}
            assert config["sire_distance_table"] == {}
        finally:
            path.unlink(missing_ok=True)


# =====================================================================
# v3: 調教師×人気帯別テーブル補正テスト
# =====================================================================


class TestTrainerNinkiTableCorrection:
    """trainer_ninki_table による調教師×人気帯別補正のテスト."""

    def setup_method(self) -> None:
        self.evaluator = ModelEvaluator()

    def _make_row(self, **kwargs) -> pd.Series:
        defaults = {
            "jockey_win_rate_year": 0.0,
            "horse_last_jyuni": 99,
            "post_umaban": 0,
            "trainer_code": "01078",
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_trainer_ninki_factor_applied(self) -> None:
        """調教師×人気帯のfactorが正しく乗算されること."""
        config: dict = {
            "enabled": True,
            "trainer_ninki_table": {"01078_D": 1.25},
            "rules": {},
        }
        row = self._make_row(trainer_code="01078")
        # ninki_rank=12 → band D
        result = self.evaluator._apply_odds_correction(10.0, row, 12, config)
        assert abs(result - 10.0 * 1.25) < 0.01

    def test_trainer_ninki_band_A(self) -> None:
        """人気1-3がバンドAに分類されること."""
        config: dict = {
            "enabled": True,
            "trainer_ninki_table": {"01078_A": 0.90},
            "rules": {},
        }
        row = self._make_row(trainer_code="01078")
        result = self.evaluator._apply_odds_correction(10.0, row, 2, config)
        assert abs(result - 10.0 * 0.90) < 0.01

    def test_trainer_ninki_band_B(self) -> None:
        """人気4-6がバンドBに分類されること."""
        config: dict = {
            "enabled": True,
            "trainer_ninki_table": {"01078_B": 1.05},
            "rules": {},
        }
        row = self._make_row(trainer_code="01078")
        result = self.evaluator._apply_odds_correction(10.0, row, 5, config)
        assert abs(result - 10.0 * 1.05) < 0.01

    def test_trainer_ninki_band_C(self) -> None:
        """人気7-9がバンドCに分類されること."""
        config: dict = {
            "enabled": True,
            "trainer_ninki_table": {"01078_C": 1.15},
            "rules": {},
        }
        row = self._make_row(trainer_code="01078")
        result = self.evaluator._apply_odds_correction(10.0, row, 8, config)
        assert abs(result - 10.0 * 1.15) < 0.01

    def test_trainer_ninki_band_D(self) -> None:
        """人気10以上がバンドDに分類されること."""
        config: dict = {
            "enabled": True,
            "trainer_ninki_table": {"01078_D": 1.30},
            "rules": {},
        }
        row = self._make_row(trainer_code="01078")
        result = self.evaluator._apply_odds_correction(10.0, row, 15, config)
        assert abs(result - 10.0 * 1.30) < 0.01

    def test_trainer_ninki_missing_key_no_effect(self) -> None:
        """テーブルにキーがない場合 factor=1.0 であること."""
        config: dict = {
            "enabled": True,
            "trainer_ninki_table": {"01078_D": 1.25},
            "rules": {},
        }
        row = self._make_row(trainer_code="99999")  # 別の調教師
        result = self.evaluator._apply_odds_correction(10.0, row, 12, config)
        assert result == 10.0

    def test_trainer_ninki_empty_table_no_effect(self) -> None:
        """trainer_ninki_table が空辞書の場合、補正なしであること."""
        config: dict = {
            "enabled": True,
            "trainer_ninki_table": {},
            "rules": {},
        }
        row = self._make_row(trainer_code="01078")
        result = self.evaluator._apply_odds_correction(10.0, row, 12, config)
        assert result == 10.0

    def test_trainer_ninki_no_table_key_in_config(self) -> None:
        """config に trainer_ninki_table キーがなくても動作する."""
        config: dict = {
            "enabled": True,
            "rules": {},
        }
        row = self._make_row(trainer_code="01078")
        result = self.evaluator._apply_odds_correction(10.0, row, 12, config)
        assert result == 10.0

    def test_trainer_ninki_empty_trainer_code(self) -> None:
        """trainer_code が空の場合、補正なしであること."""
        config: dict = {
            "enabled": True,
            "trainer_ninki_table": {"01078_D": 1.25},
            "rules": {},
        }
        row = self._make_row(trainer_code="")
        result = self.evaluator._apply_odds_correction(10.0, row, 12, config)
        assert result == 10.0

    def test_trainer_ninki_combined_with_ninki_table(self) -> None:
        """ninki_table と trainer_ninki_table が乗算される."""
        config: dict = {
            "enabled": True,
            "ninki_table": {10: 1.20},
            "trainer_ninki_table": {"01078_D": 1.25},
            "rules": {},
        }
        row = self._make_row(trainer_code="01078")
        result = self.evaluator._apply_odds_correction(10.0, row, 10, config)
        expected = 10.0 * 1.20 * 1.25
        assert abs(result - expected) < 0.01

    def test_trainer_ninki_combined_with_sire(self) -> None:
        """sire_surface_table と trainer_ninki_table が乗算される."""
        config: dict = {
            "enabled": True,
            "sire_surface_table": {"サンデーサイレンス_siba": 1.05},
            "trainer_ninki_table": {"01078_C": 1.15},
            "rules": {},
        }
        row = self._make_row(
            trainer_code="01078",
            blood_father_keito="サンデーサイレンス",
            race_track_cd="11",
            race_jyo_cd="05",
            race_distance=1600,
        )
        result = self.evaluator._apply_odds_correction(10.0, row, 8, config)
        expected = 10.0 * 1.15 * 1.05
        assert abs(result - expected) < 0.01

    def test_trainer_ninki_discount_popular(self) -> None:
        """戦略的でない厩舎の人気馬は割引されること."""
        config: dict = {
            "enabled": True,
            "trainer_ninki_table": {"00999_A": 0.85},
            "rules": {},
        }
        row = self._make_row(trainer_code="00999")
        result = self.evaluator._apply_odds_correction(10.0, row, 1, config)
        assert abs(result - 10.0 * 0.85) < 0.01


class TestNinkiBandClassification:
    """_ninki_band 分類のテスト."""

    def test_band_A(self) -> None:
        """人気1-3 はバンド A."""
        assert ModelEvaluator._ninki_band(1) == "A"
        assert ModelEvaluator._ninki_band(2) == "A"
        assert ModelEvaluator._ninki_band(3) == "A"

    def test_band_B(self) -> None:
        """人気4-6 はバンド B."""
        assert ModelEvaluator._ninki_band(4) == "B"
        assert ModelEvaluator._ninki_band(5) == "B"
        assert ModelEvaluator._ninki_band(6) == "B"

    def test_band_C(self) -> None:
        """人気7-9 はバンド C."""
        assert ModelEvaluator._ninki_band(7) == "C"
        assert ModelEvaluator._ninki_band(8) == "C"
        assert ModelEvaluator._ninki_band(9) == "C"

    def test_band_D(self) -> None:
        """人気10+ はバンド D."""
        assert ModelEvaluator._ninki_band(10) == "D"
        assert ModelEvaluator._ninki_band(18) == "D"
        assert ModelEvaluator._ninki_band(99) == "D"


class TestTrainerNinkiTableRoundtrip:
    """trainer_ninki_table の save/load テスト."""

    def _make_sample_stats(self) -> dict:
        return {
            "generated_at": "2026-02-21T12:00:00",
            "period": {"start": "2022", "end": "2024"},
            "baseline_roi": 0.775,
            "baseline_samples": 100000,
            "min_samples": 1000,
            "ninki_table": {
                "1": {"factor": 0.85, "samples": 5000, "roi": 0.659},
            },
            "style_table": {},
            "post_course_table": {},
            "sire_surface_table": {},
            "sire_distance_table": {},
            "trainer_ninki_table": {
                "01078_C": {
                    "factor": 1.25, "samples": 500, "roi": 0.969,
                },
                "01078_D": {
                    "factor": 1.40, "samples": 350, "roi": 1.085,
                },
                "01180_A": {
                    "factor": 0.88, "samples": 200, "roi": 0.682,
                },
            },
            "rules": {},
        }

    def test_trainer_ninki_table_roundtrip(self) -> None:
        """trainer_ninki_table が正しく save/load される."""
        stats = self._make_sample_stats()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_odds_correction_stats(stats, path)
            config = load_odds_correction_stats(path)

            assert "trainer_ninki_table" in config
            assert abs(config["trainer_ninki_table"]["01078_C"] - 1.25) < 0.001
            assert abs(config["trainer_ninki_table"]["01078_D"] - 1.40) < 0.001
            assert abs(config["trainer_ninki_table"]["01180_A"] - 0.88) < 0.001
        finally:
            path.unlink(missing_ok=True)

    def test_load_without_trainer_ninki_table(self) -> None:
        """trainer_ninki_table がないJSONでもエラーなくロードできる（後方互換）."""
        stats = self._make_sample_stats()
        del stats["trainer_ninki_table"]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_odds_correction_stats(stats, path)
            config = load_odds_correction_stats(path)

            assert config["trainer_ninki_table"] == {}
        finally:
            path.unlink(missing_ok=True)
