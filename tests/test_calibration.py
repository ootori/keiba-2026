"""確率キャリブレーションのテスト.

DB接続不要のユニットテスト。

実行方法:
    pytest tests/test_calibration.py -v
"""

from __future__ import annotations

import json
import pickle
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestCalibratorFit:
    """キャリブレータの学習テスト."""

    def test_fit_calibrator_returns_isotonic(self) -> None:
        """_fit_calibrator が IsotonicRegression を返す."""
        from sklearn.isotonic import IsotonicRegression
        from src.model.trainer import ModelTrainer

        trainer = ModelTrainer(calibrate=True)
        trainer.model = _make_mock_booster()

        X_valid = pd.DataFrame({"f1": [0.1, 0.5, 0.9, 0.3, 0.7]})
        y_valid = pd.Series([0, 0, 1, 0, 1])

        calibrator = trainer._fit_calibrator(X_valid, y_valid)
        assert isinstance(calibrator, IsotonicRegression)

    def test_calibrator_output_range(self) -> None:
        """キャリブレーション後の出力が [0, 1] に収まる."""
        from src.model.trainer import ModelTrainer

        trainer = ModelTrainer(calibrate=True)
        trainer.model = _make_mock_booster()

        X_valid = pd.DataFrame({"f1": np.linspace(0, 1, 100)})
        y_valid = pd.Series((np.linspace(0, 1, 100) > 0.5).astype(int))

        calibrator = trainer._fit_calibrator(X_valid, y_valid)
        raw = trainer.model.predict(X_valid)
        calibrated = calibrator.predict(raw)

        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)

    def test_calibrator_monotonic(self) -> None:
        """Isotonic Regression の出力が単調非減少."""
        from src.model.trainer import ModelTrainer

        trainer = ModelTrainer(calibrate=True)
        trainer.model = _make_mock_booster()

        X_valid = pd.DataFrame({"f1": np.linspace(0, 1, 200)})
        y_valid = pd.Series((np.linspace(0, 1, 200) > 0.4).astype(int))

        calibrator = trainer._fit_calibrator(X_valid, y_valid)
        raw_sorted = np.sort(trainer.model.predict(X_valid))
        calibrated = calibrator.predict(raw_sorted)

        diffs = np.diff(calibrated)
        assert np.all(diffs >= -1e-10)  # 単調非減少（浮動小数点誤差許容）


class TestCalibratorSaveLoad:
    """キャリブレータの保存・読み込みテスト."""

    def test_save_creates_calibrator_file(self) -> None:
        """save_model でキャリブレータファイルが作成される."""
        from src.model.trainer import ModelTrainer
        from sklearn.isotonic import IsotonicRegression

        trainer = ModelTrainer(calibrate=True)
        trainer.model = _make_mock_booster()

        X_valid = pd.DataFrame({"f1": np.linspace(0, 1, 50)})
        y_valid = pd.Series((np.linspace(0, 1, 50) > 0.5).astype(int))
        trainer.calibrator = trainer._fit_calibrator(X_valid, y_valid)
        trainer.feature_columns = ["f1"]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.model.trainer.MODEL_DIR", Path(tmpdir)):
                trainer.save_model(name="test_model")

                cal_path = Path(tmpdir) / "test_model_calibrator.pkl"
                assert cal_path.exists()

                # pkl ファイルを読み込めるか確認
                with open(cal_path, "rb") as f:
                    loaded = pickle.load(f)
                assert isinstance(loaded, IsotonicRegression)

    def test_save_meta_includes_calibrated_flag(self) -> None:
        """メタデータに calibrated フラグが含まれる."""
        from src.model.trainer import ModelTrainer

        trainer = ModelTrainer(calibrate=True)
        trainer.model = _make_mock_booster()

        X_valid = pd.DataFrame({"f1": np.linspace(0, 1, 50)})
        y_valid = pd.Series((np.linspace(0, 1, 50) > 0.5).astype(int))
        trainer.calibrator = trainer._fit_calibrator(X_valid, y_valid)
        trainer.feature_columns = ["f1"]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.model.trainer.MODEL_DIR", Path(tmpdir)):
                trainer.save_model(name="test_model")

                meta_path = Path(tmpdir) / "test_model_meta.json"
                with open(meta_path) as f:
                    meta = json.load(f)
                assert meta["calibrated"] is True

    def test_no_calibrator_meta_false(self) -> None:
        """キャリブレータなし時に calibrated=false."""
        from src.model.trainer import ModelTrainer

        trainer = ModelTrainer(calibrate=False)
        trainer.model = _make_mock_booster()
        trainer.feature_columns = ["f1"]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.model.trainer.MODEL_DIR", Path(tmpdir)):
                trainer.save_model(name="test_model")

                meta_path = Path(tmpdir) / "test_model_meta.json"
                with open(meta_path) as f:
                    meta = json.load(f)
                assert meta["calibrated"] is False

                # キャリブレータファイルは作成されない
                cal_path = Path(tmpdir) / "test_model_calibrator.pkl"
                assert not cal_path.exists()

    def test_load_restores_calibrator(self) -> None:
        """load_model でキャリブレータが復元される."""
        from src.model.trainer import ModelTrainer

        trainer = ModelTrainer(calibrate=True)
        trainer.model = _make_mock_booster()

        X_valid = pd.DataFrame({"f1": np.linspace(0, 1, 50)})
        y_valid = pd.Series((np.linspace(0, 1, 50) > 0.5).astype(int))
        trainer.calibrator = trainer._fit_calibrator(X_valid, y_valid)
        trainer.feature_columns = ["f1"]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.model.trainer.MODEL_DIR", Path(tmpdir)):
                trainer.save_model(name="test_model")

                # 新しいトレーナーでロード（lgb.Booster をモック）
                trainer2 = ModelTrainer()
                with patch("src.model.trainer.lgb.Booster", return_value=_make_mock_booster()):
                    trainer2.load_model(name="test_model")

                assert trainer2.calibrator is not None
                assert trainer2.calibrate is True

    def test_load_without_calibrator(self) -> None:
        """キャリブレータなしモデルのロード."""
        from src.model.trainer import ModelTrainer

        trainer = ModelTrainer(calibrate=False)
        trainer.model = _make_mock_booster()
        trainer.feature_columns = ["f1"]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.model.trainer.MODEL_DIR", Path(tmpdir)):
                trainer.save_model(name="test_model")

                trainer2 = ModelTrainer()
                with patch("src.model.trainer.lgb.Booster", return_value=_make_mock_booster()):
                    trainer2.load_model(name="test_model")

                assert trainer2.calibrator is None
                assert trainer2.calibrate is False


class TestEvaluatorCalibration:
    """evaluator.py のキャリブレーション対応テスト."""

    def test_evaluate_with_calibrator_adds_brier_scores(self) -> None:
        """キャリブレータ指定時に brier_score_calibrated が追加される."""
        from sklearn.isotonic import IsotonicRegression
        from src.model.evaluator import ModelEvaluator

        model = _make_mock_booster()
        calibrator = IsotonicRegression(out_of_bounds="clip")
        raw = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        labels = np.array([0, 0, 1, 0, 1])
        calibrator.fit(raw, labels)

        valid_df = pd.DataFrame({
            "f1": [0.2, 0.4, 0.6, 0.8, 0.95],
            "target": [0, 0, 1, 1, 1],
        })

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(
            model, valid_df, ["f1"],
            target_col="target",
            calibrator=calibrator,
        )

        assert "brier_score_raw" in metrics
        assert "brier_score_calibrated" in metrics
        assert "logloss_calibrated" in metrics
        assert metrics["calibrated"] is True

    def test_evaluate_without_calibrator_no_extra_metrics(self) -> None:
        """キャリブレータなし時に calibrated メトリクスが追加されない."""
        from src.model.evaluator import ModelEvaluator

        model = _make_mock_booster()
        valid_df = pd.DataFrame({
            "f1": [0.2, 0.4, 0.6, 0.8, 0.95],
            "target": [0, 0, 1, 1, 1],
        })

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(
            model, valid_df, ["f1"],
            target_col="target",
            calibrator=None,
        )

        assert "brier_score_raw" in metrics
        assert "brier_score_calibrated" not in metrics
        assert "logloss_calibrated" not in metrics
        assert metrics["calibrated"] is False

    def test_evaluate_ranking_ignores_calibrator(self) -> None:
        """LambdaRank モードではキャリブレータが無視される."""
        from sklearn.isotonic import IsotonicRegression
        from src.model.evaluator import ModelEvaluator

        model = _make_mock_booster()
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit([0.1, 0.5, 0.9], [0, 0, 1])

        valid_df = pd.DataFrame({
            "f1": [0.2, 0.4, 0.6, 0.8, 0.95],
            "target": [0, 0, 1, 1, 1],
            "target_relevance": [0, 1, 3, 4, 5],
            "_key_year": ["2025"] * 5,
            "_key_monthday": ["0101"] * 5,
            "_key_jyocd": ["01"] * 5,
            "_key_kaiji": ["01"] * 5,
            "_key_nichiji": ["01"] * 5,
            "_key_racenum": ["01"] * 5,
        })

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(
            model, valid_df, ["f1"],
            target_col="target",
            ranking=True,
            calibrator=calibrator,
        )

        # ranking モードでは brier_score 系は出力されない
        assert "brier_score_raw" not in metrics
        assert "brier_score_calibrated" not in metrics


class TestSimulateReturnCalibration:
    """simulate_return のキャリブレーション対応テスト."""

    def test_simulate_passes_calibrator(self) -> None:
        """simulate_return がキャリブレータを適用する."""
        from sklearn.isotonic import IsotonicRegression
        from src.model.evaluator import ModelEvaluator

        model = _make_mock_booster()
        calibrator = IsotonicRegression(out_of_bounds="clip")
        raw = np.linspace(0.05, 0.95, 20)
        labels = (raw > 0.5).astype(int)
        calibrator.fit(raw, labels)

        valid_df = _make_race_df()

        evaluator = ModelEvaluator()

        # キャリブレータなしで予測
        with patch.object(evaluator, "_get_harai_data", return_value={}):
            result_no_cal = evaluator.simulate_return(
                valid_df, ["f1"], model,
                strategy="top1_tansho",
                calibrator=None,
            )

        # キャリブレータありで予測（ランキングは変わらないかもしれないが
        # value_bet のEV計算には影響する）
        with patch.object(evaluator, "_get_harai_data", return_value={}):
            result_cal = evaluator.simulate_return(
                valid_df, ["f1"], model,
                strategy="top1_tansho",
                calibrator=calibrator,
            )

        # 両方とも正常に完了する
        assert "return_rate" in result_no_cal
        assert "return_rate" in result_cal


# ============================================================
# ヘルパー
# ============================================================

def _make_mock_booster():
    """テスト用のモック Booster を作成する.

    predict() は入力の最初のカラムの値をそのまま返す。
    save_model() はテキストファイルとして保存する。
    """
    import lightgbm as lgb

    class MockBooster:
        def predict(self, X):
            if isinstance(X, pd.DataFrame):
                return X.iloc[:, 0].values.astype(float)
            return np.array(X, dtype=float).flatten()

        def save_model(self, path):
            Path(path).write_text("mock_model")

        @property
        def best_iteration(self):
            return 100

        @property
        def best_score(self):
            return {"valid": {"binary_logloss": 0.5}}

        def feature_importance(self, importance_type="gain"):
            return np.array([1.0])

        def feature_name(self):
            return ["f1"]

    return MockBooster()


def _make_race_df() -> pd.DataFrame:
    """テスト用の最小限レース DataFrame を作成する."""
    n = 10
    return pd.DataFrame({
        "f1": np.linspace(0.1, 0.9, n),
        "target": [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        "post_umaban": list(range(1, n + 1)),
        "_key_year": ["2025"] * n,
        "_key_monthday": ["0101"] * n,
        "_key_jyocd": ["01"] * n,
        "_key_kaiji": ["01"] * n,
        "_key_nichiji": ["01"] * n,
        "_key_racenum": ["01"] * n,
    })
