"""Optuna ベースの LambdaRank ハイパーパラメータチューニング."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import optuna
import pandas as pd
from optuna.integration import LightGBMPruningCallback

from src.config import (
    CATEGORICAL_FEATURES,
    LGBM_PARAMS_RANKING,
    MODEL_DIR,
    TUNE_DEFAULT_NDCG_AT,
)
from src.model.trainer import ModelTrainer

logger = logging.getLogger(__name__)


class RankingTuner:
    """LambdaRank モデルのハイパーパラメータを Optuna で最適化する.

    TPESampler + MedianPruner を使い、NDCG@k を最大化するパラメータを探索する。
    lgb.Dataset は全トライアルで共有し、データ変換コストを削減する。
    """

    def __init__(
        self,
        n_trials: int = 100,
        timeout: int | None = None,
        ndcg_at: int = TUNE_DEFAULT_NDCG_AT,
        relevance_mode: str = "default",
        seed: int = 42,
    ) -> None:
        """チューナーを初期化する.

        Args:
            n_trials: Optuna の試行回数
            timeout: タイムアウト（秒）。None の場合は無制限
            ndcg_at: 最適化対象の NDCG@k（デフォルト: 3）
            relevance_mode: LambdaRank 関連度モード ("default" or "win")
            seed: 乱数シード
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.ndcg_at = ndcg_at
        self.relevance_mode = relevance_mode
        self.seed = seed

        # tune() 内で事前構築
        self._train_data: lgb.Dataset | None = None
        self._valid_data: lgb.Dataset | None = None
        self._feature_columns: list[str] = []
        self._cat_features: list[str] = []

    def tune(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Optuna でハイパーパラメータを最適化する.

        Args:
            train_df: 学習データ
            valid_df: 検証データ

        Returns:
            最適パラメータの辞書
        """
        self._prepare_datasets(train_df, valid_df)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=50,
            ),
            study_name=f"lambdarank_ndcg{self.ndcg_at}",
        )

        ndcg_at = self.ndcg_at

        def _trial_callback(
            study: optuna.Study, trial: optuna.trial.FrozenTrial
        ) -> None:
            if trial.value is not None:
                logger.info(
                    "Trial %d: NDCG@%d=%.6f (best=%.6f @ trial %d)",
                    trial.number,
                    ndcg_at,
                    trial.value,
                    study.best_value,
                    study.best_trial.number,
                )

        logger.info(
            "チューニング開始: n_trials=%d, timeout=%s, metric=NDCG@%d",
            self.n_trials,
            self.timeout,
            self.ndcg_at,
        )

        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=[_trial_callback],
            show_progress_bar=True,
        )

        completed = len(
            [
                t
                for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
        )
        pruned = len(
            [
                t
                for t in study.trials
                if t.state == optuna.trial.TrialState.PRUNED
            ]
        )

        logger.info("チューニング完了")
        logger.info("  Best trial: %d", study.best_trial.number)
        logger.info("  Best NDCG@%d: %.6f", self.ndcg_at, study.best_value)
        logger.info("  Best params: %s", study.best_trial.params)
        logger.info(
            "  Completed/Pruned/Total: %d/%d/%d",
            completed,
            pruned,
            len(study.trials),
        )

        return study.best_trial.params

    def _prepare_datasets(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
    ) -> None:
        """学習/検証データセットを事前構築する."""
        if self.relevance_mode == "win":
            relevance_col = "target_relevance_win"
        else:
            relevance_col = "target_relevance"

        if relevance_col not in train_df.columns:
            if (
                self.relevance_mode == "win"
                and "target_relevance" in train_df.columns
            ):
                logger.warning(
                    "%s が見つかりません。target_relevance にフォールバックします。",
                    relevance_col,
                )
                relevance_col = "target_relevance"
            else:
                raise ValueError(
                    f"'{relevance_col}' カラムが必要です。"
                    " parquet を --force-rebuild で再構築してください。"
                )

        # ModelTrainer のヘルパーを再利用
        helper = ModelTrainer(ranking=True, relevance_mode=self.relevance_mode)
        exclude_cols = helper._exclude_cols("target")
        self._feature_columns = helper._resolve_features(
            train_df, valid_df, exclude_cols
        )
        self._cat_features = [
            c for c in CATEGORICAL_FEATURES if c in self._feature_columns
        ]

        train_sorted, train_groups = helper._prepare_groups(train_df)
        valid_sorted, valid_groups = helper._prepare_groups(valid_df)

        X_train = train_sorted[self._feature_columns].copy()
        y_train = train_sorted[relevance_col].astype(int)
        X_valid = valid_sorted[self._feature_columns].copy()
        y_valid = valid_sorted[relevance_col].astype(int)

        for col in self._cat_features:
            X_train[col] = X_train[col].astype("category")
            X_valid[col] = X_valid[col].astype("category")

        self._train_data = lgb.Dataset(
            X_train,
            label=y_train,
            group=train_groups,
            categorical_feature=self._cat_features,
            free_raw_data=False,
        )
        self._valid_data = lgb.Dataset(
            X_valid,
            label=y_valid,
            group=valid_groups,
            reference=self._train_data,
            free_raw_data=False,
        )

        logger.info(
            "データセット準備完了: 特徴量=%d, カテゴリ=%d, "
            "学習グループ=%d, 検証グループ=%d",
            len(self._feature_columns),
            len(self._cat_features),
            len(train_groups),
            len(valid_groups),
        )

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna の目的関数."""
        params = self._sample_params(trial)

        ndcg_eval_at = [1, 2, 3, 5]
        if self.ndcg_at not in ndcg_eval_at:
            ndcg_eval_at.append(self.ndcg_at)
            ndcg_eval_at.sort()

        params.update(
            {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": ndcg_eval_at,
                "boosting_type": "gbdt",
                "verbose": -1,
                "n_jobs": -1,
                "seed": self.seed,
            }
        )

        pruning_callback = LightGBMPruningCallback(
            trial,
            f"ndcg@{self.ndcg_at}",
            valid_name="valid",
        )

        model = lgb.train(
            params,
            self._train_data,
            num_boost_round=3000,
            valid_sets=[self._valid_data],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=0),
                pruning_callback,
            ],
        )

        score = model.best_score.get("valid", {}).get(
            f"ndcg@{self.ndcg_at}", 0.0
        )
        return score

    def _sample_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Optuna trial からパラメータをサンプリングする."""
        params: dict[str, Any] = {
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.5, 1.0
            ),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.5, 1.0
            ),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 10, 200
            ),
            "lambda_l1": trial.suggest_float(
                "lambda_l1", 1e-8, 10.0, log=True
            ),
            "lambda_l2": trial.suggest_float(
                "lambda_l2", 1e-8, 10.0, log=True
            ),
            "min_split_gain": trial.suggest_float(
                "min_split_gain", 0.0, 1.0
            ),
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),
        }

        use_max_depth = trial.suggest_categorical(
            "use_max_depth", [True, False]
        )
        if use_max_depth:
            params["max_depth"] = trial.suggest_int("max_depth", 5, 15)
        else:
            params["max_depth"] = -1

        return params

    def train_best(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        best_params: dict[str, Any],
    ) -> ModelTrainer:
        """ベストパラメータでモデルを最終学習する.

        Args:
            train_df: 学習データ
            valid_df: 検証データ
            best_params: Optuna が選んだベストパラメータ

        Returns:
            学習済み ModelTrainer
        """
        full_params = dict(LGBM_PARAMS_RANKING)

        # use_max_depth は LightGBM パラメータではないため除去
        clean_params = {
            k: v for k, v in best_params.items() if k != "use_max_depth"
        }
        full_params.update(clean_params)

        logger.info("最終学習パラメータ: %s", full_params)

        trainer = ModelTrainer(
            params=full_params,
            ranking=True,
            relevance_mode=self.relevance_mode,
        )
        trainer.train(train_df, valid_df)
        return trainer

    @staticmethod
    def save_best_params(
        params: dict[str, Any],
        model_name: str,
    ) -> Path:
        """ベストパラメータを JSON に保存する.

        Args:
            params: 最適パラメータの辞書
            model_name: モデル名

        Returns:
            保存先パス
        """
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / f"{model_name}_best_params.json"
        with open(path, "w") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        logger.info("ベストパラメータ保存: %s", path)
        return path

    @staticmethod
    def load_best_params(model_name: str) -> dict[str, Any]:
        """保存済みベストパラメータをロードする.

        Args:
            model_name: モデル名

        Returns:
            パラメータの辞書
        """
        path = MODEL_DIR / f"{model_name}_best_params.json"
        if not path.exists():
            raise FileNotFoundError(
                f"ベストパラメータが見つかりません: {path}。"
                " --tune でチューニングを実行してください。"
            )
        with open(path) as f:
            params = json.load(f)
        logger.info("ベストパラメータロード: %s", path)
        return params
