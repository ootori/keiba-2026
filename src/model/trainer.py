"""LightGBMモデルの学習."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import (
    LGBM_PARAMS,
    CATEGORICAL_FEATURES,
    MODEL_DIR,
    RACE_KEY_COLS,
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """LightGBMモデルの学習を管理する."""

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        num_boost_round: int = 3000,
        early_stopping_rounds: int = 100,
    ) -> None:
        """学習器を初期化する.

        Args:
            params: LightGBMパラメータ（Noneの場合デフォルトを使用）
            num_boost_round: 最大ブースティングラウンド数
            early_stopping_rounds: 早期停止のラウンド数
        """
        self.params = params if params is not None else LGBM_PARAMS.copy()
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model: lgb.Booster | None = None
        self.feature_columns: list[str] = []

    def train(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        target_col: str = "target",
    ) -> lgb.Booster:
        """モデルを学習する.

        Args:
            train_df: 学習データ
            valid_df: 検証データ
            target_col: 目的変数カラム名

        Returns:
            学習済み Booster
        """
        # 特徴量カラムを特定（_key_ プレフィクスと target, kettonumを除外）
        exclude_cols = {target_col, "kettonum"} | {
            c for c in train_df.columns if c.startswith("_key_")
        }
        # train と valid の両方に存在するカラムのみを使用
        # （parquet の再構築タイミング差による不整合を防止）
        common_cols = set(train_df.columns) & set(valid_df.columns)
        self.feature_columns = [
            c for c in train_df.columns
            if c not in exclude_cols and c in common_cols
        ]

        # 不整合がある場合は警告
        train_only = set(train_df.columns) - set(valid_df.columns) - exclude_cols
        valid_only = set(valid_df.columns) - set(train_df.columns) - exclude_cols
        if train_only:
            logger.warning(
                "学習データのみに存在する特徴量（除外）: %s", sorted(train_only)
            )
        if valid_only:
            logger.warning(
                "検証データのみに存在する特徴量（除外）: %s", sorted(valid_only)
            )


        logger.info("特徴量数: %d", len(self.feature_columns))
        logger.info("学習データ: %d行", len(train_df))
        logger.info("検証データ: %d行", len(valid_df))

        # カテゴリ変数の処理
        cat_features = [
            c for c in CATEGORICAL_FEATURES if c in self.feature_columns
        ]
        logger.info("カテゴリ変数: %d個", len(cat_features))

        X_train = train_df[self.feature_columns].copy()
        y_train = train_df[target_col].astype(int)
        X_valid = valid_df[self.feature_columns].copy()
        y_valid = valid_df[target_col].astype(int)

        # カテゴリ変数を category 型に変換
        for col in cat_features:
            X_train[col] = X_train[col].astype("category")
            X_valid[col] = X_valid[col].astype("category")

        # LightGBM Dataset作成
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=cat_features,
            free_raw_data=False,
        )
        valid_data = lgb.Dataset(
            X_valid,
            label=y_valid,
            reference=train_data,
            free_raw_data=False,
        )

        # 学習実行
        logger.info("学習開始...")
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=[train_data, valid_data],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=self.early_stopping_rounds
                ),
                lgb.log_evaluation(period=100),
            ],
        )

        logger.info(
            "学習完了: best_iteration=%d, best_score=%.6f",
            self.model.best_iteration,
            self.model.best_score.get("valid", {}).get("binary_logloss", -1),
        )

        return self.model

    def save_model(self, name: str = "model") -> Path:
        """モデルを保存する.

        Args:
            name: ファイル名プレフィクス

        Returns:
            保存先パス
        """
        if self.model is None:
            raise RuntimeError("モデルが学習されていません")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / f"{name}.txt"
        self.model.save_model(str(model_path))
        logger.info("モデル保存: %s", model_path)

        # 特徴量名も保存
        feature_path = MODEL_DIR / f"{name}_features.txt"
        with open(feature_path, "w") as f:
            for col in self.feature_columns:
                f.write(col + "\n")
        logger.info("特徴量リスト保存: %s", feature_path)

        return model_path

    def load_model(self, name: str = "model") -> lgb.Booster:
        """保存済みモデルをロードする.

        Args:
            name: ファイル名プレフィクス

        Returns:
            ロードされた Booster
        """
        model_path = MODEL_DIR / f"{name}.txt"
        self.model = lgb.Booster(model_file=str(model_path))

        feature_path = MODEL_DIR / f"{name}_features.txt"
        if feature_path.exists():
            with open(feature_path) as f:
                self.feature_columns = [line.strip() for line in f if line.strip()]

        logger.info("モデルロード: %s (%d特徴量)", model_path, len(self.feature_columns))
        return self.model

    def get_feature_importance(
        self,
        importance_type: str = "gain",
        top_n: int = 50,
    ) -> pd.DataFrame:
        """特徴量重要度を取得する.

        Args:
            importance_type: 'gain' or 'split'
            top_n: 上位N件

        Returns:
            特徴量名と重要度の DataFrame
        """
        if self.model is None:
            raise RuntimeError("モデルが学習されていません")

        importance = self.model.feature_importance(
            importance_type=importance_type
        )
        feature_names = self.model.feature_name()

        imp_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)

        return imp_df.head(top_n).reset_index(drop=True)
