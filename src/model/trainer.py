"""LightGBMモデルの学習（二値分類 + LambdaRank 対応）."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import (
    LGBM_PARAMS,
    LGBM_PARAMS_RANKING,
    CATEGORICAL_FEATURES,
    MODEL_DIR,
    RACE_KEY_COLS,
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """LightGBMモデルの学習を管理する.

    ranking=True を指定すると LambdaRank（ランキング学習）モードになり、
    レース単位の group パラメータを用いて着順ランキングを直接学習する。
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        num_boost_round: int = 3000,
        early_stopping_rounds: int = 100,
        ranking: bool = False,
    ) -> None:
        """学習器を初期化する.

        Args:
            params: LightGBMパラメータ（Noneの場合デフォルトを使用）
            num_boost_round: 最大ブースティングラウンド数
            early_stopping_rounds: 早期停止のラウンド数
            ranking: LambdaRank モードで学習するか
        """
        self.ranking = ranking
        if params is not None:
            self.params = params
        elif ranking:
            self.params = LGBM_PARAMS_RANKING.copy()
        else:
            self.params = LGBM_PARAMS.copy()
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

        ranking=True の場合:
            - target_col は無視され、target_relevance が目的変数として使用される
            - _key_ カラムからレースグループを構築して group パラメータに渡す

        Args:
            train_df: 学習データ
            valid_df: 検証データ
            target_col: 目的変数カラム名（二値分類モード時）

        Returns:
            学習済み Booster
        """
        if self.ranking:
            return self._train_ranking(train_df, valid_df)
        return self._train_binary(train_df, valid_df, target_col)

    # ------------------------------------------------------------------
    # 二値分類モード（従来）
    # ------------------------------------------------------------------

    def _train_binary(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        target_col: str,
    ) -> lgb.Booster:
        """二値分類でモデルを学習する."""
        exclude_cols = self._exclude_cols(target_col)
        self.feature_columns = self._resolve_features(
            train_df, valid_df, exclude_cols
        )

        logger.info("特徴量数: %d", len(self.feature_columns))
        logger.info("学習データ: %d行", len(train_df))
        logger.info("検証データ: %d行", len(valid_df))

        cat_features = [
            c for c in CATEGORICAL_FEATURES if c in self.feature_columns
        ]
        logger.info("カテゴリ変数: %d個", len(cat_features))

        X_train = train_df[self.feature_columns].copy()
        y_train = train_df[target_col].astype(int)
        X_valid = valid_df[self.feature_columns].copy()
        y_valid = valid_df[target_col].astype(int)

        for col in cat_features:
            X_train[col] = X_train[col].astype("category")
            X_valid[col] = X_valid[col].astype("category")

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

        logger.info("学習開始（二値分類モード）...")
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

        best_metric = "binary_logloss"
        best_score = self.model.best_score.get("valid", {}).get(best_metric, -1)
        logger.info(
            "学習完了: best_iteration=%d, best_score=%.6f",
            self.model.best_iteration,
            best_score,
        )
        return self.model

    # ------------------------------------------------------------------
    # LambdaRank モード
    # ------------------------------------------------------------------

    def _train_ranking(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
    ) -> lgb.Booster:
        """LambdaRank でモデルを学習する.

        目的変数には target_relevance（関連度スコア）を使用し、
        レースキーから group（各レースの出走頭数）を構築して渡す。
        """
        relevance_col = "target_relevance"
        if relevance_col not in train_df.columns:
            raise ValueError(
                f"LambdaRank モードには '{relevance_col}' カラムが必要です。"
                " parquet を --force-rebuild で再構築してください。"
            )

        exclude_cols = self._exclude_cols("target")
        self.feature_columns = self._resolve_features(
            train_df, valid_df, exclude_cols
        )

        logger.info("特徴量数: %d", len(self.feature_columns))
        logger.info("学習データ: %d行", len(train_df))
        logger.info("検証データ: %d行", len(valid_df))

        cat_features = [
            c for c in CATEGORICAL_FEATURES if c in self.feature_columns
        ]
        logger.info("カテゴリ変数: %d個", len(cat_features))

        # レース単位でソートしてグループサイズを計算
        train_sorted, train_groups = self._prepare_groups(train_df)
        valid_sorted, valid_groups = self._prepare_groups(valid_df)

        logger.info(
            "学習グループ（レース）数: %d, 検証グループ数: %d",
            len(train_groups),
            len(valid_groups),
        )

        X_train = train_sorted[self.feature_columns].copy()
        y_train = train_sorted[relevance_col].astype(int)
        X_valid = valid_sorted[self.feature_columns].copy()
        y_valid = valid_sorted[relevance_col].astype(int)

        for col in cat_features:
            X_train[col] = X_train[col].astype("category")
            X_valid[col] = X_valid[col].astype("category")

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            group=train_groups,
            categorical_feature=cat_features,
            free_raw_data=False,
        )
        valid_data = lgb.Dataset(
            X_valid,
            label=y_valid,
            group=valid_groups,
            reference=train_data,
            free_raw_data=False,
        )

        logger.info("学習開始（LambdaRank モード）...")
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

        # NDCG の best score を表示
        best_scores = self.model.best_score.get("valid", {})
        ndcg_strs = [
            f"ndcg@{k.split('@')[-1]}={v:.6f}"
            for k, v in best_scores.items()
            if "ndcg" in k
        ]
        logger.info(
            "学習完了: best_iteration=%d, %s",
            self.model.best_iteration,
            ", ".join(ndcg_strs) if ndcg_strs else "score=N/A",
        )
        return self.model

    def _prepare_groups(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[int]]:
        """レースキーでソートしてグループサイズリストを生成する.

        LambdaRank では同一レースの馬が連続している必要があるため、
        レースキーでソートし、各レースの頭数をリストで返す。

        Args:
            df: 特徴量 DataFrame（_key_ カラム含む）

        Returns:
            (ソート済み DataFrame, グループサイズのリスト)
        """
        key_cols = [f"_key_{c}" for c in RACE_KEY_COLS if f"_key_{c}" in df.columns]
        if not key_cols:
            raise ValueError(
                "LambdaRank に必要なレースキー (_key_*) が見つかりません"
            )

        df_sorted = df.sort_values(key_cols).reset_index(drop=True)
        groups = df_sorted.groupby(key_cols, sort=False).size().tolist()
        return df_sorted, groups

    # ------------------------------------------------------------------
    # 共通ユーティリティ
    # ------------------------------------------------------------------

    def _exclude_cols(self, target_col: str) -> set[str]:
        """特徴量から除外するカラムのセットを返す."""
        return {
            target_col,
            "target",
            "target_relevance",
            "kakuteijyuni",
            "kettonum",
        } | {
            c for c in ["_key_year", "_key_monthday", "_key_jyocd",
                         "_key_kaiji", "_key_nichiji", "_key_racenum"]
        }

    def _resolve_features(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        exclude_cols: set[str],
    ) -> list[str]:
        """train と valid の共通カラムから特徴量リストを決定する."""
        exclude_cols = exclude_cols | {
            c for c in train_df.columns if c.startswith("_key_")
        }
        common_cols = set(train_df.columns) & set(valid_df.columns)
        feature_columns = [
            c for c in train_df.columns
            if c not in exclude_cols and c in common_cols
        ]

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
        return feature_columns

    # ------------------------------------------------------------------
    # 保存 / ロード
    # ------------------------------------------------------------------

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

        # メタデータ保存（ranking フラグなど）
        meta_path = MODEL_DIR / f"{name}_meta.json"
        meta = {
            "ranking": self.ranking,
            "objective": self.params.get("objective", "binary"),
            "num_features": len(self.feature_columns),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("メタデータ保存: %s", meta_path)

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

        # メタデータの復元
        meta_path = MODEL_DIR / f"{name}_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.ranking = meta.get("ranking", False)
            logger.info(
                "モデルロード: %s (%d特徴量, ranking=%s)",
                model_path,
                len(self.feature_columns),
                self.ranking,
            )
        else:
            logger.info(
                "モデルロード: %s (%d特徴量)",
                model_path,
                len(self.feature_columns),
            )

        return self.model

    # ------------------------------------------------------------------
    # 特徴量重要度
    # ------------------------------------------------------------------

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
