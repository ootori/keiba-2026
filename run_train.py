#!/usr/bin/env python3
"""学習実行スクリプト.

使用例:
    # 基本的な学習（デフォルト設定）
    python run_train.py

    # 特徴量構築のみ
    python run_train.py --build-features-only

    # 既存特徴量を使ってモデル学習のみ
    python run_train.py --train-only

    # 評価・回収率シミュレーションのみ（Step 4-5）
    python run_train.py --eval-only

    # オッズありモデル（デフォルトはオッズ除外）
    python run_train.py --with-odds
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    TRAIN_START_YEAR,
    TRAIN_END_YEAR,
    VALID_YEAR,
    DATA_DIR,
    MODEL_DIR,
    LGBM_PARAMS,
)
from src.db import check_connection, get_table_counts
from src.features.pipeline import FeaturePipeline
from src.model.trainer import ModelTrainer
from src.model.evaluator import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="競馬予測モデル学習")
    parser.add_argument(
        "--build-features-only",
        action="store_true",
        help="特徴量構築のみ実行",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="既存特徴量を使ってモデル学習のみ実行",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="既存モデル+特徴量で評価・回収率シミュレーションのみ実行（Step 4-5）",
    )
    parser.add_argument(
        "--no-odds",
        action="store_true",
        default=True,  # デフォルトでオッズ除外
        help="オッズ特徴量を除外する（デフォルト: 除外）",
    )
    parser.add_argument(
        "--with-odds",
        action="store_true",
        help="オッズ特徴量を含める（明示的に指定時のみ）",
    )
    parser.add_argument(
        "--model-name",
        default="model",
        help="モデル名（デフォルト: model）",
    )
    parser.add_argument(
        "--train-start",
        default=TRAIN_START_YEAR,
        help=f"学習開始年（デフォルト: {TRAIN_START_YEAR}）",
    )
    parser.add_argument(
        "--train-end",
        default=TRAIN_END_YEAR,
        help=f"学習終了年（デフォルト: {TRAIN_END_YEAR}）",
    )
    parser.add_argument(
        "--valid-year",
        default=VALID_YEAR,
        help=f"検証年（デフォルト: {VALID_YEAR}）",
    )
    return parser.parse_args()


def step_check_db() -> bool:
    """Step 0: DB接続確認."""
    logger.info("=" * 60)
    logger.info("Step 0: DB接続確認")
    logger.info("=" * 60)

    if not check_connection():
        logger.error("DB接続に失敗しました。設定を確認してください。")
        return False

    logger.info("テーブル行数:")
    counts = get_table_counts()
    for _, row in counts.iterrows():
        logger.info("  %s: %s行", row["tbl"], row["cnt"])

    return True


def step_build_features(
    args: argparse.Namespace,
    include_odds: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 1: 特徴量構築."""
    logger.info("=" * 60)
    logger.info("Step 1: 特徴量構築")
    logger.info("=" * 60)

    pipeline = FeaturePipeline(include_odds=include_odds)

    # 学習データ
    logger.info("学習データ構築中 (%s〜%s)...", args.train_start, args.train_end)
    train_df = pipeline.build_dataset(
        year_start=args.train_start,
        year_end=args.train_end,
        save_parquet=True,
        output_name="train_features",
    )

    # 検証データ
    logger.info("検証データ構築中 (%s)...", args.valid_year)
    valid_df = pipeline.build_dataset(
        year_start=args.valid_year,
        year_end=args.valid_year,
        save_parquet=True,
        output_name="valid_features",
    )

    logger.info("学習データ: %d行, 検証データ: %d行", len(train_df), len(valid_df))
    return train_df, valid_df


def step_load_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """既存の特徴量parquetを読み込む."""
    train_path = DATA_DIR / "train_features.parquet"
    valid_path = DATA_DIR / "valid_features.parquet"

    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(
            "特徴量ファイルが見つかりません。先に --build-features-only で構築してください。"
        )

    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)
    logger.info("特徴量ロード: 学習=%d行, 検証=%d行", len(train_df), len(valid_df))
    return train_df, valid_df


def step_train(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    model_name: str,
) -> None:
    """Step 2: モデル学習 + 評価."""
    logger.info("=" * 60)
    logger.info("Step 2: モデル学習")
    logger.info("=" * 60)

    # target カラムが存在するか確認
    if "target" not in train_df.columns:
        logger.error("target カラムが見つかりません")
        return

    # NaN のある行を除外
    train_df = train_df.dropna(subset=["target"]).copy()
    valid_df = valid_df.dropna(subset=["target"]).copy()

    # 学習
    trainer = ModelTrainer()
    model = trainer.train(train_df, valid_df)

    # 保存
    trainer.save_model(name=model_name)

    # 特徴量重要度
    logger.info("=" * 60)
    logger.info("Step 3: 特徴量重要度")
    logger.info("=" * 60)
    imp = trainer.get_feature_importance(top_n=30)
    logger.info("Top 30 特徴量重要度:")
    for _, row in imp.iterrows():
        logger.info("  %s: %.1f", row["feature"], row["importance"])

    # 評価
    logger.info("=" * 60)
    logger.info("Step 4: モデル評価")
    logger.info("=" * 60)
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(
        model, valid_df, trainer.feature_columns
    )

    logger.info("評価結果:")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)
        else:
            logger.info("  %s: %s", k, v)

    # 回収率シミュレーション
    logger.info("=" * 60)
    logger.info("Step 5: 回収率シミュレーション")
    logger.info("=" * 60)
    for strategy in ["top1_tansho", "top1_fukusho", "top3_fukusho", "value_bet"]:
        result = evaluator.simulate_return(
            valid_df, trainer.feature_columns, model, strategy=strategy
        )
        logger.info(
            "  [%s] 回収率: %.1f%%, 的中率: %.1f%% (%d/%d)",
            strategy,
            result["return_rate"],
            result["hit_rate"] * 100,
            result["win_count"],
            result["bet_count"],
        )


def step_eval_only(model_name: str) -> None:
    """Step 4-5 のみ: 既存モデル+特徴量で評価・回収率シミュレーションを実行."""
    # 特徴量ロード
    _, valid_df = step_load_features()
    valid_df = valid_df.dropna(subset=["target"]).copy()

    # モデルロード
    trainer = ModelTrainer()
    model = trainer.load_model(name=model_name)

    # 評価
    logger.info("=" * 60)
    logger.info("Step 4: モデル評価")
    logger.info("=" * 60)
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(
        model, valid_df, trainer.feature_columns
    )

    logger.info("評価結果:")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)
        else:
            logger.info("  %s: %s", k, v)

    # 回収率シミュレーション
    logger.info("=" * 60)
    logger.info("Step 5: 回収率シミュレーション")
    logger.info("=" * 60)
    for strategy in ["top1_tansho", "top1_fukusho", "top3_fukusho", "value_bet"]:
        result = evaluator.simulate_return(
            valid_df, trainer.feature_columns, model, strategy=strategy
        )
        logger.info(
            "  [%s] 回収率: %.1f%%, 的中率: %.1f%% (%d/%d)",
            strategy,
            result["return_rate"],
            result["hit_rate"] * 100,
            result["win_count"],
            result["bet_count"],
        )


def main() -> None:
    args = parse_args()

    # DB接続確認
    if not step_check_db():
        sys.exit(1)

    include_odds = args.with_odds  # --with-odds指定時のみオッズを含める

    if args.eval_only:
        # 評価・回収率シミュレーションのみ
        step_eval_only(args.model_name)

    elif args.train_only:
        # 既存特徴量からの学習のみ
        train_df, valid_df = step_load_features()
        step_train(train_df, valid_df, args.model_name)

    elif args.build_features_only:
        # 特徴量構築のみ
        step_build_features(args, include_odds)

    else:
        # フル実行: 特徴量構築 → 学習 → 評価
        train_df, valid_df = step_build_features(args, include_odds)
        step_train(train_df, valid_df, args.model_name)

    logger.info("完了!")


if __name__ == "__main__":
    main()
