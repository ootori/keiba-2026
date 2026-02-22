#!/usr/bin/env python3
"""学習実行スクリプト.

使用例:
    # 基本的な学習（デフォルト設定、年度別 parquet で構築）
    python run_train.py

    # 4並列で特徴量構築
    python run_train.py --workers 4

    # 特徴量構築のみ（8並列、既存を再構築）
    python run_train.py --build-features-only --workers 8 --force-rebuild

    # 既存特徴量を使ってモデル学習のみ
    python run_train.py --train-only

    # 評価・回収率シミュレーションのみ（Step 4-5）
    python run_train.py --eval-only

    # オッズありモデル（デフォルトはオッズ除外）
    python run_train.py --with-odds

    # LambdaRank（ランキング学習）モードで学習
    python run_train.py --ranking

    # LambdaRank + モデル名指定
    python run_train.py --ranking --model-name ranking_model
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
    DEFAULT_ODDS_CORRECTION_CONFIG,
    ODDS_CORRECTION_STATS_PATH,
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
        "--ranking",
        action="store_true",
        help="LambdaRank（ランキング学習）モードで学習する",
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="特徴量構築の並列ワーカー数（デフォルト: 1 = 直列）",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="既存の年度別 parquet を無視して全年度を再構築",
    )
    parser.add_argument(
        "--target",
        choices=["top3", "win"],
        default="top3",
        help="目的変数: top3=3着以内（デフォルト）, win=1着",
    )
    parser.add_argument(
        "--relevance-mode",
        choices=["default", "win"],
        default="default",
        help="LambdaRankの関連度モード: default=均等, win=1着重み寄せ",
    )
    parser.add_argument(
        "--build-supplement",
        nargs="+",
        metavar="NAME",
        help="サプリメント（差分特徴量）を構築する。名前を指定（例: mining）",
    )
    parser.add_argument(
        "--supplement",
        nargs="+",
        metavar="NAME",
        help="学習/評価時にマージするサプリメント名（例: --supplement mining）",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="確率キャリブレーション（Isotonic Regression）を適用する",
    )
    parser.add_argument(
        "--odds-correction",
        action="store_true",
        help="value_bet戦略でオッズ歪み補正を有効化（統計JSON自動ロード）",
    )
    parser.add_argument(
        "--build-odds-stats",
        action="store_true",
        help="DBからオッズ補正統計を算出してJSONに保存",
    )
    parser.add_argument(
        "--odds-stats-start",
        default="2022",
        help="オッズ統計の集計開始年（デフォルト: 2022）",
    )
    parser.add_argument(
        "--odds-stats-end",
        default="2024",
        help="オッズ統計の集計終了年（デフォルト: 2024）",
    )
    return parser.parse_args()


def _build_odds_correction_config(args: argparse.Namespace) -> dict | None:
    """CLIオプションからオッズ補正設定を構築する.

    統計JSONが存在すれば自動ロード、なければハードコードのフォールバック。
    """
    if not args.odds_correction:
        return None

    # 統計JSONからロードを試みる
    if ODDS_CORRECTION_STATS_PATH.exists():
        from src.odds_correction_stats import load_odds_correction_stats
        config = load_odds_correction_stats()
        logger.info("統計JSONからオッズ補正設定をロード")
        return config

    # フォールバック: ハードコードのデフォルト設定
    logger.warning(
        "統計JSONが見つかりません (%s)。"
        "ハードコードのデフォルト設定を使用します。"
        " --build-odds-stats で統計を構築してください。",
        ODDS_CORRECTION_STATS_PATH,
    )
    config = dict(DEFAULT_ODDS_CORRECTION_CONFIG)
    config["enabled"] = True
    return config


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


def _split_train_valid(
    all_df: pd.DataFrame,
    valid_year: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """全年度 DataFrame を学習/検証に分割する.

    Args:
        all_df: 全年度を結合した DataFrame（_key_year カラム必須）
        valid_year: 検証年（この年度のデータが検証用になる）

    Returns:
        (train_df, valid_df) のタプル
    """
    mask = all_df["_key_year"] == valid_year
    valid_df = all_df[mask].copy()
    train_df = all_df[~mask].copy()
    return train_df, valid_df


def step_build_features(
    args: argparse.Namespace,
    include_odds: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 1: 特徴量構築（年度別 parquet 方式）.

    学習期間〜検証年を一括で構築し、_key_year で分割する。
    これにより検証年も並列構築の恩恵を受けられる。
    """
    logger.info("=" * 60)
    logger.info("Step 1: 特徴量構築（年度別, workers=%d）", args.workers)
    logger.info("=" * 60)

    supplement_names = args.supplement or []

    # 学習期間〜検証年を一括構築
    logger.info(
        "全年度構築中 (%s〜%s, 検証=%s)...",
        args.train_start, args.valid_year, args.valid_year,
    )
    all_df = FeaturePipeline.build_years(
        year_start=args.train_start,
        year_end=args.valid_year,
        include_odds=include_odds,
        workers=args.workers,
        force_rebuild=args.force_rebuild,
    )

    # サプリメントマージ（一括）
    if supplement_names:
        from src.features.supplement import merge_supplements
        all_df = merge_supplements(
            all_df, supplement_names,
            args.train_start, args.valid_year,
        )

    # 学習/検証に分割
    train_df, valid_df = _split_train_valid(all_df, args.valid_year)

    logger.info("学習データ: %d行, 検証データ: %d行", len(train_df), len(valid_df))
    return train_df, valid_df


def step_load_features(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """既存の特徴量 parquet を読み込む.

    年度別 parquet（features_{year}.parquet）を優先してロードする。
    学習期間〜検証年を一括ロードし、_key_year で分割する。
    見つからない場合は旧方式（train_features.parquet / valid_features.parquet）を試みる。
    --supplement が指定されている場合はサプリメントもマージする。
    """
    supplement_names = args.supplement or []

    # 年度別 parquet の存在チェック（一括ロード → 分割）
    try:
        all_df = FeaturePipeline.load_years(
            args.train_start, args.valid_year,
            supplement_names=supplement_names,
        )
        train_df, valid_df = _split_train_valid(all_df, args.valid_year)
        logger.info(
            "年度別ロード完了: 学習=%d行, 検証=%d行",
            len(train_df), len(valid_df),
        )
        return train_df, valid_df
    except FileNotFoundError:
        pass

    # 旧方式へのフォールバック
    train_path = DATA_DIR / "train_features.parquet"
    valid_path = DATA_DIR / "valid_features.parquet"

    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(
            "特徴量ファイルが見つかりません。"
            " --build-features-only で構築してください。"
        )

    logger.info("旧方式 parquet からロード")
    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)

    # 旧方式でもサプリメントマージ
    if supplement_names:
        from src.features.supplement import merge_supplements
        train_df = merge_supplements(
            train_df, supplement_names,
            args.train_start, args.train_end,
        )
        valid_df = merge_supplements(
            valid_df, supplement_names,
            args.valid_year, args.valid_year,
        )

    logger.info("特徴量ロード: 学習=%d行, 検証=%d行", len(train_df), len(valid_df))
    return train_df, valid_df


def step_train(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    model_name: str,
    ranking: bool = False,
    target_type: str = "top3",
    odds_correction_config: dict | None = None,
    relevance_mode: str = "default",
    calibrate: bool = False,
) -> None:
    """Step 2: モデル学習 + 評価.

    Args:
        target_type: "top3"=3着以内, "win"=1着
        odds_correction_config: オッズ歪み補正設定
        relevance_mode: LambdaRank関連度モード ("default" or "win")
        calibrate: 確率キャリブレーションを適用するか
    """
    # 目的変数カラムの決定
    target_col = "target_win" if target_type == "win" else "target"

    logger.info("=" * 60)
    mode_str = "LambdaRank" if ranking else "二値分類"
    target_desc = "1着" if target_type == "win" else "3着以内"
    logger.info("Step 2: モデル学習（%s, 目的変数=%s）", mode_str, target_desc)
    logger.info("=" * 60)

    # target カラムが存在するか確認
    if target_col not in train_df.columns:
        logger.error(
            "%s カラムが見つかりません。"
            " parquet を --force-rebuild で再構築してください。",
            target_col,
        )
        return

    # LambdaRank の場合は target_relevance が必要
    if ranking and "target_relevance" not in train_df.columns:
        logger.error(
            "target_relevance カラムが見つかりません。"
            " parquet を --force-rebuild で再構築してください。"
        )
        return

    # NaN のある行を除外
    drop_cols = [target_col]
    if ranking:
        drop_cols.append("target_relevance")
    train_df = train_df.dropna(subset=drop_cols).copy()
    valid_df = valid_df.dropna(subset=drop_cols).copy()

    # 学習
    trainer = ModelTrainer(
        ranking=ranking, relevance_mode=relevance_mode, calibrate=calibrate,
    )
    model = trainer.train(train_df, valid_df, target_col=target_col)

    # 保存
    trainer.save_model(name=model_name, target_type=target_type)

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
        model, valid_df, trainer.feature_columns,
        target_col=target_col, ranking=ranking,
        calibrator=trainer.calibrator,
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
    for strategy in [
        "top1_tansho", "top1_fukusho", "top3_fukusho",
        "top2_umaren", "top2_umatan", "top3_sanrenpuku", "top3_sanrentan",
        "value_bet_tansho", "value_bet_umaren",
    ]:
        result = evaluator.simulate_return(
            valid_df, trainer.feature_columns, model,
            strategy=strategy, ranking=ranking,
            target_type=target_type,
            odds_correction_config=odds_correction_config,
            calibrator=trainer.calibrator,
        )
        logger.info(
            "  [%s] 回収率: %.1f%%, 的中率: %.1f%% (%d/%d)",
            strategy,
            result["return_rate"],
            result["hit_rate"] * 100,
            result["win_count"],
            result["bet_count"],
        )


def step_eval_only(args: argparse.Namespace) -> None:
    """Step 4-5 のみ: 既存モデル+特徴量で評価・回収率シミュレーションを実行."""
    # 特徴量ロード
    _, valid_df = step_load_features(args)

    # モデルロード
    trainer = ModelTrainer()
    model = trainer.load_model(name=args.model_name)
    ranking = trainer.ranking  # メタデータから復元
    target_type = trainer.target_type  # メタデータから復元

    if args.ranking:
        ranking = True  # CLI で明示指定された場合を優先
    if args.target != "top3":
        target_type = args.target  # CLI で明示指定された場合を優先

    # 目的変数カラムの決定
    target_col = "target_win" if target_type == "win" else "target"
    valid_df = valid_df.dropna(subset=[target_col]).copy()

    # オッズ補正設定
    odds_correction_config = _build_odds_correction_config(args)

    # 評価
    logger.info("=" * 60)
    mode_str = "LambdaRank" if ranking else "二値分類"
    target_desc = "1着" if target_type == "win" else "3着以内"
    logger.info("Step 4: モデル評価（%s, 目的変数=%s）", mode_str, target_desc)
    logger.info("=" * 60)
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(
        model, valid_df, trainer.feature_columns,
        target_col=target_col, ranking=ranking,
        calibrator=trainer.calibrator,
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
    for strategy in [
        "top1_tansho", "top1_fukusho", "top3_fukusho",
        "top2_umaren", "top2_umatan", "top3_sanrenpuku", "top3_sanrentan",
        "value_bet_tansho", "value_bet_umaren",
    ]:
        result = evaluator.simulate_return(
            valid_df, trainer.feature_columns, model,
            strategy=strategy, ranking=ranking,
            target_type=target_type,
            odds_correction_config=odds_correction_config,
            calibrator=trainer.calibrator,
        )
        logger.info(
            "  [%s] 回収率: %.1f%%, 的中率: %.1f%% (%d/%d)",
            strategy,
            result["return_rate"],
            result["hit_rate"] * 100,
            result["win_count"],
            result["bet_count"],
        )


def step_build_odds_stats(args: argparse.Namespace) -> None:
    """オッズ補正統計をDBから算出してJSONに保存する."""
    from src.odds_correction_stats import (
        build_odds_correction_stats,
        save_odds_correction_stats,
    )

    logger.info("=" * 60)
    logger.info("オッズ補正統計構築: %s〜%s年", args.odds_stats_start, args.odds_stats_end)
    logger.info("=" * 60)

    stats = build_odds_correction_stats(
        year_start=args.odds_stats_start,
        year_end=args.odds_stats_end,
    )
    save_odds_correction_stats(stats)


def step_build_supplements(args: argparse.Namespace) -> None:
    """サプリメント（差分特徴量）を構築する."""
    from src.features.supplement import (
        build_supplement_years,
        list_available_supplements,
    )

    available = list_available_supplements()
    for name in args.build_supplement:
        if name not in available:
            logger.error(
                "未知のサプリメント: %s（利用可能: %s）",
                name, available,
            )
            continue

        logger.info("=" * 60)
        logger.info("サプリメント構築: %s", name)
        logger.info("=" * 60)

        # 学習期間〜検証年を一括構築
        build_supplement_years(
            supplement_name=name,
            year_start=args.train_start,
            year_end=args.valid_year,
            workers=args.workers,
            force_rebuild=args.force_rebuild,
        )


def main() -> None:
    args = parse_args()

    # DB接続確認
    if not step_check_db():
        sys.exit(1)

    include_odds = args.with_odds  # --with-odds指定時のみオッズを含める
    ranking = args.ranking

    target_type = args.target

    # オッズ補正統計構築モード
    if args.build_odds_stats:
        step_build_odds_stats(args)
        if not args.train_only and not args.eval_only and not args.build_features_only:
            logger.info("完了!")
            return

    # サプリメント構築モード
    if args.build_supplement:
        step_build_supplements(args)
        if not args.train_only and not args.eval_only and not args.build_features_only:
            # --build-supplement のみの場合はここで終了
            logger.info("完了!")
            return

    odds_correction_config = _build_odds_correction_config(args)

    if args.eval_only:
        # 評価・回収率シミュレーションのみ
        step_eval_only(args)

    elif args.train_only:
        # 既存特徴量からの学習のみ
        train_df, valid_df = step_load_features(args)
        step_train(
            train_df, valid_df, args.model_name,
            ranking=ranking, target_type=target_type,
            odds_correction_config=odds_correction_config,
            relevance_mode=args.relevance_mode,
            calibrate=args.calibrate,
        )

    elif args.build_features_only:
        # 特徴量構築のみ
        step_build_features(args, include_odds)

    else:
        # フル実行: 特徴量構築 → 学習 → 評価
        train_df, valid_df = step_build_features(args, include_odds)
        step_train(
            train_df, valid_df, args.model_name,
            ranking=ranking, target_type=target_type,
            odds_correction_config=odds_correction_config,
            relevance_mode=args.relevance_mode,
            calibrate=args.calibrate,
        )

    logger.info("完了!")


if __name__ == "__main__":
    main()
