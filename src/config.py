"""DB接続設定・定数定義."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# プロジェクトルート
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# DB接続
# ---------------------------------------------------------------------------
DB_CONFIG: dict = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "everydb"),
    "user": os.getenv("DB_USER", "webmaster"),
    "password": os.getenv("DB_PASSWORD", "devpassword"),
}

# ---------------------------------------------------------------------------
# 対象期間
# ---------------------------------------------------------------------------
TRAIN_START_YEAR: str = "2015"
TRAIN_END_YEAR: str = "2024"
VALID_YEAR: str = "2025"

# ---------------------------------------------------------------------------
# JRA中央10場コード
# ---------------------------------------------------------------------------
JRA_JYO_CODES: list[str] = [f"{i:02d}" for i in range(1, 11)]

# ---------------------------------------------------------------------------
# レースキー構成カラム
# ---------------------------------------------------------------------------
RACE_KEY_COLS: list[str] = [
    "year",
    "monthday",
    "jyocd",
    "kaiji",
    "nichiji",
    "racenum",
]

# ---------------------------------------------------------------------------
# LightGBM カテゴリ変数
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES: list[str] = [
    "race_jyo_cd",
    "race_track_cd",
    "race_track_type",
    "race_course_dir",
    "race_grade_cd",
    "race_syubetu_cd",
    "race_jyuryo_cd",
    "race_jyoken_cd",
    "horse_sex",
    "horse_tozai",
    "horse_keiro",
    "style_type_last",
    "style_type_mode_last5",
    "jockey_code",
    "trainer_code",
    "trainer_tozai",
    "blood_father_id",
    "blood_bms_id",
    "blood_father_keito",
    "blood_bms_keito",
    "interval_category",
    "cross_dist_category_change",
]

# ---------------------------------------------------------------------------
# LightGBM デフォルトパラメータ
# ---------------------------------------------------------------------------
LGBM_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

# ---------------------------------------------------------------------------
# LightGBM LambdaRank パラメータ
# ---------------------------------------------------------------------------
LGBM_PARAMS_RANKING: dict = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3, 5],
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

# ---------------------------------------------------------------------------
# 欠損値のデフォルト
# ---------------------------------------------------------------------------
MISSING_NUMERIC: float = -1.0
MISSING_RATE: float = 0.0
MISSING_CATEGORY: str = "unknown"

# ---------------------------------------------------------------------------
# モデル保存先
# ---------------------------------------------------------------------------
MODEL_DIR: Path = PROJECT_ROOT / "models"
DATA_DIR: Path = PROJECT_ROOT / "data"
