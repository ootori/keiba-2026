"""差分特徴量（サプリメント）パイプライン.

メイン特徴量パイプライン（pipeline.py）で構築済みの年度別 parquet に対して、
追加の特徴量を別ファイルとして構築・保存し、学習/評価時にマージする仕組み。

これにより、新しい特徴量を追加するたびに全特徴量を再構築する必要がなくなる。

ディレクトリ構造:
    data/
    ├── features_2024.parquet           # メイン特徴量
    ├── features_2025.parquet
    └── supplements/
        ├── mining_2024.parquet         # マイニング特徴量（2024年）
        ├── mining_2025.parquet
        ├── pace_2024.parquet           # 将来追加: ペース特徴量
        └── ...

マージルール:
    - メイン parquet と supplement parquet はレースキー + kettonum で結合
    - 同名カラムが存在する場合は supplement 側を優先（上書き）
    - 複数 supplement を指定可能（順に left join）
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.db import query_df
from src.config import (
    RACE_KEY_COLS,
    JRA_JYO_CODES,
    DATA_DIR,
)
from src.features.base import FeatureExtractor

logger = logging.getLogger(__name__)

# サプリメント保存先ディレクトリ
SUPPLEMENT_DIR = DATA_DIR / "supplements"

# 結合キー（メインparquetとsupplementの結合に使用）
_MERGE_KEYS = [f"_key_{c}" for c in RACE_KEY_COLS] + ["kettonum"]


# ------------------------------------------------------------------
# サプリメント登録簿
# ------------------------------------------------------------------

# 利用可能なサプリメント名と対応する Extractor クラスのマッピング。
# 新しいサプリメントを追加する場合はここに登録する。
def _get_registry() -> dict[str, type[FeatureExtractor]]:
    """遅延インポートでサプリメント登録簿を返す."""
    from src.features.mining import MiningFeatureExtractor
    from src.features.bms_detail import BMSDetailFeatureExtractor

    return {
        "mining": MiningFeatureExtractor,
        "bms_detail": BMSDetailFeatureExtractor,
    }


# ------------------------------------------------------------------
# パスヘルパー
# ------------------------------------------------------------------

def supplement_parquet_path(name: str, year: str) -> Path:
    """サプリメント parquet のファイルパスを返す.

    Args:
        name: サプリメント名（例: "mining"）
        year: 年度

    Returns:
        パス（例: data/supplements/mining_2024.parquet）
    """
    return SUPPLEMENT_DIR / f"{name}_{year}.parquet"


def list_available_supplements() -> list[str]:
    """利用可能なサプリメント名の一覧を返す."""
    return list(_get_registry().keys())


# ------------------------------------------------------------------
# ビルド
# ------------------------------------------------------------------

def _build_supplement_year_worker(
    supplement_name: str,
    year: str,
    force_rebuild: bool,
) -> str:
    """ProcessPoolExecutor 用のワーカー関数."""
    path = supplement_parquet_path(supplement_name, year)
    if not force_rebuild and path.exists():
        return f"{supplement_name}/{year}: スキップ（既存あり）"

    df = build_supplement_year(supplement_name, year)
    if df.empty:
        return f"{supplement_name}/{year}: 0行"

    SUPPLEMENT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return f"{supplement_name}/{year}: {len(df)}行を保存"


def build_supplement_year(
    supplement_name: str,
    year: str,
) -> pd.DataFrame:
    """1年分のサプリメント特徴量を構築する.

    Args:
        supplement_name: サプリメント名
        year: 対象年

    Returns:
        kettonum + レースキー + 特徴量カラムを含む DataFrame
    """
    registry = _get_registry()
    if supplement_name not in registry:
        raise ValueError(
            f"未知のサプリメント: {supplement_name}。"
            f" 利用可能: {list(registry.keys())}"
        )

    extractor = registry[supplement_name]()

    # 対象レース一覧を取得
    races = _get_target_races(year, year)
    logger.info(
        "サプリメント[%s] %s年: %dレース",
        supplement_name, year, len(races),
    )

    all_rows: list[pd.DataFrame] = []
    for _, race_row in tqdm(
        races.iterrows(),
        total=len(races),
        desc=f"supplement:{supplement_name}/{year}",
    ):
        race_key = {col: str(race_row[col]).strip() for col in RACE_KEY_COLS}

        try:
            # 出走馬を取得
            uma_race_df = _get_horses(race_key)
            if uma_race_df.empty:
                continue

            feat_df = extractor.extract(race_key, uma_race_df)
            if feat_df.empty:
                continue

            # インデックスをカラムに戻す
            feat_df = feat_df.reset_index()

            # レースキーを付与（マージ用）
            for col in RACE_KEY_COLS:
                feat_df[f"_key_{col}"] = race_key[col]

            all_rows.append(feat_df)
        except Exception as e:
            logger.warning(
                "サプリメント[%s] レース処理エラー (%s): %s",
                supplement_name, race_key, e,
            )
            continue

    if not all_rows:
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)
    logger.info(
        "サプリメント[%s] %s年: %d行 × %dカラム構築完了",
        supplement_name, year, len(result), len(result.columns),
    )
    return result


def build_supplement_years(
    supplement_name: str,
    year_start: str,
    year_end: str,
    workers: int = 1,
    force_rebuild: bool = False,
) -> None:
    """複数年度のサプリメント特徴量を構築する（並列対応）.

    Args:
        supplement_name: サプリメント名
        year_start: 開始年
        year_end: 終了年
        workers: 並列ワーカー数
        force_rebuild: 既存を無視して再構築
    """
    years = [str(y) for y in range(int(year_start), int(year_end) + 1)]
    logger.info(
        "サプリメント[%s] 構築: %s〜%s (%d年分, workers=%d)",
        supplement_name, year_start, year_end, len(years), workers,
    )

    if workers <= 1:
        for year in years:
            path = supplement_parquet_path(supplement_name, year)
            if not force_rebuild and path.exists():
                logger.info("%s/%s: 既存あり → スキップ", supplement_name, year)
                continue
            df = build_supplement_year(supplement_name, year)
            if not df.empty:
                SUPPLEMENT_DIR.mkdir(parents=True, exist_ok=True)
                df.to_parquet(path, index=False)
                logger.info(
                    "%s/%s: %d行を保存", supplement_name, year, len(df),
                )
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for year in years:
                fut = executor.submit(
                    _build_supplement_year_worker,
                    supplement_name, year, force_rebuild,
                )
                futures[fut] = year

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"supplement:{supplement_name}",
            ):
                year = futures[fut]
                try:
                    msg = fut.result()
                    logger.info(msg)
                except Exception as e:
                    logger.error("%s/%s: エラー — %s", supplement_name, year, e)


# ------------------------------------------------------------------
# ロード & マージ
# ------------------------------------------------------------------

def load_supplement_years(
    supplement_name: str,
    year_start: str,
    year_end: str,
) -> pd.DataFrame:
    """年度別サプリメント parquet を結合してロードする.

    Args:
        supplement_name: サプリメント名
        year_start: 開始年
        year_end: 終了年

    Returns:
        結合された DataFrame

    Raises:
        FileNotFoundError: いずれかの年度の parquet が見つからない場合
    """
    years = [str(y) for y in range(int(year_start), int(year_end) + 1)]
    dfs: list[pd.DataFrame] = []
    missing: list[str] = []

    for year in years:
        path = supplement_parquet_path(supplement_name, year)
        if path.exists():
            dfs.append(pd.read_parquet(path))
        else:
            missing.append(year)

    if missing:
        raise FileNotFoundError(
            f"サプリメント[{supplement_name}] parquet が見つかりません: {missing}。"
            f" --build-supplement {supplement_name} で構築してください。"
        )

    result = pd.concat(dfs, ignore_index=True)
    logger.info(
        "サプリメント[%s] ロード: %s〜%s → %d行",
        supplement_name, year_start, year_end, len(result),
    )
    return result


def merge_supplements(
    main_df: pd.DataFrame,
    supplement_names: list[str],
    year_start: str,
    year_end: str,
) -> pd.DataFrame:
    """メイン DataFrame にサプリメント特徴量をマージする.

    メイン parquet の各行に対して、レースキー + kettonum で
    サプリメントの特徴量カラムを left join する。

    同名カラムが存在する場合はサプリメント側で上書きする。

    Args:
        main_df: メインの特徴量 DataFrame
        supplement_names: マージするサプリメント名のリスト
        year_start: 開始年
        year_end: 終了年

    Returns:
        サプリメント特徴量がマージされた DataFrame
    """
    if not supplement_names:
        return main_df

    result = main_df.copy()

    # マージキーの存在を確認
    available_keys = [k for k in _MERGE_KEYS if k in result.columns]
    if not available_keys:
        logger.warning(
            "マージキー(%s)がメインDFに見つかりません。マージをスキップします。",
            _MERGE_KEYS,
        )
        return result

    for name in supplement_names:
        try:
            supp_df = load_supplement_years(name, year_start, year_end)
        except FileNotFoundError as e:
            logger.warning("サプリメント[%s] のロードに失敗: %s", name, e)
            continue

        if supp_df.empty:
            logger.warning("サプリメント[%s] が空です", name)
            continue

        # 結合キーの共通部分を特定
        merge_keys = [k for k in available_keys if k in supp_df.columns]
        if not merge_keys:
            logger.warning(
                "サプリメント[%s] にマージキーが見つかりません: %s",
                name, available_keys,
            )
            continue

        # サプリメント側の新カラム（マージキー以外）
        supp_new_cols = [
            c for c in supp_df.columns if c not in merge_keys
        ]
        if not supp_new_cols:
            logger.warning("サプリメント[%s] に新しい特徴量がありません", name)
            continue

        # 既存カラムとの重複を処理
        overlap_cols = [c for c in supp_new_cols if c in result.columns]
        if overlap_cols:
            logger.info(
                "サプリメント[%s] 重複カラム（上書き）: %s",
                name, overlap_cols,
            )
            result = result.drop(columns=overlap_cols)

        # left join
        supp_merge = supp_df[merge_keys + supp_new_cols].copy()

        # マージキーの型を揃える
        for k in merge_keys:
            result[k] = result[k].astype(str).str.strip()
            supp_merge[k] = supp_merge[k].astype(str).str.strip()

        result = result.merge(supp_merge, on=merge_keys, how="left")

        logger.info(
            "サプリメント[%s] マージ完了: +%dカラム",
            name, len(supp_new_cols),
        )

    return result


# ------------------------------------------------------------------
# 内部ヘルパー（pipeline.py と同等のDB取得）
# ------------------------------------------------------------------

def _get_target_races(
    year_start: str,
    year_end: str,
) -> pd.DataFrame:
    """対象期間のレース一覧を取得する."""
    sql = """
    SELECT DISTINCT year, monthday, jyocd, kaiji, nichiji, racenum
    FROM n_race
    WHERE datakubun = '7'
      AND year >= %(year_start)s
      AND year <= %(year_end)s
      AND jyocd IN %(jyo_codes)s
    ORDER BY year, monthday, jyocd, racenum
    """
    return query_df(sql, {
        "year_start": year_start,
        "year_end": year_end,
        "jyo_codes": tuple(JRA_JYO_CODES),
    })


def _get_horses(race_key: dict[str, str]) -> pd.DataFrame:
    """出走馬のkettonumとumaban等を取得する."""
    sql = """
    SELECT kettonum, umaban, bamei
    FROM n_uma_race
    WHERE year = %(year)s AND monthday = %(monthday)s
      AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
      AND nichiji = %(nichiji)s AND racenum = %(racenum)s
      AND datakubun IN ('1','2','3','4','5','6','7')
      AND ijyocd = '0'
    ORDER BY CAST(umaban AS integer)
    """
    return query_df(sql, race_key)
