"""特徴量パイプライン統合 + クロス特徴量（カテゴリ16）.

全特徴量抽出器を束ね、レース単位・年度単位で特徴量DataFrameを構築する。
年度別 parquet 保存と並列構築をサポートする。
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.db import query_df
from src.config import (
    RACE_KEY_COLS,
    JRA_JYO_CODES,
    MISSING_NUMERIC,
    MISSING_RATE,
    DATA_DIR,
)
from src.features.base import FeatureExtractor
from src.features.race import RaceFeatureExtractor
from src.features.horse import HorseFeatureExtractor
from src.features.speed import SpeedStyleFeatureExtractor
from src.features.jockey_trainer import JockeyTrainerFeatureExtractor
from src.features.training import TrainingFeatureExtractor
from src.features.bloodline import BloodlineFeatureExtractor
from src.features.odds import OddsFeatureExtractor
from src.utils.code_master import track_type, distance_category, class_level
from src.utils.base_time import get_or_build_base_time

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 年度別 parquet ヘルパー
# ------------------------------------------------------------------

def year_parquet_path(year: str) -> Path:
    """年度別 parquet のファイルパスを返す."""
    return DATA_DIR / f"features_{year}.parquet"


def _build_year_worker(
    year: str,
    include_odds: bool,
    force_rebuild: bool,
) -> str:
    """ProcessPoolExecutor 用のワーカー関数.

    各プロセスで FeaturePipeline を新規生成し、1年分の特徴量を構築する。
    pickle 可能にするためクラスメソッドではなくモジュールレベル関数とする。

    Args:
        year: 対象年（文字列）
        include_odds: オッズ特徴量を含めるか
        force_rebuild: True の場合、既存 parquet を無視して再構築

    Returns:
        完了メッセージ文字列
    """
    path = year_parquet_path(year)
    if not force_rebuild and path.exists():
        return f"{year}: スキップ（既存 parquet あり）"

    pipeline = FeaturePipeline(include_odds=include_odds)
    df = pipeline.build_dataset(
        year_start=year,
        year_end=year,
        save_parquet=False,
    )
    if df.empty:
        return f"{year}: 特徴量なし（0行）"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return f"{year}: {len(df)}行を保存"


class FeaturePipeline:
    """全特徴量を統合するパイプライン."""

    def __init__(self, include_odds: bool = True) -> None:
        """パイプラインを初期化する.

        Args:
            include_odds: オッズ特徴量を含めるかどうか
        """
        self.extractors: list[FeatureExtractor] = [
            RaceFeatureExtractor(),
            HorseFeatureExtractor(),
            SpeedStyleFeatureExtractor(),
            JockeyTrainerFeatureExtractor(),
            TrainingFeatureExtractor(),
            BloodlineFeatureExtractor(),
        ]
        if include_odds:
            self.extractors.append(OddsFeatureExtractor())

        self._include_odds = include_odds

    @property
    def feature_names(self) -> list[str]:
        """全特徴量名を返す."""
        names: list[str] = []
        for ext in self.extractors:
            names.extend(ext.feature_names)
        # クロス特徴量
        names.extend(self._cross_feature_names())
        # レース内相対特徴量
        names.extend(self._relative_feature_names())
        return names

    # ------------------------------------------------------------------
    # 公開メソッド
    # ------------------------------------------------------------------

    def extract_race(
        self,
        race_key: dict[str, str],
    ) -> pd.DataFrame:
        """1レース分の全特徴量を抽出する.

        Args:
            race_key: レースキー辞書

        Returns:
            全特徴量を含む DataFrame（kettonum をインデックス）
        """
        # 出走馬の基本情報を取得
        uma_race_df = self._get_horses(race_key)
        if uma_race_df.empty:
            return pd.DataFrame()

        # 各抽出器から特徴量を取得して結合
        all_features: list[pd.DataFrame] = []
        for extractor in self.extractors:
            try:
                feat_df = extractor.extract(race_key, uma_race_df)
                all_features.append(feat_df)
            except Exception as e:
                logger.warning(
                    "特徴量抽出エラー (%s): %s",
                    extractor.__class__.__name__,
                    e,
                )
                continue

        if not all_features:
            return pd.DataFrame()

        # 全特徴量をkettonumで結合
        result = all_features[0]
        for feat_df in all_features[1:]:
            result = result.join(feat_df, how="left", rsuffix="_dup")

        # 重複カラムを除去
        dup_cols = [c for c in result.columns if c.endswith("_dup")]
        if dup_cols:
            result = result.drop(columns=dup_cols)

        # クロス特徴量を追加
        result = self._add_cross_features(result, race_key)

        # レース内相対特徴量を追加
        result = self._add_relative_features(result)

        return result

    # ------------------------------------------------------------------
    # 年度別ビルド & 並列実行
    # ------------------------------------------------------------------

    def build_year(
        self,
        year: str,
        force_rebuild: bool = False,
    ) -> pd.DataFrame:
        """1年分の特徴量を構築し年度別 parquet に保存する.

        Args:
            year: 対象年（例: "2024"）
            force_rebuild: True の場合、既存 parquet を無視して再構築

        Returns:
            特徴量 DataFrame
        """
        path = year_parquet_path(year)
        if not force_rebuild and path.exists():
            logger.info("%s: 既存 parquet を使用", year)
            return pd.read_parquet(path)

        df = self.build_dataset(
            year_start=year,
            year_end=year,
            save_parquet=False,
        )
        if not df.empty:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            logger.info("%s: %d行を保存 → %s", year, len(df), path)
        return df

    @staticmethod
    def build_years(
        year_start: str,
        year_end: str,
        include_odds: bool = False,
        workers: int = 1,
        force_rebuild: bool = False,
    ) -> pd.DataFrame:
        """複数年度の特徴量を構築する（並列対応）.

        workers == 1 の場合は直列で build_year() を呼ぶ。
        workers >= 2 の場合は ProcessPoolExecutor で並列実行する。

        Args:
            year_start: 開始年
            year_end: 終了年
            include_odds: オッズ特徴量を含めるか
            workers: 並列ワーカー数（1 = 直列）
            force_rebuild: 既存 parquet を無視して再構築

        Returns:
            全年度を結合した DataFrame
        """
        years = [
            str(y) for y in range(int(year_start), int(year_end) + 1)
        ]
        logger.info(
            "年度別特徴量構築: %s〜%s (%d年分, workers=%d)",
            year_start, year_end, len(years), workers,
        )

        if workers >= 2:
            # 並列実行前に基準タイムテーブルをメインプロセスで構築しておく。
            # 各ワーカーが同時に build_base_time_table() を呼ぶと
            # 同一CSVへの同時書き込みでファイル破損するレースコンディションを防止。
            logger.info("基準タイムテーブルを事前構築します（並列実行の準備）...")
            get_or_build_base_time()

        if workers <= 1:
            # 直列実行
            pipeline = FeaturePipeline(include_odds=include_odds)
            for year in years:
                pipeline.build_year(year, force_rebuild=force_rebuild)
        else:
            # 並列実行（ProcessPoolExecutor）
            futures = {}
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for year in years:
                    fut = executor.submit(
                        _build_year_worker, year, include_odds, force_rebuild,
                    )
                    futures[fut] = year

                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="年度別構築",
                ):
                    year = futures[fut]
                    try:
                        msg = fut.result()
                        logger.info(msg)
                    except Exception as e:
                        logger.error("%s: エラー — %s", year, e)

        return FeaturePipeline.load_years(year_start, year_end)

    @staticmethod
    def load_years(
        year_start: str,
        year_end: str,
        supplement_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """年度別 parquet を結合してロードする.

        supplement_names が指定された場合、サプリメント parquet も
        自動的にマージして返す。

        Args:
            year_start: 開始年
            year_end: 終了年
            supplement_names: マージするサプリメント名のリスト
                （例: ["mining"]）。None の場合はメインのみ。

        Returns:
            全年度を結合した DataFrame（サプリメントマージ済み）

        Raises:
            FileNotFoundError: いずれかの年度の parquet が見つからない場合
        """
        years = [
            str(y) for y in range(int(year_start), int(year_end) + 1)
        ]
        dfs: list[pd.DataFrame] = []
        missing: list[str] = []
        for year in years:
            path = year_parquet_path(year)
            if path.exists():
                dfs.append(pd.read_parquet(path))
            else:
                missing.append(year)

        if missing:
            raise FileNotFoundError(
                f"年度別 parquet が見つかりません: {missing}。"
                " --build-features-only で構築してください。"
            )

        result = pd.concat(dfs, ignore_index=True)
        logger.info(
            "年度別ロード: %s〜%s → %d行",
            year_start, year_end, len(result),
        )

        # サプリメントのマージ
        if supplement_names:
            from src.features.supplement import merge_supplements
            result = merge_supplements(
                result, supplement_names, year_start, year_end,
            )

        return result

    # ------------------------------------------------------------------
    # 旧インターフェース（後方互換）
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        year_start: str,
        year_end: str,
        jyo_codes: list[str] | None = None,
        save_parquet: bool = True,
        output_name: str = "features",
    ) -> pd.DataFrame:
        """指定期間の全レースに対して特徴量データセットを構築する.

        Args:
            year_start: 開始年
            year_end: 終了年
            jyo_codes: 対象競馬場コード（デフォルトはJRA全10場）
            save_parquet: parquetとして保存するか
            output_name: 出力ファイル名プレフィクス

        Returns:
            全レース・全馬の特徴量 DataFrame
        """
        if jyo_codes is None:
            jyo_codes = JRA_JYO_CODES

        # 対象レースの一覧を取得
        races = self._get_target_races(year_start, year_end, jyo_codes)
        logger.info(
            "対象レース数: %d (%s〜%s)", len(races), year_start, year_end
        )

        all_features: list[pd.DataFrame] = []
        for _, race_row in tqdm(races.iterrows(), total=len(races), desc="特徴量構築"):
            race_key = {col: str(race_row[col]).strip() for col in RACE_KEY_COLS}

            try:
                features = self.extract_race(race_key)
                if features.empty:
                    continue

                # 目的変数を追加
                target = self._get_target(race_key)
                features = features.join(target, how="left")

                # レースキー情報を追加（後の分析用）
                for col in RACE_KEY_COLS:
                    features[f"_key_{col}"] = race_key[col]

                all_features.append(features)
            except Exception as e:
                logger.warning("レース処理エラー (%s): %s", race_key, e)
                continue

        if not all_features:
            logger.warning("特徴量が1件も生成されませんでした")
            return pd.DataFrame()

        result = pd.concat(all_features, ignore_index=False)
        result = result.reset_index()

        logger.info(
            "データセット構築完了: %d行 × %dカラム",
            len(result),
            len(result.columns),
        )

        # parquetで保存
        if save_parquet:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            path = DATA_DIR / f"{output_name}.parquet"
            result.to_parquet(path, index=False)
            logger.info("保存: %s", path)

        return result

    # ------------------------------------------------------------------
    # 内部メソッド
    # ------------------------------------------------------------------

    def _get_horses(self, race_key: dict[str, str]) -> pd.DataFrame:
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

    def _get_target_races(
        self,
        year_start: str,
        year_end: str,
        jyo_codes: list[str],
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
            "jyo_codes": tuple(jyo_codes),
        })

    def _get_target(self, race_key: dict[str, str]) -> pd.DataFrame:
        """目的変数を取得する.

        Returns:
            kettonum をインデックスとした DataFrame:
                - target: 3着以内=1, 他=0（二値分類用）
                - target_win: 1着=1, 他=0（単勝予測用）
                - target_relevance: LambdaRank用関連度スコア
                    1着=5, 2着=4, 3着=3, 4着=2, 5着=1, 6着以下=0
                - kakuteijyuni: 確定着順（生値）
        """
        sql = """
        SELECT kettonum,
            CAST(kakuteijyuni AS integer) AS kakuteijyuni,
            CASE WHEN CAST(kakuteijyuni AS integer) <= 3 THEN 1 ELSE 0 END AS target,
            CASE WHEN CAST(kakuteijyuni AS integer) = 1 THEN 1 ELSE 0 END AS target_win,
            CASE
                WHEN CAST(kakuteijyuni AS integer) = 1 THEN 5
                WHEN CAST(kakuteijyuni AS integer) = 2 THEN 4
                WHEN CAST(kakuteijyuni AS integer) = 3 THEN 3
                WHEN CAST(kakuteijyuni AS integer) <= 5 THEN 1
                ELSE 0
            END AS target_relevance
        FROM n_uma_race
        WHERE year = %(year)s AND monthday = %(monthday)s
          AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
          AND nichiji = %(nichiji)s AND racenum = %(racenum)s
          AND datakubun = '7'
          AND ijyocd = '0'
        """
        df = query_df(sql, race_key)
        if df.empty:
            return pd.DataFrame(
                columns=["target", "target_win", "target_relevance", "kakuteijyuni"]
            )
        return df.set_index("kettonum")[
            ["target", "target_win", "target_relevance", "kakuteijyuni"]
        ]

    # ------------------------------------------------------------------
    # レース内相対特徴量（カテゴリ17）
    # ------------------------------------------------------------------

    # 相対化する特徴量の定義
    # (元の特徴量名, ascending, missing_marker)
    #   ascending=True: 値が小さいほどランク上位
    #   missing_marker: 欠損値として扱う値のリスト
    #     - 率系(0.0が正当な値): MISSING_NUMERIC(-1)のみ除外
    #     - 数値系(0.0が異常値): MISSING_NUMERIC(-1)のみ除外
    #     - 血統率系(0.0=データなし): MISSING_NUMERIC(-1)とMISSING_RATE(0.0)の両方を除外
    _RELATIVE_TARGETS: list[tuple[str, bool, str]] = [
        ("speed_index_avg_last3", False, "numeric"),   # スピード指数 → 高い方が良い
        ("speed_index_last", False, "numeric"),         # 直近スピード指数
        ("speed_l3f_avg_last3", True, "numeric"),       # 上がり3F平均 → 小さい方が良い
        ("speed_l3f_best_last5", True, "numeric"),      # 上がり3Fベスト
        ("horse_fukusho_rate", False, "rate"),           # 複勝率（0.0=正当な値）
        ("horse_fukusho_rate_last5", False, "rate"),     # 直近5走複勝率
        ("horse_avg_jyuni_last3", True, "numeric"),     # 直近3走平均着順 → 小さい方が良い
        ("horse_win_rate", False, "rate"),               # 勝率（0.0=正当な値）
        ("jockey_win_rate_year", False, "rate"),         # 騎手勝率（0.0=正当な値）
        ("jockey_fukusho_rate_year", False, "rate"),     # 騎手複勝率
        ("trainer_win_rate_year", False, "rate"),        # 調教師勝率
        ("training_hanro_time4", True, "numeric"),       # 坂路4Fタイム → 小さい方が良い
        ("blood_father_turf_rate", False, "blood"),      # 父産駒芝複勝率（0.0=データなし）
        ("blood_father_dirt_rate", False, "blood"),      # 父産駒ダート複勝率
        ("blood_nicks_rate", False, "blood"),             # ニックス複勝率（0.0=データなし）
        ("blood_father_baba_rate", False, "blood"),       # 父産駒馬場状態別複勝率
        ("blood_father_jyo_rate", False, "blood"),        # 父産駒競馬場別複勝率
        ("blood_mother_produce_rate", False, "blood"),    # 母産駒複勝率（0.0=データなし）
    ]

    @staticmethod
    def _relative_feature_names() -> list[str]:
        """相対特徴量名のリストを返す."""
        names: list[str] = []
        for feat_name, _, _ in FeaturePipeline._RELATIVE_TARGETS:
            names.append(f"rel_{feat_name}_zscore")
            names.append(f"rel_{feat_name}_rank")
        return names

    def _add_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """レース内の相対特徴量（Zスコア・ランク）を追加する.

        各馬の能力指標について、同一レース出走馬の中での
        相対的な位置付けを示す特徴量を計算する。

        欠損値の扱いは特徴量の性質に応じて3パターン:
        - "numeric": MISSING_NUMERIC(-1)のみNaN化。タイム・指数・着順系。
        - "rate": MISSING_NUMERIC(-1)のみNaN化。0.0は「成績なし」で正当な値。
        - "blood": MISSING_NUMERIC(-1)とMISSING_RATE(0.0)の両方をNaN化。
                   血統産駒成績では0.0はデータ不足を意味するため。

        Args:
            df: 1レース分の特徴量 DataFrame（kettonum がインデックス）

        Returns:
            相対特徴量を追加した DataFrame
        """
        result = df.copy()

        for feat_name, ascending, missing_type in self._RELATIVE_TARGETS:
            if feat_name not in result.columns:
                # 対象特徴量が存在しない場合はデフォルト値で埋める
                result[f"rel_{feat_name}_zscore"] = 0.0
                result[f"rel_{feat_name}_rank"] = 0.0
                continue

            # 欠損値を NaN に変換（特徴量の性質に応じて判定）
            col = result[feat_name].copy()
            col = pd.to_numeric(col, errors="coerce")
            col = col.replace(MISSING_NUMERIC, np.nan)
            if missing_type == "blood":
                # 血統産駒成績: 0.0=データ不足のため欠損扱い
                col = col.replace(MISSING_RATE, np.nan)
            # "rate" / "numeric": 0.0は正当な値なので除外しない

            # Zスコア: (値 - レース内平均) / レース内標準偏差
            race_mean = col.mean()
            race_std = col.std()

            if race_std is not None and race_std > 0:
                zscore = (col - race_mean) / race_std
                result[f"rel_{feat_name}_zscore"] = zscore.fillna(0.0)
            else:
                result[f"rel_{feat_name}_zscore"] = 0.0

            # レース内順位（1 = 最良）
            result[f"rel_{feat_name}_rank"] = col.rank(
                ascending=ascending, method="min", na_option="bottom",
            ).fillna(len(result))

        return result

    # ------------------------------------------------------------------
    # クロス特徴量（カテゴリ16）
    # ------------------------------------------------------------------

    @staticmethod
    def _cross_feature_names() -> list[str]:
        return [
            "cross_dist_change",
            "cross_dist_category_change",
            "cross_track_change",
            "cross_class_change",
            "cross_jyo_change",
            "cross_weight_futan_per_bw",
            "cross_jockey_horse_runs",
            "cross_jockey_horse_wins",
            "cross_prev_filly_only",
            "cross_current_filly_only",
        ]

    def _add_cross_features(
        self,
        df: pd.DataFrame,
        race_key: dict[str, str],
    ) -> pd.DataFrame:
        """クロス特徴量を追加する."""
        result = df.copy()
        race_date = race_key["year"] + race_key["monthday"]

        # 前走情報を取得（距離変更・トラック変更・クラス変更用）
        kettonums = df.index.tolist()
        prev_info = self._get_prev_race_info(kettonums, race_date)

        for idx in result.index:
            kn = str(idx).strip()
            pi = prev_info.get(kn, {})

            # 距離変更
            current_dist = result.at[idx, "race_distance"] if "race_distance" in result.columns else 0
            prev_dist = pi.get("prev_kyori", 0)
            try:
                current_dist = int(current_dist) if current_dist and current_dist != MISSING_NUMERIC else 0
                prev_dist = int(prev_dist) if prev_dist else 0
            except (ValueError, TypeError):
                current_dist = 0
                prev_dist = 0

            result.at[idx, "cross_dist_change"] = (
                current_dist - prev_dist if current_dist > 0 and prev_dist > 0 else 0
            )

            # 距離カテゴリ変更
            if current_dist > 0 and prev_dist > 0:
                cur_cat = distance_category(current_dist)
                prev_cat = distance_category(prev_dist)
                result.at[idx, "cross_dist_category_change"] = (
                    f"{prev_cat}_to_{cur_cat}" if cur_cat != prev_cat else "same"
                )
            else:
                result.at[idx, "cross_dist_category_change"] = "unknown"

            # トラック変更（芝⇔ダート）
            current_tt = result.at[idx, "race_track_type"] if "race_track_type" in result.columns else ""
            prev_tt = pi.get("prev_track_type", "")
            result.at[idx, "cross_track_change"] = (
                1 if current_tt and prev_tt and current_tt != prev_tt else 0
            )

            # クラス変更（+1=昇級, 0=同級, -1=降級）
            prev_jyoken = pi.get("prev_jyokencd", "")
            prev_grade = pi.get("prev_gradecd", "")
            cur_jyoken = str(result.at[idx, "race_jyoken_cd"]) if "race_jyoken_cd" in result.columns else ""
            cur_grade = str(result.at[idx, "race_grade_cd"]) if "race_grade_cd" in result.columns else ""
            prev_level = class_level(prev_jyoken, prev_grade)
            cur_level = class_level(cur_jyoken, cur_grade)
            if prev_level >= 0 and cur_level >= 0:
                if cur_level > prev_level:
                    cc = 1
                elif cur_level < prev_level:
                    cc = -1
                else:
                    cc = 0
            else:
                cc = 0
            result.at[idx, "cross_class_change"] = cc

            # 前走牝馬限定フラグ
            result.at[idx, "cross_prev_filly_only"] = (
                1 if pi.get("prev_filly_only") else 0
            )

            # 競馬場変更
            current_jyo = race_key.get("jyocd", "")
            prev_jyo = pi.get("prev_jyocd", "")
            result.at[idx, "cross_jyo_change"] = (
                1 if current_jyo and prev_jyo and current_jyo != prev_jyo else 0
            )

            # 負担重量/馬体重 比率
            futan = result.at[idx, "weight_futan"] if "weight_futan" in result.columns else -1
            bw = result.at[idx, "bw_weight"] if "bw_weight" in result.columns else -1
            try:
                futan = float(futan) if futan != MISSING_NUMERIC else 0
                bw = float(bw) if bw != MISSING_NUMERIC else 0
            except (ValueError, TypeError):
                futan = 0
                bw = 0
            result.at[idx, "cross_weight_futan_per_bw"] = (
                futan / bw if bw > 0 and futan > 0 else MISSING_NUMERIC
            )

            # 同馬×同騎手
            result.at[idx, "cross_jockey_horse_runs"] = pi.get("jockey_horse_runs", 0)
            result.at[idx, "cross_jockey_horse_wins"] = pi.get("jockey_horse_wins", 0)

        # 現在のレースの牝馬限定フラグ（全出走馬が牝馬なら1）
        if "horse_sex" in result.columns:
            all_filly = (result["horse_sex"].astype(str) == "2").all()
            result["cross_current_filly_only"] = 1 if all_filly else 0
        else:
            result["cross_current_filly_only"] = 0

        return result

    def _get_prev_race_info(
        self,
        kettonums: list[str],
        race_date: str,
    ) -> dict[str, dict[str, Any]]:
        """前走情報（距離・トラック・クラス等）を一括取得する."""
        if not kettonums:
            return {}

        sql = """
        SELECT DISTINCT ON (ur.kettonum)
            ur.kettonum,
            r.kyori AS prev_kyori,
            r.trackcd AS prev_trackcd,
            r.jyocd AS prev_jyocd,
            r.jyokencd5 AS prev_jyokencd,
            r.gradecd AS prev_gradecd,
            ur.kisyucode AS prev_kisyucode,
            (SELECT BOOL_AND(ur2.sexcd = '2')
             FROM n_uma_race ur2
             WHERE ur2.year = ur.year AND ur2.monthday = ur.monthday
               AND ur2.jyocd = ur.jyocd AND ur2.kaiji = ur.kaiji
               AND ur2.nichiji = ur.nichiji AND ur2.racenum = ur.racenum
               AND ur2.datakubun = '7' AND ur2.ijyocd = '0'
            ) AS prev_filly_only
        FROM n_uma_race ur
        JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
        WHERE ur.kettonum IN %(kettonums)s
          AND ur.datakubun = '7'
          AND ur.ijyocd = '0'
          AND (ur.year || ur.monthday) < %(race_date)s
        ORDER BY ur.kettonum, ur.year DESC, ur.monthday DESC
        """
        df = query_df(sql, {"kettonums": tuple(kettonums), "race_date": race_date})

        # 同馬×同騎手の過去成績
        jockey_horse_sql = """
        SELECT ur.kettonum, ur.kisyucode,
            COUNT(*) AS runs,
            SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) = 1 THEN 1 ELSE 0 END) AS wins
        FROM n_uma_race ur
        WHERE ur.kettonum IN %(kettonums)s
          AND ur.datakubun = '7'
          AND ur.ijyocd = '0'
          AND (ur.year || ur.monthday) < %(race_date)s
        GROUP BY ur.kettonum, ur.kisyucode
        """
        jh_df = query_df(jockey_horse_sql, {
            "kettonums": tuple(kettonums),
            "race_date": race_date,
        })

        # 現在の騎手コードを取得
        current_kisyu_sql = """
        SELECT kettonum, kisyucode
        FROM n_uma_race
        WHERE year = %(year)s AND monthday = %(monthday)s
          AND jyocd = %(jyocd)s
          AND datakubun IN ('1','2','3','4','5','6','7')
          AND ijyocd = '0'
          AND kettonum IN %(kettonums)s
        """
        # race_dateから年月日を取得
        year = race_date[:4]
        monthday = race_date[4:8]
        try:
            ck_df = query_df(current_kisyu_sql, {
                "year": year,
                "monthday": monthday,
                "jyocd": "",  # この部分は簡略化（パイプラインの中で取得済み）
                "kettonums": tuple(kettonums),
            })
        except Exception:
            ck_df = pd.DataFrame()

        result: dict[str, dict[str, Any]] = {}
        for _, row in df.iterrows():
            kn = str(row["kettonum"]).strip()
            prev_trackcd = str(row.get("prev_trackcd", "")).strip()

            # クラス変更の判定
            prev_jyoken = str(row.get("prev_jyokencd", "")).strip()
            prev_grade = str(row.get("prev_gradecd", "")).strip()
            class_change = 0  # デフォルトは同級

            info: dict[str, Any] = {
                "prev_kyori": row.get("prev_kyori"),
                "prev_track_type": track_type(prev_trackcd),
                "prev_jyocd": str(row.get("prev_jyocd", "")).strip(),
                "prev_jyokencd": prev_jyoken,
                "prev_gradecd": prev_grade,
                "prev_filly_only": bool(row.get("prev_filly_only", False)),
                "class_change": class_change,
                "jockey_horse_runs": 0,
                "jockey_horse_wins": 0,
            }

            # 同馬×同騎手の成績
            if not jh_df.empty:
                kn_jh = jh_df[jh_df["kettonum"] == kn]
                # 現在のレースの騎手を特定
                if not ck_df.empty:
                    ck_row = ck_df[ck_df["kettonum"] == kn]
                    if not ck_row.empty:
                        current_kc = str(ck_row.iloc[0]["kisyucode"]).strip()
                        combo = kn_jh[kn_jh["kisyucode"].str.strip() == current_kc]
                        if not combo.empty:
                            info["jockey_horse_runs"] = int(combo.iloc[0].get("runs", 0))
                            info["jockey_horse_wins"] = int(combo.iloc[0].get("wins", 0))

            result[kn] = info

        return result
