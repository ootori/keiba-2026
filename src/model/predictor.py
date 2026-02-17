"""予測実行モジュール."""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from src.config import MODEL_DIR, CATEGORICAL_FEATURES
from src.db import query_df
from src.features.pipeline import FeaturePipeline
from src.utils.code_master import JYO_CODE_MAP, GRADE_CODE_MAP, BABA_CODE_MAP

logger = logging.getLogger(__name__)


class Predictor:
    """レース予測を実行する."""

    def __init__(
        self,
        model_name: str = "model",
        include_odds: bool = False,
    ) -> None:
        """予測器を初期化する.

        Args:
            model_name: モデルファイル名（.txt拡張子なし）
            include_odds: オッズ特徴量を使うか
        """
        self.model_name = model_name
        self.include_odds = include_odds
        self.model: lgb.Booster | None = None
        self.feature_columns: list[str] = []
        self.pipeline = FeaturePipeline(include_odds=include_odds)

    def load(self) -> None:
        """モデルと特徴量リストをロードする."""
        model_path = MODEL_DIR / f"{self.model_name}.txt"
        feature_path = MODEL_DIR / f"{self.model_name}_features.txt"

        if not model_path.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

        self.model = lgb.Booster(model_file=str(model_path))

        if feature_path.exists():
            with open(feature_path) as f:
                self.feature_columns = [
                    line.strip() for line in f if line.strip()
                ]
        logger.info(
            "モデルロード完了: %s (%d特徴量)",
            model_path,
            len(self.feature_columns),
        )

    def predict_race(
        self,
        race_key: dict[str, str],
    ) -> pd.DataFrame:
        """1レースの予測を実行する.

        Args:
            race_key: レースキー辞書
                {'year', 'monthday', 'jyocd', 'kaiji', 'nichiji', 'racenum'}

        Returns:
            予測結果 DataFrame:
                umaban, bamei, pred_prob, pred_rank
        """
        if self.model is None:
            self.load()

        # 特徴量抽出
        features = self.pipeline.extract_race(race_key)
        if features.empty:
            logger.warning("特徴量を抽出できませんでした: %s", race_key)
            return pd.DataFrame()

        # 出走馬情報を取得
        horse_info = self._get_horse_info(race_key)

        # 予測
        X = self._prepare_features(features)
        probs = self.model.predict(X)

        # 結果を構成
        result = pd.DataFrame(
            {
                "kettonum": features.index,
                "pred_prob": probs,
            }
        )

        # 馬名・馬番を結合
        if not horse_info.empty:
            result = result.merge(
                horse_info[["kettonum", "umaban", "bamei"]],
                on="kettonum",
                how="left",
            )
        else:
            result["umaban"] = ""
            result["bamei"] = ""

        result["pred_rank"] = (
            result["pred_prob"].rank(ascending=False).astype(int)
        )
        result = result.sort_values("pred_rank")

        return result[
            ["pred_rank", "umaban", "bamei", "pred_prob", "kettonum"]
        ].reset_index(drop=True)

    def predict_day(
        self,
        year: str,
        monthday: str,
        jyocd: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """1日分の全レースを予測する.

        Args:
            year: 年
            monthday: 月日
            jyocd: 競馬場コード（Noneの場合は全場）

        Returns:
            レースキー文字列 → 予測結果 DataFrame の辞書
        """
        races = self._get_day_races(year, monthday, jyocd)
        results: dict[str, pd.DataFrame] = {}

        for _, race in races.iterrows():
            race_key = {
                "year": str(race["year"]).strip(),
                "monthday": str(race["monthday"]).strip(),
                "jyocd": str(race["jyocd"]).strip(),
                "kaiji": str(race["kaiji"]).strip(),
                "nichiji": str(race["nichiji"]).strip(),
                "racenum": str(race["racenum"]).strip(),
            }

            try:
                pred = self.predict_race(race_key)
                key_str = f"{race_key['jyocd']}_{race_key['racenum']}R"
                results[key_str] = pred
            except Exception as e:
                logger.warning("予測エラー: %s - %s", race_key, e)

        return results

    def format_prediction(
        self,
        race_key: dict[str, str],
        prediction: pd.DataFrame,
    ) -> str:
        """予測結果をフォーマットして出力する.

        Args:
            race_key: レースキー辞書
            prediction: predict_race() の結果

        Returns:
            フォーマットされた予測結果文字列
        """
        # レース情報を取得
        race_info = self._get_race_detail(race_key)

        jyo_name = JYO_CODE_MAP.get(race_key["jyocd"], race_key["jyocd"])
        racenum = race_key["racenum"]

        lines: list[str] = []
        lines.append(
            f"=== {race_key['year']}/{race_key['monthday'][:2]}/"
            f"{race_key['monthday'][2:]} {jyo_name}{racenum}R ==="
        )

        if race_info:
            distance = race_info.get("kyori", "?")
            track_name = "芝" if race_info.get("track_type") == "turf" else "ダート"
            baba = BABA_CODE_MAP.get(race_info.get("baba_cd", ""), "?")
            tosu = race_info.get("tosu", "?")
            grade = GRADE_CODE_MAP.get(race_info.get("gradecd", ""), "")
            race_name = race_info.get("racename", "")
            if race_name:
                lines[0] += f" {race_name.strip()}"
            lines.append(f"{track_name}{distance}m {baba} {tosu}頭 {grade}")

        lines.append("")
        lines.append(f"{'予測':>4s}  {'馬番':>4s}  {'馬名':<16s}  {'確率':>6s}")
        lines.append("-" * 40)

        for _, row in prediction.iterrows():
            rank = int(row["pred_rank"])
            umaban = str(row.get("umaban", "")).strip()
            bamei = str(row.get("bamei", "")).strip()
            prob = row["pred_prob"] * 100

            lines.append(f"{rank:>4d}  {umaban:>4s}  {bamei:<16s}  {prob:>5.1f}%")

        # 推奨買い目
        if len(prediction) >= 3:
            top3 = prediction.head(3)
            top1_umaban = str(top3.iloc[0].get("umaban", "")).strip()
            top2_umaban = str(top3.iloc[1].get("umaban", "")).strip()
            top3_umaban = str(top3.iloc[2].get("umaban", "")).strip()
            lines.append("")
            lines.append(
                f"推奨: 複勝 {top1_umaban}, {top2_umaban}  "
                f"ワイド {top1_umaban}-{top2_umaban}"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 内部メソッド
    # ------------------------------------------------------------------

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """予測用に特徴量を整備する."""
        if not self.feature_columns:
            return features

        # 学習時の特徴量順に合わせる
        X = pd.DataFrame(index=features.index)
        for col in self.feature_columns:
            if col in features.columns:
                X[col] = features[col]
            else:
                X[col] = -1  # 欠損

        # カテゴリ変数を category 型に変換
        for col in CATEGORICAL_FEATURES:
            if col in X.columns:
                X[col] = X[col].astype("category")

        return X

    def _get_horse_info(self, race_key: dict[str, str]) -> pd.DataFrame:
        sql = """
        SELECT kettonum, umaban, bamei
        FROM n_uma_race
        WHERE year = %(year)s AND monthday = %(monthday)s
          AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
          AND nichiji = %(nichiji)s AND racenum = %(racenum)s
          AND datakubun IN ('1','2','3','4','5','6','7')
          AND ijyocd = '0'
        """
        return query_df(sql, race_key)

    def _get_day_races(
        self,
        year: str,
        monthday: str,
        jyocd: str | None,
    ) -> pd.DataFrame:
        if jyocd:
            sql = """
            SELECT DISTINCT year, monthday, jyocd, kaiji, nichiji, racenum
            FROM n_race
            WHERE year = %(year)s AND monthday = %(monthday)s AND jyocd = %(jyocd)s
            ORDER BY racenum
            """
            return query_df(sql, {"year": year, "monthday": monthday, "jyocd": jyocd})
        else:
            sql = """
            SELECT DISTINCT year, monthday, jyocd, kaiji, nichiji, racenum
            FROM n_race
            WHERE year = %(year)s AND monthday = %(monthday)s
              AND jyocd IN ('01','02','03','04','05','06','07','08','09','10')
            ORDER BY jyocd, racenum
            """
            return query_df(sql, {"year": year, "monthday": monthday})

    def _get_race_detail(self, race_key: dict[str, str]) -> dict | None:
        sql = """
        SELECT kyori, trackcd, sibababacd, dirtbabacd, gradecd,
               syussotosu, hondai
        FROM n_race
        WHERE year = %(year)s AND monthday = %(monthday)s
          AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
          AND nichiji = %(nichiji)s AND racenum = %(racenum)s
        LIMIT 1
        """
        df = query_df(sql, race_key)
        if df.empty:
            return None

        row = df.iloc[0]
        from src.utils.code_master import track_type as tt_fn, baba_code_for_track

        trackcd = str(row.get("trackcd", "")).strip()
        return {
            "kyori": str(row.get("kyori", "")).strip(),
            "track_type": tt_fn(trackcd),
            "baba_cd": baba_code_for_track(
                trackcd,
                str(row.get("sibababacd", "")).strip(),
                str(row.get("dirtbabacd", "")).strip(),
            ),
            "gradecd": str(row.get("gradecd", "")).strip(),
            "tosu": str(row.get("syussotosu", "")).strip(),
            "racename": str(row.get("hondai", "")).strip(),
        }
