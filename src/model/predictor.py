"""予測実行モジュール."""

from __future__ import annotations

import json
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
        self.ranking: bool = False
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

        # メタデータから ranking フラグを復元
        meta_path = MODEL_DIR / f"{self.model_name}_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.ranking = meta.get("ranking", False)

        logger.info(
            "モデルロード完了: %s (%d特徴量, ranking=%s)",
            model_path,
            len(self.feature_columns),
            self.ranking,
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
    ) -> list[tuple[dict[str, str], pd.DataFrame]]:
        """1日分の全レースを予測する.

        Args:
            year: 年
            monthday: 月日
            jyocd: 競馬場コード（Noneの場合は全場）

        Returns:
            (レースキー辞書, 予測結果DataFrame) のリスト
        """
        races = self._get_day_races(year, monthday, jyocd)
        results: list[tuple[dict[str, str], pd.DataFrame]] = []

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
                results.append((race_key, pred))
            except Exception as e:
                logger.warning("予測エラー: %s - %s", race_key, e)

        return results

    def format_prediction(
        self,
        race_key: dict[str, str],
        prediction: pd.DataFrame,
        odds_correction_config: dict | None = None,
    ) -> str:
        """予測結果をフォーマットして出力する.

        DBからオッズを取得できた場合はオッズ・補正後オッズ・EV・人気も表示する。

        Args:
            race_key: レースキー辞書
            prediction: predict_race() の結果
            odds_correction_config: オッズ補正設定（Noneの場合は補正なし）

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

        # モデル出力をレース内合計で割って正規化（合計100%にする）
        # LambdaRank の場合は softmax、二値分類の場合は ratio 正規化
        import numpy as np
        raw_probs = prediction["pred_prob"].copy()
        if self.ranking:
            exp_scores = np.exp(raw_probs - raw_probs.max())
            total = exp_scores.sum()
            if total > 0:
                prediction = prediction.copy()
                prediction["pred_prob"] = exp_scores / total
        else:
            total = raw_probs.sum()
            if total > 0:
                prediction = prediction.copy()
                prediction["pred_prob"] = raw_probs / total

        # オッズをDBから取得
        odds_dict = self._get_odds_from_db(race_key)
        has_odds = bool(odds_dict)

        # 人気順を導出
        ninki_ranks: dict[str, int] = {}
        if has_odds:
            sorted_odds = sorted(odds_dict.items(), key=lambda x: x[1])
            for i, (uma, _) in enumerate(sorted_odds, 1):
                ninki_ranks[uma] = i

        # オッズ補正用 evaluator（補正設定がある場合のみ）
        evaluator = None
        if has_odds and odds_correction_config:
            try:
                from src.model.evaluator import ModelEvaluator
                evaluator = ModelEvaluator()
            except Exception:
                pass

        lines.append("")
        if has_odds:
            if evaluator:
                lines.append(
                    f"{'予測':>4s}  {'馬番':>4s}  {'馬名':<16s}  "
                    f"{'確率':>6s}  {'人気':>4s}  {'ｵｯｽﾞ':>6s}  "
                    f"{'補正後':>6s}  {'EV':>5s}"
                )
                lines.append("-" * 72)
            else:
                lines.append(
                    f"{'予測':>4s}  {'馬番':>4s}  {'馬名':<16s}  "
                    f"{'確率':>6s}  {'人気':>4s}  {'ｵｯｽﾞ':>6s}  {'EV':>5s}"
                )
                lines.append("-" * 64)
        else:
            if self.ranking:
                lines.append(
                    f"{'予測':>4s}  {'馬番':>4s}  {'馬名':<16s}  {'スコア':>6s}"
                )
            else:
                lines.append(
                    f"{'予測':>4s}  {'馬番':>4s}  {'馬名':<16s}  {'確率':>6s}"
                )
            lines.append("-" * 40)

        for _, row in prediction.iterrows():
            rank = int(row["pred_rank"])
            umaban = str(row.get("umaban", "")).strip()
            bamei = str(row.get("bamei", "")).strip()
            pred_prob = row["pred_prob"]

            if has_odds:
                umaban_key = umaban.zfill(2)
                raw_odds = odds_dict.get(umaban_key, 0.0)
                ninki = ninki_ranks.get(umaban_key, 0)

                if evaluator and raw_odds > 0:
                    dummy_row = pd.Series({"post_umaban": int(umaban) if umaban.isdigit() else 0})
                    corrected_odds = evaluator._apply_odds_correction(
                        raw_odds, dummy_row, ninki, odds_correction_config,
                    )
                    ev = pred_prob * corrected_odds
                    marker = " *" if ev >= 1.1 and pred_prob >= 0.025 else ""
                    if self.ranking:
                        lines.append(
                            f"{rank:>4d}  {umaban:>4s}  {bamei:<16s}  "
                            f"{pred_prob:>6.2f}  {ninki:>4d}  {raw_odds:>6.1f}  "
                            f"{corrected_odds:>6.1f}  {ev:>5.2f}{marker}"
                        )
                    else:
                        prob_pct = pred_prob * 100
                        lines.append(
                            f"{rank:>4d}  {umaban:>4s}  {bamei:<16s}  "
                            f"{prob_pct:>5.1f}%  {ninki:>4d}  {raw_odds:>6.1f}  "
                            f"{corrected_odds:>6.1f}  {ev:>5.2f}{marker}"
                        )
                elif raw_odds > 0:
                    ev = pred_prob * raw_odds
                    marker = " *" if ev >= 1.1 and pred_prob >= 0.025 else ""
                    if self.ranking:
                        lines.append(
                            f"{rank:>4d}  {umaban:>4s}  {bamei:<16s}  "
                            f"{pred_prob:>6.2f}  {ninki:>4d}  {raw_odds:>6.1f}  "
                            f"{ev:>5.2f}{marker}"
                        )
                    else:
                        prob_pct = pred_prob * 100
                        lines.append(
                            f"{rank:>4d}  {umaban:>4s}  {bamei:<16s}  "
                            f"{prob_pct:>5.1f}%  {ninki:>4d}  {raw_odds:>6.1f}  "
                            f"{ev:>5.2f}{marker}"
                        )
                else:
                    # オッズなし（出走取消等）
                    if self.ranking:
                        lines.append(
                            f"{rank:>4d}  {umaban:>4s}  {bamei:<16s}  "
                            f"{pred_prob:>6.2f}  {'--':>4s}  {'--':>6s}  {'--':>5s}"
                        )
                    else:
                        prob_pct = pred_prob * 100
                        lines.append(
                            f"{rank:>4d}  {umaban:>4s}  {bamei:<16s}  "
                            f"{prob_pct:>5.1f}%  {'--':>4s}  {'--':>6s}  {'--':>5s}"
                        )
            else:
                # オッズ取得不可（DBアクセス不可等）
                if self.ranking:
                    lines.append(
                        f"{rank:>4d}  {umaban:>4s}  {bamei:<16s}  {pred_prob:>6.2f}"
                    )
                else:
                    prob_pct = pred_prob * 100
                    lines.append(
                        f"{rank:>4d}  {umaban:>4s}  {bamei:<16s}  {prob_pct:>5.1f}%"
                    )

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

        if has_odds:
            lines.append("")
            lines.append("* = EV >= 1.1 かつ 予想勝率 >= 2.5%（value_bet候補）")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 内部メソッド
    # ------------------------------------------------------------------

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """予測用に特徴量を整備する."""
        if not self.feature_columns:
            return features

        # 学習時の特徴量順に一括構築（フラグメンテーション回避）
        col_data: dict[str, pd.Series] = {}
        for col in self.feature_columns:
            if col in features.columns:
                col_data[col] = features[col]
            else:
                col_data[col] = pd.Series(-1, index=features.index)
        X = pd.DataFrame(col_data, index=features.index)

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

    def _get_odds_from_db(
        self,
        race_key: dict[str, str],
    ) -> dict[str, float]:
        """n_odds_tanpuku から単勝オッズを取得する.

        Returns:
            umaban(ゼロ埋め) → 単勝オッズ の辞書。取得失敗時は空辞書。
        """
        sql = """
        SELECT umaban, tanodds
        FROM n_odds_tanpuku
        WHERE year = %(year)s AND monthday = %(monthday)s
          AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
          AND nichiji = %(nichiji)s AND racenum = %(racenum)s
        """
        try:
            df = query_df(sql, race_key)
        except Exception:
            logger.debug("オッズDB取得失敗: race_key=%s", race_key)
            return {}

        result: dict[str, float] = {}
        for _, row in df.iterrows():
            umaban = str(row.get("umaban", "")).strip().zfill(2)
            odds_str = str(row.get("tanodds", "")).strip()
            if not odds_str or odds_str == "0000":
                continue
            try:
                odds_val = int(odds_str) / 10.0
                if odds_val > 0:
                    result[umaban] = odds_val
            except (ValueError, TypeError):
                continue
        return result
