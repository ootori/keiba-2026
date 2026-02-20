"""モデル評価・回収率シミュレーション."""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from src.db import query_df
from src.config import RACE_KEY_COLS, CATEGORICAL_FEATURES

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """モデルの評価と回収率シミュレーションを実行する.

    ranking=True の場合、LambdaRank モデルとして評価する。
    LambdaRank の出力はランキングスコア（高いほど上位）であるため、
    logloss は算出せず NDCG とレース単位の的中率で評価する。
    """

    def evaluate(
        self,
        model: lgb.Booster,
        valid_df: pd.DataFrame,
        feature_columns: list[str],
        target_col: str = "target",
        ranking: bool = False,
    ) -> dict[str, Any]:
        """検証データでモデルを評価する.

        Args:
            model: 学習済みモデル
            valid_df: 検証データ
            feature_columns: 特徴量カラム
            target_col: 目的変数カラム（二値分類のラベル）
            ranking: LambdaRank モデルか

        Returns:
            評価指標の辞書
        """
        X_valid = valid_df[feature_columns].copy()
        y_valid = valid_df[target_col].astype(int)

        # カテゴリ変数を category 型に変換（学習時と一致させる）
        for col in CATEGORICAL_FEATURES:
            if col in X_valid.columns:
                X_valid[col] = X_valid[col].astype("category")

        # 予測
        y_pred = model.predict(X_valid)

        metrics: dict[str, Any] = {
            "n_samples": len(y_valid),
            "positive_rate": float(y_valid.mean()),
            "ranking_mode": ranking,
        }

        if ranking:
            # LambdaRank: スコアは確率ではないため logloss は不適用
            # AUC は二値ラベルに対するランキング指標として引き続き有効
            metrics["auc"] = roc_auc_score(y_valid, y_pred)
            logger.info("AUC (ranking score vs top3 label): %.6f", metrics["auc"])

            # NDCG をレース単位で算出
            ndcg_metrics = self._compute_ndcg(valid_df, y_pred)
            metrics.update(ndcg_metrics)
        else:
            # 二値分類: 従来どおり
            metrics["logloss"] = log_loss(y_valid, y_pred)
            metrics["auc"] = roc_auc_score(y_valid, y_pred)
            logger.info("LogLoss: %.6f", metrics["logloss"])
            logger.info("AUC: %.6f", metrics["auc"])

        # レース単位の評価（共通）
        valid_df = valid_df.copy()
        valid_df["pred_prob"] = y_pred

        race_metrics = self._evaluate_by_race(valid_df, target_col)
        metrics.update(race_metrics)

        return metrics

    def _compute_ndcg(
        self,
        valid_df: pd.DataFrame,
        y_pred: np.ndarray,
        ks: list[int] | None = None,
    ) -> dict[str, float]:
        """レース単位で NDCG@k を算出する.

        Args:
            valid_df: 検証データ（target_relevance カラム必須）
            y_pred: モデルの予測スコア
            ks: 評価する k のリスト

        Returns:
            NDCG 指標の辞書
        """
        if ks is None:
            ks = [1, 3, 5]

        relevance_col = "target_relevance"
        if relevance_col not in valid_df.columns:
            return {}

        df = valid_df.copy()
        df["_pred_score"] = y_pred

        key_cols = [f"_key_{c}" for c in RACE_KEY_COLS if f"_key_{c}" in df.columns]
        if not key_cols:
            return {}

        ndcg_sums: dict[int, float] = {k: 0.0 for k in ks}
        total_races = 0

        for _, group in df.groupby(key_cols):
            if len(group) < 2:
                continue
            total_races += 1

            # 予測スコアでソート（降順）
            sorted_by_pred = group.sort_values("_pred_score", ascending=False)
            pred_relevances = sorted_by_pred[relevance_col].values

            # 理想的なソート（関連度の降順）
            ideal_relevances = np.sort(group[relevance_col].values)[::-1]

            for k in ks:
                dcg = self._dcg_at_k(pred_relevances, k)
                idcg = self._dcg_at_k(ideal_relevances, k)
                if idcg > 0:
                    ndcg_sums[k] += dcg / idcg
                else:
                    ndcg_sums[k] += 1.0  # 全馬同一関連度の場合

        result: dict[str, float] = {}
        for k in ks:
            val = ndcg_sums[k] / total_races if total_races > 0 else 0.0
            result[f"ndcg@{k}"] = val
            logger.info("NDCG@%d: %.6f", k, val)

        return result

    @staticmethod
    def _dcg_at_k(relevances: np.ndarray, k: int) -> float:
        """DCG@k を算出する."""
        relevances = relevances[:k]
        if len(relevances) == 0:
            return 0.0
        discounts = np.log2(np.arange(1, len(relevances) + 1) + 1)
        return float(np.sum(relevances / discounts))

    def _evaluate_by_race(
        self,
        df: pd.DataFrame,
        target_col: str,
    ) -> dict[str, float]:
        """レース単位の予測精度を評価する."""
        key_cols = [f"_key_{c}" for c in RACE_KEY_COLS if f"_key_{c}" in df.columns]
        if not key_cols:
            return {}

        hit_top1 = 0
        hit_top3 = 0
        total_races = 0

        for _, group in df.groupby(key_cols):
            if len(group) < 2:
                continue
            total_races += 1

            # 予測上位3頭
            top3_pred = group.nlargest(3, "pred_prob")
            # 実際の3着以内
            actual_top3 = set(group[group[target_col] == 1].index)

            # 予測1位が3着以内に入ったか
            top1_idx = group["pred_prob"].idxmax()
            if top1_idx in actual_top3:
                hit_top1 += 1

            # 予測上位3頭と実際の3着以内の一致数
            pred_top3_idx = set(top3_pred.index)
            if len(pred_top3_idx & actual_top3) >= 1:
                hit_top3 += 1

        if total_races == 0:
            return {}

        return {
            "race_count": total_races,
            "top1_hit_rate": hit_top1 / total_races,
            "top3_any_hit_rate": hit_top3 / total_races,
        }

    # value_bet のデフォルト設定
    DEFAULT_VALUE_BET_CONFIG: dict[str, Any] = {
        "ev_threshold": 1.2,       # 期待値閾値（1.0より高めでマージン確保）
        "max_bets_per_race": 3,    # 1レースあたり最大単勝ベット数
    }

    def simulate_return(
        self,
        valid_df: pd.DataFrame,
        feature_columns: list[str],
        model: lgb.Booster,
        target_col: str = "target",
        strategy: str = "top1_tansho",
        ranking: bool = False,
        target_type: str = "top3",
        value_bet_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """回収率シミュレーションを実行する.

        LambdaRank モデルの場合も、予測スコアが高い馬を上位として
        同じ賭け戦略を適用する。スコアの絶対値は異なるが、
        レース内でのランキング（nlargest）は同様に機能する。

        Args:
            valid_df: 検証データ
            feature_columns: 特徴量カラム
            model: 学習済みモデル
            target_col: 目的変数カラム
            strategy: 賭け戦略
                - 'top1_tansho': 予測1位の単勝を購入
                - 'top1_fukusho': 予測1位の複勝を購入
                - 'top3_fukusho': 予測Top3の複勝を購入
                - 'top2_umaren': 予測Top2の馬連を購入
                - 'top2_umatan': 予測Top2の馬単を購入（1位→2位）
                - 'top3_sanrenpuku': 予測Top3の三連複を購入
                - 'top3_sanrentan': 予測Top3の三連単を購入（1位→2位→3位）
                - 'value_bet': 期待値ベースの購入
            ranking: LambdaRank モデルか
            target_type: 目的変数タイプ ("top3" or "win")
            value_bet_config: value_bet戦略の設定
                - ev_threshold: 期待値閾値（デフォルト1.2）
                - max_bets_per_race: 最大単勝ベット数（デフォルト3）

        Returns:
            回収率シミュレーション結果
        """
        df = valid_df.copy()

        # カテゴリ変数を category 型に変換（学習時と一致させる）
        X_pred = df[feature_columns].copy()
        for col in CATEGORICAL_FEATURES:
            if col in X_pred.columns:
                X_pred[col] = X_pred[col].astype("category")

        df["pred_prob"] = model.predict(X_pred)

        key_cols = [f"_key_{c}" for c in RACE_KEY_COLS if f"_key_{c}" in df.columns]
        if not key_cols:
            return {"error": "レースキーカラムが見つかりません"}

        # value_bet_*: モデルタイプに応じた確率変換
        # LightGBM の sigmoid 出力はレース内で合計 1.0 にならない
        # （P(win)モデルでもレース合計は 1.0 を大きく超える）ため、
        # 全モデルタイプで正規化が必要
        is_value_bet = strategy in ("value_bet_tansho", "value_bet_umaren")
        if is_value_bet:
            if target_type == "win":
                # P(win) モデル: ratio正規化で合計1.0にする
                logger.info("value_bet: P(win)モデル — ratio正規化")
                df["pred_prob"] = df.groupby(key_cols)["pred_prob"].transform(
                    self._ratio_normalize_group
                )
            elif ranking:
                # LambdaRank: softmax で確率に変換
                logger.info("value_bet: LambdaRank — softmax正規化")
                df["pred_prob"] = df.groupby(key_cols)["pred_prob"].transform(
                    self._softmax_normalize_group
                )
            else:
                # 二値分類 (top3): レース内合計で割って近似的 P(win) に変換
                logger.info("value_bet: P(top3)モデル — ratio正規化")
                df["pred_prob"] = df.groupby(key_cols)["pred_prob"].transform(
                    self._ratio_normalize_group
                )

        # value_bet 設定の確定
        vb_config = dict(self.DEFAULT_VALUE_BET_CONFIG)
        if value_bet_config:
            vb_config.update(value_bet_config)

        # 払戻データを取得
        harai_data = self._get_harai_data(df)

        total_bet = 0
        total_return = 0
        bet_count = 0
        win_count = 0

        for group_key, group in df.groupby(key_cols):
            if len(group) < 2:
                continue

            # レースキーの復元
            race_key_vals = dict(zip(RACE_KEY_COLS, group_key)) if isinstance(group_key, tuple) else {}

            if is_value_bet:
                vb_func = (
                    self._bet_value_tansho
                    if strategy == "value_bet_tansho"
                    else self._bet_value_umaren
                )
                result = vb_func(
                    group, harai_data, race_key_vals,
                    ev_threshold=vb_config["ev_threshold"],
                    max_bets=vb_config["max_bets_per_race"],
                )
            else:
                strategy_map = {
                    "top1_tansho": self._bet_top1_tansho,
                    "top1_fukusho": self._bet_top1_fukusho,
                    "top3_fukusho": self._bet_top3_fukusho,
                    "top2_umaren": self._bet_top2_umaren,
                    "top2_umatan": self._bet_top2_umatan,
                    "top3_sanrenpuku": self._bet_top3_sanrenpuku,
                    "top3_sanrentan": self._bet_top3_sanrentan,
                }
                bet_func = strategy_map.get(strategy)
                if bet_func is None:
                    continue
                result = bet_func(group, harai_data, race_key_vals)

            total_bet += result["bet"]
            total_return += result["return"]
            bet_count += result["bet_count"]
            win_count += result["win_count"]

        return_rate = (total_return / total_bet * 100) if total_bet > 0 else 0.0

        result = {
            "strategy": strategy,
            "total_bet": total_bet,
            "total_return": total_return,
            "return_rate": return_rate,
            "bet_count": bet_count,
            "win_count": win_count,
            "hit_rate": win_count / bet_count if bet_count > 0 else 0.0,
        }

        if is_value_bet:
            logger.info(
                "回収率シミュレーション [%s] (ev>=%.1f, max=%d): "
                "回収率=%.1f%%, 的中率=%.1f%% (%d/%d)",
                strategy,
                vb_config["ev_threshold"],
                vb_config["max_bets_per_race"],
                return_rate,
                result["hit_rate"] * 100,
                win_count,
                bet_count,
            )
        else:
            logger.info(
                "回収率シミュレーション [%s]: 回収率=%.1f%%, 的中率=%.1f%% (%d/%d)",
                strategy,
                return_rate,
                result["hit_rate"] * 100,
                win_count,
                bet_count,
            )

        return result

    # ------------------------------------------------------------------
    # 賭け戦略
    # ------------------------------------------------------------------

    def _bet_top1_tansho(
        self,
        group: pd.DataFrame,
        harai_data: dict,
        race_key: dict[str, str],
    ) -> dict[str, int]:
        """予測1位の単勝を購入する."""
        top1 = group.nlargest(1, "pred_prob").iloc[0]
        umaban = self._format_umaban(top1.get("post_umaban", "")) if "post_umaban" in group.columns else ""

        bet = 100
        payout = 0
        race_harai = harai_data.get(self._race_key_str(race_key), {})
        tansho = race_harai.get("tansho", {})

        if umaban and umaban in tansho:
            payout = tansho[umaban]

        return {
            "bet": bet,
            "return": payout,
            "bet_count": 1,
            "win_count": 1 if payout > 0 else 0,
        }

    def _bet_top1_fukusho(
        self,
        group: pd.DataFrame,
        harai_data: dict,
        race_key: dict[str, str],
    ) -> dict[str, int]:
        """予測1位の複勝を購入する."""
        top1 = group.nlargest(1, "pred_prob").iloc[0]
        umaban = self._format_umaban(top1.get("post_umaban", "")) if "post_umaban" in group.columns else ""

        bet = 100
        payout = 0
        race_harai = harai_data.get(self._race_key_str(race_key), {})
        fukusho = race_harai.get("fukusho", {})

        if umaban and umaban in fukusho:
            payout = fukusho[umaban]

        return {
            "bet": bet,
            "return": payout,
            "bet_count": 1,
            "win_count": 1 if payout > 0 else 0,
        }

    def _bet_top3_fukusho(
        self,
        group: pd.DataFrame,
        harai_data: dict,
        race_key: dict[str, str],
    ) -> dict[str, int]:
        """予測Top3の複勝を購入する."""
        top3 = group.nlargest(3, "pred_prob")
        race_harai = harai_data.get(self._race_key_str(race_key), {})
        fukusho = race_harai.get("fukusho", {})

        total_bet = 0
        total_payout = 0
        total_bets = 0
        total_wins = 0

        for _, row in top3.iterrows():
            umaban = self._format_umaban(row.get("post_umaban", "")) if "post_umaban" in top3.columns else ""
            if not umaban:
                continue
            total_bet += 100
            total_bets += 1
            if umaban in fukusho:
                total_payout += fukusho[umaban]
                total_wins += 1

        return {
            "bet": total_bet,
            "return": total_payout,
            "bet_count": total_bets,
            "win_count": total_wins,
        }

    def _select_value_candidates(
        self,
        group: pd.DataFrame,
        race_key: dict[str, str],
        ev_threshold: float = 1.2,
        max_bets: int = 3,
    ) -> list[str]:
        """EV条件を満たす馬番リストを返す（value_bet共通ロジック）.

        予測確率 × 単勝オッズ >= ev_threshold の馬を EV 降順でソートし、
        上位 max_bets 頭の馬番を返す。

        オッズは特徴量データ（odds_tan）を優先し、
        欠損時は n_odds_tanpuku テーブルから取得する。
        """
        odds_from_db: dict[str, float] | None = None
        has_odds_col = "odds_tan" in group.columns

        candidates: list[tuple[str, float]] = []  # (umaban, ev)

        for _, row in group.iterrows():
            pred_prob = row.get("pred_prob", 0)
            umaban = (
                self._format_umaban(row.get("post_umaban", ""))
                if "post_umaban" in group.columns
                else ""
            )
            if not umaban:
                continue

            odds_tan = 0.0
            if has_odds_col:
                try:
                    val = float(row.get("odds_tan", 0))
                    if val > 0:
                        odds_tan = val
                except (ValueError, TypeError):
                    pass

            if odds_tan <= 0:
                if odds_from_db is None:
                    odds_from_db = self._get_odds_from_db(race_key)
                odds_tan = odds_from_db.get(umaban, 0.0)

            if odds_tan > 0:
                ev = pred_prob * odds_tan
                if ev >= ev_threshold:
                    candidates.append((umaban, ev))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:max_bets]]

    def _bet_value_tansho(
        self,
        group: pd.DataFrame,
        harai_data: dict,
        race_key: dict[str, str],
        ev_threshold: float = 1.2,
        max_bets: int = 3,
    ) -> dict[str, int]:
        """期待値ベースで単勝を購入する.

        EV条件を満たす馬の単勝のみを購入する。
        """
        qualified = self._select_value_candidates(
            group, race_key, ev_threshold, max_bets,
        )
        race_harai = harai_data.get(self._race_key_str(race_key), {})
        tansho = race_harai.get("tansho", {})

        total_bet = 0
        total_payout = 0
        total_bets = 0
        total_wins = 0

        for umaban in qualified:
            total_bet += 100
            total_bets += 1
            if umaban in tansho:
                total_payout += tansho[umaban]
                total_wins += 1

        return {
            "bet": total_bet,
            "return": total_payout,
            "bet_count": total_bets,
            "win_count": total_wins,
        }

    def _bet_value_umaren(
        self,
        group: pd.DataFrame,
        harai_data: dict,
        race_key: dict[str, str],
        ev_threshold: float = 1.2,
        max_bets: int = 3,
    ) -> dict[str, int]:
        """期待値ベースで馬連を購入する.

        EV条件を満たす馬が2頭以上の場合、全組み合わせの馬連を購入する。
        """
        qualified = self._select_value_candidates(
            group, race_key, ev_threshold, max_bets,
        )
        race_harai = harai_data.get(self._race_key_str(race_key), {})
        umaren_harai = race_harai.get("umaren", {})

        total_bet = 0
        total_payout = 0
        total_bets = 0
        total_wins = 0

        if len(qualified) >= 2:
            from itertools import combinations

            for uma1, uma2 in combinations(qualified, 2):
                kumi = self._make_kumi_umaren(uma1, uma2)
                total_bet += 100
                total_bets += 1
                if kumi in umaren_harai:
                    total_payout += umaren_harai[kumi]
                    total_wins += 1

        return {
            "bet": total_bet,
            "return": total_payout,
            "bet_count": total_bets,
            "win_count": total_wins,
        }

    def _get_odds_from_db(
        self,
        race_key: dict[str, str],
    ) -> dict[str, float]:
        """n_odds_tanpuku から単勝オッズを取得する.

        特徴量にオッズが含まれていない場合のフォールバック。
        馬番（2桁ゼロ埋め） → 単勝オッズ のマッピングを返す。

        Returns:
            umaban(ゼロ埋め) → 単勝オッズ の辞書
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
            umaban = self._format_umaban(str(row.get("umaban", "")).strip())
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

    # ------------------------------------------------------------------
    # 払戻データ
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 組番フォーマット
    # ------------------------------------------------------------------

    @staticmethod
    def _make_kumi_umaren(uma1: str, uma2: str) -> str:
        """馬連の組番を生成する（小さい番号が先）."""
        a = ModelEvaluator._format_umaban(uma1)
        b = ModelEvaluator._format_umaban(uma2)
        if a > b:
            a, b = b, a
        return a + b

    @staticmethod
    def _make_kumi_umatan(uma1: str, uma2: str) -> str:
        """馬単の組番を生成する（1着→2着の順）."""
        a = ModelEvaluator._format_umaban(uma1)
        b = ModelEvaluator._format_umaban(uma2)
        return a + b

    @staticmethod
    def _make_kumi_sanren(uma1: str, uma2: str, uma3: str) -> str:
        """三連複の組番を生成する（小さい順にソート）."""
        nums = sorted([
            ModelEvaluator._format_umaban(uma1),
            ModelEvaluator._format_umaban(uma2),
            ModelEvaluator._format_umaban(uma3),
        ])
        return "".join(nums)

    @staticmethod
    def _make_kumi_sanrentan(uma1: str, uma2: str, uma3: str) -> str:
        """三連単の組番を生成する（1着→2着→3着の順）."""
        a = ModelEvaluator._format_umaban(uma1)
        b = ModelEvaluator._format_umaban(uma2)
        c = ModelEvaluator._format_umaban(uma3)
        return a + b + c

    # ------------------------------------------------------------------
    # 馬連・馬単・三連複・三連単 の賭け戦略
    # ------------------------------------------------------------------

    def _bet_top2_umaren(
        self,
        group: pd.DataFrame,
        harai_data: dict,
        race_key: dict[str, str],
    ) -> dict[str, int]:
        """予測上位2頭の馬連を購入する."""
        if len(group) < 2:
            return {"bet": 0, "return": 0, "bet_count": 0, "win_count": 0}

        top2 = group.nlargest(2, "pred_prob")
        if "post_umaban" not in top2.columns:
            return {"bet": 0, "return": 0, "bet_count": 0, "win_count": 0}

        uma_list = top2["post_umaban"].tolist()
        kumi = self._make_kumi_umaren(uma_list[0], uma_list[1])

        race_harai = harai_data.get(self._race_key_str(race_key), {})
        umaren = race_harai.get("umaren", {})

        payout = umaren.get(kumi, 0)
        return {
            "bet": 100,
            "return": payout,
            "bet_count": 1,
            "win_count": 1 if payout > 0 else 0,
        }

    def _bet_top2_umatan(
        self,
        group: pd.DataFrame,
        harai_data: dict,
        race_key: dict[str, str],
    ) -> dict[str, int]:
        """予測上位2頭の馬単を購入する（1位→2位の順）."""
        if len(group) < 2:
            return {"bet": 0, "return": 0, "bet_count": 0, "win_count": 0}

        top2 = group.nlargest(2, "pred_prob")
        if "post_umaban" not in top2.columns:
            return {"bet": 0, "return": 0, "bet_count": 0, "win_count": 0}

        uma_list = top2["post_umaban"].tolist()
        kumi = self._make_kumi_umatan(uma_list[0], uma_list[1])

        race_harai = harai_data.get(self._race_key_str(race_key), {})
        umatan = race_harai.get("umatan", {})

        payout = umatan.get(kumi, 0)
        return {
            "bet": 100,
            "return": payout,
            "bet_count": 1,
            "win_count": 1 if payout > 0 else 0,
        }

    def _bet_top3_sanrenpuku(
        self,
        group: pd.DataFrame,
        harai_data: dict,
        race_key: dict[str, str],
    ) -> dict[str, int]:
        """予測上位3頭の三連複を購入する."""
        if len(group) < 3:
            return {"bet": 0, "return": 0, "bet_count": 0, "win_count": 0}

        top3 = group.nlargest(3, "pred_prob")
        if "post_umaban" not in top3.columns:
            return {"bet": 0, "return": 0, "bet_count": 0, "win_count": 0}

        uma_list = top3["post_umaban"].tolist()
        kumi = self._make_kumi_sanren(uma_list[0], uma_list[1], uma_list[2])

        race_harai = harai_data.get(self._race_key_str(race_key), {})
        sanren = race_harai.get("sanren", {})

        payout = sanren.get(kumi, 0)
        return {
            "bet": 100,
            "return": payout,
            "bet_count": 1,
            "win_count": 1 if payout > 0 else 0,
        }

    def _bet_top3_sanrentan(
        self,
        group: pd.DataFrame,
        harai_data: dict,
        race_key: dict[str, str],
    ) -> dict[str, int]:
        """予測上位3頭の三連単を購入する（1位→2位→3位の順）."""
        if len(group) < 3:
            return {"bet": 0, "return": 0, "bet_count": 0, "win_count": 0}

        top3 = group.nlargest(3, "pred_prob")
        if "post_umaban" not in top3.columns:
            return {"bet": 0, "return": 0, "bet_count": 0, "win_count": 0}

        uma_list = top3["post_umaban"].tolist()
        kumi = self._make_kumi_sanrentan(uma_list[0], uma_list[1], uma_list[2])

        race_harai = harai_data.get(self._race_key_str(race_key), {})
        sanrentan = race_harai.get("sanrentan", {})

        payout = sanrentan.get(kumi, 0)
        return {
            "bet": 100,
            "return": payout,
            "bet_count": 1,
            "win_count": 1 if payout > 0 else 0,
        }

    # ------------------------------------------------------------------
    # 払戻データ
    # ------------------------------------------------------------------

    # 賭式ごとの検出キーワード定義
    _BET_TYPES: list[tuple[str, str]] = [
        ("tansyo", "tansho"),
        ("fukusyo", "fukusho"),
        ("umaren", "umaren"),
        ("umatan", "umatan"),
        ("sanrentan", "sanrentan"),   # sanrentan を sanren より先に検出
        ("sanren", "sanren"),
    ]

    def _get_harai_data(
        self,
        df: pd.DataFrame,
    ) -> dict[str, dict[str, dict[str, int]]]:
        """払戻データを取得する.

        n_harai テーブルのカラム名は EveryDB2 のバージョンにより異なる
        可能性があるため、スキーマから動的にカラムを検出する。
        単勝/複勝/馬連/馬単/三連複/三連単に対応。
        """
        key_cols = [f"_key_{c}" for c in RACE_KEY_COLS if f"_key_{c}" in df.columns]
        if not key_cols:
            return {}

        # 年の範囲を取得
        years = df["_key_year"].unique().tolist() if "_key_year" in df.columns else []
        if not years:
            return {}

        # Step 1: テーブルのカラム名を検出
        schema_df = query_df("SELECT * FROM n_harai LIMIT 0")
        all_cols = schema_df.columns.tolist()

        # 各賭式のカラムペアを検出
        bet_pairs: dict[str, list[tuple[str, str]]] = {}
        for col_key, result_key in self._BET_TYPES:
            pairs = self._find_pay_column_pairs(all_cols, col_key)
            bet_pairs[result_key] = pairs

        detected = {k: len(v) for k, v in bet_pairs.items() if v}
        if not detected:
            logger.warning("n_harai: 払戻カラムが検出できません (columns=%s)", all_cols)
            return {}

        logger.info("n_harai カラム検出: %s", detected)

        # Step 2: 必要なカラムだけ SELECT
        select_cols = list(RACE_KEY_COLS)
        for pairs in bet_pairs.values():
            for u, p in pairs:
                select_cols.extend([u, p])

        # 重複除去して順序を保持
        seen: set[str] = set()
        unique_cols: list[str] = []
        for c in select_cols:
            if c not in seen:
                seen.add(c)
                unique_cols.append(c)

        cols_str = ", ".join(unique_cols)
        sql = f"""
        SELECT {cols_str}
        FROM n_harai
        WHERE year IN %(years)s
          AND datakubun = '2'
        """
        harai_df = query_df(sql, {"years": tuple(years)})

        if harai_df.empty:
            return {}

        result: dict[str, dict[str, dict[str, int]]] = {}
        for _, row in harai_df.iterrows():
            rk = "_".join(str(row.get(c, "")).strip() for c in RACE_KEY_COLS)

            race_data: dict[str, dict[str, int]] = {}
            for col_key, result_key in self._BET_TYPES:
                pays: dict[str, int] = {}
                for u_col, p_col in bet_pairs[result_key]:
                    kumi_or_umaban = str(row.get(u_col, "")).strip()
                    pay = self._safe_pay(row.get(p_col))
                    if kumi_or_umaban and pay > 0:
                        pays[kumi_or_umaban] = pay
                race_data[result_key] = pays

            result[rk] = race_data

        return result

    @staticmethod
    def _linear_normalize_group(scores: pd.Series) -> pd.Series:
        """レース内のスコアを線形正規化して確率に変換する.

        LambdaRank のスコアを確率として扱うための変換。
        最小値を0にシフトしてから合計で割ることで、
        スコア差の比率をそのまま保持した確率分布を生成する。

        ※ value_bet では使用しない（最下位馬が常に0になる問題あり）。
        """
        shifted = scores - scores.min()
        total = shifted.sum()
        if total > 0:
            return shifted / total
        # 全馬同スコアの場合は均等確率
        return pd.Series(1.0 / len(scores), index=scores.index)

    @staticmethod
    def _ratio_normalize_group(scores: pd.Series) -> pd.Series:
        """レース内合計で割って確率に変換する（二値分類 top3 用）.

        P(top3) の比率をそのまま保持して合計1.0にする。
        min-shift を行わないため、全馬に正の確率が残る。
        """
        total = scores.sum()
        if total > 0:
            return scores / total
        return pd.Series(1.0 / len(scores), index=scores.index)

    @staticmethod
    def _softmax_normalize_group(scores: pd.Series) -> pd.Series:
        """Softmax でスコアを確率に変換する（LambdaRank 用）.

        LambdaRank の出力は任意のスケールのスコアであるため、
        softmax で確率分布に変換する。スコア差の指数比を保持する。
        """
        exp_scores = np.exp(scores - scores.max())  # オーバーフロー防止
        total = exp_scores.sum()
        if total > 0:
            return exp_scores / total
        return pd.Series(1.0 / len(scores), index=scores.index)

    @staticmethod
    def _format_umaban(val) -> str:
        """馬番をゼロパディングした文字列に変換する."""
        try:
            return str(int(val)).zfill(2)  # 1 -> "01", 10 -> "10"
        except (ValueError, TypeError):
            return str(val).strip()

    @staticmethod
    def _find_pay_column_pairs(
        all_cols: list[str],
        bet_type: str,
    ) -> list[tuple[str, str]]:
        """馬番/組番カラムと払戻金カラムのペアを検出する.

        単勝/複勝は 'umaban' を含むカラム、馬連/馬単/三連複/三連単は
        'umaban' または 'kumi' を含むカラムを基準に検出する。

        なお sanrentan を先に検出してから sanren を検出する必要がある
        （sanren は sanrentan にも部分一致するため、呼び出し順で制御）。
        """
        # umaban ベース（単勝/複勝、および EveryDB2 の馬連等も umaban 表記の場合）
        id_cols = sorted(
            c for c in all_cols if bet_type in c and ("umaban" in c or "kumi" in c)
        )
        pairs: list[tuple[str, str]] = []
        for id_col in id_cols:
            # sanren と sanrentan の誤マッチ防止
            if bet_type == "sanren" and "sanrentan" in id_col:
                continue
            p_col = id_col.replace("umaban", "pay").replace("kumi", "pay")
            if p_col in all_cols and p_col != id_col:
                pairs.append((id_col, p_col))
        return pairs

    @staticmethod
    def _race_key_str(race_key: dict[str, str]) -> str:
        """レースキーを文字列に変換する."""
        return "_".join(str(race_key.get(c, "")).strip() for c in RACE_KEY_COLS)

    @staticmethod
    def _safe_pay(val: Any) -> int:
        """払戻金額を安全にintに変換する."""
        if val is None:
            return 0
        try:
            return int(str(val).strip())
        except (ValueError, TypeError):
            return 0
