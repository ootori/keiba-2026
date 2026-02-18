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

    def simulate_return(
        self,
        valid_df: pd.DataFrame,
        feature_columns: list[str],
        model: lgb.Booster,
        target_col: str = "target",
        strategy: str = "top1_tansho",
        ranking: bool = False,
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

            strategy_map = {
                "top1_tansho": self._bet_top1_tansho,
                "top1_fukusho": self._bet_top1_fukusho,
                "top3_fukusho": self._bet_top3_fukusho,
                "top2_umaren": self._bet_top2_umaren,
                "top2_umatan": self._bet_top2_umatan,
                "top3_sanrenpuku": self._bet_top3_sanrenpuku,
                "top3_sanrentan": self._bet_top3_sanrentan,
                "value_bet": self._bet_value,
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

    def _bet_value(
        self,
        group: pd.DataFrame,
        harai_data: dict,
        race_key: dict[str, str],
    ) -> dict[str, int]:
        """期待値ベースで購入する.

        予測確率 × オッズ > 1.2 の馬の複勝を購入。
        """
        race_harai = harai_data.get(self._race_key_str(race_key), {})
        fukusho = race_harai.get("fukusho", {})

        total_bet = 0
        total_payout = 0
        total_bets = 0
        total_wins = 0

        for _, row in group.iterrows():
            pred_prob = row.get("pred_prob", 0)
            odds_fuku = row.get("odds_fuku_low", 0)
            try:
                odds_fuku = float(odds_fuku) if odds_fuku and odds_fuku > 0 else 0
            except (ValueError, TypeError):
                odds_fuku = 0

            if odds_fuku > 0 and pred_prob * odds_fuku > 1.2:
                umaban = self._format_umaban(row.get("post_umaban", "")) if "post_umaban" in group.columns else ""
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
