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
    """モデルの評価と回収率シミュレーションを実行する."""

    def evaluate(
        self,
        model: lgb.Booster,
        valid_df: pd.DataFrame,
        feature_columns: list[str],
        target_col: str = "target",
    ) -> dict[str, Any]:
        """検証データでモデルを評価する.

        Args:
            model: 学習済みモデル
            valid_df: 検証データ
            feature_columns: 特徴量カラム
            target_col: 目的変数カラム

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

        # 基本指標
        metrics: dict[str, Any] = {
            "logloss": log_loss(y_valid, y_pred),
            "auc": roc_auc_score(y_valid, y_pred),
            "n_samples": len(y_valid),
            "positive_rate": float(y_valid.mean()),
        }

        logger.info("LogLoss: %.6f", metrics["logloss"])
        logger.info("AUC: %.6f", metrics["auc"])

        # レース単位の評価
        valid_df = valid_df.copy()
        valid_df["pred_prob"] = y_pred

        race_metrics = self._evaluate_by_race(valid_df, target_col)
        metrics.update(race_metrics)

        return metrics

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
    ) -> dict[str, Any]:
        """回収率シミュレーションを実行する.

        Args:
            valid_df: 検証データ
            feature_columns: 特徴量カラム
            model: 学習済みモデル
            target_col: 目的変数カラム
            strategy: 賭け戦略
                - 'top1_tansho': 予測1位の単勝を購入
                - 'top1_fukusho': 予測1位の複勝を購入
                - 'top3_fukusho': 予測Top3の複勝を購入
                - 'value_bet': 期待値ベースの購入

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

            if strategy == "top1_tansho":
                result = self._bet_top1_tansho(group, harai_data, race_key_vals)
            elif strategy == "top1_fukusho":
                result = self._bet_top1_fukusho(group, harai_data, race_key_vals)
            elif strategy == "top3_fukusho":
                result = self._bet_top3_fukusho(group, harai_data, race_key_vals)
            elif strategy == "value_bet":
                result = self._bet_value(group, harai_data, race_key_vals)
            else:
                continue

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

    def _get_harai_data(
        self,
        df: pd.DataFrame,
    ) -> dict[str, dict[str, dict[str, int]]]:
        """払戻データを取得する.

        n_harai テーブルのカラム名は EveryDB2 のバージョンにより異なる
        可能性があるため、スキーマから動的に単勝/複勝カラムを検出する。
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

        tansyo_pairs = self._find_pay_column_pairs(all_cols, "tansyo")
        fukusyo_pairs = self._find_pay_column_pairs(all_cols, "fukusyo")

        if not tansyo_pairs and not fukusyo_pairs:
            logger.warning("n_harai: 単勝/複勝払戻カラムが検出できません (columns=%s)", all_cols)
            return {}

        logger.info(
            "n_harai カラム検出: 単勝 %d組, 複勝 %d組",
            len(tansyo_pairs),
            len(fukusyo_pairs),
        )

        # Step 2: 必要なカラムだけ SELECT
        select_cols = list(RACE_KEY_COLS)
        for u, p in tansyo_pairs + fukusyo_pairs:
            select_cols.extend([u, p])

        cols_str = ", ".join(select_cols)
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

            tansho: dict[str, int] = {}
            for u_col, p_col in tansyo_pairs:
                umaban = str(row.get(u_col, "")).strip()
                pay = self._safe_pay(row.get(p_col))
                if umaban and pay > 0:
                    tansho[umaban] = pay

            fukusho: dict[str, int] = {}
            for u_col, p_col in fukusyo_pairs:
                umaban = str(row.get(u_col, "")).strip()
                pay = self._safe_pay(row.get(p_col))
                if umaban and pay > 0:
                    fukusho[umaban] = pay

            result[rk] = {"tansho": tansho, "fukusho": fukusho}

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
        """馬番カラムと払戻金カラムのペアを検出する.

        カラム名の命名規則に依存せず、'umaban' を含むカラムを基準に
        対応する 'pay' カラムを推定する。
        """
        umaban_cols = sorted(
            c for c in all_cols if bet_type in c and "umaban" in c
        )
        pairs: list[tuple[str, str]] = []
        for u_col in umaban_cols:
            p_col = u_col.replace("umaban", "pay")
            if p_col in all_cols:
                pairs.append((u_col, p_col))
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
