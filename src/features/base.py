"""特徴量抽出の基底クラス."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class FeatureExtractor(ABC):
    """特徴量抽出の基底クラス.

    全ての特徴量抽出クラスはこのクラスを継承し、
    extract メソッドと feature_names プロパティを実装する。
    """

    @abstractmethod
    def extract(
        self,
        race_key: dict[str, str],
        uma_race_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """特徴量を抽出する.

        Args:
            race_key: レースキー辞書
                {'year', 'monthday', 'jyocd', 'kaiji', 'nichiji', 'racenum'}
            uma_race_df: 当該レースの出走馬情報 DataFrame
                最低限 kettonum, umaban カラムを含む

        Returns:
            kettonum をインデックスとする特徴量 DataFrame
        """
        pass

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """この抽出器が生成する特徴量名のリスト."""
        pass

    def _safe_int(self, val: str | None, default: int = -1) -> int:
        """文字列を安全にintに変換する."""
        if val is None:
            return default
        try:
            return int(str(val).strip())
        except (ValueError, TypeError):
            return default

    def _safe_float(self, val: str | None, default: float = -1.0) -> float:
        """文字列を安全にfloatに変換する."""
        if val is None:
            return default
        try:
            return float(str(val).strip())
        except (ValueError, TypeError):
            return default

    def _safe_rate(
        self, numerator: int, denominator: int, default: float = 0.0
    ) -> float:
        """安全に割合を計算する."""
        if denominator <= 0:
            return default
        return numerator / denominator
