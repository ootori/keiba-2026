"""PostgreSQL接続ユーティリティ."""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from typing import Any, Generator

import pandas as pd
import psycopg2
import psycopg2.extras

from src.config import DB_CONFIG

logger = logging.getLogger(__name__)

# pandas の psycopg2 非推奨警告を抑制
# （psycopg2 は IN %(tuple)s 展開をネイティブ対応しており、
#   SQLAlchemy text() では代替困難なため psycopg2 直接接続を維持する）
warnings.filterwarnings(
    "ignore",
    message=".*pandas only supports SQLAlchemy.*",
    category=UserWarning,
)


def get_connection() -> psycopg2.extensions.connection:
    """PostgreSQLへの接続を取得する.

    Returns:
        psycopg2 connection オブジェクト
    """
    return psycopg2.connect(**DB_CONFIG)


@contextmanager
def get_cursor(
    dict_cursor: bool = False,
) -> Generator[psycopg2.extensions.cursor, None, None]:
    """コンテキストマネージャでカーソルを取得する.

    Args:
        dict_cursor: Trueの場合 RealDictCursor を使用

    Yields:
        psycopg2 cursor オブジェクト
    """
    conn = get_connection()
    try:
        cursor_factory = (
            psycopg2.extras.RealDictCursor if dict_cursor else None
        )
        cur = conn.cursor(cursor_factory=cursor_factory)
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def query_df(sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """SQLを実行しDataFrameで結果を返す.

    psycopg2 接続を直接使用する。
    psycopg2 は IN %(tuple)s のタプル展開をネイティブでサポートしており、
    大量のIN句パラメータを安全かつ効率的に処理できる。

    Args:
        sql: 実行するSQL（名前付きプレースホルダ %(name)s 形式）
        params: SQLパラメータの辞書

    Returns:
        クエリ結果の DataFrame
    """
    conn = get_connection()
    try:
        df = pd.read_sql(sql, conn, params=params)
        return df
    finally:
        conn.close()


def execute(sql: str, params: dict[str, Any] | None = None) -> None:
    """SQLを実行する（結果を返さない）.

    Args:
        sql: 実行するSQL
        params: SQLパラメータの辞書
    """
    with get_cursor() as cur:
        cur.execute(sql, params)


def check_connection() -> bool:
    """DB接続テスト.

    Returns:
        接続成功ならTrue
    """
    try:
        with get_cursor() as cur:
            cur.execute("SELECT 1")
            result = cur.fetchone()
            logger.info("DB接続成功: %s", result)
            return True
    except Exception as e:
        logger.error("DB接続失敗: %s", e)
        return False


def get_table_counts() -> pd.DataFrame:
    """主要テーブルの行数を取得する.

    Returns:
        テーブル名と行数のDataFrame
    """
    sql = """
    SELECT 'n_race' AS tbl, COUNT(*) AS cnt FROM n_race
    UNION ALL SELECT 'n_uma_race', COUNT(*) FROM n_uma_race
    UNION ALL SELECT 'n_uma', COUNT(*) FROM n_uma
    UNION ALL SELECT 'n_kisyu', COUNT(*) FROM n_kisyu
    UNION ALL SELECT 'n_kisyu_seiseki', COUNT(*) FROM n_kisyu_seiseki
    UNION ALL SELECT 'n_chokyo', COUNT(*) FROM n_chokyo
    UNION ALL SELECT 'n_chokyo_seiseki', COUNT(*) FROM n_chokyo_seiseki
    UNION ALL SELECT 'n_hanro', COUNT(*) FROM n_hanro
    UNION ALL SELECT 'n_wood_chip', COUNT(*) FROM n_wood_chip
    UNION ALL SELECT 'n_odds_tanpuku', COUNT(*) FROM n_odds_tanpuku
    UNION ALL SELECT 'n_harai', COUNT(*) FROM n_harai
    UNION ALL SELECT 'n_hansyoku', COUNT(*) FROM n_hansyoku
    UNION ALL SELECT 'n_sanku', COUNT(*) FROM n_sanku
    UNION ALL SELECT 'n_keito', COUNT(*) FROM n_keito
    """
    return query_df(sql)
