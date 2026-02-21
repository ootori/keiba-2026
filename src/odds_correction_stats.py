"""オッズ歪み補正の統計データ算出・保存・ロード.

DBから過去レースの回収率統計を算出し、オッズ補正 factor を
JSON ファイルに保存する。evaluator / predictor が実行時に参照する。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    JRA_JYO_CODES,
    ODDS_CORRECTION_STATS_PATH,
    RACE_KEY_COLS,
)
from src.db import query_df

logger = logging.getLogger(__name__)

# 最小サンプル数（これ未満は factor = 1.0）
DEFAULT_MIN_SAMPLES: int = 1000


# =====================================================================
# 公開 API
# =====================================================================


def build_odds_correction_stats(
    year_start: str = "2022",
    year_end: str = "2024",
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> dict[str, Any]:
    """DBから統計を算出し、補正ルール設定を返す.

    Args:
        year_start: 集計開始年
        year_end: 集計終了年
        min_samples: 最小サンプル数

    Returns:
        統計データ辞書（JSON保存用）
    """
    logger.info(
        "オッズ補正統計を算出: %s〜%s年 (min_samples=%d)",
        year_start, year_end, min_samples,
    )

    # 単勝カラム名を動的検出
    tansho_umaban_col, tansho_pay_col = _detect_tansho_columns()
    logger.info(
        "n_harai 単勝カラム検出: umaban=%s, pay=%s",
        tansho_umaban_col, tansho_pay_col,
    )

    # baseline ROI
    baseline_roi, baseline_samples = _calc_baseline_roi(
        year_start, year_end, tansho_umaban_col, tansho_pay_col,
    )
    logger.info(
        "baseline ROI: %.4f (%d件)", baseline_roi, baseline_samples,
    )

    # 人気順別テーブル
    ninki_table = _calc_ninki_table(
        year_start, year_end,
        tansho_umaban_col, tansho_pay_col,
        baseline_roi, min_samples,
    )
    logger.info("ninki_table: %d人気順分", len(ninki_table))

    # 個別ルール
    jockey_stats = _calc_jockey_popular_stats(
        year_start, year_end,
        tansho_umaban_col, tansho_pay_col,
        baseline_roi, min_samples,
    )
    form_stats = _calc_form_popular_stats(
        year_start, year_end,
        tansho_umaban_col, tansho_pay_col,
        baseline_roi, min_samples,
    )
    odd_stats, even_stats = _calc_gate_parity_stats(
        year_start, year_end,
        tansho_umaban_col, tansho_pay_col,
        baseline_roi, min_samples,
    )

    # v2: 脚質別テーブル
    style_table = _calc_running_style_stats(
        year_start, year_end,
        tansho_umaban_col, tansho_pay_col,
        baseline_roi, min_samples,
    )
    logger.info("style_table: %d脚質分", len(style_table))

    # v2: 馬番×コース別テーブル
    post_course_table = _calc_post_position_course_stats(
        year_start, year_end,
        tansho_umaban_col, tansho_pay_col,
        baseline_roi, min_samples,
    )
    logger.info("post_course_table: %d区分", len(post_course_table))

    # v2: クラス変更統計
    class_stats = _calc_class_change_stats(
        year_start, year_end,
        tansho_umaban_col, tansho_pay_col,
        baseline_roi, min_samples,
    )

    # v2: 牝馬限定⇔混合遷移
    filly_stats = _calc_filly_transition_stats(
        year_start, year_end,
        tansho_umaban_col, tansho_pay_col,
        baseline_roi, min_samples,
    )

    stats: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "period": {"start": year_start, "end": year_end},
        "baseline_roi": round(baseline_roi, 6),
        "baseline_samples": baseline_samples,
        "min_samples": min_samples,
        "ninki_table": ninki_table,
        "style_table": style_table,
        "post_course_table": post_course_table,
        "rules": {
            "jockey_popular_discount": jockey_stats,
            "form_popular_discount": form_stats,
            "odd_gate_discount": odd_stats,
            "even_gate_boost": even_stats,
            "class_upgrade": class_stats.get("class_upgrade", {
                "factor": 1.0, "samples": 0, "roi": 0.0,
            }),
            "class_downgrade": class_stats.get("class_downgrade", {
                "factor": 1.0, "samples": 0, "roi": 0.0,
            }),
            "filly_to_mixed": filly_stats.get("filly_to_mixed", {
                "factor": 1.0, "samples": 0, "roi": 0.0,
            }),
            "mixed_to_filly": filly_stats.get("mixed_to_filly", {
                "factor": 1.0, "samples": 0, "roi": 0.0,
            }),
        },
    }

    # ログ出力
    for name, rule in stats["rules"].items():
        logger.info(
            "  %s: factor=%.4f, samples=%d, roi=%.4f",
            name, rule["factor"], rule["samples"], rule["roi"],
        )

    return stats


def save_odds_correction_stats(
    stats: dict[str, Any],
    path: Path | None = None,
) -> None:
    """統計をJSONファイルに保存する."""
    p = path or ODDS_CORRECTION_STATS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("統計を保存: %s", p)


def load_odds_correction_stats(
    path: Path | None = None,
) -> dict[str, Any]:
    """JSONファイルから補正設定をロードする.

    evaluator の odds_correction_config 形式で返す。

    Returns:
        {"enabled": True, "ninki_table": {int→float}, "rules": {...}}
    """
    p = path or ODDS_CORRECTION_STATS_PATH
    if not p.exists():
        raise FileNotFoundError(f"統計ファイルが見つかりません: {p}")

    with open(p, encoding="utf-8") as f:
        stats = json.load(f)

    logger.info(
        "統計をロード: %s (期間=%s〜%s, baseline_roi=%.4f)",
        p,
        stats.get("period", {}).get("start", "?"),
        stats.get("period", {}).get("end", "?"),
        stats.get("baseline_roi", 0),
    )

    # ninki_table: JSON のキーは文字列なので int に変換
    ninki_table_raw = stats.get("ninki_table", {})
    ninki_table: dict[int, float] = {}
    for k, v in ninki_table_raw.items():
        ninki_table[int(k)] = v["factor"] if isinstance(v, dict) else float(v)

    # style_table: キーは脚質コード文字列 ("1"-"4")
    style_table_raw = stats.get("style_table", {})
    style_table: dict[str, float] = {}
    for k, v in style_table_raw.items():
        style_table[k] = v["factor"] if isinstance(v, dict) else float(v)

    # post_course_table: キーは "{post_group}_{course_cat}" または "{post_group}"
    post_course_table_raw = stats.get("post_course_table", {})
    post_course_table: dict[str, float] = {}
    for k, v in post_course_table_raw.items():
        post_course_table[k] = v["factor"] if isinstance(v, dict) else float(v)

    # rules: factor 以外のキーも含めて渡す
    rules = stats.get("rules", {})

    return {
        "enabled": True,
        "ninki_table": ninki_table,
        "style_table": style_table,
        "post_course_table": post_course_table,
        "rules": rules,
    }


# =====================================================================
# n_harai カラム検出
# =====================================================================


def _detect_tansho_columns() -> tuple[str, str]:
    """n_harai から単勝の馬番/払戻カラム名を動的検出する.

    Returns:
        (umaban_col, pay_col) のタプル

    Raises:
        RuntimeError: 単勝カラムが検出できない場合
    """
    schema_df = query_df("SELECT * FROM n_harai LIMIT 0")
    all_cols = schema_df.columns.tolist()

    for col in sorted(all_cols):
        if "tansyo" in col and ("umaban" in col or "kumi" in col):
            pay_col = col.replace("umaban", "pay").replace("kumi", "pay")
            if pay_col in all_cols and pay_col != col:
                return col, pay_col

    raise RuntimeError(
        f"n_harai から単勝カラムを検出できません (columns={all_cols})"
    )


# =====================================================================
# 統計算出
# =====================================================================


def _jyo_filter(prefix: str = "") -> str:
    """JRA中央10場のフィルタ条件文字列を返す.

    Args:
        prefix: テーブルエイリアスのプレフィックス（例: "ur."）
    """
    codes = ", ".join(f"'{c}'" for c in JRA_JYO_CODES)
    return f"{prefix}jyocd IN ({codes})"


def _calc_baseline_roi(
    year_start: str,
    year_end: str,
    tansho_umaban_col: str,
    tansho_pay_col: str,
) -> tuple[float, int]:
    """全レースの単勝平均回収率を算出する.

    全出走馬に1頭100円ずつ単勝を購入したと仮定した場合の回収率。

    Returns:
        (roi, sample_count)
    """
    # 全出走馬数
    sql_count = f"""
    SELECT COUNT(*) AS cnt
    FROM n_uma_race
    WHERE datakubun = '7' AND ijyocd = '0'
      AND year BETWEEN %(start)s AND %(end)s
      AND {_jyo_filter()}
    """
    df_count = query_df(sql_count, {"start": year_start, "end": year_end})
    total = int(df_count.iloc[0]["cnt"])

    if total == 0:
        return 0.0, 0

    # 全レースの単勝払戻合計（非数値の払戻値を除外）
    sql_pay = f"""
    SELECT COALESCE(SUM(CAST({tansho_pay_col} AS int)), 0) AS total_pay
    FROM n_harai
    WHERE datakubun = '2'
      AND year BETWEEN %(start)s AND %(end)s
      AND {_jyo_filter()}
      AND {tansho_pay_col} ~ '^[0-9]+$'
    """
    df_pay = query_df(sql_pay, {"start": year_start, "end": year_end})
    total_pay = int(df_pay.iloc[0]["total_pay"])

    # 1レースあたり1頭ずつ買うのではなく、全馬に買った場合の回収率
    # ただし実際は「1レースに1枚だけ買う」のが単勝の標準
    # ここでは baseline = (全レースの単勝払戻合計) / (全レース数 × 100)
    sql_race_count = f"""
    SELECT COUNT(DISTINCT year || monthday || jyocd || kaiji || nichiji || racenum)
        AS race_cnt
    FROM n_uma_race
    WHERE datakubun = '7' AND ijyocd = '0'
      AND year BETWEEN %(start)s AND %(end)s
      AND {_jyo_filter()}
    """
    df_rc = query_df(sql_race_count, {"start": year_start, "end": year_end})
    race_count = int(df_rc.iloc[0]["race_cnt"])

    # 全馬に100円ずつ単勝を買った場合の回収率
    total_bet = total * 100
    roi = total_pay / total_bet if total_bet > 0 else 0.0

    return roi, total


def _calc_ninki_table(
    year_start: str,
    year_end: str,
    tansho_umaban_col: str,
    tansho_pay_col: str,
    baseline_roi: float,
    min_samples: int,
) -> dict[str, dict[str, Any]]:
    """人気順1-18それぞれの単勝回収率テーブルを算出する.

    Returns:
        {"1": {"factor": 0.98, "samples": 30000, "roi": 0.76}, ...}
    """
    # 人気順別の出走数と単勝的中時の払戻を集計
    # n_odds_tanpuku の tanninki から人気順を取得
    sql = f"""
    SELECT
        CAST(o.tanninki AS int) AS ninki,
        COUNT(*) AS cnt,
        SUM(CASE WHEN ur.kakuteijyuni ~ '^[0-9]+$' AND CAST(ur.kakuteijyuni AS int) = 1
            THEN COALESCE(CAST(o.tanodds AS numeric) / 10.0, 0)
            ELSE 0 END) * 100 AS total_pay
    FROM n_uma_race ur
    JOIN n_odds_tanpuku o
        ON ur.year = o.year AND ur.monthday = o.monthday
        AND ur.jyocd = o.jyocd AND ur.kaiji = o.kaiji
        AND ur.nichiji = o.nichiji AND ur.racenum = o.racenum
        AND ur.umaban = o.umaban
    WHERE ur.datakubun = '7' AND ur.ijyocd = '0'
      AND ur.year BETWEEN %(start)s AND %(end)s
      AND {_jyo_filter("ur.")}
      AND o.tanninki ~ '^[0-9]+$'
      AND CAST(o.tanninki AS int) BETWEEN 1 AND 18
      AND o.tanodds ~ '^[0-9]+$'
      AND CAST(o.tanodds AS int) > 0
    GROUP BY CAST(o.tanninki AS int)
    ORDER BY ninki
    """
    df = query_df(sql, {"start": year_start, "end": year_end})

    result: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        ninki = int(row["ninki"])
        cnt = int(row["cnt"])
        total_pay = float(row["total_pay"])
        total_bet = cnt * 100
        roi = total_pay / total_bet if total_bet > 0 else 0.0

        if cnt >= min_samples and baseline_roi > 0:
            factor = roi / baseline_roi
        else:
            factor = 1.0

        result[str(ninki)] = {
            "factor": round(factor, 6),
            "samples": cnt,
            "roi": round(roi, 6),
        }
        logger.info(
            "  ninki=%d: factor=%.4f, roi=%.4f, samples=%d",
            ninki, factor, roi, cnt,
        )

    return result


def _calc_jockey_popular_stats(
    year_start: str,
    year_end: str,
    tansho_umaban_col: str,
    tansho_pay_col: str,
    baseline_roi: float,
    min_samples: int,
    jockey_wr_threshold: float = 0.15,
    ninki_threshold: int = 3,
) -> dict[str, Any]:
    """人気騎手×人気馬の回収率を算出する."""
    # Step 1: 騎手勝率テーブル
    sql_jockey = f"""
    SELECT kisyucode,
        SUM(CASE WHEN kakuteijyuni ~ '^[0-9]+$' AND CAST(kakuteijyuni AS int) = 1
            THEN 1 ELSE 0 END)::float / COUNT(*) AS win_rate
    FROM n_uma_race
    WHERE datakubun = '7' AND ijyocd = '0'
      AND year BETWEEN %(start)s AND %(end)s
      AND {_jyo_filter()}
    GROUP BY kisyucode
    HAVING COUNT(*) >= 100
    """
    df_jockey = query_df(sql_jockey, {"start": year_start, "end": year_end})
    popular_jockeys = set(
        df_jockey[df_jockey["win_rate"] >= jockey_wr_threshold]["kisyucode"]
        .astype(str).str.strip().tolist()
    )

    if not popular_jockeys:
        return {
            "jockey_win_rate_threshold": jockey_wr_threshold,
            "ninki_threshold": ninki_threshold,
            "factor": 1.0, "samples": 0, "roi": 0.0,
        }

    # Step 2: 該当馬の単勝回収率
    jockey_list = ", ".join(f"'{j}'" for j in popular_jockeys)
    sql = f"""
    SELECT COUNT(*) AS cnt,
        SUM(CASE WHEN ur.kakuteijyuni ~ '^[0-9]+$' AND CAST(ur.kakuteijyuni AS int) = 1
            THEN COALESCE(CAST(o.tanodds AS numeric) / 10.0, 0)
            ELSE 0 END) * 100 AS total_pay
    FROM n_uma_race ur
    JOIN n_odds_tanpuku o
        ON ur.year = o.year AND ur.monthday = o.monthday
        AND ur.jyocd = o.jyocd AND ur.kaiji = o.kaiji
        AND ur.nichiji = o.nichiji AND ur.racenum = o.racenum
        AND ur.umaban = o.umaban
    WHERE ur.datakubun = '7' AND ur.ijyocd = '0'
      AND ur.year BETWEEN %(start)s AND %(end)s
      AND {_jyo_filter("ur.")}
      AND TRIM(ur.kisyucode) IN ({jockey_list})
      AND o.tanninki ~ '^[0-9]+$'
      AND CAST(o.tanninki AS int) <= %(ninki_th)s
      AND o.tanodds ~ '^[0-9]+$'
      AND CAST(o.tanodds AS int) > 0
    """
    df = query_df(
        sql,
        {"start": year_start, "end": year_end, "ninki_th": ninki_threshold},
    )

    cnt = int(df.iloc[0]["cnt"])
    total_pay = float(df.iloc[0]["total_pay"])
    total_bet = cnt * 100
    roi = total_pay / total_bet if total_bet > 0 else 0.0
    factor = (roi / baseline_roi) if (cnt >= min_samples and baseline_roi > 0) else 1.0

    return {
        "jockey_win_rate_threshold": jockey_wr_threshold,
        "ninki_threshold": ninki_threshold,
        "factor": round(factor, 6),
        "samples": cnt,
        "roi": round(roi, 6),
    }


def _calc_form_popular_stats(
    year_start: str,
    year_end: str,
    tansho_umaban_col: str,
    tansho_pay_col: str,
    baseline_roi: float,
    min_samples: int,
    last_jyuni_threshold: int = 3,
    ninki_threshold: int = 3,
) -> dict[str, Any]:
    """前走好走×人気馬の回収率を算出する."""
    # 前走着順をウィンドウ関数で取得
    start_minus_1 = str(int(year_start) - 1)

    sql = f"""
    WITH ranked AS (
        SELECT kettonum, year, monthday, jyocd, kaiji, nichiji, racenum,
            kakuteijyuni,
            LAG(CASE WHEN kakuteijyuni ~ '^[0-9]+$'
                     THEN CAST(kakuteijyuni AS int) END) OVER (
                PARTITION BY kettonum
                ORDER BY year, monthday
            ) AS prev_jyuni
        FROM n_uma_race
        WHERE datakubun = '7' AND ijyocd = '0'
          AND year BETWEEN %(start_m1)s AND %(end)s
          AND {_jyo_filter()}
    )
    SELECT COUNT(*) AS cnt,
        SUM(CASE WHEN r.kakuteijyuni ~ '^[0-9]+$' AND CAST(r.kakuteijyuni AS int) = 1
            THEN COALESCE(CAST(o.tanodds AS numeric) / 10.0, 0)
            ELSE 0 END) * 100 AS total_pay
    FROM ranked r
    JOIN n_odds_tanpuku o
        ON r.year = o.year AND r.monthday = o.monthday
        AND r.jyocd = o.jyocd AND r.kaiji = o.kaiji
        AND r.nichiji = o.nichiji AND r.racenum = o.racenum
    JOIN n_uma_race ur
        ON r.year = ur.year AND r.monthday = ur.monthday
        AND r.jyocd = ur.jyocd AND r.kaiji = ur.kaiji
        AND r.nichiji = ur.nichiji AND r.racenum = ur.racenum
        AND r.kettonum = ur.kettonum
    WHERE r.year BETWEEN %(start)s AND %(end)s
      AND r.prev_jyuni IS NOT NULL
      AND r.prev_jyuni <= %(last_jyuni_th)s
      AND o.tanninki ~ '^[0-9]+$'
      AND CAST(o.tanninki AS int) <= %(ninki_th)s
      AND o.tanodds ~ '^[0-9]+$'
      AND CAST(o.tanodds AS int) > 0
      AND ur.umaban = o.umaban
    """
    df = query_df(
        sql,
        {
            "start_m1": start_minus_1,
            "start": year_start,
            "end": year_end,
            "last_jyuni_th": last_jyuni_threshold,
            "ninki_th": ninki_threshold,
        },
    )

    cnt = int(df.iloc[0]["cnt"])
    total_pay = float(df.iloc[0]["total_pay"])
    total_bet = cnt * 100
    roi = total_pay / total_bet if total_bet > 0 else 0.0
    factor = (roi / baseline_roi) if (cnt >= min_samples and baseline_roi > 0) else 1.0

    return {
        "last_jyuni_threshold": last_jyuni_threshold,
        "ninki_threshold": ninki_threshold,
        "factor": round(factor, 6),
        "samples": cnt,
        "roi": round(roi, 6),
    }


def _calc_gate_parity_stats(
    year_start: str,
    year_end: str,
    tansho_umaban_col: str,
    tansho_pay_col: str,
    baseline_roi: float,
    min_samples: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """奇数/偶数ゲート別回収率を算出する.

    Returns:
        (odd_stats, even_stats)
    """
    sql = f"""
    SELECT
        CAST(ur.umaban AS int) %% 2 AS parity,
        COUNT(*) AS cnt,
        SUM(CASE WHEN ur.kakuteijyuni ~ '^[0-9]+$' AND CAST(ur.kakuteijyuni AS int) = 1
            THEN COALESCE(CAST(o.tanodds AS numeric) / 10.0, 0)
            ELSE 0 END) * 100 AS total_pay
    FROM n_uma_race ur
    JOIN n_odds_tanpuku o
        ON ur.year = o.year AND ur.monthday = o.monthday
        AND ur.jyocd = o.jyocd AND ur.kaiji = o.kaiji
        AND ur.nichiji = o.nichiji AND ur.racenum = o.racenum
        AND ur.umaban = o.umaban
    WHERE ur.datakubun = '7' AND ur.ijyocd = '0'
      AND ur.year BETWEEN %(start)s AND %(end)s
      AND {_jyo_filter("ur.")}
      AND ur.umaban ~ '^[0-9]+$'
      AND CAST(ur.umaban AS int) > 0
      AND o.tanodds ~ '^[0-9]+$'
      AND CAST(o.tanodds AS int) > 0
    GROUP BY CAST(ur.umaban AS int) %% 2
    """
    df = query_df(sql, {"start": year_start, "end": year_end})

    odd_stats: dict[str, Any] = {"factor": 1.0, "samples": 0, "roi": 0.0}
    even_stats: dict[str, Any] = {"factor": 1.0, "samples": 0, "roi": 0.0}

    for _, row in df.iterrows():
        parity = int(row["parity"])
        cnt = int(row["cnt"])
        total_pay = float(row["total_pay"])
        total_bet = cnt * 100
        roi = total_pay / total_bet if total_bet > 0 else 0.0
        factor = (roi / baseline_roi) if (cnt >= min_samples and baseline_roi > 0) else 1.0

        entry = {
            "factor": round(factor, 6),
            "samples": cnt,
            "roi": round(roi, 6),
        }
        if parity == 1:  # 奇数
            odd_stats = entry
        else:  # 偶数
            even_stats = entry

    return odd_stats, even_stats


# =====================================================================
# v2 統計: 脚質・馬番×コース・クラス変更・牝馬限定遷移
# =====================================================================


def _calc_running_style_stats(
    year_start: str,
    year_end: str,
    tansho_umaban_col: str,
    tansho_pay_col: str,
    baseline_roi: float,
    min_samples: int,
) -> dict[str, dict[str, Any]]:
    """前走脚質別の単勝回収率テーブルを算出する.

    LAGウィンドウ関数で前走の脚質区分を取得し、
    脚質1〜4ごとの回収率 → factor を算出する。

    Returns:
        {"1": {"factor": ..., "samples": ..., "roi": ...}, ...}
        キーは前走脚質 (1=逃げ, 2=先行, 3=差し, 4=追込)
    """
    start_minus_1 = str(int(year_start) - 1)

    sql = f"""
    WITH ranked AS (
        SELECT kettonum, year, monthday, jyocd, kaiji, nichiji, racenum,
            umaban, kakuteijyuni,
            LAG(kyakusitukubun) OVER (
                PARTITION BY kettonum
                ORDER BY year, monthday
            ) AS prev_style
        FROM n_uma_race
        WHERE datakubun = '7' AND ijyocd = '0'
          AND year BETWEEN %(start_m1)s AND %(end)s
          AND {_jyo_filter()}
    )
    SELECT
        r.prev_style,
        COUNT(*) AS cnt,
        SUM(CASE WHEN r.kakuteijyuni ~ '^[0-9]+$'
                      AND CAST(r.kakuteijyuni AS int) = 1
            THEN COALESCE(CAST(o.tanodds AS numeric) / 10.0, 0)
            ELSE 0 END) * 100 AS total_pay
    FROM ranked r
    JOIN n_odds_tanpuku o
        ON r.year = o.year AND r.monthday = o.monthday
        AND r.jyocd = o.jyocd AND r.kaiji = o.kaiji
        AND r.nichiji = o.nichiji AND r.racenum = o.racenum
        AND r.umaban = o.umaban
    WHERE r.year BETWEEN %(start)s AND %(end)s
      AND r.prev_style IN ('1','2','3','4')
      AND o.tanodds ~ '^[0-9]+$'
      AND CAST(o.tanodds AS int) > 0
    GROUP BY r.prev_style
    ORDER BY r.prev_style
    """
    df = query_df(
        sql,
        {"start_m1": start_minus_1, "start": year_start, "end": year_end},
    )

    result: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        style = str(row["prev_style"]).strip()
        cnt = int(row["cnt"])
        total_pay = float(row["total_pay"])
        total_bet = cnt * 100
        roi = total_pay / total_bet if total_bet > 0 else 0.0

        if cnt >= min_samples and baseline_roi > 0:
            factor = roi / baseline_roi
        else:
            factor = 1.0

        result[style] = {
            "factor": round(factor, 6),
            "samples": cnt,
            "roi": round(roi, 6),
        }
        logger.info(
            "  style=%s: factor=%.4f, roi=%.4f, samples=%d",
            style, factor, roi, cnt,
        )

    return result


def _post_group_case() -> str:
    """SQL CASE 式: 馬番を4グループに分類する."""
    return """
        CASE
            WHEN CAST(ur.umaban AS int) BETWEEN 1 AND 3 THEN 'inner'
            WHEN CAST(ur.umaban AS int) BETWEEN 4 AND 6 THEN 'mid_inner'
            WHEN CAST(ur.umaban AS int) BETWEEN 7 AND 9 THEN 'mid_outer'
            ELSE 'outer'
        END"""


def _course_cat_case() -> str:
    """SQL CASE 式: コースカテゴリを分類する."""
    return """
        CASE
            WHEN ur.jyocd = '04' AND r.trackcd = '10' THEN 'niigata_straight'
            WHEN r.trackcd IN ('10','11','12') THEN 'turf_left'
            WHEN r.trackcd IN ('17','18') THEN 'turf_right'
            WHEN r.trackcd = '23' THEN 'dirt_left'
            WHEN r.trackcd = '24' THEN 'dirt_right'
            ELSE 'other'
        END"""


def _calc_post_position_course_stats(
    year_start: str,
    year_end: str,
    tansho_umaban_col: str,
    tansho_pay_col: str,
    baseline_roi: float,
    min_samples: int,
) -> dict[str, dict[str, Any]]:
    """馬番グループ × コースカテゴリ別の単勝回収率テーブルを算出する.

    馬番グループ: inner(1-3), mid_inner(4-6), mid_outer(7-9), outer(10+)
    コースカテゴリ: turf_left, turf_right, dirt_left, dirt_right, niigata_straight, other

    詳細 (post_group × course_cat) とフォールバック (post_group のみ) を両方返す。

    Returns:
        {"inner_turf_left": {...}, ..., "inner": {...}(フォールバック), ...}
    """
    pg = _post_group_case()
    cc = _course_cat_case()

    # 詳細: post_group × course_cat
    sql_detail = f"""
    SELECT
        {pg} AS post_group,
        {cc} AS course_cat,
        COUNT(*) AS cnt,
        SUM(CASE WHEN ur.kakuteijyuni ~ '^[0-9]+$'
                      AND CAST(ur.kakuteijyuni AS int) = 1
            THEN COALESCE(CAST(o.tanodds AS numeric) / 10.0, 0)
            ELSE 0 END) * 100 AS total_pay
    FROM n_uma_race ur
    JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
    JOIN n_odds_tanpuku o
        ON ur.year = o.year AND ur.monthday = o.monthday
        AND ur.jyocd = o.jyocd AND ur.kaiji = o.kaiji
        AND ur.nichiji = o.nichiji AND ur.racenum = o.racenum
        AND ur.umaban = o.umaban
    WHERE ur.datakubun = '7' AND ur.ijyocd = '0'
      AND ur.year BETWEEN %(start)s AND %(end)s
      AND {_jyo_filter("ur.")}
      AND ur.umaban ~ '^[0-9]+$'
      AND CAST(ur.umaban AS int) > 0
      AND o.tanodds ~ '^[0-9]+$'
      AND CAST(o.tanodds AS int) > 0
    GROUP BY post_group, course_cat
    """
    df_detail = query_df(sql_detail, {"start": year_start, "end": year_end})

    # フォールバック: post_group のみ
    sql_fallback = f"""
    SELECT
        {pg} AS post_group,
        COUNT(*) AS cnt,
        SUM(CASE WHEN ur.kakuteijyuni ~ '^[0-9]+$'
                      AND CAST(ur.kakuteijyuni AS int) = 1
            THEN COALESCE(CAST(o.tanodds AS numeric) / 10.0, 0)
            ELSE 0 END) * 100 AS total_pay
    FROM n_uma_race ur
    JOIN n_odds_tanpuku o
        ON ur.year = o.year AND ur.monthday = o.monthday
        AND ur.jyocd = o.jyocd AND ur.kaiji = o.kaiji
        AND ur.nichiji = o.nichiji AND ur.racenum = o.racenum
        AND ur.umaban = o.umaban
    WHERE ur.datakubun = '7' AND ur.ijyocd = '0'
      AND ur.year BETWEEN %(start)s AND %(end)s
      AND {_jyo_filter("ur.")}
      AND ur.umaban ~ '^[0-9]+$'
      AND CAST(ur.umaban AS int) > 0
      AND o.tanodds ~ '^[0-9]+$'
      AND CAST(o.tanodds AS int) > 0
    GROUP BY post_group
    """
    df_fallback = query_df(sql_fallback, {"start": year_start, "end": year_end})

    result: dict[str, dict[str, Any]] = {}

    # 詳細テーブル
    for _, row in df_detail.iterrows():
        post_grp = str(row["post_group"]).strip()
        course = str(row["course_cat"]).strip()
        key = f"{post_grp}_{course}"
        cnt = int(row["cnt"])
        total_pay = float(row["total_pay"])
        total_bet = cnt * 100
        roi = total_pay / total_bet if total_bet > 0 else 0.0
        factor = (roi / baseline_roi) if (cnt >= min_samples and baseline_roi > 0) else 1.0

        result[key] = {
            "factor": round(factor, 6),
            "samples": cnt,
            "roi": round(roi, 6),
        }
        logger.info(
            "  post_course=%s: factor=%.4f, roi=%.4f, samples=%d",
            key, factor, roi, cnt,
        )

    # フォールバックテーブル（post_group のみ）
    for _, row in df_fallback.iterrows():
        post_grp = str(row["post_group"]).strip()
        cnt = int(row["cnt"])
        total_pay = float(row["total_pay"])
        total_bet = cnt * 100
        roi = total_pay / total_bet if total_bet > 0 else 0.0
        factor = (roi / baseline_roi) if (cnt >= min_samples and baseline_roi > 0) else 1.0

        result[post_grp] = {
            "factor": round(factor, 6),
            "samples": cnt,
            "roi": round(roi, 6),
        }
        logger.info(
            "  post_group=%s (fallback): factor=%.4f, roi=%.4f, samples=%d",
            post_grp, factor, roi, cnt,
        )

    return result


def _calc_class_change_stats(
    year_start: str,
    year_end: str,
    tansho_umaban_col: str,
    tansho_pay_col: str,
    baseline_roi: float,
    min_samples: int,
) -> dict[str, dict[str, Any]]:
    """クラス昇降級別の単勝回収率を算出する.

    LAGで前走の jyokencd5 + gradecd を取得し、クラスレベルの比較で
    昇級 (upgrade) / 降級 (downgrade) を分類する。

    Returns:
        {"class_upgrade": {...}, "class_downgrade": {...}}
    """
    start_minus_1 = str(int(year_start) - 1)

    # クラスレベル判定をSQL内で行う
    # gradecd in (A,B,C,D) → 1000, jyokencd5=999 → 900,
    # jyokencd5 in (701,702,703) → 100, else → jyokencd5 + 100
    class_level_expr = """
        CASE
            WHEN {grade} IN ('A','B','C','D') THEN 1000
            WHEN {jyoken} = '999' THEN 900
            WHEN {jyoken} IN ('701','702','703') THEN 100
            WHEN {jyoken} ~ '^[0-9]+$'
                 AND CAST({jyoken} AS int) BETWEEN 1 AND 100
                 THEN CAST({jyoken} AS int) + 100
            ELSE -1
        END"""

    cur_level_expr = class_level_expr.format(
        grade="r_cur.gradecd", jyoken="r_cur.jyokencd5",
    )
    prev_level_expr = class_level_expr.format(
        grade="prev_gradecd", jyoken="prev_jyokencd",
    )

    sql = f"""
    WITH horse_races AS (
        SELECT
            ur.kettonum, ur.year, ur.monthday, ur.jyocd, ur.kaiji,
            ur.nichiji, ur.racenum, ur.umaban, ur.kakuteijyuni,
            r.jyokencd5, r.gradecd
        FROM n_uma_race ur
        JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
        WHERE ur.datakubun = '7' AND ur.ijyocd = '0'
          AND ur.year BETWEEN %(start_m1)s AND %(end)s
          AND {_jyo_filter("ur.")}
    ),
    with_prev AS (
        SELECT *,
            LAG(jyokencd5) OVER (PARTITION BY kettonum ORDER BY year, monthday)
                AS prev_jyokencd,
            LAG(gradecd) OVER (PARTITION BY kettonum ORDER BY year, monthday)
                AS prev_gradecd
        FROM horse_races
    ),
    classified AS (
        SELECT wp.*,
            {cur_level_expr.replace('r_cur.gradecd', 'wp.gradecd').replace('r_cur.jyokencd5', 'wp.jyokencd5')} AS cur_level,
            {prev_level_expr.replace('prev_gradecd', 'wp.prev_gradecd').replace('prev_jyokencd', 'wp.prev_jyokencd')} AS prev_level
        FROM with_prev wp
        WHERE wp.prev_jyokencd IS NOT NULL
          AND wp.year BETWEEN %(start)s AND %(end)s
    )
    SELECT
        CASE
            WHEN cur_level > prev_level AND cur_level >= 0 AND prev_level >= 0
                THEN 'class_upgrade'
            WHEN cur_level < prev_level AND cur_level >= 0 AND prev_level >= 0
                THEN 'class_downgrade'
        END AS change_type,
        COUNT(*) AS cnt,
        SUM(CASE WHEN c.kakuteijyuni ~ '^[0-9]+$'
                      AND CAST(c.kakuteijyuni AS int) = 1
            THEN COALESCE(CAST(o.tanodds AS numeric) / 10.0, 0)
            ELSE 0 END) * 100 AS total_pay
    FROM classified c
    JOIN n_odds_tanpuku o
        ON c.year = o.year AND c.monthday = o.monthday
        AND c.jyocd = o.jyocd AND c.kaiji = o.kaiji
        AND c.nichiji = o.nichiji AND c.racenum = o.racenum
        AND c.umaban = o.umaban
    WHERE (cur_level > prev_level OR cur_level < prev_level)
      AND cur_level >= 0 AND prev_level >= 0
      AND o.tanodds ~ '^[0-9]+$'
      AND CAST(o.tanodds AS int) > 0
    GROUP BY change_type
    """
    df = query_df(
        sql,
        {"start_m1": start_minus_1, "start": year_start, "end": year_end},
    )

    result: dict[str, dict[str, Any]] = {}
    for label in ("class_upgrade", "class_downgrade"):
        matched = df[df["change_type"] == label]
        if matched.empty:
            result[label] = {"factor": 1.0, "samples": 0, "roi": 0.0}
            continue
        r = matched.iloc[0]
        cnt = int(r["cnt"])
        total_pay = float(r["total_pay"])
        total_bet = cnt * 100
        roi = total_pay / total_bet if total_bet > 0 else 0.0
        factor = (roi / baseline_roi) if (cnt >= min_samples and baseline_roi > 0) else 1.0

        result[label] = {
            "factor": round(factor, 6),
            "samples": cnt,
            "roi": round(roi, 6),
        }
        logger.info(
            "  %s: factor=%.4f, roi=%.4f, samples=%d",
            label, factor, roi, cnt,
        )

    return result


def _calc_filly_transition_stats(
    year_start: str,
    year_end: str,
    tansho_umaban_col: str,
    tansho_pay_col: str,
    baseline_roi: float,
    min_samples: int,
) -> dict[str, dict[str, Any]]:
    """牝馬限定⇔混合遷移別の単勝回収率を算出する.

    BOOL_AND(sexcd='2') でレースが牝馬限定かを判定し、
    LAGで前走との遷移を検出する。対象は牝馬（sexcd='2'）のみ。

    Returns:
        {"filly_to_mixed": {...}, "mixed_to_filly": {...}}
    """
    start_minus_1 = str(int(year_start) - 1)

    sql = f"""
    WITH race_gender AS (
        SELECT year, monthday, jyocd, kaiji, nichiji, racenum,
            BOOL_AND(sexcd = '2') AS is_filly_only
        FROM n_uma_race
        WHERE datakubun = '7' AND ijyocd = '0'
          AND year BETWEEN %(start_m1)s AND %(end)s
          AND {_jyo_filter()}
        GROUP BY year, monthday, jyocd, kaiji, nichiji, racenum
    ),
    horse_races AS (
        SELECT ur.kettonum, ur.year, ur.monthday, ur.jyocd, ur.kaiji,
            ur.nichiji, ur.racenum, ur.umaban, ur.kakuteijyuni, ur.sexcd,
            rg.is_filly_only
        FROM n_uma_race ur
        JOIN race_gender rg USING (year, monthday, jyocd, kaiji, nichiji, racenum)
        WHERE ur.datakubun = '7' AND ur.ijyocd = '0'
          AND ur.year BETWEEN %(start_m1)s AND %(end)s
          AND {_jyo_filter("ur.")}
    ),
    with_prev AS (
        SELECT *,
            LAG(is_filly_only) OVER (
                PARTITION BY kettonum ORDER BY year, monthday
            ) AS prev_filly_only
        FROM horse_races
    )
    SELECT
        CASE
            WHEN prev_filly_only = TRUE AND is_filly_only = FALSE
                THEN 'filly_to_mixed'
            WHEN prev_filly_only = FALSE AND is_filly_only = TRUE
                THEN 'mixed_to_filly'
        END AS transition_type,
        COUNT(*) AS cnt,
        SUM(CASE WHEN wp.kakuteijyuni ~ '^[0-9]+$'
                      AND CAST(wp.kakuteijyuni AS int) = 1
            THEN COALESCE(CAST(o.tanodds AS numeric) / 10.0, 0)
            ELSE 0 END) * 100 AS total_pay
    FROM with_prev wp
    JOIN n_odds_tanpuku o
        ON wp.year = o.year AND wp.monthday = o.monthday
        AND wp.jyocd = o.jyocd AND wp.kaiji = o.kaiji
        AND wp.nichiji = o.nichiji AND wp.racenum = o.racenum
        AND wp.umaban = o.umaban
    WHERE wp.year BETWEEN %(start)s AND %(end)s
      AND wp.sexcd = '2'
      AND wp.prev_filly_only IS NOT NULL
      AND (
          (wp.prev_filly_only = TRUE AND wp.is_filly_only = FALSE)
          OR (wp.prev_filly_only = FALSE AND wp.is_filly_only = TRUE)
      )
      AND o.tanodds ~ '^[0-9]+$'
      AND CAST(o.tanodds AS int) > 0
    GROUP BY transition_type
    """
    df = query_df(
        sql,
        {"start_m1": start_minus_1, "start": year_start, "end": year_end},
    )

    result: dict[str, dict[str, Any]] = {}
    for label in ("filly_to_mixed", "mixed_to_filly"):
        matched = df[df["transition_type"] == label]
        if matched.empty:
            result[label] = {"factor": 1.0, "samples": 0, "roi": 0.0}
            continue
        r = matched.iloc[0]
        cnt = int(r["cnt"])
        total_pay = float(r["total_pay"])
        total_bet = cnt * 100
        roi = total_pay / total_bet if total_bet > 0 else 0.0
        factor = (roi / baseline_roi) if (cnt >= min_samples and baseline_roi > 0) else 1.0

        result[label] = {
            "factor": round(factor, 6),
            "samples": cnt,
            "roi": round(roi, 6),
        }
        logger.info(
            "  %s: factor=%.4f, roi=%.4f, samples=%d",
            label, factor, roi, cnt,
        )

    return result
