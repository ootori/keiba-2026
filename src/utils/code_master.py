"""コード表変換ユーティリティ.

EveryDB2のコード値を意味のある文字列やカテゴリに変換するヘルパー関数群。
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# 競馬場コード (JyoCD)
# ---------------------------------------------------------------------------
JYO_CODE_MAP: dict[str, str] = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}

# ---------------------------------------------------------------------------
# グレードコード (GradeCD)
# ---------------------------------------------------------------------------
GRADE_CODE_MAP: dict[str, str] = {
    "A": "G1",
    "B": "G2",
    "C": "G3",
    "D": "重賞",
    "E": "特別",
    " ": "一般",
    "": "一般",
}

# ---------------------------------------------------------------------------
# 馬場状態コード
# ---------------------------------------------------------------------------
BABA_CODE_MAP: dict[str, str] = {
    "1": "良",
    "2": "稍重",
    "3": "重",
    "4": "不良",
}

# ---------------------------------------------------------------------------
# 天候コード
# ---------------------------------------------------------------------------
TENKO_CODE_MAP: dict[str, str] = {
    "1": "晴",
    "2": "曇",
    "3": "雨",
    "4": "小雨",
    "5": "雪",
    "6": "小雪",
}

# ---------------------------------------------------------------------------
# 性別コード
# ---------------------------------------------------------------------------
SEX_CODE_MAP: dict[str, str] = {
    "1": "牡",
    "2": "牝",
    "3": "セン",
}

# ---------------------------------------------------------------------------
# 異常区分コード
# ---------------------------------------------------------------------------
IJYO_CODE_MAP: dict[str, str] = {
    "0": "異常なし",
    "1": "出走取消",
    "2": "発走除外",
    "3": "競走除外",
    "4": "競走中止",
    "5": "失格",
    "6": "落馬再騎乗",
    "7": "降着",
}


def track_type(track_cd: str) -> str:
    """トラックコードから芝/ダート/障害を判定する.

    Args:
        track_cd: TrackCD値

    Returns:
        'turf', 'dirt', 'jump', または 'unknown'
    """
    try:
        cd = int(track_cd)
    except (ValueError, TypeError):
        return "unknown"
    if 10 <= cd <= 22:
        return "turf"
    elif 23 <= cd <= 29:
        return "dirt"
    elif 51 <= cd <= 59:
        return "jump"
    return "unknown"


def course_direction(track_cd: str) -> str:
    """トラックコードから左回り/右回り/直線を判定する.

    Args:
        track_cd: TrackCD値

    Returns:
        'left', 'right', 'straight', または 'unknown'
    """
    try:
        cd = int(track_cd)
    except (ValueError, TypeError):
        return "unknown"
    if cd in (10, 29):
        return "straight"
    elif cd in (11, 12, 13, 14, 15, 16, 23, 25, 27):
        return "left"
    elif cd in (17, 18, 19, 20, 21, 22, 24, 26, 28):
        return "right"
    return "unknown"


def distance_category(kyori: int) -> str:
    """距離を距離カテゴリに分類する.

    Args:
        kyori: 距離(m)

    Returns:
        'short', 'mile', 'middle', 'long'
    """
    if kyori <= 1400:
        return "short"
    elif kyori <= 1800:
        return "mile"
    elif kyori <= 2200:
        return "middle"
    else:
        return "long"


def time_to_sec(time_str: str) -> float | None:
    """走破タイム文字列を秒に変換する.

    Args:
        time_str: '1234' 形式のタイム文字列（1分23.4秒）

    Returns:
        秒数 (例: 83.4) または None
    """
    if not time_str or not time_str.strip():
        return None
    t = time_str.strip()
    if len(t) < 4:
        return None
    try:
        minutes = int(t[0])
        seconds = int(t[1:3])
        tenths = int(t[3])
        return minutes * 60 + seconds + tenths * 0.1
    except (ValueError, IndexError):
        return None


def haron_time_to_sec(haron_str: str) -> float | None:
    """ハロンタイム文字列を秒に変換する.

    Args:
        haron_str: '999' 形式（99.9秒）または '9999' 形式

    Returns:
        秒数 または None
    """
    if not haron_str or not haron_str.strip():
        return None
    t = haron_str.strip()
    try:
        if len(t) == 3:
            return int(t[:2]) + int(t[2]) * 0.1
        elif len(t) == 4:
            return int(t[:3]) + int(t[3]) * 0.1
        return None
    except (ValueError, IndexError):
        return None


def interval_category(days: int) -> str:
    """前走からの日数を間隔カテゴリに分類する.

    Args:
        days: 前走からの日数

    Returns:
        カテゴリ文字列
    """
    if days < 0:
        return "unknown"
    elif days <= 6:
        return "rento"        # 連闘
    elif days <= 14:
        return "1_2weeks"     # 中1-2週
    elif days <= 28:
        return "3_4weeks"     # 中3-4週
    elif days <= 56:
        return "5_8weeks"     # 中5-8週
    elif days <= 90:
        return "9plus_weeks"  # 中9週以上
    else:
        return "kyuumei"      # 休み明け（半年以上）


def class_level(jyokencd5: str, gradecd: str) -> int:
    """条件コード+グレードからクラス序列値を返す.

    大きい値ほど高いクラスを表す。

    Args:
        jyokencd5: 競走条件コード（最若年）
        gradecd: グレードコード

    Returns:
        序列値（-1=判定不可, 100=新馬/未勝利, 条件戦は数値そのまま,
        900=オープン, 1000=重賞）
    """
    gradecd = str(gradecd or "").strip()
    if gradecd in ("A", "B", "C", "D"):
        return 1000
    jyoken_str = str(jyokencd5 or "").strip()
    if not jyoken_str:
        return -1
    try:
        jyoken = int(jyoken_str)
    except (ValueError, TypeError):
        return -1
    if jyoken == 999:
        return 900
    if jyoken in (701, 702, 703):
        return 100
    if 1 <= jyoken <= 100:
        return jyoken + 100  # 条件戦: 収得賞金ベース (201-200)
    return -1


def baba_code_for_track(track_cd: str, siba_baba_cd: str, dirt_baba_cd: str) -> str:
    """トラック種別に応じた馬場状態コードを返す.

    Args:
        track_cd: TrackCD
        siba_baba_cd: SibaBabaCD
        dirt_baba_cd: DirtBabaCD

    Returns:
        馬場状態コード ('1'-'4')
    """
    tt = track_type(track_cd)
    if tt == "turf":
        return siba_baba_cd if siba_baba_cd else "0"
    elif tt == "dirt":
        return dirt_baba_cd if dirt_baba_cd else "0"
    return "0"
