# 特徴量設計書

## 概要

EveryDB2のPostgreSQLデータからLightGBMの特徴量を抽出する設計。
すべての特徴量は「予測対象レースの発走前に取得可能な情報」のみで構成する（データリーク防止）。

---

## 目的変数

```sql
-- 3着以内なら1、それ以外は0
CASE WHEN CAST(kakuteijyuni AS integer) <= 3 THEN 1 ELSE 0 END AS target
```

対象条件:
- `datakubun = '7'`（確定成績）
- `ijyocd = '0'`（正常出走のみ）
- `jyocd` IN ('01'～'10')（JRA中央10場のみ）

---

## 特徴量一覧

### カテゴリ1: 馬基本属性（horse_basic_*）

ソーステーブル: `n_uma_race`, `n_uma`

| # | 特徴量名 | 型 | 抽出SQL/ロジック | 説明 |
|---|---------|---|----------------|------|
| 1 | horse_sex | cat | `sexcd` | 性別（1:牡, 2:牝, 3:セン） |
| 2 | horse_age | int | `CAST(barei AS int)` | 馬齢 |
| 3 | horse_tozai | cat | `tozaicd` | 東西所属（1:美浦, 2:栗東） |
| 4 | horse_blinker | int | `CAST(blinker AS int)` | ブリンカー（0:不使用, 1:使用） |
| 5 | horse_keiro | cat | `keirocd` | 毛色コード |

SQL例:
```sql
SELECT kettonum, sexcd, CAST(barei AS integer) AS horse_age,
       tozaicd, blinker, keirocd
FROM n_uma_race
WHERE year = :year AND monthday = :monthday
  AND jyocd = :jyocd AND racenum = :racenum
  AND datakubun IN ('1','2','7')
```

---

### カテゴリ2: 過去成績（horse_perf_*）

ソーステーブル: `n_uma_race`, `n_race`

**算出の基本方針:** 当該レースの `year + monthday` より前の確定成績データ（`datakubun='7'`）からN走分を取得。

| # | 特徴量名 | 型 | 集計期間 | 説明 |
|---|---------|---|---------|------|
| 6 | horse_run_count | int | 全走 | 通算出走回数 |
| 7 | horse_win_count | int | 全走 | 通算1着回数 |
| 8 | horse_win_rate | float | 全走 | 通算勝率 |
| 9 | horse_rentai_rate | float | 全走 | 通算連対率（2着以内率） |
| 10 | horse_fukusho_rate | float | 全走 | 通算複勝率（3着以内率） |
| 11 | horse_win_rate_last5 | float | 直近5走 | 直近5走勝率 |
| 12 | horse_rentai_rate_last5 | float | 直近5走 | 直近5走連対率 |
| 13 | horse_fukusho_rate_last5 | float | 直近5走 | 直近5走複勝率 |
| 14 | horse_avg_jyuni_last5 | float | 直近5走 | 直近5走平均着順 |
| 15 | horse_avg_jyuni_last3 | float | 直近3走 | 直近3走平均着順 |
| 16 | horse_last_jyuni | int | 前走 | 前走着順 |
| 17 | horse_last2_jyuni | int | 前々走 | 前々走着順 |
| 18 | horse_best_jyuni_last5 | int | 直近5走 | 直近5走の最高着順 |

SQL例（直近5走取得）:
```sql
SELECT kettonum, kakuteijyuni, time, harontimel3, kyori, trackcd,
       year, monthday, jyocd
FROM n_uma_race ur
JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
WHERE ur.kettonum = :kettonum
  AND ur.datakubun = '7'
  AND ur.ijyocd = '0'
  AND (ur.year || ur.monthday) < :race_date
ORDER BY ur.year DESC, ur.monthday DESC
LIMIT 5
```

---

### カテゴリ3: 条件別成績（horse_cond_*）

ソーステーブル: `n_uma_race`, `n_race`

| # | 特徴量名 | 型 | 説明 |
|---|---------|---|------|
| 19 | horse_turf_fukusho_rate | float | 芝レースでの複勝率（TrackCD 10-22） |
| 20 | horse_dirt_fukusho_rate | float | ダートレースでの複勝率（TrackCD 23-29） |
| 21 | horse_dist_short_rate | float | 短距離(～1400m)複勝率 |
| 22 | horse_dist_mile_rate | float | マイル(1401-1800m)複勝率 |
| 23 | horse_dist_middle_rate | float | 中距離(1801-2200m)複勝率 |
| 24 | horse_dist_long_rate | float | 長距離(2201m～)複勝率 |
| 25 | horse_same_jyo_rate | float | 同一競馬場での複勝率 |
| 26 | horse_same_dist_rate | float | 同一距離(±100m)での複勝率 |
| 27 | horse_same_track_rate | float | 同一トラック種別での複勝率 |
| 28 | horse_heavy_rate | float | 重・不良馬場での複勝率（BabaCD 3-4） |
| 29 | horse_good_rate | float | 良馬場での複勝率（BabaCD 1） |
| 30 | horse_grade_rate | float | 重賞（GradeCD A-H）での複勝率 |
| 31 | horse_same_jyo_runs | int | 同一競馬場での出走回数 |
| 32 | horse_same_dist_runs | int | 同一距離での出走回数 |

**距離カテゴリの判定:**
```python
def distance_category(kyori: int) -> str:
    if kyori <= 1400: return 'short'
    elif kyori <= 1800: return 'mile'
    elif kyori <= 2200: return 'middle'
    else: return 'long'
```

---

### カテゴリ4: スピード指数（speed_*）

ソーステーブル: `n_uma_race`, `n_race`

| # | 特徴量名 | 型 | 説明 |
|---|---------|---|------|
| 33 | speed_time_last | float | 前走走破タイム（秒換算） |
| 34 | speed_time_avg_last3 | float | 直近3走の平均走破タイム |
| 35 | speed_l3f_last | float | 前走上がり3ハロン（秒） |
| 36 | speed_l3f_avg_last3 | float | 直近3走の平均上がり3F |
| 37 | speed_l3f_best_last5 | float | 直近5走の最速上がり3F |
| 38 | speed_l3f_rank_last | int | 前走レース内での上がり3F順位 |
| 39 | speed_timediff_last | float | 前走の1着馬とのタイム差（※DBカラム名は `timediff`。リファレンスの `TimeDIFN` は誤り） |
| 40 | speed_timediff_avg_last3 | float | 直近3走平均タイム差 |
| 41 | speed_index_last | float | 前走スピード指数（※後述の算出式） |
| 42 | speed_index_avg_last3 | float | 直近3走平均スピード指数 |
| 43 | speed_index_max_last5 | float | 直近5走最高スピード指数 |
| 44 | speed_l3f_time_ratio | float | 前走の上がり3F / レースラップ後3F比 |

**スピード指数の算出式（簡易版）:**
```python
def calc_speed_index(time_sec, distance, track_type, baba_cd, base_time_dict):
    """
    スピード指数 = (基準タイム - 走破タイム) / 基準タイム * 1000 + 馬場補正
    base_time_dict: 距離×トラック×馬場別の平均走破タイム（過去3年分等から算出）
    """
    key = (distance, track_type, baba_cd)
    base_time = base_time_dict.get(key, time_sec)
    raw_index = (base_time - time_sec) / base_time * 1000
    return raw_index
```

**走破タイムの秒変換:**
```python
def time_to_sec(time_str: str) -> float:
    """'1234' -> 1分23.4秒 -> 83.4秒"""
    if not time_str or time_str.strip() == '':
        return None
    t = time_str.strip()
    minutes = int(t[0])
    seconds = int(t[1:3])
    tenths = int(t[3])
    return minutes * 60 + seconds + tenths * 0.1
```

---

### カテゴリ5: 脚質（style_*）

**実装ファイル:** `src/features/speed.py`（SpeedStyleFeatureExtractor に統合。スピード指数と脚質は同じ過去成績データを参照するため。）

ソーステーブル: `n_uma_race`

| # | 特徴量名 | 型 | 説明 |
|---|---------|---|------|
| 45 | style_type_last | cat | 前走脚質（KyakusituKubun: 1逃,2先,3差,4追） |
| 46 | style_type_mode_last5 | cat | 直近5走の最頻脚質 |
| 47 | style_avg_pos_1c_last3 | float | 直近3走の1コーナー平均順位 |
| 48 | style_avg_pos_3c_last3 | float | 直近3走の3コーナー平均順位 |
| 49 | style_avg_pos_4c_last3 | float | 直近3走の4コーナー平均順位 |
| 50 | style_pos_change_last | float | 前走の4角順位 → 最終着順の変動 |
| 51 | style_front_ratio_last5 | float | 直近5走で3角3番手以内だった割合 |

---

### カテゴリ6: レース条件（race_*）

ソーステーブル: `n_race`（予測対象レースの情報）

| # | 特徴量名 | 型 | 抽出元 | 説明 |
|---|---------|---|-------|------|
| 52 | race_jyo_cd | cat | JyoCD | 競馬場コード（01-10） |
| 53 | race_distance | int | Kyori | 距離(m) |
| 54 | race_track_cd | cat | TrackCD | トラックコード |
| 55 | race_track_type | cat | — | 芝/ダート/障害（TrackCDから派生） |
| 56 | race_course_dir | cat | — | 左回り/右回り/直線（TrackCDから派生） |
| 57 | race_baba_cd | int | SibaBabaCD/DirtBabaCD | 馬場状態（1:良～4:不良） |
| 58 | race_tenko_cd | int | TenkoCD | 天候（1:晴～6:小雪） |
| 59 | race_grade_cd | cat | GradeCD | グレード |
| 60 | race_syubetu_cd | cat | SyubetuCD | 競走種別 |
| 61 | race_jyuryo_cd | cat | JyuryoCD | 重量種別（1:ﾊﾝﾃﾞ,2:別定,3:馬齢,4:定量） |
| 62 | race_jyoken_cd | cat | JyokenCD5 | 競走条件（701新馬～999OP） |
| 63 | race_tosu | int | SyussoTosu | 出走頭数 |
| 64 | race_month | int | — | 開催月（MonthDayの上2桁） |
| 65 | race_is_tokubetsu | int | TokuNum | 特別戦フラグ（TokuNum != '0000'） |

**トラック種別の派生:**
```python
def track_type(track_cd: str) -> str:
    cd = int(track_cd)
    if 10 <= cd <= 22: return 'turf'
    elif 23 <= cd <= 29: return 'dirt'
    elif 51 <= cd <= 59: return 'jump'
    return 'unknown'

def course_direction(track_cd: str) -> str:
    cd = int(track_cd)
    if cd in (10, 29): return 'straight'
    elif cd in (11,12,13,14,15,16,23,25,27): return 'left'
    elif cd in (17,18,19,20,21,22,24,26,28): return 'right'
    return 'unknown'
```

---

### カテゴリ7: 枠順・馬番（post_*）

ソーステーブル: `n_uma_race`

| # | 特徴量名 | 型 | 説明 |
|---|---------|---|------|
| 66 | post_wakuban | int | 枠番（1-8） |
| 67 | post_umaban | int | 馬番（1-18） |
| 68 | post_umaban_norm | float | 馬番 / 出走頭数（正規化）|
| 69 | post_is_inner | int | 内枠フラグ（枠番1-3=1, 他=0） |
| 70 | post_is_outer | int | 外枠フラグ（枠番6-8=1, 他=0） |

---

### カテゴリ8: 負担重量（weight_*）

ソーステーブル: `n_uma_race`

| # | 特徴量名 | 型 | 説明 |
|---|---------|---|------|
| 71 | weight_futan | float | 負担重量(kg)= CAST(futan AS numeric)/10 |
| 72 | weight_futan_diff | float | 前走との負担重量差 |
| 73 | weight_futan_vs_avg | float | 同レース出走馬の平均負担重量との差 |

---

### カテゴリ9: 馬体重（bw_*）

ソーステーブル: `n_uma_race`

| # | 特徴量名 | 型 | 説明 |
|---|---------|---|------|
| 74 | bw_weight | int | 馬体重(kg) |
| 75 | bw_change | int | 増減差(kg) 符号付き |
| 76 | bw_abs_change | int | 増減差の絶対値 |
| 77 | bw_weight_vs_avg | float | 自身の過去平均馬体重との差 |
| 78 | bw_is_big_change | int | 大幅増減フラグ（|増減| >= 10kg） |

注意: 馬体重は前日発表の場合もあり、速報系テーブル `s_uma_race` から取得する場合がある。

---

### カテゴリ10: 騎手（jockey_*）

ソーステーブル: `n_kisyu`, `n_kisyu_seiseki`, `n_uma_race`

| # | 特徴量名 | 型 | 説明 |
|---|---------|---|------|
| 79 | jockey_code | cat | 騎手コード |
| 80 | jockey_win_rate_year | float | 当年勝率（KISYU_SEISEKI） |
| 81 | jockey_fukusho_rate_year | float | 当年複勝率 |
| 82 | jockey_minarai | int | 見習減量（0:なし, 1:1kg, 2:2kg, 3:3kg） |
| 83 | jockey_win_rate_jyo | float | 当該競馬場での勝率（過去2年） |
| 84 | jockey_same_horse_rate | float | 同馬騎乗時の複勝率 |
| 85 | jockey_change_flag | int | 乗り替わりフラグ（前走と騎手が異なる=1） |
| 86 | jockey_avg_ninki_diff | float | 直近20走の（着順 - 人気）平均 |

**騎手成績の計算:**
```sql
SELECT
    CAST(heichichakukaisu1 AS integer) AS wins,
    CAST(heichichakukaisu1 AS integer)
      + CAST(heichichakukaisu2 AS integer)
      + CAST(heichichakukaisu3 AS integer) AS top3,
    CAST(heichichakukaisu1 AS integer)
      + CAST(heichichakukaisu2 AS integer)
      + CAST(heichichakukaisu3 AS integer)
      + CAST(heichichakukaisu4 AS integer)
      + CAST(heichichakukaisu5 AS integer)
      + CAST(heichichakukaisu6 AS integer) AS total_runs
FROM n_kisyu_seiseki
WHERE kisyucode = :code AND setyear = :year
```

---

### カテゴリ11: 調教師（trainer_*）

ソーステーブル: `n_chokyo`, `n_chokyo_seiseki`, `n_uma_race`

| # | 特徴量名 | 型 | 説明 |
|---|---------|---|------|
| 87 | trainer_code | cat | 調教師コード |
| 88 | trainer_win_rate_year | float | 当年勝率 |
| 89 | trainer_fukusho_rate_year | float | 当年複勝率 |
| 90 | trainer_win_rate_jyo | float | 当該競馬場での勝率（過去2年） |
| 91 | trainer_tozai | cat | 東西所属 |
| 92 | trainer_jockey_combo_rate | float | 騎手×調教師コンビの複勝率（過去2年） |
| 93 | trainer_jockey_combo_runs | int | コンビの出走回数 |

---

### カテゴリ12: 調教データ（training_*）

ソーステーブル: `n_hanro`, `n_wood_chip`

| # | 特徴量名 | 型 | 説明 |
|---|---------|---|------|
| 94 | training_hanro_time4 | float | 坂路4Fタイム（最終追切、秒） |
| 95 | training_hanro_time3 | float | 坂路3Fタイム |
| 96 | training_hanro_lap1 | float | 坂路最終1F（200M-0M）ラップ |
| 97 | training_hanro_accel | float | 坂路加速度（LapTime4 - LapTime1） |
| 98 | training_wc_time_best | float | ウッドチップ最速5Fタイム（直近2週） |
| 99 | training_days_from_last | int | 最終追切から発走日までの日数 |
| 100 | training_count_2weeks | int | 直近2週間の調教本数 |

**坂路最終追切の取得:**
```sql
SELECT harontime4, harontime3, laptime4, laptime3, laptime2, laptime1
FROM n_hanro
WHERE kettonum = :kettonum
  AND chokyodate <= :race_date
ORDER BY chokyodate DESC, chokyotime DESC
LIMIT 1
```

---

### カテゴリ13: 血統（blood_*）

ソーステーブル: `n_uma`, `n_hansyoku`, `n_sanku`, `n_keito`

| # | 特徴量名 | 型 | 説明 |
|---|---------|---|------|
| 101 | blood_father_id | cat | 父馬繁殖登録番号 |
| 102 | blood_bms_id | cat | 母父馬繁殖登録番号（BMS） |
| 103 | blood_father_keito | cat | 父系統名（KEITOテーブル） |
| 104 | blood_bms_keito | cat | 母父系統名 |
| 105 | blood_father_turf_rate | float | 父産駒の芝複勝率（過去3年集計） |
| 106 | blood_father_dirt_rate | float | 父産駒のダート複勝率 |
| 107 | blood_father_dist_rate | float | 父産駒の同距離帯複勝率 |
| 108 | blood_bms_turf_rate | float | 母父産駒の芝複勝率 |
| 109 | blood_bms_dirt_rate | float | 母父産駒のダート複勝率 |
| 110 | blood_inbreed_flag | int | 近親交配フラグ（3代以内に同一祖先） |

**父系統の取得:**
```sql
-- UMAマスタの3代血統情報から父馬の繁殖登録番号を取得
-- → KEITOテーブルで系統名を検索
SELECT k.keitoname
FROM n_uma u
JOIN n_keito k ON k.hansyokunum = u.ketto3infohanNum1  -- 父の繁殖番号
WHERE u.kettonum = :kettonum
```

**父産駒成績の集計（血統適性）:**
```sql
-- 父馬の繁殖登録番号を基に、産駒の全成績を集計
SELECT
    COUNT(*) AS total,
    SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) <= 3 THEN 1 ELSE 0 END) AS top3
FROM n_uma u
JOIN n_uma_race ur ON u.kettonum = ur.kettonum
JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
WHERE u.ketto3infohanNum1 = :father_hansyoku_num  -- 同じ父
  AND ur.datakubun = '7' AND ur.ijyocd = '0'
  AND CAST(r.trackcd AS int) BETWEEN 10 AND 22  -- 芝
  AND (r.year || r.monthday) < :race_date
```

---

### カテゴリ14: 間隔・ローテーション（interval_*）

ソーステーブル: `n_uma_race`

| # | 特徴量名 | 型 | 説明 |
|---|---------|---|------|
| 111 | interval_days | int | 前走からの中N日 |
| 112 | interval_category | cat | 連闘(0-6日)/中1-2週/中3-4週/中5-8週/中9週以上/休み明け(半年以上) |
| 113 | interval_days_prev2 | int | 前々走からのN日（ローテーション） |
| 114 | interval_is_rensho | int | 連闘フラグ（7日以内=1） |
| 115 | interval_is_kyuumei | int | 休み明けフラグ（90日以上=1） |

---

### カテゴリ15: オッズ・人気（odds_*）

ソーステーブル: `n_odds_tanpuku`, `n_uma_race`

| # | 特徴量名 | 型 | 説明 |
|---|---------|---|------|
| 116 | odds_tan | float | 単勝オッズ |
| 117 | odds_ninki | int | 人気順 |
| 118 | odds_log | float | ln(単勝オッズ) — スケール調整 |
| 119 | odds_fuku_low | float | 複勝最低オッズ |
| 120 | odds_fuku_high | float | 複勝最高オッズ |
| 121 | odds_is_favorite | int | 1番人気フラグ |
| 122 | odds_is_top3_ninki | int | 3番人気以内フラグ |

**注意:** オッズは発走直前に確定するため、予測のタイミングによって使用可否が異なる。
- **前日予測:** 前日売オッズを使用
- **当日予測:** 締切直前オッズを使用
- **オッズなし予測:** オッズ特徴量を除外したモデルも別途作成推奨（`--no-odds` オプション）

**DB実装上の注意:**
- `n_odds_tanpuku`（明細テーブル）には `DataKubun` カラムが存在しない
- `DataKubun` はヘッダテーブル `n_odds_tanpukuwaku_head` にのみ存在
- 明細テーブルはEveryDB2側で最新のオッズに上書きされるため、そのまま取得すれば最新データが得られる

---

### カテゴリ16: クロス特徴量（cross_*）

他カテゴリから派生する交差特徴量。

| # | 特徴量名 | 型 | 説明 |
|---|---------|---|------|
| 123 | cross_dist_change | int | 前走との距離差（m） |
| 124 | cross_dist_category_change | cat | 距離カテゴリ変更（短→マイルなど） |
| 125 | cross_track_change | int | 芝↔ダート変更フラグ |
| 126 | cross_class_change | int | クラス変更（昇級=1, 同級=0, 降級=-1） |
| 127 | cross_jyo_change | int | 競馬場変更フラグ |
| 128 | cross_weight_futan_per_bw | float | 負担重量/馬体重 比率 |
| 129 | cross_jockey_horse_runs | int | 同馬×同騎手の過去出走回数 |
| 130 | cross_jockey_horse_wins | int | 同馬×同騎手の過去勝利回数 |

---

## 特徴量の合計

| カテゴリ | 特徴量数 |
|---------|---------|
| 馬基本属性 | 5 |
| 過去成績 | 13 |
| 条件別成績 | 14 |
| スピード指数 | 12 |
| 脚質 | 7 |
| レース条件 | 14 |
| 枠順・馬番 | 5 |
| 負担重量 | 3 |
| 馬体重 | 5 |
| 騎手 | 8 |
| 調教師 | 7 |
| 調教データ | 7 |
| 血統 | 10 |
| 間隔 | 5 |
| オッズ | 7 |
| クロス特徴量 | 8 |
| **合計** | **130** |

---

## 欠損値の処理方針

| 状況 | 処理 |
|------|------|
| 新馬（過去成績なし） | 数値特徴量は-1、率系は0.0 |
| 調教データ不明 | -1（LightGBMはNaN対応だが明示的に） |
| 海外遠征歴 | 対象外（JRA10場のみ集計） |
| 騎手コード不明（外国人短期免許等） | 全体平均で代替 |
| 血統不明 | "unknown"カテゴリ |

LightGBMの設定で `use_missing=true`, `zero_as_missing=false` とすることで、
-1を欠損マーカーとして自然に扱える。

---

## 特徴量重要度による選別（実装後）

初期実装後、以下の方法で特徴量を選別する:

1. LightGBMの `feature_importance(importance_type='gain')` で重要度ランキング
2. 上位80特徴量程度に絞ったモデルと全特徴量モデルを比較
3. Permutation importanceで相互検証
4. 相関が高い特徴量ペア（r > 0.95）は片方を除去

---

## カテゴリ別 実装ファイル対応表

| カテゴリ | 実装ファイル | クラス名 |
|---------|-----------|---------|
| 1. 馬基本属性 | src/features/horse.py | HorseFeatureExtractor |
| 2. 過去成績 | src/features/horse.py | HorseFeatureExtractor |
| 3. 条件別成績 | src/features/horse.py | HorseFeatureExtractor |
| 4. スピード指数 | src/features/speed.py | SpeedStyleFeatureExtractor |
| 5. 脚質 | src/features/speed.py | SpeedStyleFeatureExtractor |
| 6. レース条件 | src/features/race.py | RaceFeatureExtractor |
| 7. 枠順・馬番 | src/features/race.py | RaceFeatureExtractor |
| 8. 負担重量 | src/features/race.py（一部horse.py） | RaceFeatureExtractor |
| 9. 馬体重 | src/features/horse.py | HorseFeatureExtractor |
| 10. 騎手 | src/features/jockey_trainer.py | JockeyTrainerFeatureExtractor |
| 11. 調教師 | src/features/jockey_trainer.py | JockeyTrainerFeatureExtractor |
| 12. 調教データ | src/features/training.py | TrainingFeatureExtractor |
| 13. 血統 | src/features/bloodline.py | BloodlineFeatureExtractor |
| 14. 間隔 | src/features/horse.py | HorseFeatureExtractor |
| 15. オッズ | src/features/odds.py | OddsFeatureExtractor |
| 16. クロス特徴量 | src/features/pipeline.py | FeaturePipeline._add_cross_features() |
