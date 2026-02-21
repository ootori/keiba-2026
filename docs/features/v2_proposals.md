# 特徴量深掘り提案（v2: 2026-02-20策定）

本ドキュメントは `docs/feature_design.md` の補足資料。重要度分析に基づく追加特徴量の詳細設計を記載する。

---

## 背景：現行モデルの特徴量重要度分析

2015-2024学習 / 2025検証の LightGBM gain 重要度 Top 30 から得られた知見:

```
1位  trainer_code: 126575.9      ← ID特徴量が上位4つを独占
2位  blood_bms_id: 122868.9      ← BMS条件別の特徴量が不足
3位  blood_father_id: 81528.2
4位  jockey_code: 79222.2
5位  rel_horse_avg_jyuni_last3_zscore: 68535.1  ← 相対特徴量が5-10位に集中
6位  rel_horse_fukusho_rate_zscore: 37105.2
7位  rel_horse_avg_jyuni_last3_rank: 35291.7
8位  horse_last_jyuni: 17370.8    ← フォーム/モメンタム系が弱い
9位  rel_speed_index_avg_last3_rank: 16069.3
10位 rel_horse_fukusho_rate_last5_zscore: 15868.5
```

**主要な課題:**

1. **ID特徴量への過度な依存:** trainer_code, blood_bms_id, blood_father_id, jockey_code の4つのID変数が全体の重要度の大部分を占める。モデルがID経由で「暗黙の条件別成績」を学習しており、明示的に条件別分解することでID依存を減らし汎化性能を改善できる
2. **BMS（母父）の条件別特徴量不足:** father側は6つの条件別特徴量（turf/dirt/dist/baba/jyo/nicks）があるが、BMS側はturf/dirtの2つのみ。blood_bms_idがfather_idより重要度が高い（122868 > 81528）のは、BMS情報がまだ十分に特徴量化されていないことを示唆
3. **フォームモメンタムの欠如:** horse_last_jyuni（8位）は「前走の着順」だが、「改善中か悪化中か」のトレンド情報がない
4. **レース内ペース構造の未活用:** 脚質は個馬単位だが、レース全体の展開予測（逃げ馬の多寡等）が未使用

---

## 提案A: 調教師条件別パフォーマンス（trainer_code深掘り）

**目的:** trainer_code（重要度1位: 126575）が凝縮している情報を明示的に分解し、未知の調教師への汎化を改善する。

**実装先:** `src/features/jockey_trainer.py` の `JockeyTrainerFeatureExtractor`

**既存との差分:** 現在は `trainer_win_rate_year`, `trainer_fukusho_rate_year`, `trainer_win_rate_jyo` の3つのみ。条件別分解が不十分。

| # | 特徴量名 | 型 | 集計期間 | 説明 |
|---|---------|---|---------|------|
| A1 | trainer_win_rate_track_type | float | 過去2年 | 調教師の今回トラック種別（芝/ダート）での勝率 |
| A2 | trainer_fukusho_rate_track_type | float | 過去2年 | 調教師の今回トラック種別での複勝率 |
| A3 | trainer_win_rate_dist_cat | float | 過去2年 | 調教師の今回距離帯（短/マイル/中/長）での勝率 |
| A4 | trainer_win_rate_baba | float | 過去2年 | 調教師の今回馬場状態（良/稍重/重/不良）での勝率 |
| A5 | trainer_recent_form_30d | float | 直近30日 | 直近30日間の勝率（好不調指標） |
| A6 | trainer_grade_rate | float | 過去3年 | 重賞レース（GradeCD A-D）での複勝率 |
| A7 | trainer_kyuumei_rate | float | 過去3年 | 休み明け馬（90日以上間隔）での複勝率 |

**SQL例（A1: 芝/ダート別成績）:**
```sql
SELECT ur.chokyosicode,
    COUNT(*) AS total,
    SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) = 1 THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) <= 3 THEN 1 ELSE 0 END) AS top3
FROM n_uma_race ur
JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
WHERE ur.chokyosicode IN %(codes)s
  AND ur.datakubun = '7'
  AND ur.ijyocd = '0'
  AND CAST(r.trackcd AS int) BETWEEN %(track_lo)s AND %(track_hi)s
  AND ur.year >= %(year_start)s
  AND (ur.year || ur.monthday) < %(race_date)s
GROUP BY ur.chokyosicode
```

**SQL例（A5: 直近30日成績）:**
```sql
SELECT ur.chokyosicode,
    COUNT(*) AS total,
    SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) = 1 THEN 1 ELSE 0 END) AS wins
FROM n_uma_race ur
WHERE ur.chokyosicode IN %(codes)s
  AND ur.datakubun = '7'
  AND ur.ijyocd = '0'
  AND (ur.year || ur.monthday) >= %(date_30d_ago)s
  AND (ur.year || ur.monthday) < %(race_date)s
GROUP BY ur.chokyosicode
```

**SQL例（A7: 休み明け成績）:**
```sql
WITH horse_races AS (
    SELECT ur.chokyosicode, ur.kettonum, ur.kakuteijyuni,
           ur.year || ur.monthday AS race_date,
           LAG(ur.year || ur.monthday) OVER (
               PARTITION BY ur.kettonum ORDER BY ur.year, ur.monthday
           ) AS prev_race_date
    FROM n_uma_race ur
    WHERE ur.datakubun = '7' AND ur.ijyocd = '0'
      AND ur.chokyosicode IN %(codes)s
      AND ur.year >= %(year_start)s
      AND (ur.year || ur.monthday) < %(race_date)s
)
SELECT chokyosicode,
    COUNT(*) AS total,
    SUM(CASE WHEN CAST(kakuteijyuni AS int) <= 3 THEN 1 ELSE 0 END) AS top3
FROM horse_races
WHERE prev_race_date IS NOT NULL
  AND CAST(race_date AS int) - CAST(prev_race_date AS int) >= 90
GROUP BY chokyosicode
```
※ 日付差の正確な計算にはPostgreSQLのdate型変換が望ましい（年跨ぎ対応）。実装時は `TO_DATE(race_date, 'YYYYMMDD') - TO_DATE(prev_race_date, 'YYYYMMDD') >= 90` の形で日数を計算すること。

**実装方針:**
- `_get_trainer_track_stats()`, `_get_trainer_dist_stats()`, `_get_trainer_baba_stats()` 等のメソッドを追加
- レース情報（trackcd, kyori, baba_cd）の取得は `extract()` の冒頭で1回だけ行い、各メソッドに渡す
- `_FEATURES` リストに7つの特徴量名を追加
- 欠損値: `MISSING_RATE` (0.0)

---

## 提案B: BMS（母父）条件別パフォーマンス（blood_bms_id深掘り） ✅ 実装済み

> **実装状態:** 2026-02-21 にサプリメントとして実装完了。
> **実装ファイル:** `src/features/bms_detail.py`（BMSDetailFeatureExtractor）
> **テスト:** `tests/test_bms_detail_supplement.py`（10テスト）
> **使用方法:** `--build-supplement bms_detail` で構築、`--supplement bms_detail` でマージ

**目的:** blood_bms_id（重要度2位: 122868）が持つ情報を条件別に展開する。father側の6条件別特徴量に対し、BMS側は2つのみという非対称を解消する。

**実装先:** `src/features/bms_detail.py`（サプリメント方式）
※ 当初は `bloodline.py` への組み込みを予定していたが、メイン parquet の再構築なしに独立して構築・実験できるサプリメント方式を採用した。

**既存との差分:** BMS側は `blood_bms_turf_rate`, `blood_bms_dirt_rate` の2つのみ。father側の `_get_sire_stats()`, `_get_sire_baba_stats()`, `_get_sire_jyo_stats()`, `_get_sire_dist_stats()` と同等のメソッドをBMS用に追加する。

| # | 特徴量名 | 型 | 集計期間 | 説明 |
|---|---------|---|---------|------|
| B1 | blood_bms_dist_rate | float | 過去3年 | 母父産駒の今回距離帯での複勝率 |
| B2 | blood_bms_baba_rate | float | 過去3年 | 母父産駒の今回馬場状態での複勝率 |
| B3 | blood_bms_jyo_rate | float | 過去3年 | 母父産駒の今回競馬場での複勝率 |
| B4 | blood_father_age_rate | float | 過去5年 | 父産駒の同馬齢（今回のbarei）での複勝率 |
| B5 | blood_nicks_track_rate | float | 過去5年 | 父×母父ニックスの芝/ダート別（今回トラック種別に対応）複勝率 |
| B6 | blood_father_class_rate | float | 過去3年 | 父産駒のクラス別成績（今回のJyokenCD5相当で条件戦/OP/重賞を分類） |

**SQL例（B1: BMS距離帯別成績）:**
```sql
SELECT
    s.mfnum AS bms_num,
    COUNT(*) AS total,
    SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) <= 3 THEN 1 ELSE 0 END) AS top3
FROM n_sanku s
JOIN n_uma_race ur ON s.kettonum = ur.kettonum
JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
WHERE s.mfnum IN %(bms_nums)s
  AND ur.datakubun = '7' AND ur.ijyocd = '0'
  AND r.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
  AND r.year >= %(year_start)s
  AND (r.year || r.monthday) < %(race_date)s
  AND CAST(r.kyori AS int) BETWEEN %(dist_lo)s AND %(dist_hi)s
GROUP BY s.mfnum
```

**SQL例（B4: 父産駒の馬齢別成績）:**
```sql
SELECT
    s.fnum AS sire_num,
    COUNT(*) AS total,
    SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) <= 3 THEN 1 ELSE 0 END) AS top3
FROM n_sanku s
JOIN n_uma_race ur ON s.kettonum = ur.kettonum
WHERE s.fnum IN %(sire_nums)s
  AND ur.datakubun = '7' AND ur.ijyocd = '0'
  AND CAST(ur.barei AS int) = %(barei)s
  AND ur.year >= %(year_start)s
  AND (ur.year || ur.monthday) < %(race_date)s
GROUP BY s.fnum
```

**SQL例（B5: ニックスのトラック種別別成績）:**
```sql
SELECT
    s.fnum AS father_num,
    s.mfnum AS bms_num,
    COUNT(*) AS total,
    SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) <= 3 THEN 1 ELSE 0 END) AS top3
FROM n_sanku s
JOIN n_uma_race ur ON s.kettonum = ur.kettonum
JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
WHERE s.fnum IN %(father_nums)s
  AND s.mfnum IN %(bms_nums)s
  AND ur.datakubun = '7' AND ur.ijyocd = '0'
  AND r.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
  AND CAST(r.trackcd AS int) BETWEEN %(track_lo)s AND %(track_hi)s
  AND r.year >= %(year_start)s
  AND (r.year || r.monthday) < %(race_date)s
GROUP BY s.fnum, s.mfnum
```

**実装方針（実装済み — 以下は実際の実装内容）:**
- `_get_bms_dist_stats()`, `_get_bms_baba_stats()`, `_get_bms_jyo_stats()` を追加（father版のメソッドを参考に `s.fnum` を `s.mfnum` に変更）
- `_get_sire_age_stats()` を新規追加（`n_uma_race.barei` でフィルタ）
- `_get_nicks_track_stats()` を新規追加（既存 `_get_nicks_stats()` を拡張、trackcdフィルタ追加）
- `_get_sire_class_stats()` を新規追加（`n_race.jyokencd5` + `gradecd` でクラス分類）
- `_FEATURES` リストに6つの特徴量名を追加
- **欠損値: NaN（LightGBMネイティブ欠損）** ← 当初の `MISSING_RATE=0.0` からNaNに変更。サンプル不足と複勝率0%を区別するため
- **ノイズ抑制: MIN_SAMPLES=20、MIN_SAMPLES_NICKS=30。** サンプル数が閾値未満の場合はNaNを返す（小サンプルの率 1/1=100% 等を除去）

**クラス分類の定義（B6用 — 実装版）:**
```python
def _classify_class(gradecd: str, jyokencd5: str) -> str:
    """GradeCD と JyokenCD5 からクラスを分類する."""
    gradecd = (gradecd or "").strip()
    if gradecd in ("A", "B", "C", "D"):
        return "grade"   # 重賞
    cd = int(jyokencd5) if (jyokencd5 or "").strip() else 0
    if cd >= 500:
        return "open"    # オープン
    return "jouken"      # 条件戦
```
※ 当初設計から変更: GradeCD による重賞判定を追加し、JyokenCD5 の閾値を 500 に調整。

---

## 提案C: 騎手条件別パフォーマンス（jockey_code深掘り）

**目的:** jockey_code（重要度4位: 79222）の情報を条件別に分解する。明示的な騎手特徴量で30位以内に入っているのは jockey_win_rate_jyo（12位, 11919）と jockey_avg_ninki_diff（22位, 2575）のみであり、条件別分解の余地が大きい。

**実装先:** `src/features/jockey_trainer.py` の `JockeyTrainerFeatureExtractor`

**既存との差分:** 騎手の競馬場別成績（`_get_jockey_jyo_stats()`）は既存。トラック種別・距離帯・直近フォーム・重賞・脚質適性が未実装。

| # | 特徴量名 | 型 | 集計期間 | 説明 |
|---|---------|---|---------|------|
| C1 | jockey_win_rate_track_type | float | 過去2年 | 騎手の今回トラック種別（芝/ダート）での勝率 |
| C2 | jockey_fukusho_rate_track_type | float | 過去2年 | 騎手の今回トラック種別での複勝率 |
| C3 | jockey_win_rate_dist_cat | float | 過去2年 | 騎手の今回距離帯での勝率 |
| C4 | jockey_recent_form_30d | float | 直近30日 | 直近30日間の勝率（好不調指標） |
| C5 | jockey_grade_rate | float | 過去3年 | 重賞レースでの複勝率 |

**SQL例（C1: 芝/ダート別成績）:**
```sql
SELECT ur.kisyucode,
    COUNT(*) AS total,
    SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) = 1 THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) <= 3 THEN 1 ELSE 0 END) AS top3
FROM n_uma_race ur
JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
WHERE ur.kisyucode IN %(codes)s
  AND ur.datakubun = '7' AND ur.ijyocd = '0'
  AND CAST(r.trackcd AS int) BETWEEN %(track_lo)s AND %(track_hi)s
  AND ur.year >= %(year_start)s
  AND (ur.year || ur.monthday) < %(race_date)s
GROUP BY ur.kisyucode
```

**実装方針:**
- `_get_jockey_track_stats()`, `_get_jockey_dist_stats()`, `_get_jockey_recent_stats()`, `_get_jockey_grade_stats()` を追加
- 調教師版（提案A）と同構造のクエリを騎手用に変換
- `_FEATURES` リストに5つの特徴量名を追加

---

## 提案D: フォームサイクル・モメンタム（horse_last_jyuni深掘り）

**目的:** horse_last_jyuni（重要度8位: 17370）は前走の着順値だが、「改善中か悪化中か」のトレンド情報がない。モメンタム（勢い）を捉える特徴量を追加する。

**実装先:** `src/features/horse.py` の `HorseFeatureExtractor._calc_past_performance()`

**既存との差分:** 現在は着順の静的な統計量（平均、最高、勝率等）のみ。時系列的なトレンドが未実装。

| # | 特徴量名 | 型 | 集計期間 | 説明 |
|---|---------|---|---------|------|
| D1 | horse_jyuni_trend_slope | float | 直近5走 | 着順の回帰傾斜（負=改善傾向、正=悪化傾向）。直近走に近いほど重みが大きいインデックス(0,1,2,3,4)に対する線形回帰の傾き |
| D2 | horse_days_since_last_win | int | 全走 | 最後の1着からの日数。未勝利馬は-1 |
| D3 | horse_consecutive_top3 | int | 直近 | 現在の連続3着以内回数（ストリーク）。前走が4着以下なら0 |
| D4 | horse_last_vs_best | int | 直近5走 | 前走着順 - 直近5走最高着順（ピークからの乖離） |
| D5 | horse_improving_flag | int | 直近3走 | 直近3走が連続して着順改善なら1（例: 8着→5着→3着） |
| D6 | horse_class_first_flag | int | - | 今回のクラスでの初出走フラグ（昇級初戦=1）。`n_race.jyokencd5` で判定 |

**算出ロジック（D1: 着順トレンド傾斜）:**
```python
def calc_jyuni_trend(jyuni_series: pd.Series) -> float:
    """直近5走の着順傾斜を算出する.

    index 0 = 最新走, 1 = 2走前, ...
    回帰の傾きが負なら改善傾向（直近の方が着順が良い）。

    Returns:
        傾斜値。データ不足の場合は0.0。
    """
    vals = jyuni_series.head(5).values
    valid = vals[vals < 99]
    if len(valid) < 2:
        return 0.0
    x = np.arange(len(valid))
    slope = np.polyfit(x, valid, 1)[0]
    return float(slope)
```

**算出ロジック（D3: 連続3着以内ストリーク）:**
```python
def calc_consecutive_top3(jyuni_series: pd.Series) -> int:
    """直近からの連続3着以内回数を返す."""
    count = 0
    for j in jyuni_series:
        j_int = int(j) if j and j != 99 else 99
        if j_int <= 3:
            count += 1
        else:
            break
    return count
```

**実装方針:**
- `_calc_past_performance()` 内に追加ロジックを埋め込む（既存の `h_past` DataFrameを使うため新規SQLは不要）
- D6のみ `race_info` から `jyokencd5` を取得し、過去走の `jyokencd5` と比較する必要がある → `_get_past_results()` の SELECT に `r.jyokencd5` を追加
- `_FEATURES` リストに6つの特徴量名を追加
- 欠損値: D1=0.0, D2=MISSING_NUMERIC, D3=0, D4=MISSING_NUMERIC, D5=0, D6=0

---

## 提案E: レース内ペース構造（脚質インタラクション）

**目的:** 現在の脚質特徴量は個馬単位だが、「このレースに逃げ馬が何頭いるか」「予想ペースはどうか」というレース構造が未使用。展開予測は競馬予想の核であり、これを数値化する。

**実装先:** `src/features/pipeline.py` の `FeaturePipeline._add_cross_features()` またはクロス特徴量として追加

**既存との差分:** style_type_mode_last5（脚質モード）、style_front_ratio_last5（先行率）は個馬単位で存在するが、レース全体の集計がない。

| # | 特徴量名 | 型 | 説明 |
|---|---------|---|------|
| E1 | race_n_front_runners | int | レース内の逃げ・先行馬の頭数（style_type_mode_last5 が "1" または "2" の馬数） |
| E2 | race_pace_expected | float | 予想ペース指標（全出走馬の style_front_ratio_last5 の平均値。高いほどハイペース予想） |
| E3 | horse_style_vs_pace | float | 自馬の脚質と予想ペースの相性指標。差し・追込馬(style_type_mode_last5="3","4")はrace_pace_expected が高いほど有利（プラス）、逃げ・先行馬は低いほど有利 |
| E4 | race_avg_speed_index | float | レース出走馬全体のspeed_index_avg_last3の平均（レースレベル指標）|
| E5 | race_max_speed_index | float | レース出走馬中のspeed_index_avg_last3の最大値（最強馬のレベル）|

**算出ロジック:**
```python
def add_pace_features(df: pd.DataFrame) -> pd.DataFrame:
    """1レース分のDataFrameにペース構造特徴量を追加する."""
    result = df.copy()

    # E1: 逃げ・先行馬の頭数
    if "style_type_mode_last5" in result.columns:
        front = result["style_type_mode_last5"].isin(["1", "2", 1, 2])
        result["race_n_front_runners"] = int(front.sum())
    else:
        result["race_n_front_runners"] = 0

    # E2: 予想ペース
    if "style_front_ratio_last5" in result.columns:
        avg_front = pd.to_numeric(
            result["style_front_ratio_last5"], errors="coerce"
        ).mean()
        result["race_pace_expected"] = avg_front if pd.notna(avg_front) else 0.0
    else:
        result["race_pace_expected"] = 0.0

    # E3: 脚質×ペース相性
    pace_val = float(result["race_pace_expected"].iloc[0]) if "race_pace_expected" in result.columns else 0.0

    def style_pace_affinity(row):
        style = str(row.get("style_type_mode_last5", "")).strip()
        if style in ("3", "4"):   # 差し・追込 → ハイペースほど有利
            return pace_val
        elif style in ("1", "2"): # 逃げ・先行 → スローほど有利
            return -pace_val
        return 0.0

    if "style_type_mode_last5" in result.columns:
        result["horse_style_vs_pace"] = result.apply(style_pace_affinity, axis=1)
    else:
        result["horse_style_vs_pace"] = 0.0

    # E4, E5: レースレベル
    if "speed_index_avg_last3" in result.columns:
        si = pd.to_numeric(
            result["speed_index_avg_last3"].replace(MISSING_NUMERIC, np.nan),
            errors="coerce",
        )
        result["race_avg_speed_index"] = si.mean() if si.notna().any() else 0.0
        result["race_max_speed_index"] = si.max() if si.notna().any() else 0.0
    else:
        result["race_avg_speed_index"] = 0.0
        result["race_max_speed_index"] = 0.0

    return result
```

**実装方針:**
- `pipeline.py` の `extract_race()` で、`_add_relative_features()` の前に `_add_pace_features()` を呼ぶ
- 全て既存特徴量のレース内集計から算出するため、新規SQLは不要
- `_cross_feature_names()` に5つ追加

---

## 提案F: 相対特徴量の拡張（rel_*追加）

**目的:** rel_*（相対特徴量）が5位〜10位に集中しており、この仕組みの効果が実証済み。まだ相対化していない重要特徴量にも拡張する。

**実装先:** `src/features/pipeline.py` の `_RELATIVE_TARGETS`

**既存との差分:** 現在18個の対象。以下を追加する。

| # | 追加対象の元特徴量 | ascending | missing_type | 根拠 |
|---|-----------------|-----------|-------------|------|
| F1 | bw_weight | False | numeric | 馬体重（重要度21位: 2639）。大型馬/小型馬のレース内相対比較 |
| F2 | trainer_jockey_combo_rate | False | rate | コンビ成績（重要度24位: 2067）。レース内での相対的な信頼度 |
| F3 | cross_weight_futan_per_bw | False | numeric | 斤量/体重比（重要度25位: 1576）。レース内での相対負担 |
| F4 | interval_days | True | numeric | 休養日数。小さい方が実戦感覚が高い |
| F5 | horse_jyuni_trend_slope | True | numeric | 着順トレンド（提案D1）。負（改善傾向）の方が良い → ascending=True |
| F6 | trainer_win_rate_track_type | False | rate | 調教師芝ダ別勝率（提案A1）。条件別実力のレース内比較 |
| F7 | blood_bms_dist_rate | False | blood | BMS距離帯別成績（提案B1）。血統適性のレース内比較 |

**実装方針:**
- `_RELATIVE_TARGETS` に7つのタプルを追加するだけ
- 提案D,A,Bの特徴量が先に実装されている必要がある（F5, F6, F7）
- F1〜F4は既存特徴量なので即時追加可能

---

## 実装優先順位

コスト対効果と依存関係を考慮した推奨実装順序:

| 順位 | 提案 | 追加特徴量数 | 実装コスト | 期待効果 | 状態 | 理由 |
|-----|------|-----------|----------|---------|------|------|
| 1 | B: BMS条件別 | 6 | 中 | 高 | **✅ 実装済み** | 重要度2位のID依存を解消。サプリメント方式で実装 |
| 2 | A: 調教師条件別 | 7 | 中 | 高 | 未実装 | 重要度1位のID依存を解消。同様のクエリパターン |
| 3 | D: フォームモメンタム | 6 | 低 | 中〜高 | 未実装 | 既存DataFrameから算出、新規SQL最小 |
| 4 | E: ペース構造 | 5 | 低 | 中 | 未実装 | 既存特徴量の集計のみ、SQL不要 |
| 5 | F: 相対特徴量拡張 | 7 | 極低 | 中 | 未実装 | タプル追加のみ（A,B,D完了後にF5-F7も追加） |
| 6 | C: 騎手条件別 | 5 | 中 | 中 | 未実装 | A,Bと同構造だが重要度が若干低い |

**残り追加: +37特徴量（A+C+D+E+F）→ 全v2完成時 約229特徴量**
（B実装済みの現行: 約192特徴量）

## サプリメント vs メインパイプライン

- **提案B:** ✅ **サプリメントとして実装済み**（`bms_detail.py` → `BMSDetailFeatureExtractor`）。当初はメインパイプラインへの組み込みを予定していたが、メイン parquet の再構築なしに独立して構築・実験できるサプリメント方式を採用した
- **提案A, C:** メインパイプラインに組み込む（既存のExtractorクラスに追加メソッドを実装）。理由: 既存の `extract()` 内でレース情報を取得済みで、それを条件別クエリに渡すため。または提案Bと同様にサプリメント方式も検討可
- **提案D:** メインパイプラインに組み込む（`horse.py` の `_calc_past_performance()` に追加ロジック）
- **提案E:** メインパイプラインに組み込む（`pipeline.py` のクロス特徴量に追加）
- **提案F:** メインパイプラインに組み込む（`pipeline.py` の `_RELATIVE_TARGETS` にタプル追加）

## parquet再構築時の注意

- **サプリメント（提案B等）:** メイン parquet の再構築は不要。`--build-supplement {name} --force-rebuild` でサプリメントのみ再構築し、`--supplement {name}` で学習/評価時にマージする
- **メインパイプライン組み込み（提案A,C,D,E,F）:** 実装後に `python run_train.py --build-features-only --force-rebuild --workers 4` で全年度を再構築する必要がある
- 段階的に実装する場合は、各提案の実装完了ごとに対象年度のみ再構築可能（例: `--train-start 2025 --train-end 2025 --force-rebuild` で検証年度のみ先行確認）
