# 芝ダート替わり補正 開発仕様書

## 背景

初ダート・初芝の回収率分析（`analysis/first_dirt_report.md`, `analysis/first_turf_report.md`）により、以下が判明した:

### 初ダート

| 指標 | 初ダート | 芝のみ | 差分 |
|------|---------|--------|------|
| 勝率 | 5.90% | 7.34% | -1.44pt |
| 複勝率 | 16.76% | 22.00% | -5.24pt |
| 単勝回収率 | 74.2% | 71.9% | **+2.3pt** |
| 複勝回収率 | 68.0% | 72.6% | -4.6pt |

- 成績は明確に劣るが、人気薄になりやすく単勝回収率はわずかにプラス
- 複勝回収率は11年中10年でマイナスと安定的に低い

### 初芝

| 指標 | 初芝 | ダートのみ | 差分 |
|------|------|----------|------|
| 勝率 | 6.29% | 7.01% | -0.72pt |
| 複勝率 | 18.86% | 21.01% | -2.15pt |
| 単勝回収率 | 69.4% | 73.0% | **-3.6pt** |
| 複勝回収率 | 67.4% | 73.5% | -6.1pt |

- 全指標でマイナス。初ダートと異なり単勝でも穴妙味なし
- 複勝回収率は11年中10年でマイナス

### 2回目以降の替わり

| 指標 | 2回目以降ダート vs 芝のみ | 2回目以降芝 vs ダートのみ |
|------|------------------------|------------------------|
| 単勝回収率差 | +0.9pt | -0.7pt |
| 複勝回収率差 | +2.0pt | +0.1pt |

- ブリンカー2回目以降（単勝+6.6pt）のような明確な歪みはなし
- 2回目以降は補正対象としない

### 補正方針

| カテゴリ | 方向 | 根拠 |
|---------|------|------|
| 初ダート | 微弱な過小評価（単勝+2.3pt） | 人気薄になりやすく、単勝でわずかに有利だが不安定 |
| 初芝 | 過大評価（単勝-3.6pt, 複勝-6.1pt） | 全指標でマイナス、安定した低回収率 |

初ダートの単勝+2.3ptは年度間で不安定（-24.1pt〜+16.8pt）であり、factor=1.03程度の微弱な上乗せにとどめる。初芝は安定的にマイナスのためfactor < 1.0で割引する。

## 実装概要

既存のオッズ歪み補正フレームワーク（v1/v2）に「芝ダート替わり補正」をv3ルールとして追加する。
変更は以下の5箇所。

| # | ファイル | 変更内容 |
|---|---------|---------|
| 1 | `src/features/horse.py` | `horse_first_dirt`, `horse_first_turf` 特徴量を追加 |
| 2 | `src/odds_correction_stats.py` | `_calc_first_surface_stats()` を追加、`build_odds_correction_stats()` に統合 |
| 3 | `src/model/evaluator.py` | `_apply_odds_correction()` に初ダート/初芝ルールを追加 |
| 4 | `src/config.py` | `DEFAULT_ODDS_CORRECTION_CONFIG` にデフォルト値を追加 |
| 5 | `tests/test_odds_correction.py` | テストケースを追加 |

## 1. 特徴量の追加 (`src/features/horse.py`)

### 新特徴量

| 特徴量名 | 型 | 値 | 説明 |
|---------|---|---|------|
| `horse_first_dirt` | int | 0 or 1 | 当該レースが生涯初のダート出走なら1 |
| `horse_first_turf` | int | 0 or 1 | 当該レースが生涯初の芝出走なら1 |

### 判定ロジック

```
horse_first_dirt = 1  ←  今走の track_type == "dirt"
                         AND 過去走に track_type == "dirt" が一度もない

horse_first_turf = 1  ←  今走の track_type == "turf"
                         AND 過去走に track_type == "turf" が一度もない
```

### 実装方針

`_get_past_results()` で取得する過去走データには既に `trackcd` 情報が含まれている（条件別成績のカテゴリ3で芝/ダート別成績を算出するため）。この既存データを活用する。

```python
from src.utils.code_master import track_type

# 過去走のトラック種別を取得
past_track_types = [track_type(str(r.get("trackcd", ""))) for r in past_results]
has_past_dirt = "dirt" in past_track_types
has_past_turf = "turf" in past_track_types

# 今走のトラック種別（race_track_type は pipeline で既に設定済み）
current_tt = track_type(str(current_trackcd))

feat["horse_first_dirt"] = 1 if current_tt == "dirt" and not has_past_dirt else 0
feat["horse_first_turf"] = 1 if current_tt == "turf" and not has_past_turf else 0
```

**注意:** `_get_past_results()` のSQLで `trackcd` を取得していない場合は、JOINまたはサブクエリで `n_race.trackcd` を追加する必要がある。既存の `horse.py` を確認し、`trackcd` が取得済みか確認すること。

### docs/feature_design.md への追記

カテゴリ1（馬基本属性）に `horse_first_dirt`, `horse_first_turf` を追加（合計特徴量数 +2）。

## 2. 統計算出 (`src/odds_correction_stats.py`)

### 新関数: `_calc_first_surface_stats()`

初ダート・初芝それぞれについて `factor = category_roi / baseline_roi` を算出する。

#### SQL概要

```sql
WITH horse_races AS (
    SELECT
        ur.year, ur.monthday, ur.jyocd, ur.kaiji, ur.nichiji, ur.racenum,
        ur.kettonum, ur.umaban, ur.kakuteijyuni,
        CASE
            WHEN CAST(r.trackcd AS int) BETWEEN 10 AND 22 THEN 'turf'
            WHEN CAST(r.trackcd AS int) BETWEEN 23 AND 29 THEN 'dirt'
            ELSE 'other'
        END AS track_type
    FROM n_uma_race ur
    JOIN n_race r ON r.year = ur.year AND r.monthday = ur.monthday
        AND r.jyocd = ur.jyocd AND r.kaiji = ur.kaiji
        AND r.nichiji = ur.nichiji AND r.racenum = ur.racenum
        AND r.datakubun = '7'
    WHERE ur.datakubun = '7'
      AND ur.ijyocd = '0'
      AND ur.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
      AND CAST(ur.year AS integer) >= 2001
),
with_history AS (
    SELECT
        hr.*,
        -- 過去走で当該トラック種別を走った回数
        (SELECT COUNT(*)
         FROM horse_races hr2
         WHERE hr2.kettonum = hr.kettonum
           AND (hr2.year < hr.year OR (hr2.year = hr.year AND hr2.monthday < hr.monthday))
           AND hr2.track_type = hr.track_type
        ) AS prev_same_surface_count
    FROM horse_races hr
    WHERE CAST(hr.year AS integer) BETWEEN %(start)s AND %(end)s
)
```

初ダート条件: `track_type = 'dirt' AND prev_same_surface_count = 0`
初芝条件: `track_type = 'turf' AND prev_same_surface_count = 0`

この集合に対して単勝オッズ（`n_odds_tanpuku`）と払戻（`n_harai`）を結合し、ROI を算出する。

**パフォーマンス注意:** 相関サブクエリは重い可能性がある。実装時に `ROW_NUMBER()` OVER (PARTITION BY kettonum, track_type ORDER BY year, monthday) を使い、rn=1のレコードを初挑戦として検出する方式も検討すること。

#### 出力

```json
"first_dirt_boost": {
    "factor": <float>,    // 例: 1.03 (= 74.2 / 71.9 ≒ baseline_roi比)
    "samples": <int>,     // 例: 41473
    "roi": <float>        // 例: 0.742
},
"first_turf_discount": {
    "factor": <float>,    // 例: 0.95 (= 69.4 / 73.0 ≒ baseline_roi比)
    "samples": <int>,     // 例: 38564
    "roi": <float>        // 例: 0.694
}
```

### `build_odds_correction_stats()` への統合

```python
first_surface = _calc_first_surface_stats(
    year_start, year_end, tansho_umaban_col, tansho_pay_col,
    baseline_roi, min_samples
)
stats["rules"]["first_dirt_boost"] = first_surface["first_dirt"]
stats["rules"]["first_turf_discount"] = first_surface["first_turf"]
```

## 3. 補正適用 (`src/model/evaluator.py`)

### `_apply_odds_correction()` への追加

既存ルールの末尾に追加:

```python
# --- 初ダート/初芝補正 (v3) ---
# 初ダート
r = rules.get("first_dirt_boost", {})
if r and r.get("factor", 1.0) != 1.0:
    first_dirt = int(row.get("horse_first_dirt", 0) or 0)
    if first_dirt == 1:
        factor *= r["factor"]

# 初芝
r = rules.get("first_turf_discount", {})
if r and r.get("factor", 1.0) != 1.0:
    first_turf = int(row.get("horse_first_turf", 0) or 0)
    if first_turf == 1:
        factor *= r["factor"]
```

### 適用条件

| ルール名 | 条件 | factorの方向 | 説明 |
|---------|------|-------------|------|
| `first_dirt_boost` | `horse_first_dirt == 1` | factor > 1.0 | 初ダート馬はやや過小評価（単勝+2.3pt） |
| `first_turf_discount` | `horse_first_turf == 1` | factor < 1.0 | 初芝馬は過大評価（単勝-3.6pt） |

## 4. デフォルト設定 (`src/config.py`)

`DEFAULT_ODDS_CORRECTION_CONFIG["rules"]` に追加:

```python
"first_dirt_boost": {"factor": 1.03},
"first_turf_discount": {"factor": 0.95},
```

factor の根拠:
- `first_dirt_boost`: 74.2 / 71.9 ≒ 1.032 → 1.03（控えめに設定。年度間変動が大きいため）
- `first_turf_discount`: 69.4 / 73.0 ≒ 0.951 → 0.95（安定的にマイナスのため適用）

## 5. テスト (`tests/test_odds_correction.py`)

### 追加テストケース

1. **統計算出テスト:** `_calc_first_surface_stats()` が正しい factor を返すことを確認（初ダート・初芝それぞれ）
2. **補正適用テスト:**
   - `horse_first_dirt=1` の場合に `first_dirt_boost` の factor が適用されること
   - `horse_first_turf=1` の場合に `first_turf_discount` の factor が適用されること
   - `horse_first_dirt=0` / `horse_first_turf=0` の場合に factor が適用されないこと
   - ルールが存在しない場合に無影響であること
3. **排他性テスト:** 同一レースで `horse_first_dirt=1` かつ `horse_first_turf=1` は論理的にありえないことの確認（あるレースは芝かダートのどちらか一方）
4. **他ルールとの組み合わせテスト:** ninki_table + first_dirt_boost の乗算が正しいこと

## 6. parquet 再構築

`horse_first_dirt`, `horse_first_turf` は特徴量構築時に生成されるため、開発完了後に `--force-rebuild` で全年度の parquet を再構築する必要がある。

```bash
python run_train.py --build-features-only --workers 8 --force-rebuild
```

## 7. CLAUDE.md への追記

### 個別ルール一覧テーブルに追加

| ルール名 | 条件 | 説明 | バージョン |
|---------|------|------|----------|
| `first_dirt_boost` | horse_first_dirt == 1 | 初ダート馬はやや過小評価 | v3 |
| `first_turf_discount` | horse_first_turf == 1 | 初芝馬は過大評価 | v3 |

### 特徴量カテゴリ（カテゴリ1）に追記

馬基本属性の特徴量数を更新し、`horse_first_dirt`, `horse_first_turf` を追加。

### 注意事項に追記

- **初芝/初ダート特徴量のparquet依存:** `horse_first_dirt`, `horse_first_turf` は `horse.py` の `_build_horse_features()` で構築時に生成される。旧parquetには含まれないため、この特徴量を使うには `--force-rebuild` で再構築が必要

## 実装優先順位

1. `horse.py` — 特徴量追加（他の変更の前提）
2. `odds_correction_stats.py` — 統計算出
3. `evaluator.py` — 補正適用
4. `config.py` — デフォルト値
5. `tests/` — テスト
6. parquet 再構築
7. `CLAUDE.md` / `docs/feature_design.md` 更新

## 既存特徴量との関係

既に `cross_track_change`（カテゴリ16）が芝⇔ダート替わりを示すフラグ（0 or 1）として存在する。ただし `cross_track_change` は「替わり」を示すのみで「初めてか2回目以降か」を区別しない。本仕様の `horse_first_dirt` / `horse_first_turf` はより詳細な情報を持つ。

- `cross_track_change == 1` かつ `horse_first_dirt == 1` → 初ダート
- `cross_track_change == 1` かつ `horse_first_dirt == 0` → 2回目以降のダート替わり
- `cross_track_change == 0` → 前走と同じトラック種別（替わりなし）

オッズ補正ではこの区別が重要（初ダートと2回目以降で回収率傾向が異なるため）。
