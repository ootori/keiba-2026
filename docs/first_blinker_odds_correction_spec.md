# 初ブリンカー補正 開発仕様書

## 背景

初ブリンカー馬の回収率分析（`analysis/first_blinker_report.md`）により、以下が判明した:

- **初ブリンカー馬は過大評価されている:** 単勝回収率 67.9%（ブリンカーなし 72.2% に対し -4.3pt）
- **2回目以降のブリンカー馬は過小評価されている:** 単勝回収率 78.8%（+6.6pt）

この歪みをオッズ補正に組み込む。

## 実装概要

既存のオッズ歪み補正フレームワーク（v1/v2）に「初ブリンカー補正」をv3として追加する。
変更は以下の5箇所。

| # | ファイル | 変更内容 |
|---|---------|---------|
| 1 | `src/features/horse.py` | `horse_blinker_first` 特徴量を追加 |
| 2 | `src/odds_correction_stats.py` | `_calc_first_blinker_stats()` を追加、`build_odds_correction_stats()` に統合 |
| 3 | `src/model/evaluator.py` | `_apply_odds_correction()` に初ブリンカールールを追加 |
| 4 | `src/config.py` | `DEFAULT_ODDS_CORRECTION_CONFIG` にデフォルト値を追加 |
| 5 | `tests/test_odds_correction.py` | テストケースを追加 |

## 1. 特徴量の追加 (`src/features/horse.py`)

### 新特徴量

| 特徴量名 | 型 | 値 | 説明 |
|---------|---|---|------|
| `horse_blinker_first` | int | 0 or 1 | 当該レースが生涯初のブリンカー装着なら1 |

### 判定ロジック

```
horse_blinker_first = 1  ←  blinker == '1' （当該レースでブリンカー使用）
                           AND 過去走の blinker が全て '0'（過去にブリンカー使用歴なし）
```

### 実装方針

`_get_past_results()` で取得する過去走データに `blinker` カラムを追加し、
`_build_horse_features()` 内で過去走にブリンカー使用があるかを判定する。

```python
# 過去走のブリンカー使用履歴を確認
past_blinkers = [str(r.get("blinker", "0")).strip() for r in past_results]
has_past_blinker = any(b == "1" for b in past_blinkers)

# 初ブリンカー判定
current_blinker = self._safe_int(horse.get("blinker"), default=0)
feat["horse_blinker_first"] = 1 if current_blinker == 1 and not has_past_blinker else 0
```

### docs/feature_design.md への追記

カテゴリ1（馬基本属性）に `horse_blinker_first` を追加（合計特徴量数 +1）。

## 2. 統計算出 (`src/odds_correction_stats.py`)

### 新関数: `_calc_first_blinker_stats()`

既存ルールと同じ `factor = category_roi / baseline_roi` パターンで算出する。

#### SQL概要

```sql
WITH ordered AS (
    SELECT
        ur.year, ur.monthday, ur.jyocd, ur.kaiji, ur.nichiji, ur.racenum,
        ur.umaban, ur.blinker, ur.kakuteijyuni,
        -- 過去走でブリンカーを使ったことがあるか
        COALESCE(
            MAX(CASE WHEN ur2.blinker = '1' THEN 1 ELSE 0 END),
            0
        ) AS has_past_blinker
    FROM n_uma_race ur
    LEFT JOIN n_uma_race ur2
        ON ur2.kettonum = ur.kettonum
        AND (ur2.year < ur.year OR (ur2.year = ur.year AND ur2.monthday < ur.monthday))
        AND ur2.datakubun = '7'
        AND ur2.ijyocd = '0'
    WHERE ur.datakubun = '7'
      AND ur.ijyocd = '0'
      AND ur.jyocd IN ('01',...,'10')
      AND CAST(ur.year AS integer) BETWEEN %(start)s AND %(end)s
    GROUP BY ur.year, ur.monthday, ur.jyocd, ur.kaiji, ur.nichiji,
             ur.racenum, ur.umaban, ur.blinker, ur.kakuteijyuni, ur.kettonum
)
```

初ブリンカー条件: `blinker = '1' AND has_past_blinker = 0`

この集合に対して単勝オッズ（`n_odds_tanpuku`）と払戻（`n_harai`）を結合し、ROI を算出する。

#### 出力

```json
"first_blinker_discount": {
    "factor": <float>,    // 例: 0.94 (= 67.9 / 72.2 ≒ baseline_roi比)
    "samples": <int>,     // 例: 11394
    "roi": <float>        // 例: 0.679
}
```

### `build_odds_correction_stats()` への統合

```python
stats["rules"]["first_blinker_discount"] = _calc_first_blinker_stats(
    year_start, year_end, tansho_umaban_col, tansho_pay_col,
    baseline_roi, min_samples
)
```

## 3. 補正適用 (`src/model/evaluator.py`)

### `_apply_odds_correction()` への追加

既存ルールの末尾（filly_transition の後）に追加:

```python
# --- 初ブリンカー補正 (v3) ---
r = rules.get("first_blinker_discount", {})
if r and r.get("factor", 1.0) != 1.0:
    blinker_first = int(row.get("horse_blinker_first", 0) or 0)
    if blinker_first == 1:
        factor *= r["factor"]
```

### 適用条件

| 条件 | 値 |
|------|---|
| `horse_blinker_first == 1` | 当該レースが初ブリンカー |

factor < 1.0 のため、初ブリンカー馬のオッズ（=期待値）を割り引く方向に補正される。

## 4. デフォルト設定 (`src/config.py`)

`DEFAULT_ODDS_CORRECTION_CONFIG["rules"]` に追加:

```python
"first_blinker_discount": {"factor": 0.94},
```

factor の根拠: 分析結果の単勝回収率比 67.9 / 72.2 ≒ 0.94

## 5. テスト (`tests/test_odds_correction.py`)

### 追加テストケース

1. **統計算出テスト:** `_calc_first_blinker_stats()` が正しい factor を返すことを確認
2. **補正適用テスト:**
   - `horse_blinker_first=1` の場合に factor が適用されること
   - `horse_blinker_first=0` の場合に factor が適用されないこと
   - `first_blinker_discount` ルールが存在しない場合に無影響であること
3. **他ルールとの組み合わせテスト:** ninki_table + first_blinker_discount の乗算が正しいこと

## 6. parquet 再構築

`horse_blinker_first` は特徴量構築時に生成されるため、開発完了後に `--force-rebuild` で全年度の parquet を再構築する必要がある。

```bash
python run_train.py --build-features-only --workers 8 --force-rebuild
```

## 7. CLAUDE.md への追記

### 個別ルール一覧テーブルに追加

| ルール名 | 条件 | 説明 | バージョン |
|---------|------|------|----------|
| `first_blinker_discount` | horse_blinker_first == 1 | 初ブリンカー馬は過大評価 | v3 |

### 特徴量カテゴリ（カテゴリ1）に追記

馬基本属性の特徴量数を (5) → (6) に更新し、`horse_blinker_first` を追加。

### 注意事項に追記

- **初ブリンカー特徴量のparquet依存:** `horse_blinker_first` は `horse.py` の `_build_horse_features()` で構築時に生成される。旧parquetには含まれないため、この特徴量を使うには `--force-rebuild` で再構築が必要

## 実装優先順位

1. `horse.py` — 特徴量追加（他の変更の前提）
2. `odds_correction_stats.py` — 統計算出
3. `evaluator.py` — 補正適用
4. `config.py` — デフォルト値
5. `tests/` — テスト
6. parquet 再構築
7. `CLAUDE.md` / `docs/feature_design.md` 更新
