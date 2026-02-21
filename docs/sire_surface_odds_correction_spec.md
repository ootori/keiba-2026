# 種牡馬系統×サーフェス/距離帯 オッズ歪み補正 開発仕様書

## 1. 背景・目的

### 調査結果サマリ

種牡馬別回収率分析（`analysis/sire_roi_report.md`、2019-2024年 213,399出走）により、
以下のオッズ歪みが確認された:

- **芝/ダート間:** 同一種牡馬で回収率差が最大190ポイント（ヴァンセンヌ: 芝65.6% vs ダート255.8%）。上位20種牡馬すべてで50ポイント以上の乖離
- **洋芝（札幌・函館）:** 回収率の標準偏差42.4%（芝34.4%, ダート36.0%）で最大。回収率>100%の種牡馬が25.9%と高い
- **長距離:** 標準偏差49.1%でスプリント（37.5%）の1.3倍。回収率>100%の割合も25.0%と最大
- **ローカル場:** 福島・新潟・小倉で300%超の回収率の種牡馬×競馬場の組み合わせが多い

市場（ファン）が種牡馬のサーフェス・距離適性をオッズに十分反映できていないことを示している。

### 目的

この歪みをオッズ補正の新テーブルとして既存のオッズ歪み補正システム（v2）に組み込み、value_bet戦略の精度を向上させる。

---

## 2. 既存のオッズ補正システム概要

### アーキテクチャ（3層構造）

```
統計生成層 (odds_correction_stats.py)
  DB → SQL → ROI計算 → factor算出 → JSON保存

設定ロード層 (odds_correction_stats.py)
  JSON → 辞書変換 → config辞書

補正適用層 (evaluator.py)
  _apply_odds_correction(odds, row, ninki_rank, config)
  → adjusted_odds = raw_odds × Π(factors)
```

### 現行の補正チェーン（v2）

```
adjusted_odds = raw_odds × Π(factors)

Step 1: ninki_table[ninki_rank]               ← v1
Step 2: jockey_popular_discount               ← v1
Step 3: form_popular_discount                 ← v1
Step 4: style_table[style_type_last]          ← v2
Step 5: post_course_table[post_group×course]  ← v2
Step 6: class_upgrade / class_downgrade       ← v2
Step 7: filly_to_mixed / mixed_to_filly       ← v2
```

### factor算出の共通方式

```
factor = category_roi / baseline_roi
```
- `baseline_roi`: 全馬に100円ずつ単勝を買った場合の回収率
- `category_roi`: 対象セグメントの単勝回収率
- factor < 1.0 → 市場が過大評価（オッズを割り引く）
- factor > 1.0 → 市場が過小評価（オッズを上乗せ）
- 最小サンプル数未満 → factor = 1.0（補正なし）

### JSON構造（data/odds_correction_stats.json）

```json
{
  "generated_at": "ISO形式",
  "period": {"start": "2022", "end": "2024"},
  "baseline_roi": 0.8234,
  "baseline_samples": 450000,
  "min_samples": 1000,
  "ninki_table": {"1": {"factor": 0.98, "samples": 30000, "roi": 0.806}, ...},
  "style_table": {"1": {"factor": 1.02, "samples": 45000, "roi": 0.840}, ...},
  "post_course_table": {"inner_turf_left": {"factor": 1.05, ...}, ...},
  "rules": {"jockey_popular_discount": {"factor": 0.90, ...}, ...}
}
```

---

## 3. 実装仕様

### 追加するテーブル

| テーブル名 | 軸 | 効果 |
|-----------|-----|------|
| `sire_surface_table` | 父系統 × サーフェス | 芝/ダート/洋芝の血統適性歪みを補正 |
| `sire_distance_table` | 父系統 × 距離帯 | 距離適性の血統歪みを補正 |

### 集約キーの設計

種牡馬**個体ID**（`blood_father_id`）ではなく**父系統名**（`blood_father_keito`）を使用する。

**理由:**
- 個体IDだとセグメントが細かくなりすぎ、サンプル不足でノイジーなfactorが多発する
- 系統レベル（92系統）で集約することで、新種牡馬や出走数の少ない種牡馬にも適用可能
- 血統の適性傾向は系統単位で共通する部分が大きい（例: サンデーサイレンス系は芝向き）
- 現行の特徴量 `blood_father_keito` をそのまま利用でき、追加のDB結合が不要

### DBテーブル情報

**n_keito テーブル:**
- カラム: `hansyokunum`（繁殖番号）, `keitoid`（階層構造ID）, `keitoname`（系統名）
- 登録数: 92系統
- `blood_father_keito` 特徴量には `keitoname`（系統名文字列）が格納されている
  - 例: "サンデーサイレンス", "キングカメハメハ", "ディープインパクト"
- 結合方法: `n_sanku.fnum = n_keito.hansyokunum` で父馬の繁殖番号から系統名を取得

**特徴量のカラム名:**
- `blood_father_keito` — 父系統名（文字列）、v1から全parquetに存在
- `race_track_cd` — トラックコード（芝/ダート判定に使用）
- `race_jyo_cd` — 競馬場コード（洋芝判定に使用）
- `race_distance` — 距離（整数）

---

## 4. sire_surface_table 仕様

### テーブル定義

| 項目 | 内容 |
|------|------|
| キー | `{keitoname}_{surface}` （例: `"サンデーサイレンス_siba"`, `"キングカメハメハ_dirt"`） |
| surface分類 | `yousiba`（jyocd∈{01,02} かつ trackcd<23）, `siba`（その他のtrackcd<23）, `dirt`（trackcd>=23） |
| factor算出 | `factor = category_roi / baseline_roi` |
| 最小サンプル閾値 | 500件 |
| サンプル不足時 | factor = 1.0（補正なし） |

### SQL設計

```sql
WITH harai_tansho AS (
    -- 単勝払戻サブクエリ（既存の _calc_ninki_table() と同じパターン）
    SELECT year, monthday, jyocd, kaiji, nichiji, racenum,
           paytansyoumaban1, paytansyopay1,
           paytansyoumaban2, paytansyopay2,
           paytansyoumaban3, paytansyopay3
    FROM n_harai
    WHERE datakubun IN ('1','2')
      AND year BETWEEN %(y1)s AND %(y2)s
      AND jyocd IN ('01','02','03','04','05','06','07','08','09','10')
)
SELECT
    kt.keitoname AS father_keito,
    CASE
        WHEN CAST(r.trackcd AS integer) >= 23 THEN 'dirt'
        WHEN r.jyocd IN ('01','02') THEN 'yousiba'
        ELSE 'siba'
    END AS surface,
    COUNT(*) AS samples,
    -- 1着馬の払戻合計（投資100円あたり）
    SUM(CASE WHEN CAST(ur.kakuteijyuni AS integer) = 1
         THEN (払戻額取得ロジック) ELSE 0 END) AS total_pay
FROM n_uma_race ur
JOIN n_race r
  ON ur.year = r.year AND ur.monthday = r.monthday
 AND ur.jyocd = r.jyocd AND ur.kaiji = r.kaiji
 AND ur.nichiji = r.nichiji AND ur.racenum = r.racenum
JOIN n_sanku sk ON ur.kettonum = sk.kettonum
JOIN n_keito kt ON sk.fnum = kt.hansyokunum
WHERE ur.datakubun = '7'
  AND ur.ijyocd = '0'
  AND ur.kakuteijyuni ~ '^[0-9]+$'
  AND ur.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
  AND ur.year BETWEEN %(y1)s AND %(y2)s
  AND r.trackcd ~ '^[0-9]+$'
GROUP BY kt.keitoname, surface
```

※ 払戻額の取得は `n_harai` の `paytansyoumaban1`〜`3` と `paytansyopay1`〜`3` を使用。馬番はゼロパディング2桁（"01", "02"等）。既存の `_calc_ninki_table()` 内の払戻取得パターンを踏襲すること。

---

## 5. sire_distance_table 仕様

### テーブル定義

| 項目 | 内容 |
|------|------|
| キー | `{keitoname}_{dist_cat}` （例: `"サンデーサイレンス_sprint"`, `"キングカメハメハ_long"`） |
| 距離帯分類 | `sprint`（~1400m）, `mile`（1401-1800m）, `middle`（1801-2200m）, `long`（2201m~） |
| factor算出 | `factor = category_roi / baseline_roi` |
| 最小サンプル閾値 | 500件 |
| サンプル不足時 | factor = 1.0（補正なし） |

### SQL設計

sire_surface_table と同構造で、GROUP BY を以下に変更:

```sql
    CASE
        WHEN CAST(r.kyori AS integer) <= 1400 THEN 'sprint'
        WHEN CAST(r.kyori AS integer) <= 1800 THEN 'mile'
        WHEN CAST(r.kyori AS integer) <= 2200 THEN 'middle'
        ELSE 'long'
    END AS dist_cat
```

---

## 6. 実装変更箇所

### 6.1 `src/odds_correction_stats.py`

#### 新規関数（2つ）

```python
def _calc_sire_surface_stats(
    year_start: str, year_end: str, baseline_roi: float, min_samples: int = 500
) -> dict[str, dict]:
    """父系統 × サーフェス別の単勝回収率factorテーブルを算出する."""

def _calc_sire_distance_stats(
    year_start: str, year_end: str, baseline_roi: float, min_samples: int = 500
) -> dict[str, dict]:
    """父系統 × 距離帯別の単勝回収率factorテーブルを算出する."""
```

各関数の返り値形式:
```python
{
    "サンデーサイレンス_siba": {"factor": 1.05, "samples": 12000, "roi": 0.865},
    "サンデーサイレンス_dirt": {"factor": 0.88, "samples": 8000, "roi": 0.724},
    ...
}
```

#### `build_odds_correction_stats()` への追加

既存の `style_table`, `post_course_table` 算出の後に呼び出しを追加:

```python
stats["sire_surface_table"] = _calc_sire_surface_stats(
    year_start, year_end, baseline_roi, min_samples=500
)
stats["sire_distance_table"] = _calc_sire_distance_stats(
    year_start, year_end, baseline_roi, min_samples=500
)
```

### 6.2 `src/model/evaluator.py`

#### `_apply_odds_correction()` への追加

既存のStep 7（牝馬限定遷移ルール）の後に追加:

```python
# Step 8: 父系統×サーフェス別テーブル
sire_surface_table = config.get("sire_surface_table", {})
if sire_surface_table:
    father_keito = str(row.get("blood_father_keito", "") or "")
    track_cd_val = str(row.get("race_track_cd", "") or "")
    jyo_cd = str(row.get("race_jyo_cd", "") or "")
    try:
        track_cd_int = int(track_cd_val)
    except (ValueError, TypeError):
        track_cd_int = 0
    if track_cd_int >= 23:
        surface = "dirt"
    elif jyo_cd in ("01", "02"):
        surface = "yousiba"
    else:
        surface = "siba"
    key = f"{father_keito}_{surface}"
    if key in sire_surface_table:
        f = sire_surface_table[key].get("factor", 1.0)
        factor *= f

# Step 9: 父系統×距離帯別テーブル
sire_distance_table = config.get("sire_distance_table", {})
if sire_distance_table:
    father_keito = str(row.get("blood_father_keito", "") or "")
    kyori = int(row.get("race_distance", 0) or 0)
    if kyori <= 1400:
        dist_cat = "sprint"
    elif kyori <= 1800:
        dist_cat = "mile"
    elif kyori <= 2200:
        dist_cat = "middle"
    else:
        dist_cat = "long"
    key = f"{father_keito}_{dist_cat}"
    if key in sire_distance_table:
        f = sire_distance_table[key].get("factor", 1.0)
        factor *= f
```

### 6.3 `src/config.py`

変更不要。テーブルベースの補正はJSONにキーがなければ空辞書 `{}` となり、factor=1.0にフォールバック。既存の `ninki_table` / `style_table` / `post_course_table` と同じ方式。

### 6.4 `data/odds_correction_stats.json`（出力）

`--build-odds-stats` 実行時に以下のキーが追加される:

```json
{
  "...既存キー...",
  "sire_surface_table": {
    "サンデーサイレンス_siba": {"factor": 1.05, "samples": 12000, "roi": 0.865},
    "サンデーサイレンス_dirt": {"factor": 0.88, "samples": 8000, "roi": 0.724},
    "サンデーサイレンス_yousiba": {"factor": 1.12, "samples": 2000, "roi": 0.922},
    "キングカメハメハ_siba": {"factor": 0.92, "samples": 5000, "roi": 0.757},
    ...
  },
  "sire_distance_table": {
    "サンデーサイレンス_sprint": {"factor": 0.95, "samples": 10000, "roi": 0.782},
    "サンデーサイレンス_mile": {"factor": 1.03, "samples": 11000, "roi": 0.848},
    "ディープインパクト_middle": {"factor": 1.08, "samples": 7000, "roi": 0.889},
    "ディープインパクト_long": {"factor": 1.15, "samples": 3000, "roi": 0.947},
    ...
  }
}
```

---

## 7. 補正チェーン全体像（v3）

```
adjusted_odds = raw_odds × Π(factors)

Step 1: ninki_table[ninki_rank]                       ← v1
Step 2: jockey_popular_discount                       ← v1
Step 3: form_popular_discount                         ← v1
Step 4: style_table[style_type_last]                  ← v2
Step 5: post_course_table[post_group × course_cat]    ← v2
Step 6: class_upgrade / class_downgrade               ← v2
Step 7: filly_to_mixed / mixed_to_filly               ← v2
Step 8: sire_surface_table[father_keito × surface]    ← v3 NEW
Step 9: sire_distance_table[father_keito × dist_cat]  ← v3 NEW
```

---

## 8. テスト仕様

### `tests/test_odds_correction.py` に追加

```python
# === sire_surface_table ===
def test_sire_surface_factor_applied():
    """父系統×サーフェスのfactorが正しく乗算されること."""

def test_sire_surface_missing_key_no_effect():
    """テーブルにキーがない場合 factor=1.0 であること."""

def test_sire_surface_empty_table_no_effect():
    """sire_surface_table が空辞書の場合、補正なしであること."""

def test_sire_surface_yousiba_classification():
    """札幌(01)・函館(02)の芝コースが yousiba に分類されること."""

def test_sire_surface_dirt_classification():
    """trackcd >= 23 が dirt に分類されること."""

# === sire_distance_table ===
def test_sire_distance_factor_applied():
    """父系統×距離帯のfactorが正しく乗算されること."""

def test_sire_distance_boundary_1400():
    """距離1400mが sprint に分類されること."""

def test_sire_distance_boundary_1401():
    """距離1401mが mile に分類されること."""

def test_sire_distance_boundary_2200():
    """距離2200mが middle に分類されること."""

def test_sire_distance_boundary_2201():
    """距離2201mが long に分類されること."""

def test_sire_distance_missing_key_no_effect():
    """テーブルにキーがない場合 factor=1.0 であること."""

# === 統計生成テスト ===
def test_build_sire_surface_stats():
    """sire_surface_table が正しく生成されること."""

def test_build_sire_distance_stats():
    """sire_distance_table が正しく生成されること."""

def test_sire_stats_min_samples_filter():
    """最小サンプル数（500）未満のセグメントが除外される（factor=1.0）こと."""
```

---

## 9. 後方互換性

| 条件 | 動作 |
|------|------|
| 旧JSONファイル（v1/v2のみ） | `sire_surface_table` / `sire_distance_table` が存在しない → 空辞書として扱われ factor=1.0。デグレードなし |
| 旧parquet | `blood_father_keito` はv1から全parquetに存在。`race_track_cd`, `race_jyo_cd`, `race_distance` も同様。追加のparquet再構築は不要 |
| JSONキー不一致 | 系統名やサーフェスの組み合わせが辞書に存在しなければ factor=1.0。未知の系統でも安全 |

---

## 10. CLIオプション

追加のCLIオプションは不要。既存のコマンドで自動対応。

```bash
# 統計データ再構築（新テーブル含む）
python run_train.py --build-odds-stats

# 新テーブルを含む補正で評価
python run_train.py --eval-only --odds-correction

# 本番予測でオッズ補正付き
python run_predict.py --year 2025 --monthday 0622 --all-day --odds-correction
```

---

## 11. 検証計画

1. `--build-odds-stats` で統計JSONを再生成し、新テーブルの内容を確認
2. `--eval-only --odds-correction` でv2（テーブルなし）とv3（テーブルあり）の回収率を比較
3. 各サーフェス・距離帯でvalue_bet戦略の的中率・回収率の変化を確認
4. factorの分布を確認し、極端な値（0.5未満 or 2.0超）がないかチェック
5. 必要に応じて min_samples を調整（500 → 1000 等）

---

## 12. 注意事項

- **factor乗算の累積:** sire_surface と sire_distance の両方が同時に適用されるため、同じ系統の情報が二重に効く可能性がある。ただしサーフェスと距離は独立した軸でありクロス効果は各テーブル単独では捉えきれないため、乗算で問題ない

- **系統名の粒度:** n_keito は92系統登録。大系統（サンデーサイレンス系全体）はサンプル豊富だが歪みが平均化される可能性がある。一方、末端系統（ダンスインザダーク等）は分類が細かくサンプル不足になりうる。min_samples=500 でフィルタリングすることで対処

- **洋芝のサンプル数:** 洋芝は札幌・函館のみで出走数が限られる。系統×洋芝の組み合わせでmin_samples=500を満たせないケースが多いと予想される。その場合は factor=1.0（補正なし）となるが、設計上問題ない

- **keitoname の文字列一致:** `blood_father_keito`（特徴量）と統計テーブルのキーは `keitoname` の文字列で結合する。全角半角・空白の正規化に注意すること（既存の `bloodline.py` では `.strip()` 済み）
