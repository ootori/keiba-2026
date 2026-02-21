# 種牡馬×芝ダ×馬場状態 特徴量仕様書

**策定日:** 2026-02-21
**ステータス:** 未実装
**カテゴリ:** v2提案の拡張（既存カテゴリ13の強化）

---

## 背景

### 分析結果（2019-2024年、6年間）

種牡馬別に芝/ダート×馬場状態（良/稍重/重/不良）の単勝回収率を分析した結果、
**馬場状態による回収率の振れが種牡馬ごとに極めて大きい**ことが判明した。

#### 全体傾向

| コース | 馬場 | 出走数 | 回収率 |
|--------|------|--------|--------|
| 芝 | 良 | 69,800 | 74.5% |
| 芝 | 稍重 | 10,094 | 70.9% |
| 芝 | 重 | 3,068 | 69.6% |
| ダート | 良 | 60,526 | 72.9% |
| ダート | 稍重 | 15,331 | 64.7% |
| ダート | 重 | 7,280 | **85.0%** |
| ダート | 不良 | 2,397 | 72.0% |

#### 種牡馬別の馬場状態による回収率差（ダート重馬場が顕著）

| 種牡馬 | ダート良 | ダート重 | 差 |
|--------|---------|---------|-----|
| パイロ | 70.8% | **244.4%** | +173.6% |
| ジャスタウェイ | 70.2% | **211.6%** | +141.4% |
| オルフェーヴル | 73.9% | **199.7%** | +125.8% |
| ホッコータルマエ | 124.1% | 95.7% | -28.4% |

#### 芝/ダートによる適性差

| 種牡馬 | 芝 ROI | ダート ROI | 差 |
|--------|--------|-----------|-----|
| エピファネイア | 87.5% | 39.7% | +47.8% |
| カレンブラックヒル | 54.4% | 111.4% | -57.0% |

### 既存特徴量との関係

既存の `blood_father_baba_rate` は**父産駒の今回馬場状態での複勝率**を返すが、
芝/ダートを区別していない。重馬場でも芝重とダート重では全く異なる適性が求められるため、
芝ダ×馬場の交差項を明示的に特徴量化する価値がある。

| 既存特徴量 | 内容 | 課題 |
|-----------|------|------|
| `blood_father_turf_rate` | 父産駒の芝複勝率 | 馬場状態を区別しない |
| `blood_father_dirt_rate` | 父産駒のダート複勝率 | 馬場状態を区別しない |
| `blood_father_baba_rate` | 父産駒の今回馬場での複勝率 | 芝/ダートを区別しない |
| `blood_bms_baba_rate` | BMS産駒の今回馬場での複勝率（サプリメント） | 芝/ダートを区別しない |

**本提案では、芝ダ×馬場状態を交差させた複勝率特徴量を追加する。**

---

## 特徴量定義

### 新規特徴量（+4）

| # | 特徴量名 | 型 | 集計期間 | 説明 |
|---|---------|---|---------|------|
| 1 | `blood_father_track_baba_rate` | float | 過去3年 | 父産駒の「今回コース種別×今回馬場状態」での複勝率 |
| 2 | `blood_bms_track_baba_rate` | float | 過去3年 | 母父産駒の「今回コース種別×今回馬場状態」での複勝率 |
| 3 | `blood_father_heavy_rate` | float | 過去3年 | 父産駒の「今回コース種別×重不良馬場」での複勝率 |
| 4 | `blood_bms_heavy_rate` | float | 過去3年 | 母父産駒の「今回コース種別×重不良馬場」での複勝率 |

**設計方針:**
- 特徴量1,2は「芝×良」「芝×稍重」「芝×重」「ダート×良」「ダート×稍重」「ダート×重」「ダート×不良」の7パターンで条件マッチ
- 特徴量3,4は馬場状態を「重+不良」にまとめた集約版。重馬場のサンプル数不足を補う目的
- 芝の不良はほぼ開催されないため、芝×不良は重に含めて扱う

### 実装方式の選択

| 方式 | メリット | デメリット |
|------|---------|----------|
| **サプリメント方式（推奨）** | メインparquet再構築不要、独立実験可能 | 結合キーの管理が必要 |
| bloodline.py組み込み | 既存のSQLパターンを拡張しやすい | 全年度parquet再構築が必要 |

**推奨: サプリメント方式** (`sire_track_baba.py`)

BMS条件別（`bms_detail.py`）と同様にサプリメントとして実装する。
既存の `blood_father_baba_rate` / `blood_bms_baba_rate` と共存させ、
LightGBMに芝ダ×馬場の交差情報を直接渡す。

---

## 算出ロジック

### blood_father_track_baba_rate / blood_bms_track_baba_rate

```
対象: 同じ父（または母父）を持つ全産駒
条件: 今回と同じコース種別（芝 or ダート）AND 同じ馬場状態コード
期間: レース日より過去3年
算出: 複勝率 = 3着以内数 / 出走数
欠損: MIN_SAMPLES（20）未満 → NaN
```

### blood_father_heavy_rate / blood_bms_heavy_rate

```
対象: 同じ父（または母父）を持つ全産駒
条件: 今回と同じコース種別（芝 or ダート）AND 馬場状態が重(3)または不良(4)
      ※芝の場合は sibababacd IN ('3','4')、ダートは dirtbabacd IN ('3','4')
期間: レース日より過去3年
算出: 複勝率 = 3着以内数 / 出走数
欠損: MIN_SAMPLES（20）未満 → NaN
注意: 今回の馬場が良/稍重の場合でも値を返す（今後の道悪替わり耐性として使用）
```

---

## SQL例

### 父産駒の芝ダ×馬場状態別複勝率

```sql
-- 芝の場合 (trackcd BETWEEN 10 AND 22)
SELECT
    s.fnum AS sire_num,
    COUNT(*) AS total,
    SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) <= 3
        THEN 1 ELSE 0 END) AS top3
FROM n_sanku s
JOIN n_uma_race ur ON s.kettonum = ur.kettonum
JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
WHERE s.fnum IN %(sire_nums)s
  AND ur.datakubun = '7'
  AND ur.ijyocd = '0'
  AND r.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
  AND r.year >= %(year_start)s
  AND (r.year || r.monthday) < %(race_date)s
  AND CAST(r.trackcd AS int) BETWEEN 10 AND 22
  AND r.sibababacd = %(baba_cd)s
GROUP BY s.fnum
```

### 父産駒の芝ダ×重不良複勝率

```sql
-- ダートの場合 (trackcd BETWEEN 23 AND 29)
SELECT
    s.fnum AS sire_num,
    COUNT(*) AS total,
    SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) <= 3
        THEN 1 ELSE 0 END) AS top3
FROM n_sanku s
JOIN n_uma_race ur ON s.kettonum = ur.kettonum
JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
WHERE s.fnum IN %(sire_nums)s
  AND ur.datakubun = '7'
  AND ur.ijyocd = '0'
  AND r.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
  AND r.year >= %(year_start)s
  AND (r.year || r.monthday) < %(race_date)s
  AND CAST(r.trackcd AS int) BETWEEN 23 AND 29
  AND r.dirtbabacd IN ('3', '4')
GROUP BY s.fnum
```

### BMS側も同様のクエリ構造

```sql
-- s.fnum の代わりに n_sanku の mfnum (母父) を使用
WHERE s.mfnum IN %(bms_nums)s
```

---

## ノイズ抑制

`bms_detail.py` と同じ方針を踏襲する。

| パラメータ | 値 | 理由 |
|-----------|---|------|
| MIN_SAMPLES | 20 | 父/BMS単独のフィルタ |
| NaN | LightGBMネイティブ欠損 | 「データなし」と「複勝率0%」を区別 |

**懸念:** 芝×重/不良はサンプルが少ない種牡馬が多い（芝重は全体で3,068件）。
`blood_father_heavy_rate` で重+不良をまとめることで緩和する。

---

## 実装計画

### ファイル構成

| ファイル | 変更内容 |
|---------|---------|
| `src/features/sire_track_baba.py` | 新規作成。サプリメント実装 |
| `src/features/supplement.py` | `_get_registry()` に `sire_track_baba` を登録 |
| `src/config.py` | `CATEGORICAL_FEATURES` への追加なし（全てfloat型） |
| `docs/feature_design.md` | カテゴリ20として追加 |
| `tests/test_sire_track_baba_supplement.py` | テスト作成 |

### CLIフロー

```bash
# サプリメント構築
python run_train.py --build-supplement sire_track_baba --workers 4

# 既存サプリメントと組み合わせて学習
python run_train.py --train-only --supplement mining bms_detail sire_track_baba

# 評価
python run_train.py --eval-only --supplement mining bms_detail sire_track_baba
```

### クラス設計

```python
class SireTrackBabaFeatureExtractor(FeatureExtractor):
    """種牡馬×芝ダ×馬場状態 特徴量（サプリメント）."""

    _FEATURES: list[str] = [
        "blood_father_track_baba_rate",
        "blood_bms_track_baba_rate",
        "blood_father_heavy_rate",
        "blood_bms_heavy_rate",
    ]

    def extract(self, race_key, uma_race_df) -> pd.DataFrame:
        # 1. レース情報取得（trackcd, sibababacd/dirtbabacd）
        # 2. 出走馬の kettonum → n_sanku で fnum/mfnum 取得
        # 3. 父/BMS × コース種別×馬場のバッチクエリ実行
        # 4. MIN_SAMPLES 閾値でフィルタ → NaN or 率
        ...
```

---

## レース内相対特徴量への展開

追加した4特徴量のうち `blood_father_track_baba_rate` と `blood_bms_track_baba_rate` は
`_add_relative_features()` でZスコア+ランクに展開する候補。

| 特徴量 | missing_type | 理由 |
|--------|-------------|------|
| `blood_father_track_baba_rate` | `blood` | サンプル不足由来のNaN → Zスコア除外 |
| `blood_bms_track_baba_rate` | `blood` | 同上 |

ただし相対特徴量の追加数が増えすぎるリスクがあるため、初期段階では相対化せず、
重要度分析の結果を見て追加を検討する。

---

## 期待される効果

1. **道悪レースの予測精度向上:** パイロ、オルフェーヴル等の道悪巧者種牡馬の産駒を
   重馬場レースで適切に評価できる
2. **芝ダ替わりの判断精度向上:** エピファネイア（芝得意・ダート不得意）等の
   コース適性を馬場状態と組み合わせて捕捉
3. **既存 `blood_father_baba_rate` の限界克服:** 芝重とダート重を混同しない
4. **ID依存の軽減:** `blood_father_id` に暗黙的にエンコードされていた
   芝ダ×馬場状態の情報を明示的に特徴量化
