# 特徴量・アルゴリズム改善提案書

## 現状分析サマリ

現在の実装は約130特徴量（16カテゴリ）をLightGBMの二値分類（3着以内確率）で予測する構成。
特徴量は馬基本属性・過去成績・スピード指数・脚質・レース条件・騎手/調教師・調教・血統・オッズなど幅広くカバーしている。

本提案では、コード解析とDB定義の精査から抽出した **未活用データ・未実装ロジック・モデル改善** の3軸で改善案を整理する。

---

## 改善提案一覧（優先度順）

| # | カテゴリ | 提案 | 期待効果 | 実装難度 | 状況 |
|---|---------|------|---------|---------|------|
| 1 | 特徴量追加 | レース内相対特徴量 | ★★★★★ | 中 | ✅ 実装済 |
| 2 | 特徴量追加 | レースラップ・ペース特徴量 | ★★★★ | 中 | 🔲 未着手 |
| 3 | 特徴量追加 | JRA-VANデータマイニング予想の活用 | ★★★★ | 低 | ✅ 実装済（サプリメント方式） |
| 4 | 特徴量改善 | 直近走の「重み付き」成績集計 | ★★★★ | 低 | 🔲 未着手 |
| 5 | 特徴量追加 | 票数（投票比率）ベースの特徴量 | ★★★★ | 中 | 🔲 未着手 |
| 6 | 特徴量追加 | 時系列オッズの変動特徴量 | ★★★ | 中 | 🔲 未着手 |
| 7 | 特徴量追加 | 馬主・生産者の成績特徴量 | ★★★ | 低 | 🔲 未着手 |
| 8 | 特徴量改善 | 血統特徴量の強化（母系、ニックス） | ★★★ | 中 | ✅ 実装済 |
| 9 | 特徴量追加 | コース区分（A/B/C/D）の活用 | ★★★ | 低 | 🔲 未着手 |
| 10 | 特徴量追加 | 競走馬セール価格 | ★★ | 低 | 🔲 未着手 |
| 11 | 特徴量改善 | 調教特徴量の強化 | ★★★ | 中 | 🔲 未着手 |
| 12 | モデル改善 | 目的変数の多様化（LambdaRank等） | ★★★★★ | 高 | ✅ 実装済（案A: LambdaRank） |
| 13 | モデル改善 | 確率キャリブレーション | ★★★★ | 低 | 🔲 未着手 |
| 14 | モデル改善 | 特徴量自動選択パイプライン | ★★★ | 中 | 🔲 未着手 |
| 15 | モデル改善 | オッズ有無の2モデル体制 | ★★★ | 中 | 🔲 未着手 |
| 16 | 特徴量追加 | 賞金ベース特徴量 | ★★★ | 低 | 🔲 未着手 |
| 17 | 特徴量追加 | 枠順×コース適性クロス特徴量 | ★★★ | 中 | 🔲 未着手 |

---

## 1. レース内相対特徴量（最重要）

### 問題点
現在の特徴量は各馬の「絶対的な能力値」のみで構成されている。しかし競馬は相対的な競争であり、
**同レース出走馬の中での相対的な位置付け**が極めて重要。

### 提案内容

レース内の各特徴量について、平均・偏差ベースの相対化を行う。

```python
# レース内特徴量の追加例（pipeline.py の _add_cross_features に追加）
relative_features = [
    "speed_index_avg_last3",
    "horse_fukusho_rate",
    "horse_avg_jyuni_last3",
    "jockey_win_rate_year",
    "blood_father_turf_rate",
    "training_hanro_time4",
]

for feat_name in relative_features:
    if feat_name in result.columns:
        race_mean = result[feat_name].replace(MISSING_NUMERIC, np.nan).mean()
        race_std = result[feat_name].replace(MISSING_NUMERIC, np.nan).std()
        if race_std and race_std > 0:
            result[f"rel_{feat_name}_zscore"] = (
                (result[feat_name] - race_mean) / race_std
            ).fillna(0)
        # レース内順位
        result[f"rel_{feat_name}_rank"] = (
            result[feat_name].rank(ascending=False, method="min")
        )
```

### 追加特徴量一覧（約20個）

| 特徴量名 | 説明 |
|---------|------|
| `rel_speed_index_zscore` | スピード指数のレース内Zスコア |
| `rel_speed_index_rank` | スピード指数のレース内順位 |
| `rel_fukusho_rate_zscore` | 複勝率のレース内Zスコア |
| `rel_fukusho_rate_rank` | 複勝率のレース内順位 |
| `rel_avg_jyuni_zscore` | 直近平均着順のレース内Zスコア |
| `rel_jockey_rate_zscore` | 騎手勝率のレース内Zスコア |
| `rel_jockey_rate_rank` | 騎手勝率のレース内順位 |
| `rel_trainer_rate_zscore` | 調教師勝率のレース内Zスコア |
| `rel_blood_father_zscore` | 父産駒成績のレース内Zスコア |
| `rel_training_time_rank` | 調教タイムのレース内順位 |

### 根拠
LightGBMは分岐でのthreshold比較なので、絶対値では「同レースのライバルより上か下か」が判断しにくい。
Zスコアやランク特徴量を入れることで、相対的な実力差がダイレクトにモデルに伝わる。

### 実装ノート（2026-02-18 実装済み）

- **実装箇所:** `src/features/pipeline.py` の `_add_relative_features()` メソッド
- **相対化対象:** 14特徴量 × 2（Zスコア + ランク）= 28特徴量を追加
- **提案からの変更点:** 提案では `relative_features` リストを `_add_cross_features` 内に直接記述するコード例だったが、メンテナンス性を考慮してクラス変数 `_RELATIVE_TARGETS` にターゲット特徴量と方向性（ascending）を定義する方式に変更
- **欠損値処理（missing_type 3パターン方式）:**
  - `_RELATIVE_TARGETS` は3タプル `(feat_name, ascending, missing_type)` で定義
  - `missing_type="numeric"`: `MISSING_NUMERIC`(-1) のみNaN化。0.0は有効値として保持
  - `missing_type="rate"`: `MISSING_NUMERIC`(-1) のみNaN化。0.0は「勝率0%」等の正当な値なので保持
  - `missing_type="blood"`: `MISSING_NUMERIC`(-1) と `MISSING_RATE`(0.0) の両方をNaN化。血統適性はデータ不足時に0.0が設定されるため
  - Zスコアは0.0、ランクは最下位で埋める
- **テスト:** `tests/test_features.py` に7件のテストを追加（Zスコア計算、ランク計算、欠損カラム、欠損値処理、rate型0.0保持、blood型0.0除外）
- **注意:** parquet を再構築しないと新特徴量が含まれない。`--force-rebuild` で全年度を再構築すること

### バグ修正ノート: MISSING_RATE=0.0 の誤NaN化（2026-02-18）

**問題:** 初期実装では全特徴量に対して `MISSING_NUMERIC`(-1) と `MISSING_RATE`(0.0) の両方をNaNに変換していた。
しかし、rate系特徴量（`horse_fukusho_rate`, `horse_win_rate`, `jockey_win_rate_year` 等）では
0.0は「勝率0%」「複勝率0%」を意味する正当な値である。これをNaN化すると、弱い馬のZスコアが
0.0（平均）になってしまい、本来マイナスであるべき相対評価が失われた。

**影響:** モデル性能が低下（回収率-5%、的中率-0.3%）。弱い馬が平均並みの評価を受けることで、
予測精度が全体的に悪化していた。

**修正:** 特徴量ごとに欠損値の意味が異なることを認識し、`missing_type` パラメータで3パターンに分類。
rate系では0.0を有効値として保持し、blood系のみ0.0をNaN化する方式に変更。

**教訓:** 欠損値のセンチネル値（特に0.0）が特徴量のドメインによって意味が異なるケースでは、
一律の欠損値処理は危険。特徴量ごとの意味論を考慮した処理が必要。

---

## 2. レースラップ・ペース特徴量

### 問題点
n_raceテーブルには`LapTime1`〜`LapTime20`（1ハロンごとのラップ）、`HaronTimeS3`(前3F)、`HaronTimeL3`(後3F)が
格納されているが、**現在これらは一切使用されていない**。
レース全体のペースは各馬の着順に大きく影響する（ハイペース→差し有利、スローペース→逃げ有利）。

### 提案内容

```python
# 過去走のレースペース情報を取得
class PaceFeatureExtractor:
    """レースペース特徴量（新規モジュール）"""

    _FEATURES = [
        "pace_s3f_last",           # 前走レースの前3F（秒）
        "pace_l3f_last",           # 前走レースの後3F（秒）
        "pace_s3f_l3f_ratio_last", # 前3F/後3F比（>1ならスロー）
        "pace_type_last",          # 前走ペースタイプ（H/M/S）
        "pace_horse_style_pace_match",  # 脚質×ペース適性スコア
        "pace_avg_front_ratio",    # 直近5走でハイペースレースでの好走率
        "pace_avg_slow_ratio",     # 直近5走でスローペースレースでの好走率
    ]
```

SQL例：
```sql
SELECT harontimes3, harontimel3, laptime1, laptime2, laptime3
FROM n_race
WHERE year = %(year)s AND monthday = %(monthday)s
  AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
  AND nichiji = %(nichiji)s AND racenum = %(racenum)s
```

### ペースタイプの判定ロジック
```python
def classify_pace(s3f: float, l3f: float) -> str:
    """前3F/後3F比でペースを分類する."""
    if s3f <= 0 or l3f <= 0:
        return "unknown"
    ratio = s3f / l3f
    if ratio < 0.97:    # 前傾ラップ → ハイペース
        return "high"
    elif ratio > 1.03:  # 後傾ラップ → スローペース
        return "slow"
    else:
        return "middle"
```

### 脚質×ペース適性マトリクス
```python
# 馬の主要脚質に対してペースタイプごとの好走率を計算
# 逃げ馬: スローペース→好走率↑, ハイペース→好走率↓
# 差し馬: スローペース→好走率↓, ハイペース→好走率↑
pace_style_matrix = {
    ("1", "slow"): 1.0,   # 逃げ×スロー → 有利
    ("1", "high"): -1.0,  # 逃げ×ハイ → 不利
    ("4", "slow"): -1.0,  # 追込×スロー → 不利
    ("4", "high"): 1.0,   # 追込×ハイ → 有利
    # ...
}
```

---

## 3. JRA-VANデータマイニング予想の活用

### 問題点
DBには`MINING`テーブル（JRA-VAN公式のデータマイニング予想タイム・順位）と
`TAISENGATA_MINING`テーブル（対戦型マイニングスコア）が存在するが、**完全に未使用**。
UMA_RACEテーブルにも`DMTime`（マイニング予想走破タイム）、`DMJyuni`（マイニング予想順位）が
格納されているが未利用。

### 提案内容

```python
class MiningFeatureExtractor:
    _FEATURES = [
        "mining_dm_time",           # DM予想走破タイム
        "mining_dm_jyuni",          # DM予想順位
        "mining_dm_gosa_range",     # DM予想誤差幅（信頼度の指標）
        "mining_dm_jyuni_rank",     # DM予想順位のレース内順位
        "mining_tm_score",          # 対戦型マイニングスコア
        "mining_tm_score_rank",     # 対戦型スコアのレース内順位
    ]
```

SQL（UMA_RACEから）：
```sql
SELECT kettonum, dmtime, dmgosap, dmgosam, dmjyuni, dmkubun
FROM n_uma_race
WHERE year = %(year)s AND monthday = %(monthday)s
  AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
  AND nichiji = %(nichiji)s AND racenum = %(racenum)s
  AND datakubun IN ('1','2','3','4','5','6','7')
```

### 注意点
- `DMKubun`が1=前日、2=当日、3=直前。予測タイミングに応じて使い分ける
- オッズと同様にデータリーク防止の観点から使用タイミングを管理する必要あり

### 実装ノート（2026-02-20 実装済み）

- **実装箇所:** `src/features/mining.py`（MiningFeatureExtractor）+ `src/features/supplement.py`（サプリメントシステム）
- **サプリメント方式の採用:** メイン parquet の再構築を避けるため、差分特徴量として独立した parquet に保存し、学習/評価時にマージする方式を採用
- **提案からの変更点:**
  - 提案の6特徴量に加え `mining_dm_gosa_p`、`mining_dm_gosa_m` を個別カラムとして追加（計7特徴量）
  - `mining_dm_jyuni_rank`（レース内順位）と `mining_tm_score_rank` は相対特徴量システムに委譲可能なため、Extractor には含めていない
  - n_mining / n_taisengata_mining はDB依存の横持ちカラム名のため、スキーマから動的検出して縦持ちに変換
- **使い方:**
  - `python run_train.py --build-supplement mining` でサプリメント構築
  - `python run_train.py --train-only --supplement mining` でマージして学習
  - `python run_train.py --eval-only --supplement mining` でマージして評価
- **テスト:** `tests/test_mining_supplement.py`（16テスト）

---

## 4. 直近走の「重み付き」成績集計

### 問題点
現在の過去成績は「直近5走の均等平均」で計算しているが、
前走の結果は2走前より重要度が高く、古い走ほど情報の鮮度が落ちる。

### 提案内容

指数重み（Exponential Decay）を適用した成績集計を追加する。

```python
def weighted_avg(values: list[float], decay: float = 0.7) -> float:
    """指数減衰重み付き平均.

    最新走 = 1.0, 1走前 = 0.7, 2走前 = 0.49, ...
    """
    weights = [decay ** i for i in range(len(values))]
    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w
```

### 追加特徴量

| 特徴量名 | 説明 |
|---------|------|
| `horse_fukusho_rate_weighted5` | 直近5走の重み付き複勝率 |
| `horse_avg_jyuni_weighted5` | 直近5走の重み付き平均着順 |
| `speed_index_weighted3` | 直近3走の重み付きスピード指数 |
| `speed_l3f_weighted3` | 直近3走の重み付き上がり3F |

---

## 5. 票数（投票比率）ベースの特徴量

### 問題点
現在はオッズのみを使用しているが、DBには`HYOSU_TANPUKU`テーブルに各馬の
**単勝票数・複勝票数**が格納されている。票数比率はオッズの裏返しだが、
票数の「絶対量」からレースの注目度・集中度などの追加情報が得られる。

### 提案内容

```python
class VoteFeatureExtractor:
    _FEATURES = [
        "vote_tan_ratio",          # 単勝票数シェア（その馬÷全馬合計）
        "vote_fuku_ratio",         # 複勝票数シェア
        "vote_concentration",      # 票の集中度（上位3頭の票数シェア合計）
        "vote_entropy",            # 票数分布のエントロピー（混戦度）
        "vote_tan_fuku_diff",      # 単勝人気と複勝人気の乖離
    ]
```

SQL：
```sql
SELECT umaban,
       CAST(tanhyo AS bigint) AS tan_hyo,
       CAST(tanninki AS int) AS tan_ninki,
       CAST(fukuhyo AS bigint) AS fuku_hyo,
       CAST(fukuninki AS int) AS fuku_ninki
FROM n_hyosu_tanpuku
WHERE year = %(year)s AND monthday = %(monthday)s
  AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
  AND nichiji = %(nichiji)s AND racenum = %(racenum)s
```

### エントロピーの計算
```python
import numpy as np

def vote_entropy(votes: list[int]) -> float:
    """票数分布のシャノンエントロピー. 高い=混戦, 低い=本命がいる."""
    total = sum(votes)
    if total == 0:
        return 0.0
    probs = [v / total for v in votes if v > 0]
    return -sum(p * np.log(p) for p in probs)
```

---

## 6. 時系列オッズの変動特徴量

### 問題点
DBには`JODDS_TANPUKU`テーブルで時系列オッズ（時刻ごとのオッズ推移）が記録されているが未活用。
オッズの動き（急低下＝大口投入、じわじわ下がる＝安定した支持）は重要な情報。

### 提案内容

```python
class OddsTimeSeriesFeatureExtractor:
    _FEATURES = [
        "jodds_early_odds",        # 序盤のオッズ（発売開始から2時間以内）
        "jodds_final_odds",        # 最終オッズ
        "jodds_change_ratio",      # 序盤→最終のオッズ変化率
        "jodds_drop_flag",         # 急激な人気上昇フラグ（オッズが半分以下に）
        "jodds_rank_change",       # 序盤→最終の人気順位変動
    ]
```

SQL：
```sql
SELECT umaban, tanodds, happyotime
FROM n_jodds_tanpuku
WHERE year = %(year)s AND monthday = %(monthday)s
  AND jyocd = %(jyocd)s AND kaiji = %(kaiji)s
  AND nichiji = %(nichiji)s AND racenum = %(racenum)s
ORDER BY happyotime
```

---

## 7. 馬主・生産者の成績特徴量

### 問題点
`BANUSI`（馬主マスタ）と`SEISAN`（生産者マスタ）にはそれぞれ成績データが含まれるが、
現在の実装では全く使用されていない。有力馬主・有力生産者の馬は統計的に好成績の傾向がある。

### 提案内容

```python
class OwnerBreederFeatureExtractor:
    _FEATURES = [
        "owner_code",              # 馬主コード（カテゴリ変数）
        "owner_win_rate_year",     # 馬主の当年勝率
        "owner_fukusho_rate_year", # 馬主の当年複勝率
        "breeder_code",            # 生産者コード（カテゴリ変数）
        "breeder_win_rate_year",   # 生産者の当年勝率
    ]
```

SQL（馬主成績）：
```sql
SELECT banusicode,
       CAST(h_chakukaisu1 AS int) AS wins,
       CAST(h_chakukaisu1 AS int) + CAST(h_chakukaisu2 AS int)
         + CAST(h_chakukaisu3 AS int) AS top3,
       CAST(h_chakukaisu1 AS int) + CAST(h_chakukaisu2 AS int)
         + CAST(h_chakukaisu3 AS int) + CAST(h_chakukaisu4 AS int)
         + CAST(h_chakukaisu5 AS int) + CAST(h_chakukaisu6 AS int) AS total
FROM n_banusi
WHERE banusicode IN %(codes)s
```

---

## 8. 血統特徴量の強化

### 問題点

現在の血統特徴量は父・母父の2軸のみ。以下の改善余地がある：

- **母系（ファミリーライン）** が未活用
- **ニックス**（父×母父の相性）の概念がない
- **近親交配**が単純なフラグのみ（どの祖先が重複しているかの情報なし）
- 父産駒成績の集計が全体の複勝率のみで、**馬場状態別×距離帯別のクロス集計**がない

### 提案内容

```python
# 追加特徴量
blood_new_features = [
    "blood_mother_keito",           # 母系統名
    "blood_nicks_rate",             # 父×母父コンビの産駒複勝率
    "blood_nicks_runs",             # 父×母父コンビの産駒出走数
    "blood_father_baba_rate",       # 父産駒の「今回の馬場状態」での複勝率
    "blood_father_jyo_rate",        # 父産駒の「今回の競馬場」での複勝率
    "blood_inbreed_generation",     # 近親交配が発生した世代（2代/3代/なし）
    "blood_mother_produce_rate",    # 母の産駒成績（兄弟姉妹の複勝率）
]
```

### ニックス（父×母父相性）の計算
```sql
-- 同じ父×母父の組み合わせを持つ馬の成績を集計
SELECT s1.fnum AS father, s2.fnum AS bms,
    COUNT(*) AS total,
    SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) <= 3 THEN 1 ELSE 0 END) AS top3
FROM n_sanku s1
JOIN n_sanku s2 ON s1.mfnum = s2.fnum  -- s2は母の父
JOIN n_uma_race ur ON s1.kettonum = ur.kettonum
WHERE ur.datakubun = '7' AND ur.ijyocd = '0'
  AND s1.fnum = %(father_num)s
  AND s2.fnum = %(bms_num)s
  AND (ur.year || ur.monthday) < %(race_date)s
GROUP BY s1.fnum, s2.fnum
```

### 実装ノート（2026-02-19 実装済み）

- **実装箇所:** `src/features/bloodline.py` を大幅拡張（既存10特徴量 + 新規7特徴量 = 合計17特徴量）
- **変更ファイル:**
  - `src/features/bloodline.py`: 新規特徴量7個の追加、`_get_race_info()` の拡張、`_check_inbreeding()` の世代判定追加、`_get_nicks_stats()` / `_get_sire_baba_stats()` / `_get_sire_jyo_stats()` / `_get_mother_produce_stats()` の新規メソッド追加
  - `src/config.py`: `CATEGORICAL_FEATURES` に `blood_mother_keito` を追加
  - `src/features/pipeline.py`: `_RELATIVE_TARGETS` に `blood_nicks_rate`, `blood_father_baba_rate`, `blood_father_jyo_rate`, `blood_mother_produce_rate` を追加（blood型欠損値処理）
  - `tests/test_features.py`: 近親交配世代判定テスト4件、相対特徴量テスト2件を追加
  - `docs/feature_design.md`: 特徴量一覧に新規7特徴量を追加
- **提案からの変更点:**
  - ニックスの集計期間を5年に設定（3年では父×母父の組み合わせのサンプル数が不足しがちなため）
  - 母産駒成績の集計期間を10年に設定（兄弟姉妹の走歴が長期に渡るため）
  - 母産駒成績は当該馬自身を除外して計算（自身の成績が含まれるとデータリークに近い効果が出るため）
  - 近親交配世代は `_check_inbreeding()` 静的メソッドとして分離し、テスト可能な設計に
- **相対特徴量（レース内Zスコア・ランク）:** 新規4特徴量（nicks_rate, father_baba_rate, father_jyo_rate, mother_produce_rate）は blood 型欠損値処理で 0.0=データなしとして NaN 化
- **注意:** parquet を再構築しないと新特徴量が含まれない。`--force-rebuild` で全年度を再構築すること

### 設計判断ノート: blood_mother_id の削除（2026-02-19）

**経緯:** 初期実装では `blood_mother_id`（母馬繁殖登録番号）を特徴量として含め、`CATEGORICAL_FEATURES` に登録していた。
しかし母馬IDは数千～数万種類のユニーク値を持つ高カーディナリティ変数であり、以下の問題があった。

- カテゴリ変数として扱う場合: LightGBMのカテゴリ分割が訓練データ固有のIDパターンに過学習する
- 数値変数として扱う場合: ID番号の大小に意味がなく、分割閾値に根拠がない

母馬の情報は `blood_mother_keito`（母系統）と `blood_mother_produce_rate`（母産駒成績）で十分にカバーされるため、
`blood_mother_id` は特徴量から完全に削除した。

**参考:**
- `blood_father_id`（種牡馬ID）: 活躍する種牡馬は限られるため数百種類に収まる → カテゴリOK
- `blood_bms_id`（母父ID）: 同上 → カテゴリOK

**教訓:** 高カーディナリティのID変数は、カテゴリ変数では過学習、数値変数では無意味な分割を招く。
IDの情報を活用したい場合は、系統名など上位の抽象度にマッピングするか、そのIDに紐づく成績統計量に変換して使用する。

### 性能改善ノート: ニックスクエリのバッチ化（2026-02-19）

**問題:** `_get_nicks_stats()` がペアごとに個別SQLを発行しており、
1レースにN頭の異なる父×母父の組み合わせがあると最大N回のDBクエリが走っていた。
1レースあたり十数回の追加クエリはDB負荷と特徴量構築時間の増大を招く。

**修正:** `GROUP BY s.fnum, s.mfnum` の1バッチクエリに統合。
全ペアの父番号・母父番号をそれぞれ `IN` 句で渡し、一括取得後にペアでフィルタする方式に変更。
これにより1レースあたりのニックスDBクエリが N回 → 1回 に削減された。

---

## 9. コース区分（A/B/C/D）の活用

### 問題点
n_raceテーブルの`CourseKubunCD`はコースの使用区分（内回り/外回り/Aコースなど）を示すが、未使用。
芝コースのレールポジションによって内枠有利・外枠有利が大きく変わる。

### 提案内容

```python
# race.py に追加
"race_course_kubun"      # コース区分（A/B/C/D/E）
"cross_wakuban_course"   # 枠番×コース区分の交互作用
```

---

## 10. 競走馬セール価格

### 問題点
`SALE`テーブルに競走馬の市場取引価格が記録されているが未使用。
セール価格は馬の血統的ポテンシャルの代理指標として有用（特に新馬戦・若い馬）。

### 提案内容

```python
class SaleFeatureExtractor:
    _FEATURES = [
        "sale_price",          # セール価格（万円）
        "sale_price_log",      # log(セール価格)
        "sale_price_rank",     # レース内での価格順位
    ]
```

SQL：
```sql
SELECT kettonum, CAST(saleprice AS bigint) AS price
FROM n_sale
WHERE kettonum IN %(kettonums)s
```

---

## 11. 調教特徴量の強化

### 問題点

現在の調教特徴量は以下の問題を抱えている：

- **坂路のみ**最終追切を取得しているが、ウッドチップの最終追切タイム（5F/3F）は最速タイムのみ
- 調教の**トレセン区分**（美浦/栗東）が未活用
- **調教タイムの相対評価**（同日・同コースの他馬との比較）がない
- ウッドチップの**コース区分**（A/B/C/D/E）と**馬場周り**（左/右）が未活用

### 提案内容

```python
training_new_features = [
    "training_wc_time3_last",      # ウッドチップ最終追切3Fタイム
    "training_wc_accel",           # ウッドチップ最終追切の加速度
    "training_tresen_kubun",       # トレセン区分（美浦=0/栗東=1）
    "training_hanro_rank_day",     # 坂路追切タイムの同日順位
    "training_intensity_score",    # 調教強度スコア（本数×平均タイム）
    "training_pattern",            # 調教パターン（坂路のみ/WCのみ/併用）
]
```

---

## 12. 目的変数の多様化・LambdaRank（最重要モデル改善）

### 問題点

現在の問題設定は「3着以内か否か」の二値分類。しかし：

- 1着と3着の差を区別できない
- 着順が近い馬同士の相対的な比較ができない
- 回収率最適化には「1着を当てる」精度が重要

### 提案内容

#### 案A: LambdaRankによるランキング学習

LightGBMには`lambdarank`目的関数があり、レース内での着順をランキングとして直接学習できる。

```python
LGBM_PARAMS_RANKING = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3, 5],
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "min_child_samples": 50,
}

# group パラメータにレースごとの馬頭数を渡す
train_data = lgb.Dataset(
    X_train, label=y_train,
    group=train_group_sizes,  # [16, 14, 18, ...] 各レースの出走頭数
)
```

目的変数は着順の逆数や対数変換で重み付け：
```python
# 着順ベースの関連度スコア
# 1着=5, 2着=4, 3着=3, 4着=2, 5着=1, 6着以下=0
def relevance_score(jyuni: int) -> int:
    if jyuni <= 0:
        return 0
    if jyuni == 1:
        return 5
    elif jyuni == 2:
        return 4
    elif jyuni == 3:
        return 3
    elif jyuni <= 5:
        return 1
    return 0
```

### 実装ノート（2026-02-18 実装済み）

- **実装箇所:** `src/model/trainer.py`（LambdaRank モード追加）、`src/model/evaluator.py`（NDCG評価追加）、`src/features/pipeline.py`（`_get_target()`に関連度スコア追加）、`src/config.py`（`LGBM_PARAMS_RANKING`追加）、`run_train.py`（`--ranking`オプション追加）、`src/model/predictor.py`（メタデータからranking自動検出）
- **提案内容の採用:** 案A（LambdaRank）を採用。案B（マルチタスク学習）は将来の拡張として残す
- **関連度スコア:** 提案どおり 1着=5, 2着=4, 3着=3, 4着=2, 5着=1, 6着以下=0（SQLで直接計算）
- **group パラメータ:** `trainer.py` の `_prepare_groups()` でレースキーによるソートとグループサイズ計算を実装
- **評価指標:** LambdaRank モードでは logloss の代わりに NDCG@1/3/5 をレース単位で算出。AUC は二値ラベルに対するランキング品質指標として引き続き使用
- **メタデータ保存:** `{model_name}_meta.json` に `ranking` フラグを記録。`--eval-only` 時に自動検出
- **使い方:** `python run_train.py --ranking --model-name ranking_model`（parquet に `target_relevance` が必要なため `--force-rebuild` で再構築すること）
- **value_bet 戦略の注意:** LambdaRank の出力は確率ではないため、期待値ベットは二値分類モデルと組み合わせて使用することを推奨

#### 性能改善の確認結果

LambdaRank モデルの導入により、性能改善が確認された。

**変更されたモジュール（7ファイル）:**

| ファイル | 変更内容 |
|---------|---------|
| `src/config.py` | `LGBM_PARAMS_RANKING` 定数を追加 |
| `src/features/pipeline.py` | `_get_target()` で `target_relevance`, `kakuteijyuni` を追加取得 |
| `src/model/trainer.py` | `ranking` モード、`_train_ranking()`, `_prepare_groups()`, `_meta.json` 保存を追加 |
| `src/model/evaluator.py` | `_compute_ndcg()`, `_dcg_at_k()` を追加。ranking モード時の評価分岐 |
| `src/model/predictor.py` | `_meta.json` からの `ranking` フラグ自動復元、スコア表示切替 |
| `run_train.py` | `--ranking` CLI オプション追加、全ステップで ranking 伝播 |
| `CLAUDE.md` | LambdaRank 関連の記述・CLIオプション・注意事項を追加 |

**設計上のポイント:**

1. **後方互換性の維持:** `ranking=False`（デフォルト）では従来の二値分類モードがそのまま動作する。既存の parquet にも `target` カラムは引き続き含まれる
2. **parquet の再構築が必要:** `target_relevance` と `kakuteijyuni` は `_get_target()` で SQL から直接計算するため、旧 parquet には含まれない。`--force-rebuild` が必要
3. **group パラメータの生成:** `_prepare_groups()` でレースキーによるソートとグループサイズ計算を一括処理。LambdaRank は同一レースの馬が連続している前提なのでソート必須
4. **メタデータによるモード自動検出:** `{model_name}_meta.json` に `ranking` フラグを保存。`--eval-only` 時や `predictor.py` のロード時にモードを自動検出
5. **回収率シミュレーション:** LambdaRank のスコアは確率ではないがレース内の大小関係は保たれるため、`nlargest` ベースの賭け戦略は変更なしで動作する。`value_bet` のみ確率が必要なので注意

**今後の拡張候補:**

- 案B（マルチタスク学習）との組み合わせによるアンサンブル
- 二値分類モデルの確率と LambdaRank のランクスコアを統合した最終予測
- 関連度スコアの重み調整（例: 1着の重みをさらに大きくする等）のハイパーパラメータ化

#### 案B: マルチタスク学習

複数の目的変数を同時に予測する：
- target_top1: 1着フラグ
- target_top3: 3着以内フラグ
- target_jyuni: 着順（回帰）

別々のモデルを学習し、アンサンブルで最終予測を生成。

---

## 13. 確率キャリブレーション

### 問題点
LightGBMの出力はlogloss最適化されたスコアだが、実際の確率として完全に校正されているわけではない。
回収率シミュレーションで`pred_prob × odds > 1.2`のような期待値ベットをする場合、
確率の精度が回収率に直結する。

### 提案内容

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

# Platt Scaling（ロジスティック回帰）
platt = CalibratedClassifierCV(
    estimator=lgb_model, method="sigmoid", cv="prefit"
)
platt.fit(X_valid, y_valid)

# Isotonic Regression（ノンパラメトリック）
iso_reg = IsotonicRegression(out_of_bounds="clip")
iso_reg.fit(y_pred_raw, y_valid)
calibrated_prob = iso_reg.predict(y_pred_raw)
```

### 評価方法
- **Reliability Diagram**（信頼度図）で校正度を可視化
- **Brier Score** で確率精度を定量評価

---

## 14. 特徴量自動選択パイプライン

### 問題点
feature_design.mdに「特徴量重要度による選別」の方針が書かれているが、実装されていない。
130+特徴量にさらに本提案の追加特徴量を加えると200近くになり、過学習リスクが増大する。

### 提案内容

```python
class FeatureSelector:
    def select(self, model, X_train, y_train, X_valid, y_valid):
        # Step 1: Gain importance で下位30%を除去
        importance = model.feature_importance(importance_type='gain')
        threshold = np.percentile(importance, 30)
        mask = importance > threshold

        # Step 2: Permutation Importance で重要度0以下を除去
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(model, X_valid, y_valid, n_repeats=5)
        mask &= perm.importances_mean > 0

        # Step 3: 相関0.95以上のペアから片方を除去
        corr_matrix = X_train.loc[:, mask].corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        drop_cols = [c for c in upper.columns if any(upper[c] > 0.95)]

        return selected_features
```

---

## 15. オッズ有無の2モデル体制

### 問題点
現在は`include_odds`フラグで1モデルを切り替えるだけ。しかし：
- オッズありモデルとオッズなしモデルでは**最適な特徴量セットとハイパーパラメータが異なる**
- 前日予測（オッズ不安定）と当日予測（オッズ安定）でもモデルを分けるべき

### 提案内容

```python
# config.py
MODEL_CONFIGS = {
    "no_odds": {
        "params": {**LGBM_PARAMS, "num_leaves": 47},
        "include_odds": False,
        "include_mining": False,
        "description": "前日予測用（オッズ・マイニング除外）",
    },
    "with_odds": {
        "params": {**LGBM_PARAMS, "num_leaves": 63},
        "include_odds": True,
        "include_mining": True,
        "description": "当日予測用（全特徴量使用）",
    },
}
```

---

## 16. 賞金ベース特徴量

### 問題点
n_uma_raceに`Honsyokin`（獲得本賞金）、n_raceに`Honsyokin1`〜`Honsyokin7`（本賞金体系）が
あるが、賞金関連の特徴量が一切ない。

### 提案内容

```python
prize_features = [
    "horse_total_prize",           # 通算獲得賞金
    "horse_avg_prize_last5",       # 直近5走の平均獲得賞金
    "race_prize_1st",              # 当該レースの1着賞金
    "horse_prize_vs_race_class",   # 通算賞金÷レースクラス賞金（格の一致度）
]
```

賞金はクラス分けの直接的な基準であり、馬の実力レベルを間接的に反映する。
同クラスでも賞金上位（昇級間近）と賞金下位（降級せず安定）では好走率が異なる。

---

## 17. 枠順×コース適性クロス特徴量

### 問題点
現在の枠順特徴量は`post_is_inner`/`post_is_outer`の2値フラグのみ。
しかし枠順の有利不利は**競馬場×距離×トラック**の組み合わせで大きく異なる。

### 提案内容

コース別の枠順有利不利統計を事前集計し、特徴量として利用する。

```python
# 事前集計SQL: コース×枠番ゾーンの複勝率
# 例：東京芝2400m内枠(1-3枠)複勝率 vs 外枠(6-8枠)複勝率
def _calc_waku_advantage(jyocd, kyori, trackcd, wakuban, race_date):
    """枠番の有利不利スコア（過去3年のコース別統計）"""
    sql = """
    SELECT
        CASE WHEN CAST(ur.wakuban AS int) <= 3 THEN 'inner'
             WHEN CAST(ur.wakuban AS int) >= 6 THEN 'outer'
             ELSE 'middle' END AS waku_zone,
        COUNT(*) AS total,
        SUM(CASE WHEN CAST(ur.kakuteijyuni AS int) <= 3
            THEN 1 ELSE 0 END) AS top3
    FROM n_uma_race ur
    JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
    WHERE r.jyocd = %(jyocd)s
      AND r.kyori = %(kyori)s
      AND r.trackcd = %(trackcd)s
      AND r.datakubun = '7'
      AND ur.ijyocd = '0'
      AND r.year >= %(year_start)s
      AND (r.year || r.monthday) < %(race_date)s
    GROUP BY waku_zone
    """
```

追加特徴量：
```python
"cross_waku_advantage"     # 枠番のコース別有利不利スコア
"cross_umaban_advantage"   # 馬番のコース別有利不利スコア
```

---

## 実装優先順位（ロードマップ）

### フェーズ1（高ROI・低コスト）
1. **レース内相対特徴量**（#1） — パイプラインの後処理で追加可能、全特徴量に波及
2. **重み付き成績集計**（#4） — horse.pyの小改修
3. **JRA-VANマイニング予想**（#3） — UMA_RACEから取得するだけ
4. **確率キャリブレーション**（#13） — evaluator.pyへの追加

### フェーズ2（中程度の改修）
5. **レースラップ・ペース特徴量**（#2） — 新モジュール追加
6. **票数ベース特徴量**（#5） — 新モジュール追加
7. **特徴量自動選択**（#14） — 学習パイプラインの拡張
8. **2モデル体制**（#15） — 設定・学習フローの分離

### フェーズ3（血統・DB活用の深化）
9. **血統強化**（#8） — bloodline.pyの大幅拡張
10. **馬主・生産者**（#7） — 新モジュール追加
11. **調教強化**（#11） — training.pyの拡張
12. **時系列オッズ**（#6） — 新モジュール追加

### フェーズ4（モデル構造の抜本改善）
13. ~~**LambdaRank**（#12） — trainer.py/evaluator.pyの大改修~~ ✅ **実装済み**
14. **賞金ベース特徴量**（#16） — 新規追加
15. **セール価格**（#10） — 新規追加
16. **コース区分・枠順クロス**（#9, #17） — 小改修

---

## 期待される改善効果の見積もり

| 改善項目 | AUC改善幅（推定） | 回収率への影響 |
|---------|----------------|--------------|
| レース内相対特徴量 | +0.005〜0.015 | 大（相対評価がモデルに直結） |
| LambdaRank | +0.010〜0.020 | 大（着順予測精度の向上） |
| ペース特徴量 | +0.003〜0.008 | 中（展開予測の精度向上） |
| マイニング予想活用 | +0.005〜0.010 | 中（JRA-VAN公式の知見を取り込み） |
| 重み付き成績 | +0.002〜0.005 | 小〜中 |
| 確率キャリブレーション | AUCは不変 | 大（期待値ベットの精度が向上） |
| 票数特徴量 | +0.002〜0.005 | 中（オッズと相補的な情報） |
| 血統強化 | +0.002〜0.005 | 小〜中（新馬戦での改善大） |
| 特徴量自動選択 | +0.001〜0.005 | 中（過学習抑制） |

※ AUC改善幅は個別適用時の推定値。複合適用による相乗効果・相殺効果は別途検証が必要。

---

## まとめ

現在の実装は堅実な基盤が構築されているが、以下の3点が主要な改善余地：

1. **レース内相対化の欠如** — 各馬の絶対値だけでは相対的な実力差が伝わりにくい
2. **未活用DBテーブルの多さ** — MINING、票数、時系列オッズ、馬主、生産者、セール等の豊富なデータが眠っている
3. **モデル設計の余地** — 二値分類からランキング学習への発展、確率キャリブレーション、2モデル体制

フェーズ1の4項目は既存コードの小改修で実現可能かつ効果が大きいため、最優先で着手することを推奨する。
