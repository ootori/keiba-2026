# 種牡馬×芝ダ×馬場状態 オッズ歪み補正仕様書

**策定日:** 2026-02-21
**ステータス:** 未実装
**関連:** `src/odds_correction_stats.py`, `src/model/evaluator.py`

---

## 背景

### 分析で判明したオッズの歪み

2019-2024年（6年間）の分析により、種牡馬×芝ダ×馬場状態の組み合わせで
**市場（オッズ）が適正に評価できていないパターン**が複数確認された。

#### ダート重馬場の全体的な過小評価

| コース×馬場 | 出走数 | 回収率 | 全体平均との差 |
|------------|--------|--------|-------------|
| ダート良 | 60,526 | 72.9% | 基準 |
| ダート稍重 | 15,331 | 64.7% | -8.2% |
| **ダート重** | **7,280** | **85.0%** | **+12.1%** |
| ダート不良 | 2,397 | 72.0% | -0.9% |

ダート重馬場は全体ROI 85.0%で、テイク率込みでもほぼ公正な回収率に近い。
市場がダート重馬場の適性を系統的に過小評価している。

#### 種牡馬別の極端な歪み（ダート重）

| 種牡馬 | ダート良 ROI | ダート重 ROI | 重馬場 factor |
|--------|------------|------------|-------------|
| パイロ | 70.8% | 244.4% | 3.45 |
| ジャスタウェイ | 70.2% | 211.6% | 3.01 |
| オルフェーヴル | 73.9% | 199.7% | 2.70 |
| リオンディーズ | 49.9% | 186.5% | 3.74 |
| ドゥラメンテ | 86.7% | 34.4% | 0.40 |
| ヘニーヒューズ | 71.4% | 58.8% | 0.82 |

#### 芝×馬場状態の歪み

| 種牡馬 | 芝良 ROI | 芝重 ROI | 歪み方向 |
|--------|---------|---------|---------|
| オルフェーヴル | 70.9% | 103.4% | 重で過小評価 |
| ハービンジャー | 70.5% | 100.4% | 重で過小評価 |
| モーリス | 78.5% → 稍重 122.5% | 重 50.8% | 稍重で過小評価、重で過大評価 |
| エピファネイア | 92.8% | 62.0% | 重で過大評価 |

### 既存のオッズ補正との関係

現行のオッズ補正v2は以下のfactorを乗算適用する:

```
adjusted_odds = raw_odds × ninki_factor × rule_factors × style_factor × post_course_factor
```

本提案では新たに **sire_track_baba_factor** を追加する:

```
adjusted_odds = raw_odds × ninki_factor × rule_factors × style_factor
                × post_course_factor × sire_track_baba_factor  ← NEW
```

---

## 補正テーブル設計

### sire_track_baba_table

種牡馬（father）×コース種別（芝/ダート）×馬場重さグループ（良稍/重不）の
組み合わせ別にROIベースのfactorを算出する。

#### 馬場グループの定義

サンプル数確保のため、馬場状態を2グループに集約する。

| グループ | 馬場状態コード | 名称 |
|---------|-------------|------|
| `normal` | 良(1), 稍重(2) | 良好馬場 |
| `heavy` | 重(3), 不良(4) | 重馬場 |

**理由:** 個別の馬場状態（良/稍重/重/不良 の4段階）だと、
重×マイナー種牡馬で1000件の最小サンプルに満たない組み合わせが多い。
2グループに集約することでサンプル数を確保する。

#### キー形式

`{sire_id}_{track_type}_{baba_group}`

例:
- `0001234567_turf_heavy` → 種牡馬0001234567の芝重馬場factor
- `0001234567_dirt_normal` → 種牡馬0001234567のダート良好馬場factor

#### フォールバックチェーン

種牡馬個別のサンプルが不足する場合に段階的にフォールバックする:

```
1. sire_id × track_type × baba_group  (完全一致)
   ↓ サンプル不足
2. track_type × baba_group  (全種牡馬合算のベースライン)
   ↓ ない場合
3. factor = 1.0  (補正なし)
```

---

## 統計算出ロジック

### `_calc_sire_track_baba_stats()` の追加

`odds_correction_stats.py` に以下の関数を追加する。

#### 入力

| パラメータ | 型 | 説明 |
|-----------|---|------|
| year_start | str | 集計開始年 |
| year_end | str | 集計終了年 |
| tansho_umaban_col | str | n_haraiの単勝馬番カラム |
| tansho_pay_col | str | n_haraiの単勝払戻カラム |
| baseline_roi | float | 全体基準回収率 |
| min_samples | int | 最小サンプル数（デフォルト1000） |
| sire_min_samples | int | 種牡馬個別の最小サンプル数（デフォルト300） |

#### 算出フロー

```
1. コース種別（芝/ダート）× 馬場グループ（normal/heavy）の全体ROIを算出
   → ベースライン用の track_baba_baseline テーブル

2. 種牡馬別の出走数が多い上位N種牡馬を特定
   → 最小出走数 sire_min_samples 以上の種牡馬のみ対象

3. 各種牡馬 × track_type × baba_group の単勝ROIを算出
   → factor = sire_roi / baseline_roi

4. 全体の track_type × baba_group factor も算出（フォールバック用）
   → factor = track_baba_roi / overall_baseline_roi
```

#### SQL例

```sql
-- 種牡馬 × コース種別 × 馬場グループ別の単勝回収率
SELECT
    s.fnum AS sire_id,
    CASE
        WHEN CAST(r.trackcd AS int) BETWEEN 10 AND 22 THEN 'turf'
        WHEN CAST(r.trackcd AS int) BETWEEN 23 AND 29 THEN 'dirt'
    END AS track_type,
    CASE
        WHEN (CASE WHEN CAST(r.trackcd AS int) BETWEEN 10 AND 22
                   THEN r.sibababacd ELSE r.dirtbabacd END) IN ('3','4')
        THEN 'heavy'
        ELSE 'normal'
    END AS baba_group,
    COUNT(*) AS cnt,
    SUM(CASE WHEN ur.kakuteijyuni ~ '^[0-9]+$'
                  AND CAST(ur.kakuteijyuni AS int) = 1
        THEN COALESCE(CAST(o.tanodds AS numeric) / 10.0, 0)
        ELSE 0 END) * 100 AS total_pay
FROM n_uma_race ur
JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
JOIN n_sanku s ON ur.kettonum = s.kettonum
JOIN n_odds_tanpuku o
    ON ur.year = o.year AND ur.monthday = o.monthday
    AND ur.jyocd = o.jyocd AND ur.kaiji = o.kaiji
    AND ur.nichiji = o.nichiji AND ur.racenum = o.racenum
    AND ur.umaban = o.umaban
WHERE ur.datakubun = '7' AND ur.ijyocd = '0'
  AND ur.year BETWEEN %(start)s AND %(end)s
  AND ur.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
  AND o.tanodds ~ '^[0-9]+$'
  AND CAST(o.tanodds AS int) > 0
  AND CAST(r.trackcd AS int) BETWEEN 10 AND 29
GROUP BY s.fnum, track_type, baba_group
HAVING COUNT(*) >= %(sire_min_samples)s
```

---

## JSON出力形式

`data/odds_correction_stats.json` に以下のセクションを追加:

```json
{
  "sire_track_baba_table": {
    "0001234567_turf_heavy": {
      "factor": 1.35,
      "samples": 1520,
      "roi": 0.098,
      "sire_name": "オルフェーヴル"
    },
    "0001234567_dirt_heavy": {
      "factor": 1.72,
      "samples": 636,
      "roi": 0.125
    },
    "turf_heavy": {
      "factor": 0.95,
      "samples": 45000,
      "roi": 0.069
    },
    "dirt_heavy": {
      "factor": 1.16,
      "samples": 68000,
      "roi": 0.084
    }
  }
}
```

**キーの命名規則:**
- 種牡馬個別: `{sire_id}_{track_type}_{baba_group}` (例: `0001234567_dirt_heavy`)
- 全体フォールバック: `{track_type}_{baba_group}` (例: `dirt_heavy`)

---

## evaluator.py での適用ロジック

### `_apply_odds_correction()` への追加

```python
def _apply_sire_track_baba_correction(
    self,
    odds: float,
    sire_id: str,
    track_type: str,
    baba_group: str,
    sire_track_baba_table: dict[str, float],
) -> float:
    """種牡馬×芝ダ×馬場のfactorを適用する."""
    # 完全一致キーを探す
    full_key = f"{sire_id}_{track_type}_{baba_group}"
    factor = sire_track_baba_table.get(full_key)

    if factor is None:
        # フォールバック: コース×馬場の全体factor
        fallback_key = f"{track_type}_{baba_group}"
        factor = sire_track_baba_table.get(fallback_key, 1.0)

    return odds * factor
```

### 必要な情報の取得

evaluator / predictor がレース評価時に以下の情報を取得する必要がある:

| 情報 | 取得方法 | 既存で利用可能か |
|------|---------|---------------|
| 種牡馬ID (fnum) | parquetの `blood_father_id` | Yes（特徴量として既存） |
| コース種別 | parquetの `race_track_type` | Yes |
| 馬場状態 | parquetの `race_baba_cd` | Yes |

**→ 追加のDB問い合わせ不要。** parquetの既存カラムから全て判定可能。

### 馬場グループの判定

```python
def _baba_group(baba_cd: int) -> str:
    """馬場状態コードを2グループに分類する."""
    if baba_cd in (3, 4):
        return "heavy"
    return "normal"
```

---

## フォールバックと後方互換性

| 状況 | 動作 |
|------|------|
| JSONに `sire_track_baba_table` がない（旧JSON） | factor=1.0（補正なし） |
| 種牡馬キーがない | コース×馬場の全体factorにフォールバック |
| コース×馬場の全体キーもない | factor=1.0 |
| `blood_father_id` が parquet にない | factor=1.0 |
| 馬場状態コードが 0 or 不明 | factor=1.0（normalとして扱わない） |

---

## 補正の多段適用における注意

現行の補正は乗算方式で多段適用される。本テーブルの追加により、
同じレース内で以下のfactorが全て乗算される可能性がある:

```
adjusted_odds = raw_odds
    × ninki_factor           (人気順別)
    × jockey_popular_factor  (人気騎手×人気馬)
    × form_popular_factor    (前走好走×人気馬)
    × style_factor           (前走脚質別)
    × post_course_factor     (馬番×コース別)
    × class_change_factor    (昇級/降級)
    × filly_transition_factor (牝馬限定⇔混合)
    × sire_track_baba_factor (種牡馬×芝ダ×馬場)  ← NEW
```

**過補正リスク:**
- 種牡馬×馬場の歪みは、ninki_factorやpost_course_factorと部分的に相関する可能性がある
  （例: 道悪巧者の種牡馬産駒は道悪レースで人気になりやすい → ninki_factorと二重効果）
- **対策:** factorのクリッピングを導入する

### factorクリッピング

```python
# 個別factorの上下限
SIRE_TRACK_BABA_FACTOR_MIN = 0.70
SIRE_TRACK_BABA_FACTOR_MAX = 1.50

factor = max(SIRE_TRACK_BABA_FACTOR_MIN,
             min(SIRE_TRACK_BABA_FACTOR_MAX, raw_factor))
```

**理由:** 分析で観測されたパイロのダート重factor 3.45 等は、
そのまま適用すると過補正になる。factor 1.5 を上限としてリスクを限定する。

---

## 実装計画

### 変更対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| `src/odds_correction_stats.py` | `_calc_sire_track_baba_stats()` 追加、`build_odds_correction_stats()` に組込み |
| `src/model/evaluator.py` | `_apply_odds_correction()` に sire_track_baba_factor 適用ロジック追加 |
| `src/config.py` | `DEFAULT_ODDS_CORRECTION_CONFIG` に `sire_track_baba` のデフォルト値追加 |
| `tests/test_odds_correction.py` | sire_track_baba テーブルのテスト追加 |
| `docs/feature_design.md` | オッズ歪み補正v3セクション追加 |

### `--build-odds-stats` のフロー拡張

```python
# 既存の統計算出の後に追加
sire_track_baba_table = _calc_sire_track_baba_stats(
    year_start, year_end,
    tansho_umaban_col, tansho_pay_col,
    baseline_roi, min_samples,
    sire_min_samples=300,
)
stats["sire_track_baba_table"] = sire_track_baba_table
```

### `load_odds_correction_stats()` の拡張

```python
# sire_track_baba_table のロード
sire_track_baba_raw = stats.get("sire_track_baba_table", {})
sire_track_baba_table: dict[str, float] = {}
for k, v in sire_track_baba_raw.items():
    sire_track_baba_table[k] = v["factor"] if isinstance(v, dict) else float(v)

return {
    ...
    "sire_track_baba_table": sire_track_baba_table,
}
```

---

## 検証計画

### A/Bテスト

| 条件 | 設定 |
|------|------|
| ベースライン | 既存のオッズ補正v2 |
| テスト | v2 + sire_track_baba_factor |
| 評価指標 | value_bet_tansho の回収率 |
| 評価期間 | 2025年（検証データ） |

### 確認項目

1. **全体の回収率変化:** sire_track_baba_factor 追加後にvalue_bet全体の回収率が改善するか
2. **道悪レースの回収率:** 馬場状態が重/不良のレースに限定した回収率
3. **ベット数の変化:** factor追加により過剰なベットが発生していないか
4. **過補正の検出:** 他のfactorとの乗算により極端な補正値になっていないか
5. **世代交代の影響:** 2019-2024統計が2025年でも有効か（種牡馬の引退・新種牡馬の登場）

### コース×馬場の全体factorのみの評価

種牡馬個別factorを入れる前に、まず全体の `track_type × baba_group` factorのみで
回収率改善を確認する（シンプルな4パターンの補正テーブル）。
これで改善が見られた場合に、種牡馬個別factorを追加する段階的アプローチを推奨。

---

## 分析データの保存先

本仕様策定に使用した分析スクリプトと結果:

- **分析スクリプト:** `analysis/sire_track_baba_roi.py`
- **集計期間:** 2019-2024年
- **最小サンプル数:** 100件（分析表示用）

分析の再実行:
```bash
venv/bin/python analysis/sire_track_baba_roi.py
```
