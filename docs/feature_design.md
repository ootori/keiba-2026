# 特徴量設計書

## 概要

EveryDB2のPostgreSQLデータからLightGBMの特徴量を抽出する設計。
すべての特徴量は「予測対象レースの発走前に取得可能な情報」のみで構成する（データリーク防止）。

各カテゴリの詳細（特徴量テーブル・SQL例・算出ロジック）は以下の分割ドキュメントを参照:

- **[カテゴリ詳細（v1: 1〜17）](features/categories_v1.md)** — 現行特徴量の定義・SQL・ロジック
- **[深掘り提案（v2: A〜F）](features/v2_proposals.md)** — 重要度分析に基づく追加特徴量設計

---

## 目的変数

| モード | 目的変数 | 定義 |
|-------|---------|------|
| 二値分類（デフォルト） | `target` | 3着以内=1, 他=0 |
| 二値分類・単勝 | `target_win` | 1着=1, 他=0 |
| LambdaRank | `target_relevance` | 1着=5, 2着=4, 3着=3, 4-5着=1, 6着以下=0 |

**共通の対象条件:** `datakubun='7'`（確定成績）, `ijyocd='0'`（正常出走）, `jyocd` IN ('01'～'10')（JRA中央10場）

---

## 特徴量サマリ（v1: 現行）

| Cat | カテゴリ名 | 数 | 実装ファイル | 主要特徴量 |
|-----|----------|---|-----------|----------|
| 1 | 馬基本属性 | 5 | horse.py | horse_sex, horse_age, horse_tozai |
| 2 | 過去成績 | 13 | horse.py | horse_fukusho_rate, horse_avg_jyuni_last3, horse_last_jyuni |
| 3 | 条件別成績 | 14 | horse.py | horse_turf/dirt_fukusho_rate, horse_same_jyo_rate |
| 4 | スピード指数 | 12 | speed.py | speed_index_last/avg_last3, speed_l3f_last |
| 5 | 脚質 | 7 | speed.py | style_type_mode_last5, style_front_ratio_last5 |
| 6 | レース条件 | 14 | race.py | race_distance, race_track_type, race_baba_cd, race_tosu |
| 7 | 枠順・馬番 | 5 | race.py | post_umaban, post_umaban_norm |
| 8 | 負担重量 | 3 | race.py / horse.py | weight_futan, weight_futan_diff |
| 9 | 馬体重 | 5 | horse.py | bw_weight, bw_change |
| 10 | 騎手 | 8 | jockey_trainer.py | jockey_code, jockey_win_rate_jyo, jockey_avg_ninki_diff |
| 11 | 調教師 | 7 | jockey_trainer.py | trainer_code, trainer_jockey_combo_rate |
| 12 | 調教データ | 7 | training.py | training_hanro_time4, training_count_2weeks |
| 13 | 血統 | 17 | bloodline.py | blood_father_id, blood_bms_id, blood_nicks_rate |
| 14 | 間隔 | 5 | horse.py | interval_days, interval_is_kyuumei |
| 15 | オッズ | 7 | odds.py | odds_tan, odds_ninki（※デフォルト除外） |
| 16 | クロス特徴量 | 8 | pipeline.py | cross_dist_change, cross_weight_futan_per_bw |
| 17 | レース内相対 | 36 | pipeline.py | rel_*_zscore, rel_*_rank（18指標×2） |
| | **現行合計** | **~180** | | |

---

## v2 深掘り提案サマリ（2026-02-20策定、未実装）

重要度分析の課題: ID特徴量（trainer_code, blood_bms_id, blood_father_id, jockey_code）が上位4位を独占 → 条件別分解でID依存を減らし汎化改善。

| 優先順 | 提案 | 追加数 | 実装先 | 概要 |
|-------|------|-------|--------|------|
| 1 | B: BMS条件別 | +6 | bloodline.py | BMS距離帯別/馬場別/競馬場別/父馬齢別/ニックス芝ダ別/父クラス別 |
| 2 | A: 調教師条件別 | +7 | jockey_trainer.py | 芝ダ別/距離帯別/馬場別/直近30日/重賞/休み明け |
| 3 | D: フォームモメンタム | +6 | horse.py | 着順トレンド傾斜/最終勝利日数/連続3着以内/ピーク比較/改善フラグ/昇級初戦 |
| 4 | E: ペース構造 | +5 | pipeline.py | 逃げ先行馬数/予想ペース/脚質×ペース相性/レースレベル |
| 5 | F: 相対特徴量拡張 | +14 | pipeline.py | 馬体重/コンビ成績/斤量比/休養日数/トレンド等のZスコア+ランク |
| 6 | C: 騎手条件別 | +5 | jockey_trainer.py | 芝ダ別/距離帯別/直近30日/重賞 |
| | **v2追加合計** | **+43** | | **→ 約223特徴量** |

詳細（SQL例・算出ロジック・実装方針）は **[docs/features/v2_proposals.md](features/v2_proposals.md)** を参照。

---

## レース内相対特徴量の設計ポイント

カテゴリ17は特に重要（重要度5〜10位を占める）なため、設計上の要点をここに記載する。

**missing_type の3パターン:**

| missing_type | 0.0の意味 | 対象例 |
|-------------|----------|-------|
| `numeric` | 有効な数値 | speed_index_*, horse_avg_jyuni_* |
| `rate` | 「勝率0%」= 正当な値 | horse_fukusho_rate, jockey_win_rate_year |
| `blood` | データ不足 = 欠損 | blood_father_turf_rate, blood_nicks_rate |

**重要:** rate系で0.0をNaN化すると弱い馬のZスコアが平均に引き上げられ、回収率5%低下の実績あり。missing_typeの設定は慎重に。

---

## 欠損値の処理方針

| 状況 | 処理 |
|------|------|
| 新馬（過去成績なし） | 数値=-1、率系=0.0 |
| 調教データ不明 | -1 |
| 海外遠征歴 | 対象外（JRA10場のみ集計） |
| 騎手コード不明 | 全体平均で代替 |
| 血統不明 | "unknown"カテゴリ |

LightGBM: `use_missing=true`, `zero_as_missing=false` で -1 を欠損マーカーとして扱う。

---

## 特徴量重要度による選別（実装後）

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
| 17. レース内相対 | src/features/pipeline.py | FeaturePipeline._add_relative_features() |
