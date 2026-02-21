# 競馬予想プロジェクト - CLAUDE.md

## プロジェクト概要

JRA競馬の着順予測を行うLightGBMベースの予想システム。
EveryDB2（PostgreSQL）に蓄積されたJRA-VAN DataLabデータを特徴量に変換し、
各馬の着順（3着以内確率）を予測する。

## 技術スタック

- **言語:** Python 3.10+
- **ML:** LightGBM
- **DB:** PostgreSQL（EveryDB2で構築済み、Dockerコンテナ）
- **DB接続:** psycopg2（直接接続）
  - SQLAlchemyは不使用。理由: プロジェクト全体でpsycopg2ネイティブの `IN %(tuple)s` タプル展開を多用しており、SQLAlchemy `text()` では代替困難なため
  - pandasの `pd.read_sql()` に psycopg2 connectionを渡す際のUserWarningは `warnings.filterwarnings` で抑制済み
- **DB接続先:** `localhost:5432`, DB名=`everydb`, ユーザ=`webmaster`, パスワード=`devpassword`
- **データ処理:** pandas, numpy
- **評価:** scikit-learn

## ディレクトリ構成

```
everydb2/
├── CLAUDE.md                          # このファイル（プロジェクト定義）
├── everydb2_database_reference.md     # DBテーブル定義・コード表リファレンス
├── requirements.txt                   # Python依存パッケージ
├── .gitignore
├── run_train.py                       # 学習実行スクリプト（CLIエントリポイント）
├── run_predict.py                     # 予測実行スクリプト
├── docs/
│   ├── feature_design.md              # 特徴量設計書（インデックス）
│   ├── implementation_plan.md         # 実装計画書
│   └── features/
│       ├── categories_v1.md           # v1特徴量カテゴリ詳細（Cat 1〜17）
│       └── v2_proposals.md            # v2深掘り提案詳細（A〜F）
├── src/
│   ├── config.py                      # DB接続設定・定数定義
│   ├── db.py                          # DB接続ユーティリティ（psycopg2直接接続）
│   ├── features/
│   │   ├── __init__.py
│   │   ├── base.py                    # 特徴量抽出の基底クラス
│   │   ├── horse.py                   # 馬の基本属性・過去成績（カテゴリ1,2,3,9,14,8一部）
│   │   ├── race.py                    # レース条件（カテゴリ6,7,8）
│   │   ├── jockey_trainer.py          # 騎手・調教師（カテゴリ10,11）
│   │   ├── speed.py                   # スピード指数+脚質（カテゴリ4,5）
│   │   ├── training.py               # 調教データ（カテゴリ12）
│   │   ├── bloodline.py              # 血統（カテゴリ13）
│   │   ├── odds.py                   # オッズ・人気（カテゴリ15）
│   │   ├── mining.py                # JRA-VANデータマイニング予想（カテゴリ18: サプリメント）
│   │   ├── bms_detail.py            # BMS条件別パフォーマンス（v2提案B: サプリメント）
│   │   ├── supplement.py            # 差分特徴量（サプリメント）パイプライン
│   │   └── pipeline.py              # 特徴量パイプライン統合+クロス特徴量（カテゴリ16）
│   ├── model/
│   │   ├── __init__.py
│   │   ├── trainer.py                # LightGBMの学習（二値分類+LambdaRank対応）
│   │   ├── predictor.py              # 予測実行（メタデータからranking自動検出）
│   │   └── evaluator.py             # モデル評価・回収率シミュレーション（NDCG対応）
│   ├── odds_correction_stats.py      # オッズ歪み補正統計（DB算出 → JSON保存/ロード）
│   └── utils/
│       ├── __init__.py
│       ├── code_master.py            # コード表変換ユーティリティ
│       └── base_time.py              # 基準タイムテーブル（スピード指数用）
├── notebooks/
│   └── exploration.ipynb             # データ探索用ノートブック
├── models/                           # 学習済みモデル保存先（*.txt, *_features.txt, *_meta.json）
├── data/                             # 中間データキャッシュ（*.parquet, base_time_table.csv）
│   └── supplements/                  # サプリメント（差分特徴量）保存先
│       ├── mining_2024.parquet       #   マイニング特徴量（年度別）
│       ├── bms_detail_2024.parquet   #   BMS条件別特徴量（年度別）
│       └── ...
├── data/odds_correction_stats.json   # オッズ歪み補正統計（--build-odds-stats で生成）
└── tests/
    ├── test_features.py              # 30テスト（pytest）
    ├── test_mining_supplement.py     # マイニング・サプリメントテスト（16テスト）
    ├── test_bms_detail_supplement.py # BMS条件別サプリメントテスト（10テスト）
    └── test_odds_correction.py      # オッズ歪み補正テスト（102テスト）
```

## データベース接続

### テーブル・カラム命名規則
- 蓄積系: `n_テーブル名` — 過去データ
- 速報系: `s_テーブル名` — 当日データ
- **EveryDB2はPostgreSQL上にテーブルを作成する際、テーブル名・カラム名のケースはDBMS依存。**
  PostgreSQLではデフォルトで小文字に正規化されるため、通常は小文字でアクセスできる。
  ただし、もし大文字で作成されている場合は `"Year"` のようにダブルクォートが必要。
- **★実装前に必ず `\dt n_*` と `\d n_race` で実際のテーブル名・カラム名のケースを確認すること。**

### 重要な注意事項
- **全カラムがvarchar型。** 数値は必ず `CAST(col AS integer)` や `CAST(col AS numeric)` でキャストする
- **レースキー:** `year, monthday, jyocd, kaiji, nichiji, racenum` の6カラムで結合
  （※実際のカラム名のケースはDB確認後に確定。本ドキュメントでは小文字表記で統一）
- **血統登録番号（kettonum）:** 馬を一意に特定する10桁キー
- **datakubun = '7'** が確定成績データ。予測時は出馬表段階のデータ（datakubun='1' or '2'）を使う
- **コード表**は `everydb2_database_reference.md` の「10. コード表」セクションを参照

### DBリファレンスと実際のカラム名の差異（実装で判明した注意事項）

| リファレンス記載 | 実際のDBカラム名 | 所在テーブル | 備考 |
|---------------|---------------|-----------|------|
| `TimeDIFN` | `timediff` | n_uma_race | タイム差カラム。リファレンスは`TimeDIFN`だがDBでは`timediff` |
| `DataKubun` | *(存在しない)* | n_odds_tanpuku | 明細テーブルにはDataKubunなし。ヘッダ(`n_odds_tanpukuwaku_head`)にのみ存在 |
| `PayTansyo*` 等 | *(動的検出)* | n_harai | 単勝/複勝/馬連/馬単/三連複/三連単の払戻カラムの命名規則がバージョン依存のため、`evaluator.py`・`odds_correction_stats.py`ではスキーマから動的検出（`umaban` または `kumi` キーワードで探索） |
| `TanNinki` | `tanninki` | n_odds_tanpuku | 単勝人気順。値に `--` 等の非数値が含まれる場合があり、CASTの前に `~ '^[0-9]+$'` のフィルタが必要 |
| `TanOdds` | `tanodds` | n_odds_tanpuku | 単勝オッズ（10倍値）。同上、非数値データが含まれる |
| `KakuteiJyuni` | `kakuteijyuni` | n_uma_race | 確定着順。同上、非数値データが含まれる場合がある |

### テーブル別 DataKubun の扱い

| テーブル | DataKubunあり | 値の意味 |
|---------|-------------|---------|
| n_uma_race | ○ | '7'=確定成績 |
| n_race | ○ | '7'=確定 |
| n_harai | ○ | '1'=速報(払戻確定), '2'=成績, '9'=中止, '0'=削除 |
| n_odds_tanpuku | **×** | 明細テーブルにはなし（最新データで上書き） |
| n_odds_tanpukuwaku_head | ○ | '1'=中間, '2'=前日売最終, '3'=最終, '4'=確定 |

### 主要テーブル（予測に使用）

| テーブル | 用途 | 結合キー |
|---------|------|---------|
| n_race | レース条件取得 | レースキー |
| n_uma_race | 馬毎成績・出走情報 | レースキー + umaban/kettonum |
| n_uma | 競走馬マスタ（血統） | kettonum |
| n_kisyu | 騎手マスタ | kisyucode |
| n_kisyu_seiseki | 騎手成績 | kisyucode |
| n_chokyo | 調教師マスタ | chokyosicode |
| n_chokyo_seiseki | 調教師成績 | chokyosicode |
| n_hanro | 坂路調教タイム | kettonum + chokyodate |
| n_wood_chip | ウッドチップ調教タイム | kettonum + chokyodate |
| n_odds_tanpuku | 単複オッズ | レースキー + umaban |
| n_harai | 払戻（回収率計算用） | レースキー |
| n_hansyoku | 繁殖馬マスタ（血統系統） | hansyokunum |
| n_sanku | 産駒マスタ（4代血統） | kettonum |
| n_keito | 系統情報 | hansyokunum |

## 予測タスク定義

### 問題設定
- **入力:** 1レースの全出走馬の特徴量
- **出力（二値分類モード）:** 各馬の3着以内確率（LightGBMの二値分類）
- **出力（二値分類・単勝モード）:** 各馬の1着確率（`--target win` 指定時）
- **出力（LambdaRankモード）:** 各馬のランキングスコア（高いほど上位予測）
- **目的変数（二値分類・top3）:** `target` — `KakuteiJyuni` が 1, 2, 3 なら 1、それ以外は 0
- **目的変数（二値分類・win）:** `target_win` — `KakuteiJyuni` が 1 なら 1、それ以外は 0
- **目的変数（LambdaRank）:** `target_relevance` — 関連度スコア（1着=5, 2着=4, 3着=3, 4着=2, 5着=1, 6着以下=0）
- **データリーク防止:** 特徴量は必ず「当該レースより過去のデータ」のみで構成

### 学習/評価の時間分割
- 学習: 2015-2024年（10年分）
- 検証: 2025年
- 時系列で分割し、未来のデータが学習に混入しないことを保証
- **一括構築・分割方式:** 特徴量構築は学習期間〜検証年（2015-2025）を一括で `build_years()` に渡し、構築後に `_key_year` で学習/検証に分割する。これにより検証年も `--workers N` の並列構築対象となる。また本番運用時（例: 2016-2025学習 → 2026予測）でも同一フォーマットのparquetを再利用できる

## 特徴量カテゴリ（概要）

詳細は `docs/feature_design.md` を参照。

1. **馬基本属性(5):** 性別, 馬齢, 品種, 毛色, 東西所属
2. **過去成績(13):** 着順, 勝率, 連対率, 複勝率（全走/直近N走）
3. **条件別成績(14):** 芝ダ別, 距離帯別, 場別, 馬場状態別, 重賞成績
4. **スピード指数(12):** 走破タイム, 上がり3F, タイム差, スピード指数
5. **脚質(7):** 逃げ/先行/差し/追込傾向, コーナー通過順平均
6. **レース条件(14):** 距離, トラック, 馬場状態, 天候, グレード, 頭数, 月
7. **枠順・馬番(5):** 枠番, 馬番, 正規化馬番, 内外フラグ
8. **負担重量(3):** 斤量, 前走差, レース内平均差
9. **馬体重(5):** 当日体重, 増減, 適正体重差, 大幅増減フラグ
10. **騎手(8):** 勝率, 複勝率, 場別成績, 乗替フラグ, 期待値偏差
11. **調教師(7):** 勝率, 複勝率, 場別成績, 騎手×調教師コンビ
12. **調教(7):** 坂路タイム, ウッドチップタイム, 調教強度, 本数
13. **血統(17):** 父系統, 母父系統, 母系統, 芝ダ適性, 距離適性, 馬場状態別適性, 競馬場別適性, ニックス（父×母父相性）, 母産駒成績, 近交フラグ・世代
14. **間隔(5):** 中N日, 休み明け区分, ローテーション
15. **オッズ(7):** 単勝オッズ, 複勝オッズ, 人気順（※予測タイミング依存）
16. **クロス特徴量(10):** 距離変更, 芝ダ変更, クラス変更, 斤量/体重比, 前走牝馬限定フラグ, 現在レース牝馬限定フラグ
17. **レース内相対特徴量(36):** 主要能力指標のレース内Zスコア・ランク（血統4指標追加）
18. **マイニング予想(7):** DM予想タイム, DM予想順位, DM誤差幅, 対戦型スコア（サプリメント）
19. **BMS条件別(6):** BMS距離帯別/馬場別/競馬場別/父馬齢別/ニックス芝ダ別/父クラス別（サプリメント）

合計 **約194特徴量**（v1: ~182 + マイニング: 7 + BMS条件別: 6 ※サプリメント含む。クロス特徴量8→10で+2。詳細は `docs/feature_design.md` 参照）

### v2 深掘り提案（2026-02-20策定）

重要度分析に基づく追加特徴量（詳細は `docs/feature_design.md` の「特徴量深掘り提案（v2）」セクション参照）:

| 提案 | 追加数 | 実装先 | 状態 | 概要 |
|-----|-------|--------|------|------|
| B: BMS条件別 | +6 | bms_detail.py（サプリメント） | **実装済み** | BMS距離帯別/馬場別/競馬場別/父馬齢別/ニックス芝ダ別/父クラス別 |
| A: 調教師条件別 | +7 | jockey_trainer.py | 未実装 | 芝ダ別/距離帯別/馬場別/直近30日/重賞/休み明け |
| D: フォームモメンタム | +6 | horse.py | 未実装 | 着順トレンド傾斜/最終勝利日数/連続3着以内/ピーク比較/改善フラグ/昇級初戦 |
| E: ペース構造 | +5 | pipeline.py | 未実装 | 逃げ先行馬数/予想ペース/脚質×ペース相性/レースレベル |
| F: 相対特徴量拡張 | +14 | pipeline.py | 未実装 | 馬体重/コンビ成績/斤量比/休養日数/トレンド等のZスコア+ランク |
| C: 騎手条件別 | +5 | jockey_trainer.py | 未実装 | 芝ダ別/距離帯別/直近30日/重賞 |

実装優先順位: ~~B~~ → A → D → E → F → C

## コーディング規約

- Python 3.10+、型ヒント必須
- docstringはGoogle style
- SQLは読みやすさ重視で大文字キーワード
- 特徴量名は `{カテゴリ}_{内容}_{集計方法}` 形式（例: `horse_win_rate_last5`）
- テスト: pytest
- pandasのSettingWithCopyWarningが出ないよう `.copy()` を適切に使用

## コード表の扱い

コード値はそのまま数値として使える場合（馬場状態: 1-4）はそのまま使い、
カテゴリ変数（競馬場コード、トラックコードなど）は
LightGBMのcategorical_feature機能を活用する。

```python
# カテゴリ変数の指定例
categorical_features = [
    'race_jyo_cd',        # 競馬場コード
    'race_track_cd',      # トラックコード
    'horse_sex_cd',       # 性別
    'horse_tozai_cd',     # 東西所属
    'horse_father_keito', # 父系統
    'horse_bms_keito',    # 母父系統
]
```

## CLIオプション（run_train.py）

```bash
# フル実行（特徴量構築 → 学習 → 評価 → 回収率シミュレーション）
# ※デフォルトでオッズ特徴量は除外される
python run_train.py

# 期間を指定して実行（短時間テスト用）
python run_train.py --train-start 2024 --train-end 2024 --valid-year 2025

# 特徴量構築のみ（parquet保存まで）
python run_train.py --build-features-only

# 4並列で特徴量構築（学習期間〜検証年を一括並列実行）
python run_train.py --build-features-only --workers 4

# 既存parquetを無視して全年度を再構築（8並列）
python run_train.py --build-features-only --workers 8 --force-rebuild

# 特定年度だけ再構築（他の年はスキップ）
python run_train.py --build-features-only --train-start 2020 --train-end 2020 --valid-year 2020 --force-rebuild

# 本番運用向け: 2016-2025で学習し2026年を予測する場合の特徴量構築
python run_train.py --build-features-only --train-start 2016 --train-end 2025 --valid-year 2025 --workers 4

# 既存特徴量からモデル学習のみ
python run_train.py --train-only

# 評価・回収率シミュレーションのみ（Step 4-5、保存済みモデル+特徴量を使用）
python run_train.py --eval-only

# オッズ特徴量を明示的に含める（非推奨：データリークの危険あり）
python run_train.py --with-odds

# モデル名を指定
python run_train.py --model-name my_model

# LambdaRank（ランキング学習）モードで学習
# ※parquetにtarget_relevanceカラムが必要（--force-rebuildで再構築）
python run_train.py --ranking --model-name ranking_model

# LambdaRank + 既存特徴量から学習のみ
python run_train.py --ranking --train-only --model-name ranking_model

# LambdaRank モデルの評価のみ（メタデータからranking=Trueが自動検出される）
python run_train.py --eval-only --model-name ranking_model

# 1着予測モデルで学習（value_bet戦略向け）
# ※parquetにtarget_winカラムが必要（--force-rebuildで再構築）
python run_train.py --target win --model-name win_model

# 1着予測モデル + 既存特徴量から学習のみ
python run_train.py --target win --train-only --model-name win_model

# 1着予測モデルの評価のみ（メタデータからtarget_type=winが自動検出される）
python run_train.py --eval-only --model-name win_model

# === サプリメント（差分特徴量）===
# マイニング特徴量をサプリメントとして構築（学習期間〜検証年を一括構築）
python run_train.py --build-supplement mining

# 4並列でサプリメント構築
python run_train.py --build-supplement mining --workers 4

# 既存サプリメントを再構築
python run_train.py --build-supplement mining --force-rebuild

# マイニング特徴量をマージして学習（既存parquet + サプリメント）
python run_train.py --train-only --supplement mining

# マイニング特徴量をマージして評価
python run_train.py --eval-only --supplement mining

# フル実行 + マイニングサプリメント
python run_train.py --supplement mining

# BMS条件別特徴量をサプリメントとして構築
python run_train.py --build-supplement bms_detail --force-rebuild

# BMS条件別をマージして学習
python run_train.py --train-only --supplement bms_detail

# 複数サプリメントを同時にマージ
python run_train.py --train-only --supplement mining bms_detail

# === オッズ歪み補正 ===
# 統計データ構築（直近3年分、デフォルト2022-2024）
python run_train.py --build-odds-stats

# 期間指定
python run_train.py --build-odds-stats --odds-stats-start 2020 --odds-stats-end 2024

# 統計ベースの補正で評価（JSONから自動ロード）
python run_train.py --eval-only --odds-correction

# サプリメント + オッズ補正の組み合わせ
python run_train.py --eval-only --supplement mining bms_detail --odds-correction

# 本番予測でオッズ補正付きEV情報を表示
python run_predict.py --year 2025 --monthday 0622 --all-day --odds-correction
```

## 特徴量の年度別保存と並列構築

特徴量は年度別の parquet ファイル（`data/features_{year}.parquet`）に保存される。
これにより以下の利点がある:

- **差分再構築:** 特徴量設計を変更した場合、変更が必要な年度だけ `--force-rebuild` で再構築できる
- **並列構築:** `--workers N` で N 年分を同時に構築でき、学習+検証の全年度（例: 11年分）の構築時間を大幅に短縮
- **増分追加:** 新年度のデータが追加された場合、その年度だけ構築すればよい

```
data/
├── features_2015.parquet    # 年度別（現行方式）
├── features_2016.parquet
├── ...
├── supplements/             # サプリメント（差分特徴量）
│   ├── mining_2015.parquet  #   マイニング特徴量
│   ├── mining_2016.parquet
│   ├── bms_detail_2015.parquet  # BMS条件別特徴量
│   ├── bms_detail_2016.parquet
│   └── ...
├── features_2025.parquet
├── train_features.parquet   # 旧方式（後方互換、--train-only/--eval-only のフォールバック用）
└── valid_features.parquet   # 旧方式
```

**並列化の仕組み:**
- `ProcessPoolExecutor` で年度ごとに独立したプロセスを起動
- 各プロセスで `FeaturePipeline` を新規生成し、独自の DB 接続を使用
- `psycopg2` は `query_df()` 呼び出しごとに接続を生成・破棄するためプロセス間の競合なし

**API:**
- `FeaturePipeline.build_years(year_start, year_end, workers=N)` — 並列構築
- `FeaturePipeline.load_years(year_start, year_end, supplement_names=["mining"])` — 年度別 parquet を結合ロード（サプリメントマージ対応）
- `FeaturePipeline.build_year(year)` — 単年度構築（直列用）

## サプリメント（差分特徴量）システム

新しい特徴量を追加する際、メインの parquet を全年度再構築する必要がないよう、
差分特徴量（サプリメント）として別ファイルに保存し、学習/評価時にマージする仕組み。

**メリット:**
- メイン parquet の再構築不要（構築に数時間かかる場合でも、サプリメントは独立に高速構築）
- 複数のサプリメントを自由に組み合わせて実験可能
- サプリメントの追加・削除がメインパイプラインに影響しない

**仕組み:**
- サプリメントは `data/supplements/{name}_{year}.parquet` に保存
- 各 parquet はレースキー + kettonum をキーとして持つ
- `load_years()` や `merge_supplements()` で自動的に left join
- 同名カラムがある場合はサプリメント側で上書き

**サプリメント登録簿（supplement.py）:**
- `mining`: JRA-VANデータマイニング予想特徴量（MiningFeatureExtractor）
- `bms_detail`: BMS条件別パフォーマンス特徴量（BMSDetailFeatureExtractor）
- 新しいサプリメントを追加するには `_get_registry()` に登録

**API:**
- `build_supplement_years(name, year_start, year_end, workers=N)` — サプリメント構築
- `load_supplement_years(name, year_start, year_end)` — サプリメントロード
- `merge_supplements(main_df, names, year_start, year_end)` — メインDFにマージ

### マイニング特徴量（カテゴリ18）

JRA-VANが提供するデータマイニング予想を特徴量として活用する。

| 特徴量名 | 説明 | データソース |
|---------|------|------------|
| `mining_dm_time` | DM予想走破タイム（秒） | n_uma_race.dmtime / n_mining |
| `mining_dm_jyuni` | DM予想順位 | n_uma_race.dmjyuni |
| `mining_dm_gosa_range` | DM予想誤差幅（信頼度指標） | dmgosap + dmgosam |
| `mining_dm_gosa_p` | DM予想誤差（+側） | n_uma_race.dmgosap |
| `mining_dm_gosa_m` | DM予想誤差（-側） | n_uma_race.dmgosam |
| `mining_dm_kubun` | DM区分（1=前日, 2=当日, 3=直前） | n_uma_race.dmkubun |
| `mining_tm_score` | 対戦型マイニングスコア（0〜100） | n_taisengata_mining |

**注意:**
- DMKubun が 1=前日、2=当日、3=直前。データリーク防止の観点からタイミングに注意
- n_uma_race の DM カラムを優先取得し、n_mining テーブルで補完
- n_mining / n_taisengata_mining は横持ちデータのため、スキーマから動的に縦持ちに変換

### BMS条件別特徴量（カテゴリ19 / v2提案B）

blood_bms_id（重要度2位）が持つ情報を条件別に展開するサプリメント。
father側の6条件別特徴量に対し、BMS側は2つのみという非対称を解消する。

| 特徴量名 | 説明 | データソース |
|---------|------|------------|
| `blood_bms_dist_rate` | 母父産駒の今回距離帯での複勝率（過去3年） | n_sanku + n_uma_race + n_race |
| `blood_bms_baba_rate` | 母父産駒の今回馬場状態での複勝率（過去3年） | n_sanku + n_uma_race + n_race |
| `blood_bms_jyo_rate` | 母父産駒の今回競馬場での複勝率（過去3年） | n_sanku + n_uma_race + n_race |
| `blood_father_age_rate` | 父産駒の同馬齢での複勝率（過去5年） | n_sanku + n_uma_race |
| `blood_nicks_track_rate` | 父×母父ニックスの芝/ダート別複勝率（過去5年） | n_sanku + n_uma_race + n_race |
| `blood_father_class_rate` | 父産駒のクラス別成績（過去3年） | n_sanku + n_uma_race + n_race |

**ノイズ抑制策:**
- 最小サンプル数閾値: `MIN_SAMPLES=20`（一般）, `MIN_SAMPLES_NICKS=30`（ニックス）
- サンプル不足の場合は NaN を返す（`MISSING_RATE=0.0` ではなく LightGBM ネイティブ欠損）
- これにより「データなし」と「複勝率0%」を区別し、小サンプル由来のノイジーな率（1/1=100%等）を除去

**クラス分類（blood_father_class_rate用）:**
- GradeCD が A-D → `"grade"`（重賞）
- JyokenCD5 >= 500 → `"open"`（オープン）
- それ以外 → `"jouken"`（条件戦）

## LightGBM categorical_feature の注意

学習時に `categorical_feature` として指定したカラムは、予測時にも同一の `category` 型に変換する必要がある。
`trainer.py` で学習時に変換し、`evaluator.py` / `predictor.py` で予測時にも同じ変換を適用している。
この変換が抜けると `train and valid dataset categorical_feature do not match` エラーになる。

## 注意事項・制約

- **出走取消馬の除外:** `IJyoCD != '0'` の馬は訓練・予測から除外する
- **地方競馬の除外:** `JyoCD` が 01-10（JRA中央10場）のみ対象
- **新馬戦の扱い:** 過去成績なしのため別途ロジックが必要（デフォルト値 or 血統ベース）
- **2001年以降データ:** 馬齢表記が満年齢に変更されているため、2001年以降のみ使用推奨
- **DataKubun確認:** 学習データは必ず `DataKubun = '7'`（確定成績）のレコードを使用
- **n_odds_tanpuku にDataKubunなし:** オッズ明細テーブルは最新データで上書きされるため、フィルタ不要。**この仕様により確定オッズがリークするため、オッズ特徴量はデフォルトで除外している（`--with-odds` で明示的に含めない限り使用しない）**
- **n_harai のカラム名:** EveryDB2バージョンにより命名規則が異なる可能性あり。`evaluator.py` ではスキーマから動的検出する方式を採用
- **馬番のゼロパディング:** n_haraiの馬番は2桁ゼロ埋め（"01", "02"）だが、予測側のpost_umabanはint由来（"1", "2"）。`evaluator.py` の `_format_umaban()` でゼロパディング変換を行い、払戻データと正しくマッチさせている
- **組番フォーマット（馬連/馬単/三連複/三連単）:** n_haraiの組番は2桁ゼロ埋め馬番の連結（馬連/馬単: 4桁 "0102"、三連複/三連単: 6桁 "010203"）。馬連・三連複はソート済み（小さい番号が先）、馬単・三連単は着順通り
- **特徴量の train/valid 不整合防止:** `trainer.py` では train_df と valid_df の両方に存在するカラムのみを特徴量として使用する。parquet の再構築タイミング差でカラム不整合が起きた場合はWARNINGログで通知。全特徴量を使いたい場合は `--force-rebuild` で全年度を再構築すること
- **レース内相対特徴量のparquet依存:** 相対特徴量（`rel_*`）は `pipeline.py` の `_add_relative_features()` で構築時に計算される。既存 parquet には含まれないため、この特徴量を使うには対象年度の parquet を `--force-rebuild` で再構築する必要がある
- **LambdaRank の parquet 依存:** LambdaRank モード（`--ranking`）には `target_relevance` と `kakuteijyuni` カラムが必要。これらは `_get_target()` で構築時に生成される。旧 parquet には含まれないため、`--force-rebuild` で再構築が必要
- **1着予測モデルの parquet 依存:** `--target win` モードには `target_win` カラムが必要。`_get_target()` で構築時に生成される。旧 parquet には含まれないため、`--force-rebuild` で再構築が必要
- **1着予測モデルのメタデータ:** `trainer.py` はモデル保存時に `{name}_meta.json` に `target_type` フラグ（"top3" or "win"）を記録する。`--eval-only` 実行時にこのメタデータから自動的に目的変数を検出する
- **LambdaRank のモデルメタデータ:** `trainer.py` はモデル保存時に `{name}_meta.json` を出力し、`ranking` フラグを記録する。`--eval-only` 実行時にこのメタデータから自動的に LambdaRank モードを検出する
- **LambdaRank の group パラメータ:** LambdaRank では同一レースの馬が連続している必要がある。`trainer.py` の `_prepare_groups()` でレースキーによるソートとグループサイズ計算を行う
- **value_bet のモデルタイプ別正規化:** `value_bet` 戦略は `target_type` と `ranking` に応じて適切な確率変換を適用する:
  - `target_type="win"`: P(win)モデルの生出力をレース内合計で割る ratio 正規化（`_ratio_normalize_group`）
  - `target_type="top3"` + `ranking=False`: レース内合計で割る ratio 正規化（`_ratio_normalize_group`）
  - `ranking=True`: softmax 正規化（`_softmax_normalize_group`）
  - ※ LightGBM の sigmoid 出力はレース内で合計 1.0 にならないため、全モデルタイプで正規化が必要
  - ※ 旧方式の線形正規化（min-shift）は最下位馬が常に確率0になる問題があり、value_betでは使用しない
- **value_bet のEV閾値とベット上限:** デフォルトで `ev_threshold=1.2`（マージン確保）、`max_bets_per_race=3`（ベット数膨張防止）。`value_bet_config` 辞書で設定変更可能
- **value_bet のオッズ取得:** `value_bet` 戦略は特徴量の `odds_tan` を優先し、欠損時（オッズ特徴量未使用時）は `n_odds_tanpuku` テーブルからレース単位で単勝オッズを取得する。これにより `--with-odds` なしでも value_bet が動作する
- **MISSING_RATE=0.0 の特徴量ごとの意味の違い:** `_add_relative_features()` では `missing_type` パラメータで欠損値処理を3パターンに分類している。rate系特徴量（勝率・複勝率等）では0.0は「0%」という正当な値であり、NaN化すると弱い馬のZスコアが平均に引き上げられ性能が低下する。新しい相対特徴量を追加する際は `missing_type` を必ず適切に設定すること（"numeric"/"rate"/"blood"）
- **サプリメント特徴量の欠損値戦略:** `bms_detail` サプリメントでは `MISSING_RATE=0.0` ではなく `NaN`（LightGBMネイティブ欠損）を使用している。率系特徴量（複勝率）で「サンプル数不足（MIN_SAMPLES未満）でデータなし」と「本当に複勝率0%」を区別するため。新しいサプリメントで率系特徴量を追加する場合も、同様に最小サンプル数閾値 + NaN 方式を推奨
- **オッズ補正v2のparquet依存:** `cross_class_change`（修正版）, `cross_prev_filly_only`, `cross_current_filly_only` はパイプライン構築時に生成される。旧parquetでは `cross_class_change` が常に0、他2つは存在しないため、v2補正ルール（クラス変更・牝馬限定遷移）を有効にするには `--force-rebuild` でparquetを再構築する必要がある
- **高カーディナリティIDの特徴量除外:** `blood_mother_id`（母馬繁殖登録番号）は数千種のユニーク値を持つため特徴量から削除した。カテゴリ変数では過学習し、数値変数ではID番号の大小に意味がないため分割が無意味になる。母馬の情報は `blood_mother_keito`（母系統）と `blood_mother_produce_rate`（母産駒成績）でカバーしている。同様に、LightGBMの `categorical_feature` に指定するのはユニーク数が数十〜100程度までのカラムに限定すること

## 回収率シミュレーション戦略

`evaluator.py` の `simulate_return()` で使用可能な戦略:

| 戦略名 | 賭式 | ロジック |
|--------|------|---------|
| `top1_tansho` | 単勝 | 予測1位の単勝を購入 |
| `top1_fukusho` | 複勝 | 予測1位の複勝を購入 |
| `top3_fukusho` | 複勝 | 予測Top3の複勝を各100円購入 |
| `top2_umaren` | 馬連 | 予測Top2の馬連を購入 |
| `top2_umatan` | 馬単 | 予測Top2の馬単を購入（1位→2位の順） |
| `top3_sanrenpuku` | 三連複 | 予測Top3の三連複を購入 |
| `top3_sanrentan` | 三連単 | 予測Top3の三連単を購入（1位→2位→3位の順） |
| `value_bet_tansho` | 単勝 | 期待値ベースの購入（モデルタイプ別正規化後、確率×単勝オッズ≧ev_threshold の馬をEV降順で最大max_bets頭まで単勝購入。オッズは特徴量優先、欠損時はn_odds_tanpukuから取得） |
| `value_bet_umaren` | 馬連 | 期待値ベースの購入（上記と同じEV条件で選出した馬が2頭以上の場合、全組み合わせの馬連を購入） |

**n_harai からの払戻データ取得:**
`_get_harai_data()` は6賭式（tansyo/fukusyo/umaren/umatan/sanren/sanrentan）の払戻カラムをスキーマから動的検出する。検出時は `sanrentan` を `sanren` より先に検出することで部分一致の誤マッチを防いでいる。

**オッズ歪み補正（`--odds-correction`）:**
value_bet戦略のEV計算前にオッズを補正ルールで調整する。`adjusted_odds = raw_odds × Π(correction_factors)` で補正後オッズを算出し、`EV = pred_prob × adjusted_odds` で購入判断する。

**統計データ駆動の factor 算出:**
`python run_train.py --build-odds-stats` でDBから過去レースの回収率統計を算出し、`data/odds_correction_stats.json` に保存する。`--odds-correction` 使用時にJSONから自動ロードされる。

- **baseline ROI:** 全馬に100円ずつ単勝を買った場合の回収率（テイク率の基準線）
- **factor 算出:** `factor = category_roi / baseline_roi`（factor < 1.0 → 過大評価、> 1.0 → 過小評価）
- **最小サンプル閾値:** 1000件未満は factor = 1.0（補正なし）

補正は多段階で適用される（乗算）:

1. **人気順別テーブル（ninki_table）:** 人気1位〜18位それぞれの単勝回収率から個別 factor を算出。人気馬の割引も人気薄の上乗せもデータ駆動で自動算出される
2. **個別ルール（v1）:** 人気騎手・前走好走の追加補正
3. **前走脚質別テーブル（style_table）:** 前走脚質区分（逃げ/先行/差し/追込）ごとのROIベースfactor（v2）
4. **馬番×コース別テーブル（post_course_table）:** 馬番グループ×コースカテゴリ別のROIベースfactor（v2）
5. **クラス変更ルール:** 昇級/降級時の補正（v2）
6. **牝馬限定⇔混合遷移ルール:** 牝馬限定戦↔混合戦の移行時の補正（v2）
7. **調教師×人気帯別テーブル（trainer_ninki_table）:** 調教師コード×人気帯(A-D)別のROIベースfactor（v3）。戦略的厩舎（人気薄で期待以上の成績を出す厩舎）の検出に基づく
8. **父系統×サーフェス別テーブル（sire_surface_table）:** 父系統×サーフェス(洋芝/芝/ダート)別のROIベースfactor（v3）
9. **父系統×距離帯別テーブル（sire_distance_table）:** 父系統×距離帯(sprint/mile/middle/long)別のROIベースfactor（v3）

### 個別ルール一覧

| ルール名 | 条件 | 説明 | バージョン |
|---------|------|------|----------|
| `jockey_popular_discount` | 騎手勝率≧15% かつ 人気≦3位 | 人気騎手×人気馬は過大評価 | v1 |
| `form_popular_discount` | 前走3着以内 かつ 人気≦3位 | 前走好走の人気馬は過大評価 | v1 |
| `odd_gate_discount` | 馬番が奇数 | 奇数ゲートは回収率が低い（レガシー: post_course_tableがない場合のフォールバック） | v1 |
| `even_gate_boost` | 馬番が偶数 | 偶数ゲートは過小評価されがち（レガシー: 同上） | v1 |
| `class_upgrade` | cross_class_change > 0 | 昇級馬は過大評価される傾向 | v2 |
| `class_downgrade` | cross_class_change < 0 | 降級馬は過小評価される傾向 | v2 |
| `filly_to_mixed` | 牝馬 + 前走牝限 + 今走混合 | 牝馬限定戦から混合戦への移行は過大評価 | v2 |
| `mixed_to_filly` | 牝馬 + 前走混合 + 今走牝限 | 混合戦から牝馬限定戦への移行は過小評価 | v2 |

### v2 テーブル補正

**前走脚質別テーブル（style_table）:**
- 前走の脚質区分（KyakusituKubun: 1=逃げ, 2=先行, 3=差し, 4=追込）ごとに単勝回収率からfactorを算出
- 追込・差しは展開の影響を受けやすく、オッズに歪みが生じやすい
- 特徴量 `style_type_last` を参照して適用

**馬番×コース別テーブル（post_course_table）:**
- 馬番グループ: inner(1-3), mid_inner(4-6), mid_outer(7-9), outer(10+)
- コースカテゴリ: turf_left, turf_right, dirt_left, dirt_right, niigata_straight, other
- `{post_group}_{course_cat}` の組み合わせキーで詳細factor → post_group単独のフォールバックfactor → レガシーgate parityの順で適用
- 新潟直線コース（jyocd=04, trackcd=10）は外枠有利の特殊ケースとして個別に扱う
- ヘルパーメソッド: `_post_group(umaban)`, `_course_category(jyo_cd, track_cd)`

### v3 テーブル補正

**調教師×人気帯別テーブル（trainer_ninki_table）:**
- 調教師コード（chokyosicode）× 人気帯（A=1-3人気, B=4-6, C=7-9, D=10+）ごとに単勝回収率からfactorを算出
- 戦略的厩舎（人気薄で期待以上の成績を安定的に出す厩舎）の回収率歪みを補正
- 穴馬(C,D帯)で factor > 1.0 の厩舎はオッズが過小評価されている
- `{trainer_code}_{band}` のキーで検索、ない場合は factor=1.0
- 最小サンプル数200件、factor ≈ 1.0 のエントリは省略してJSONサイズを抑制
- 特徴量 `trainer_code` と人気順（`_derive_ninki_rank()` で導出）を参照して適用
- ヘルパーメソッド: `_ninki_band(ninki)`

**父系統×サーフェス別テーブル（sire_surface_table）:**
- 父系統（keitoname）× サーフェス（yousiba/siba/dirt）ごとのROIベースfactor
- 特徴量 `blood_father_keito`, `race_track_cd`, `race_jyo_cd` を参照

**父系統×距離帯別テーブル（sire_distance_table）:**
- 父系統（keitoname）× 距離帯（sprint/mile/middle/long）ごとのROIベースfactor
- 特徴量 `blood_father_keito`, `race_distance` を参照

### フォールバックチェーン

```
post_course_table あり → post_group × course_cat の完全一致
  ↓ キーなし
post_group 単独フォールバック
  ↓ post_course_table 自体がなし
レガシー gate parity ルール（odd_gate_discount / even_gate_boost）
```

### クラス変更・牝馬限定遷移の判定

- **クラス変更:** `class_level()` 関数（`src/utils/code_master.py`）で前走・今走の条件コード+グレードから序列値を算出し比較。特徴量 `cross_class_change` (+1=昇級, 0=同級, -1=降級) を使用
- **牝馬限定判定:** `BOOL_AND(sexcd='2')` で同レース全出走馬の性別から判定（KigoCDコード表2006が未ドキュメントのため）。特徴量 `cross_prev_filly_only`（前走）と `cross_current_filly_only`（今走）を使用
- **牝馬限定⇔混合ルールは `horse_sex == "2"`（牝馬）のみ適用**

### 後方互換性

- 旧JSONファイル（v1のみ）は `style_table` / `post_course_table` が欠如するが、フォールバックにより既存のgate parityルールが適用される
- 旧parquet（`cross_prev_filly_only`, `cross_current_filly_only` なし）ではC3/C4ルールが factor=1.0（無効）として安全にデグレードする
- `cross_class_change` が修正済みparquetでのみ正しく動作する（旧parquetでは常に0のため昇級/降級が検出されず factor=1.0）
- **新特徴量を有効にするには `--force-rebuild` でparquetを再構築する必要がある**

人気順はオッズ辞書からレース内で低い順に導出する（`_derive_ninki_rank()`）。統計JSONがない場合は `config.py` の `DEFAULT_ODDS_CORRECTION_CONFIG`（ハードコード値）にフォールバックする。

**関連ファイル:**
- `src/odds_correction_stats.py` — DB統計算出・JSON保存/ロード（v2: `_calc_running_style_stats`, `_calc_post_position_course_stats`, `_calc_class_change_stats`, `_calc_filly_transition_stats`、v3: `_calc_sire_surface_stats`, `_calc_sire_distance_stats`, `_calc_trainer_ninki_stats`）
- `src/model/evaluator.py` — `_apply_odds_correction()` でninki_table + 全ルール適用（v2: `_post_group()`, `_course_category()`、v3: `_ninki_band()`）
- `src/utils/code_master.py` — `class_level()` クラス序列ヘルパー
- `src/features/pipeline.py` — `cross_class_change`（修正）, `cross_prev_filly_only`, `cross_current_filly_only`（新規）
- `src/config.py` — `DEFAULT_ODDS_CORRECTION_CONFIG`（v2ルールのデフォルト値追加）
- `data/odds_correction_stats.json` — 統計データ（`--build-odds-stats`で生成、v2テーブル + v3テーブル含む）
