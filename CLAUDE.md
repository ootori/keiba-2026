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
│   ├── feature_design.md              # 特徴量設計書
│   └── implementation_plan.md         # 実装計画書
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
│   │   └── pipeline.py              # 特徴量パイプライン統合+クロス特徴量（カテゴリ16）
│   ├── model/
│   │   ├── __init__.py
│   │   ├── trainer.py                # LightGBMの学習
│   │   ├── predictor.py              # 予測実行
│   │   └── evaluator.py             # モデル評価・回収率シミュレーション
│   └── utils/
│       ├── __init__.py
│       ├── code_master.py            # コード表変換ユーティリティ
│       └── base_time.py              # 基準タイムテーブル（スピード指数用）
├── notebooks/
│   └── exploration.ipynb             # データ探索用ノートブック
├── models/                           # 学習済みモデル保存先（*.txt, *_features.txt）
├── data/                             # 中間データキャッシュ（*.parquet, base_time_table.csv）
└── tests/
    └── test_features.py              # 23テスト（pytest）
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
| `PayTansyo*` | *(動的検出)* | n_harai | 払戻カラムの命名規則がバージョン依存のため、`evaluator.py`ではスキーマから動的検出 |

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
- **出力:** 各馬の3着以内確率（LightGBMの二値分類）
- **目的変数:** `KakuteiJyuni` が 1, 2, 3 なら 1、それ以外は 0
- **データリーク防止:** 特徴量は必ず「当該レースより過去のデータ」のみで構成

### 学習/評価の時間分割
- 学習: 2015-2024年（10年分）
- 検証: 2025年
- 時系列で分割し、未来のデータが学習に混入しないことを保証

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
13. **血統(10):** 父系統, 母父系統, 芝ダ適性, 距離適性, 近交フラグ
14. **間隔(5):** 中N日, 休み明け区分, ローテーション
15. **オッズ(7):** 単勝オッズ, 複勝オッズ, 人気順（※予測タイミング依存）
16. **クロス特徴量(8):** 距離変更, 芝ダ変更, クラス変更, 斤量/体重比

合計 **約130特徴量**（詳細は `docs/feature_design.md` 参照）

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

# 既存特徴量からモデル学習のみ
python run_train.py --train-only

# 評価・回収率シミュレーションのみ（Step 4-5、保存済みモデル+特徴量を使用）
python run_train.py --eval-only

# オッズ特徴量を明示的に含める（非推奨：データリークの危険あり）
python run_train.py --with-odds

# モデル名を指定
python run_train.py --model-name my_model
```

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
