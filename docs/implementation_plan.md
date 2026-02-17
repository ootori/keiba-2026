# 実装計画書

## 全体スケジュール

```
Phase 0: 環境構築・DB接続確認          → Step 1-2
Phase 1: データ探索・基礎集計          → Step 3-4
Phase 2: 特徴量エンジニアリング        → Step 5-9
Phase 3: モデル学習・評価              → Step 10-13
Phase 4: 予測パイプライン・運用        → Step 14-16
```

---

## Phase 0: 環境構築・DB接続確認

### Step 1: プロジェクト初期化

**作業内容:**
- ディレクトリ構成の作成（CLAUDE.mdに記載の構造）
- `requirements.txt` の作成
- `.gitignore` の作成

**requirements.txt:**
```
lightgbm>=4.0
psycopg2-binary>=2.9
sqlalchemy>=2.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
tqdm>=4.65
```

**成果物:** プロジェクトディレクトリ一式

---

### Step 2: DB接続・基本動作確認

**作業内容:**
- `src/config.py` — DB接続情報の設定（環境変数から読み込み）
- `src/db.py` — PostgreSQL接続ユーティリティ
- 接続テスト：各テーブルの行数確認、サンプルデータ取得

**config.py の雛形:**
```python
import os

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'database': os.getenv('DB_NAME', 'everydb'),
    'user': os.getenv('DB_USER', 'webmaster'),
    'password': os.getenv('DB_PASSWORD', 'devpassword'),
}

# 対象期間
TRAIN_START_YEAR = '2015'
TRAIN_END_YEAR = '2024'
VALID_YEAR = '2025'

# JRA中央10場
JRA_JYO_CODES = [f'{i:02d}' for i in range(1, 11)]
```

**db.py の雛形:**
```python
import psycopg2
import pandas as pd
from config import DB_CONFIG

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def query_df(sql: str, params: dict = None) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(sql, conn, params=params)
```

**確認SQL:**
```sql
-- テーブル存在確認＆行数カウント
SELECT 'n_race' AS tbl, COUNT(*) FROM n_race
UNION ALL SELECT 'n_uma_race', COUNT(*) FROM n_uma_race
UNION ALL SELECT 'n_uma', COUNT(*) FROM n_uma
UNION ALL SELECT 'n_kisyu', COUNT(*) FROM n_kisyu
UNION ALL SELECT 'n_hanro', COUNT(*) FROM n_hanro
UNION ALL SELECT 'n_odds_tanpuku', COUNT(*) FROM n_odds_tanpuku
UNION ALL SELECT 'n_harai', COUNT(*) FROM n_harai;
```

**完了条件:** 全テーブルへの接続成功、サンプルデータ取得確認

---

## Phase 1: データ探索・基礎集計

### Step 3: データ品質確認

**作業内容:**
- 各テーブルの欠損率確認
- DataKubun別のレコード数確認
- 年度別のレース数・出走馬数の推移確認
- 異常値の検出（タイムが0、着順が空白 等）

**確認項目チェックリスト:**
```
□ n_race: DataKubun='7'のレース数（年度別）
□ n_uma_race: IJyoCD分布（出走取消・除外の割合）
□ n_uma_race: Time欄の欠損率
□ n_uma_race: HaronTimeL3の欠損率
□ n_uma_race: BaTaijyuの欠損率
□ n_hanro: 年度別レコード数（調教データの充足度）
□ n_odds_tanpuku: 年度別レコード数
□ n_race: TrackCD分布（芝/ダート比率）
□ n_race: GradeCD分布（クラス分布）
```

**成果物:** `notebooks/exploration.ipynb` にデータ品質レポート

---

### Step 4: 基準タイムテーブルの作成

**作業内容:**
- スピード指数算出に必要な「距離×トラック×馬場別の平均走破タイム」を集計
- 過去3年のデータから算出し、辞書として保持

**SQL:**
```sql
SELECT
    r.kyori,
    CASE WHEN CAST(r.trackcd AS int) BETWEEN 10 AND 22 THEN 'turf'
         WHEN CAST(r.trackcd AS int) BETWEEN 23 AND 29 THEN 'dirt'
         ELSE 'other' END AS track_type,
    CASE WHEN CAST(r.trackcd AS int) BETWEEN 10 AND 22 THEN r.sibababacd
         ELSE r.dirtbabacd END AS baba_cd,
    AVG(
        CAST(SUBSTRING(ur.time FROM 1 FOR 1) AS int) * 60
        + CAST(SUBSTRING(ur.time FROM 2 FOR 2) AS int)
        + CAST(SUBSTRING(ur.time FROM 4 FOR 1) AS numeric) * 0.1
    ) AS avg_time,
    COUNT(*) AS sample_count
FROM n_uma_race ur
JOIN n_race r USING (year, monthday, jyocd, kaiji, nichiji, racenum)
WHERE ur.datakubun = '7'
  AND ur.ijyocd = '0'
  AND ur.kakuteijyuni = '01'  -- 1着馬のタイムで基準作成
  AND r.jyocd IN ('01','02','03','04','05','06','07','08','09','10')
  AND r.year >= :base_year_start
GROUP BY r.kyori, track_type, baba_cd
HAVING COUNT(*) >= 10
ORDER BY r.kyori, track_type, baba_cd;
```

**成果物:** `data/base_time_table.csv` + `src/utils/base_time.py`

---

## Phase 2: 特徴量エンジニアリング

### Step 5: 特徴量基底クラスの実装

**作業内容:**
- `src/features/base.py` — 全特徴量クラスの基底
- 共通インターフェース定義

```python
from abc import ABC, abstractmethod
import pandas as pd

class FeatureExtractor(ABC):
    """特徴量抽出の基底クラス"""

    @abstractmethod
    def extract(self, race_key: dict, horse_keys: list[dict]) -> pd.DataFrame:
        """
        Args:
            race_key: {'year', 'monthday', 'jyocd', 'kaiji', 'nichiji', 'racenum'}
            horse_keys: [{'kettonum', 'umaban', ...}, ...]
        Returns:
            DataFrame with kettonum as index, feature columns
        """
        pass

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        pass
```

---

### Step 6: レース条件＆枠順特徴量の実装

**作業内容:**
- `src/features/race.py` — カテゴリ6（race_*）＋カテゴリ7（post_*）
- 当該レースのn_raceとn_uma_raceから直接取得する特徴量
- 最もシンプルなので最初に実装

**テスト:**
```python
# 特定レースで特徴量が正しく抽出されるか
def test_race_features():
    extractor = RaceFeatureExtractor(db)
    race_key = {'year': '2024', 'monthday': '0623', 'jyocd': '09',
                'kaiji': '03', 'nichiji': '08', 'racenum': '11'}
    features = extractor.extract(race_key, horse_keys)
    assert 'race_distance' in features.columns
    assert features['race_distance'].iloc[0] == 2200  # 宝塚記念
```

---

### Step 7: 過去成績＆スピード指数の実装

**作業内容:**
- `src/features/horse.py` — カテゴリ2,3（horse_perf_*, horse_cond_*）
- `src/features/speed.py` — カテゴリ4（speed_*）
- ここが最もSQLが複雑で実装量が多い
- パフォーマンス対策: 一括で対象馬全頭の過去成績を取得し、Python側で集計

**最適化方針:**
```python
# 悪い例: 馬ごとにSQL発行（N+1問題）
for horse in horses:
    past_races = query_df(sql, {'kettonum': horse['kettonum']})

# 良い例: 一括取得してgroupby
all_past = query_df(sql, {'kettonums': tuple(h['kettonum'] for h in horses)})
grouped = all_past.groupby('kettonum')
```

---

### Step 8: 騎手・調教師・調教・血統特徴量の実装

**作業内容:**
- `src/features/jockey_trainer.py` — カテゴリ10,11
- `src/features/training.py` — カテゴリ12
- `src/features/bloodline.py` — カテゴリ13

**実装順序（依存関係順）:**
1. 騎手・調教師（KISYU_SEISEKIから直接取得、比較的シンプル）
2. 調教データ（HANROとWOOD_CHIPから取得）
3. 血統（UMA→HANSYOKU→KEITOの複数テーブル結合が必要）

---

### Step 9: 特徴量パイプライン統合

**作業内容:**
- `src/features/pipeline.py` — 全特徴量の統合
- `src/features/odds.py` — カテゴリ15（オッズ）
- クロス特徴量（カテゴリ16）の算出
- 欠損値処理の適用
- 学習用データセット全体の構築

**パイプラインの構造:**
```python
class FeaturePipeline:
    def __init__(self, db, include_odds: bool = True):
        self.extractors = [
            RaceFeatureExtractor(db),
            HorsePerformanceExtractor(db),
            SpeedFeatureExtractor(db),
            StyleFeatureExtractor(db),
            JockeyFeatureExtractor(db),
            TrainerFeatureExtractor(db),
            TrainingFeatureExtractor(db),
            BloodlineFeatureExtractor(db),
            IntervalFeatureExtractor(db),
        ]
        if include_odds:
            self.extractors.append(OddsFeatureExtractor(db))

    def build_dataset(self, year_range: tuple, jyo_codes: list) -> pd.DataFrame:
        """指定期間の全レースに対して特徴量を構築"""
        races = self._get_target_races(year_range, jyo_codes)
        all_features = []
        for race_key in tqdm(races):
            horses = self._get_horses(race_key)
            features = self._extract_all(race_key, horses)
            features['target'] = self._get_target(race_key, horses)
            all_features.append(features)
        return pd.concat(all_features, ignore_index=True)
```

**成果物:**
- `data/train_features.parquet` — 学習用特徴量（2015-2024年）
- `data/valid_features.parquet` — 検証用特徴量（2025年）

---

## Phase 3: モデル学習・評価

### Step 10: LightGBMベースラインモデル

**作業内容:**
- `src/model/trainer.py` — LightGBMの学習ロジック
- まずデフォルトパラメータでベースライン構築

**LightGBMパラメータ（初期値）:**
```python
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 50,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}
```

**学習コード骨格:**
```python
import lightgbm as lgb

train_data = lgb.Dataset(X_train, label=y_train,
                         categorical_feature=categorical_features)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=3000,
    valid_sets=[train_data, valid_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100),
    ],
)
```

**完了条件:** ベースラインモデルの学習完了、loglossが出力される

---

### Step 11: モデル評価

**作業内容:**
- `src/model/evaluator.py` — 評価指標の計算
- 検証データ（2025年）での評価

**評価指標:**
```python
from sklearn.metrics import log_loss, roc_auc_score

def evaluate_model(model, X_valid, y_valid, valid_df):
    """
    Returns:
        - logloss
        - AUC
        - 的中率（予測確率Top3に実際の3着以内馬が含まれる割合）
        - 回収率シミュレーション
    """
    y_pred = model.predict(X_valid)

    metrics = {
        'logloss': log_loss(y_valid, y_pred),
        'auc': roc_auc_score(y_valid, y_pred),
    }

    # レース単位の評価
    valid_df['pred_prob'] = y_pred
    for race_key, group in valid_df.groupby(['year','monthday','jyocd','racenum']):
        # 予測確率上位3頭 vs 実際の3着以内
        top3_pred = group.nlargest(3, 'pred_prob')
        top3_actual = group[group['target'] == 1]
        ...

    return metrics
```

**回収率シミュレーション:**
```python
def simulate_return(valid_df, harai_df, strategy='top1_tansho'):
    """
    戦略:
    - top1_tansho: 予測1位の単勝を購入
    - top3_fukusho: 予測Top3の複勝を購入
    - threshold: 期待値が一定以上の馬のみ購入
    """
    total_bet = 0
    total_return = 0
    for race_key, group in valid_df.groupby([...]):
        ...
    return total_return / total_bet * 100  # 回収率(%)
```

---

### Step 12: ハイパーパラメータチューニング

**作業内容:**
- Optunaを使用したベイズ最適化
- 検証データでのloglossを最小化

**チューニング対象:**
```python
def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
    }
    ...
```

---

### Step 13: 特徴量選別・モデル改善

**作業内容:**
- Feature importance分析
- 不要特徴量の除去
- 特徴量追加の検討

```python
# 特徴量重要度
importance = model.feature_importance(importance_type='gain')
feature_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importance,
}).sort_values('importance', ascending=False)

# 上位N特徴量でリトレーニング
top_n = 80
selected = feature_imp.head(top_n)['feature'].tolist()
```

---

## Phase 4: 予測パイプライン・運用

### Step 14: 予測実行スクリプト

**作業内容:**
- `src/model/predictor.py` — 予測の実行
- 当日の出馬表データから特徴量を構築し予測

```python
class Predictor:
    def __init__(self, model_path: str, db):
        self.model = lgb.Booster(model_file=model_path)
        self.pipeline = FeaturePipeline(db, include_odds=False)

    def predict_race(self, race_key: dict) -> pd.DataFrame:
        """
        Returns:
            DataFrame with columns: umaban, bamei, pred_prob, pred_rank
        """
        horses = self._get_entry_horses(race_key)
        features = self.pipeline.extract_race(race_key, horses)
        probs = self.model.predict(features[self.feature_cols])

        result = pd.DataFrame({
            'umaban': horses['umaban'],
            'bamei': horses['bamei'],
            'pred_prob': probs,
        })
        result['pred_rank'] = result['pred_prob'].rank(ascending=False).astype(int)
        return result.sort_values('pred_rank')
```

---

### Step 15: モデル更新スクリプト

**作業内容:**
- 新しいレース結果が蓄積されたらモデルを再学習するスクリプト
- スライディングウィンドウ方式で学習期間を更新

---

### Step 16: 出力フォーマット

**作業内容:**
- 予測結果のフォーマット整形
- 推奨買い目の出力

出力例:
```
=== 2024/12/22 中山11R 有馬記念 ===
芝2500m 良 16頭

予測  馬番  馬名            確率    単勝オッズ
  1    5   ドウデュース      28.3%   3.2
  2   12   ジャスティンパレス 18.5%   5.8
  3    3   スターズオンアース 14.2%   8.1
  ...

推奨: 複勝 5, 12  ワイド 5-12
```

---

## 実装の優先順位まとめ

| 優先度 | Step | 内容 | 見積時間 |
|--------|------|------|---------|
| ★★★ | 1-2 | 環境構築・DB接続 | 2h |
| ★★★ | 3 | データ品質確認 | 3h |
| ★★★ | 4 | 基準タイムテーブル | 2h |
| ★★★ | 5-6 | 特徴量基盤・レース条件 | 3h |
| ★★★ | 7 | 過去成績・スピード指数 | 8h |
| ★★☆ | 8 | 騎手・調教師・調教・血統 | 6h |
| ★★★ | 9 | パイプライン統合 | 4h |
| ★★★ | 10 | ベースラインモデル | 2h |
| ★★★ | 11 | モデル評価 | 3h |
| ★★☆ | 12 | ハイパラチューニング | 3h |
| ★★☆ | 13 | 特徴量選別 | 3h |
| ★★☆ | 14 | 予測スクリプト | 3h |
| ★☆☆ | 15 | モデル更新 | 2h |
| ★☆☆ | 16 | 出力フォーマット | 2h |
| | | **合計** | **約46h** |

---

## Claude Codeへの指示テンプレート

各Stepの実装時にClaude Codeへ指示する際の例:

```
Step 7の実装をお願いします。
CLAUDE.mdとdocs/feature_design.mdの「カテゴリ2: 過去成績」「カテゴリ3: 条件別成績」
「カテゴリ4: スピード指数」を参照してください。

以下を実装してください:
1. src/features/horse.py — 過去成績特徴量の抽出クラス
2. src/features/speed.py — スピード指数特徴量の抽出クラス
3. tests/test_features.py — 上記のテスト

PostgreSQLへの接続はsrc/db.pyのquery_dfを使ってください。
パフォーマンスを考慮し、N+1問題を避けてIN句で一括取得してください。
```

---

## 実装状況（2026-02-17時点）

### Phase 0-4: 全Phase実装完了

| Phase | Step | 状況 | 成果物 |
|-------|------|------|--------|
| 0 | 1: プロジェクト初期化 | ✅ 完了 | requirements.txt, .gitignore, ディレクトリ構成 |
| 0 | 2: DB接続 | ✅ 完了 | src/config.py, src/db.py |
| 1 | 3: データ品質確認 | ⏭️ スキップ | 実DBテストで代替 |
| 1 | 4: 基準タイムテーブル | ✅ 完了 | src/utils/base_time.py |
| 2 | 5: 特徴量基底クラス | ✅ 完了 | src/features/base.py |
| 2 | 6: レース条件＆枠順 | ✅ 完了 | src/features/race.py |
| 2 | 7: 過去成績＆スピード指数 | ✅ 完了 | src/features/horse.py, speed.py |
| 2 | 8: 騎手・調教師・調教・血統 | ✅ 完了 | jockey_trainer.py, training.py, bloodline.py |
| 2 | 9: パイプライン統合 | ✅ 完了 | src/features/pipeline.py, odds.py |
| 3 | 10: ベースラインモデル | ✅ 完了 | src/model/trainer.py |
| 3 | 11: モデル評価 | ✅ 完了 | src/model/evaluator.py |
| 3 | 12: ハイパラチューニング | 🔲 未着手 | Optuna統合は今後 |
| 3 | 13: 特徴量選別 | 🔲 未着手 | feature importance分析は今後 |
| 4 | 14: 予測スクリプト | ✅ 完了 | src/model/predictor.py, run_predict.py |
| 4 | 15: モデル更新 | 🔲 未着手 | |
| 4 | 16: 出力フォーマット | ✅ 完了 | predictor.py内のformat_prediction() |

### 設計変更点

| 項目 | 当初計画 | 実装 | 理由 |
|------|---------|------|------|
| DB接続 | psycopg2 or SQLAlchemy | psycopg2直接 | IN句タプル展開がpsycopg2ネイティブで必要。pandas UserWarningは `warnings.filterwarnings` で抑制 |
| 脚質特徴量 | style.py（独立ファイル） | speed.py に統合 | スピード指数と脚質は同じ過去成績データを参照するため統合が効率的 |
| オッズ取得 | datakubun優先度で取得 | datakubunフィルタなし | n_odds_tanpuku明細テーブルにはdatakubunカラムが存在しない |
| 払戻データ取得 | カラム名ハードコード | スキーマから動的検出 | n_haraiのカラム命名規則がEveryDB2バージョン依存のため |
| run_train.py | 基本オプションのみ | --eval-only追加 | Step 4-5の再実行を高速化するため |
| 特徴量保存 | 全年度一括 parquet | 年度別 parquet + 並列構築 | 10年分の構築に丸一日かかるため。年度別分割で差分再構築・並列化に対応（`--workers N`、`--force-rebuild`） |
| 回収率シミュレーション | 単勝/複勝のみ | 単勝/複勝/馬連/馬単/三連複/三連単 | 多券種での回収率比較を可能にするため。n_haraiから6賭式の払戻を動的検出 |
| 払戻カラム検出 | `umaban` キーワードのみ | `umaban` + `kumi` キーワード | 馬連/馬単/三連複/三連単は組番（kumi）を使う可能性があるため。`sanrentan`/`sanren` の部分一致誤検出も防止 |

### テスト状況

- pytest: 23テスト全パス
- 構文チェック: 全25 Pythonファイル問題なし
- 特徴量数: 130（重複なし確認済み）

---

## トラブルシューティング（実装中に遭遇した問題と解決策）

### 1. pandas UserWarning: "pandas only supports SQLAlchemy connectable"

**原因:** `pd.read_sql()` にpsycopg2の生connectionを渡すと警告が出る（pandas 2.0+）
**解決:** `db.py` で `warnings.filterwarnings("ignore", message=".*pandas only supports SQLAlchemy.*", category=UserWarning)` を追加。SQLAlchemyへの移行はIN句パターンの大規模書き換えが必要なため不採用。

### 2. column ur.timedifn does not exist

**原因:** DBリファレンスでは `TimeDIFN` と記載されているが、実際のDBカラムは `timediff`
**解決:** `speed.py` のSQL・Pythonの両方を `timediff` に修正

### 3. column "datakubun" does not exist (n_odds_tanpuku)

**原因:** `n_odds_tanpuku`（オッズ明細テーブル）には `datakubun` カラムが存在しない。ヘッダテーブル `n_odds_tanpukuwaku_head` にのみ存在
**解決:** `odds.py` から datakubun のSELECT/WHERE/Pythonフィルタを全て削除。明細テーブルはEveryDB2で最新データに上書きされるため不要

### 4. train and valid dataset categorical_feature do not match

**原因:** `trainer.py` でカテゴリ変数を `category` dtype に変換してLightGBMに渡していたが、`evaluator.py` の `model.predict()` 時に同じ変換をしていなかった
**解決:** `evaluator.py` の `evaluate()` と `simulate_return()` で予測前に `CATEGORICAL_FEATURES` のカラムを `category` 型に変換する処理を追加

### 5. column "paytansyo1umaban" does not exist (n_harai)

**原因:** `n_harai` テーブルの払戻カラム命名規則（番号の位置等）がDBリファレンスの `PayTansyo*` ワイルドカード表記では不明確で、実際のDBと一致しなかった
**解決:** `evaluator.py` の `_get_harai_data()` を書き換え、`SELECT * FROM n_harai LIMIT 0` でスキーマを取得し、`tansyo`+`umaban` パターンでカラムを動的検出する方式に変更。`_find_pay_column_pairs()` ヘルパーメソッドを追加。後に馬連/馬単/三連複/三連単にも拡張し、`kumi` キーワードにも対応。

### 6. 特徴量構築が遅い（10年分で丸一日）

**原因:** `build_dataset()` が全年度を直列処理していた
**解決:** 年度別に parquet を分割保存し、`ProcessPoolExecutor` で並列構築する方式に変更。`--workers N` で並列度を指定可能。既存 parquet がある年度は `--force-rebuild` なしならスキップされるため、差分再構築も高速
