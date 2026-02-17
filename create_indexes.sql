-- ============================================================================
-- 競馬予想プロジェクト: パフォーマンス改善用インデックス作成スクリプト
--
-- 目的: run_train.py の実行時間短縮（推定40時間→数十分〜数時間）
--
-- 使い方:
--   psql -h localhost -p 5432 -U webmaster -d everydb -f create_indexes.sql
--   または Docker 経由:
--   docker exec -i <container_name> psql -U webmaster -d everydb -f - < create_indexes.sql
--
-- 注意:
--   - 作成には数分〜十数分かかります（特に n_uma_race, n_hanro）
--   - 既存のインデックスがある場合は IF NOT EXISTS で安全にスキップされます
--   - ディスク容量に余裕があることを確認してください（推定 2-4GB 追加）
-- ============================================================================

BEGIN;

-- ============================================================================
-- [Priority 1] n_uma_race (1,298,099行) — 最頻出・最重要テーブル
-- ============================================================================

-- (1-A) レースキー完全一致（16クエリ中10クエリで使用）
-- horse._get_basic_info, race._get_horse_info, jockey_trainer._get_horse_info,
-- speed._get_last_race_l3f, odds._get_horse_umaban, pipeline._get_horses,
-- pipeline._get_target, predictor._get_horse_info 等
CREATE INDEX IF NOT EXISTS idx_uma_race_racekey
    ON n_uma_race (year, monthday, jyocd, kaiji, nichiji, racenum);

-- (1-B) kettonum での過去成績検索（最重要：全馬の過去走を遡るクエリ）
-- horse._get_past_results, speed._get_past_results_with_style,
-- pipeline._get_prev_race_info, jockey_trainer._get_past_jockey_info 等
-- datakubun='7', ijyocd='0' で絞り込み + year DESC, monthday DESC でソート
CREATE INDEX IF NOT EXISTS idx_uma_race_kettonum_hist
    ON n_uma_race (kettonum, datakubun, ijyocd, year DESC, monthday DESC);

-- (1-C) 騎手コード別・場コード別の成績集計
-- jockey_trainer._get_jockey_jyo_stats
CREATE INDEX IF NOT EXISTS idx_uma_race_kisyu_jyo
    ON n_uma_race (kisyucode, jyocd, datakubun, ijyocd, year);

-- (1-D) 調教師コード別・場コード別の成績集計
-- jockey_trainer._get_trainer_jyo_stats
CREATE INDEX IF NOT EXISTS idx_uma_race_chokyo_jyo
    ON n_uma_race (chokyosicode, jyocd, datakubun, ijyocd, year);

-- (1-E) 騎手×調教師コンビの成績集計
-- jockey_trainer._get_jockey_trainer_combo
CREATE INDEX IF NOT EXISTS idx_uma_race_kisyu_chokyo
    ON n_uma_race (kisyucode, chokyosicode, datakubun, ijyocd, year);


-- ============================================================================
-- [Priority 2] n_hanro (7,737,376行) — 最大行数テーブル
-- ============================================================================

-- (2-A) 直近坂路調教タイム取得（DISTINCT ON + ORDER BY で最新1件）
-- training._get_last_hanro
CREATE INDEX IF NOT EXISTS idx_hanro_kettonum_date
    ON n_hanro (kettonum, chokyodate DESC, chokyotime DESC);


-- ============================================================================
-- [Priority 3] n_race (106,308行) — レース条件テーブル
-- ============================================================================

-- (3-A) レースキー完全一致（USING結合の右辺 + 単体検索）
-- 全特徴量モジュールのJOIN/WHERE で使用
CREATE INDEX IF NOT EXISTS idx_race_racekey
    ON n_race (year, monthday, jyocd, kaiji, nichiji, racenum);

-- (3-B) 対象レース一覧取得（年範囲 + 場コード + datakubun）
-- pipeline._get_target_races
CREATE INDEX IF NOT EXISTS idx_race_year_dk_jyo
    ON n_race (datakubun, year, jyocd);


-- ============================================================================
-- [Priority 4] n_odds_tanpuku (771,113行)
-- ============================================================================

-- (4-A) レースキー一致でオッズ取得
-- odds._get_odds
CREATE INDEX IF NOT EXISTS idx_odds_tanpuku_racekey
    ON n_odds_tanpuku (year, monthday, jyocd, kaiji, nichiji, racenum);


-- ============================================================================
-- [Priority 5] n_wood_chip (639,793行)
-- ============================================================================

-- (5-A) kettonum + 日付範囲で調教データ検索
-- training._get_best_woodchip, training._get_training_count
CREATE INDEX IF NOT EXISTS idx_wood_chip_kettonum_date
    ON n_wood_chip (kettonum, chokyodate);


-- ============================================================================
-- [Priority 6] n_sanku (116,344行) — 血統データ
-- ============================================================================

-- (6-A) kettonum で血統情報取得
-- bloodline._get_blood_info
CREATE INDEX IF NOT EXISTS idx_sanku_kettonum
    ON n_sanku (kettonum);

-- (6-B) 種牡馬番号で産駒成績集計
-- bloodline._get_sire_stats, bloodline._get_sire_dist_stats
CREATE INDEX IF NOT EXISTS idx_sanku_fnum
    ON n_sanku (fnum);


-- ============================================================================
-- [Priority 7] n_uma (210,057行) — 競走馬マスタ
-- ============================================================================

-- (7-A) kettonum で馬情報取得（血統フォールバック用）
-- bloodline._get_blood_info_fallback
CREATE INDEX IF NOT EXISTS idx_uma_kettonum
    ON n_uma (kettonum);


-- ============================================================================
-- [Priority 8] 小規模マスタテーブル（行数少ないが頻繁にアクセス）
-- ============================================================================

-- n_kisyu_seiseki: 騎手年度別成績
CREATE INDEX IF NOT EXISTS idx_kisyu_seiseki_code_year
    ON n_kisyu_seiseki (kisyucode, setyear);

-- n_chokyo_seiseki: 調教師年度別成績
CREATE INDEX IF NOT EXISTS idx_chokyo_seiseki_code_year
    ON n_chokyo_seiseki (chokyosicode, setyear);

-- n_kisyu: 騎手マスタ
CREATE INDEX IF NOT EXISTS idx_kisyu_code
    ON n_kisyu (kisyucode);

-- n_chokyo: 調教師マスタ
CREATE INDEX IF NOT EXISTS idx_chokyo_code
    ON n_chokyo (chokyosicode);

-- n_keito: 系統情報
CREATE INDEX IF NOT EXISTS idx_keito_hansyokunum
    ON n_keito (hansyokunum);

-- n_harai: 払戻データ（評価時に使用）
CREATE INDEX IF NOT EXISTS idx_harai_year_dk
    ON n_harai (year, datakubun);

COMMIT;

-- ============================================================================
-- インデックス作成後の統計情報更新（クエリプランナー最適化に必要）
-- ============================================================================
ANALYZE n_uma_race;
ANALYZE n_hanro;
ANALYZE n_race;
ANALYZE n_odds_tanpuku;
ANALYZE n_wood_chip;
ANALYZE n_sanku;
ANALYZE n_uma;
ANALYZE n_kisyu_seiseki;
ANALYZE n_chokyo_seiseki;
ANALYZE n_kisyu;
ANALYZE n_chokyo;
ANALYZE n_keito;
ANALYZE n_harai;

-- ============================================================================
-- 確認用: 作成されたインデックス一覧
-- ============================================================================
SELECT tablename, indexname,
       pg_size_pretty(pg_relation_size(indexname::regclass)) AS index_size
FROM pg_indexes
WHERE schemaname = 'public'
  AND indexname LIKE 'idx_%'
ORDER BY tablename, indexname;
