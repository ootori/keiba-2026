# EveryDB2 データベースリファレンス

JRA-VAN DataLabデータをPostgreSQL上に構築するEveryDB2のテーブル定義・マスタ・コード表をまとめたリファレンスです。
競馬予想ソフト開発のためにClaude Codeから参照することを目的としています。

> **テーブル命名規則:** 蓄積系テーブルは `N_テーブル名`（または `n_テーブル名`）、速報系テーブルは `S_テーブル名`（または `s_テーブル名`）のプレフィックスが付きます。
> PostgreSQLではデフォルトで識別子が小文字に正規化されるため、実際のテーブル名・カラム名は小文字になっている場合が多いです。
> **実装前に `\dt n_*` および `\d n_race` で実際のケースを確認してください。**
> 本ドキュメントのフィールド名（RecordSpec, DataKubun等）はEveryDB2公式マニュアルの表記に準じています。
>
> **データ型:** すべてのカラムは `varchar` 型で格納されます。数値として利用する場合はキャスト（`CAST(column AS integer)` など）が必要です。

---

## 目次

1. [テーブル概要一覧](#1-テーブル概要一覧)
2. [レース関連テーブル](#2-レース関連テーブル)
3. [馬毎レース情報](#3-馬毎レース情報)
4. [払戻・票数テーブル](#4-払戻票数テーブル)
5. [オッズテーブル](#5-オッズテーブル)
6. [時系列オッズテーブル](#6-時系列オッズテーブル)
7. [マスタテーブル](#7-マスタテーブル)
8. [調教データ](#8-調教データ)
9. [その他のテーブル](#9-その他のテーブル)
10. [コード表](#10-コード表)

---

## 1. テーブル概要一覧

| No | テーブル名 | 説明 | レコード種別 | 項目数 |
|----|-----------|------|-------------|--------|
| 01 | TOKU_RACE | 特別レース | TK | 36 |
| 02 | TOKU | 特別登録馬 | TK | 17 |
| 03 | RACE | レース詳細 | RA | 110 |
| 04 | UMA_RACE | 馬毎レース情報 | SE | 73 |
| 05 | HARAI | 払戻 | HR | 199 |
| 06 | HYOSU | 票数 | H1 | 77 |
| 07 | HYOSU_TANPUKU | 票数_単複 | H1 | 12 |
| 08 | HYOSU_WAKU | 票数_枠連 | H1 | 10 |
| 09 | HYOSU_UMARENWIDE | 票数_馬連_ワイド | H1 | 12 |
| 10 | HYOSU_UMATAN | 票数_馬単 | H1 | 10 |
| 11 | HYOSU_SANREN | 票数_3連複 | H1 | 10 |
| 12 | HYOSU2 | 票数2 | H6 | 32 |
| 13 | HYOSU_SANRENTAN | 票数_3連単 | H6 | 10 |
| 14 | ODDS_TANPUKUWAKU_HEAD | オッズ_単複枠_ヘッダ | O1 | 19 |
| 15 | ODDS_TANPUKU | オッズ_単複 | O1 | 13 |
| 16 | ODDS_WAKU | オッズ_枠連 | O1 | 10 |
| 17 | ODDS_UMAREN_HEAD | オッズ_馬連_ヘッダ | O2 | 14 |
| 18 | ODDS_UMAREN | オッズ_馬連 | O2 | 10 |
| 19 | ODDS_WIDE_HEAD | オッズ_ワイド_ヘッダ | O3 | 14 |
| 20 | ODDS_WIDE | オッズ_ワイド | O3 | 11 |
| 21 | ODDS_UMATAN_HEAD | オッズ_馬単_ヘッダ | O4 | 14 |
| 22 | ODDS_UMATAN | オッズ_馬単 | O4 | 10 |
| 23 | ODDS_SANREN_HEAD | オッズ_3連複_ヘッダ | O5 | 14 |
| 24 | ODDS_SANREN | オッズ_3連複 | O5 | 10 |
| 25 | ODDS_SANRENTAN_HEAD | オッズ_3連単_ヘッダ | O6 | 14 |
| 26 | ODDS_SANRENTAN | オッズ_3連単 | O6 | 10 |
| 27 | UMA | 競走馬マスタ | UM | 45+ |
| 28 | KISYU | 騎手マスタ | KS | 47+ |
| 29 | KISYU_SEISEKI | 騎手マスタ_成績 | KS | 50+ |
| 30 | CHOKYO | 調教師マスタ | CH | 42 |
| 31 | CHOKYO_SEISEKI | 調教師マスタ_成績 | CH | 50+ |
| 32 | SEISAN | 生産者マスタ | BR | 27 |
| 33 | BANUSI | 馬主マスタ | BN | 27 |
| 34 | HANSYOKU | 繁殖馬マスタ | HN | 19 |
| 35 | SANKU | 産駒マスタ | SK | 26 |
| 36 | RECORD | レコードマスタ | RC | 48 |
| 37 | HANRO | 坂路調教 | HC | 14 |
| 38 | BATAIJYU | 馬体重 | WH | 40+ |
| 39 | TENKO_BABA | 天候馬場状態 | WE | 16 |
| 40 | TORIKESI_JYOGAI | 出走取消・競走除外 | AV | 13 |
| 41 | KISYU_CHANGE | 騎手変更 | JC | 20 |
| 42 | HASSOU_JIKOKU_CHANGE | 発走時刻変更 | TC | 14 |
| 43 | COURSE_CHANGE | コース変更 | CC | 15 |
| 44 | MINING | データマイニング予想 | DM | 46 |
| 45 | SCHEDULE | 開催スケジュール | YS | 45 |
| 46 | JODDS_TANPUKUWAKU_HEAD | 時系列オッズ_単複枠_ヘッダ | O1 | 19 |
| 47 | JODDS_TANPUKU | 時系列オッズ_単複 | O1 | 14 |
| 48 | JODDS_WAKU | 時系列オッズ_枠連 | O1 | 11 |
| 49 | JODDS_UMAREN_HEAD | 時系列オッズ_馬連_ヘッダ | O2 | 14 |
| 50 | JODDS_UMAREN | 時系列オッズ_馬連 | O2 | 11 |
| 51 | SALE | 競走馬市場取引価格 | HS | 14 |
| 52 | BAMEIORIGIN | 馬名の意味由来 | HY | 6 |
| 53 | KEITO | 系統情報 | BT | 7 |
| 54 | COURSE | コース情報 | CS | 8 |
| 55 | TAISENGATA_MINING | 対戦型データマイニング予想 | TM | 46 |
| 56 | JYUSYOSIKI_HEAD | 重勝式_ヘッダ | WF | 38 |
| 57 | JYUSYOSIKI | 重勝式 | WF | 6 |
| 58 | JOGAIBA | 競走馬除外情報 | JG | 14 |
| 59 | WOOD_CHIP | ウッドチップ調教 | WC | 29 |

---

## 2. レース関連テーブル

### 2.1 TOKU_RACE（特別レース）

特別競走（重賞など）のレース基本情報。

| No | カラム名 | フィールド名 | 型 | 桁 | PK | 内容 |
|----|---------|-------------|---|---|---|------|
| 1 | レコード種別ID | RecordSpec | varchar | 2 | | "TK" |
| 2 | データ区分 | DataKubun | varchar | 1 | | 1:ハンデ発表前, 2:ハンデ発表後, 0:削除 |
| 3 | データ作成年月日 | MakeDate | varchar | 8 | | yyyymmdd |
| 4 | 開催年 | Year | varchar | 4 | ○ | yyyy |
| 5 | 開催月日 | MonthDay | varchar | 4 | ○ | mmdd |
| 6 | 競馬場コード | JyoCD | varchar | 2 | ○ | コード表2001参照 |
| 7 | 開催回 | Kaiji | varchar | 2 | ○ | 第N回 |
| 8 | 開催日目 | Nichiji | varchar | 2 | ○ | N日目 |
| 9 | レース番号 | RaceNum | varchar | 2 | ○ | |
| 10 | 曜日コード | YoubiCD | varchar | 1 | | コード表2002参照 |
| 11 | 特別競走番号 | TokuNum | varchar | 4 | | 重賞のみ設定 |
| 12 | 競走名本題 | Hondai | varchar | 60 | | 全角30文字 |
| 13 | 競走名副題 | Fukudai | varchar | 60 | | 全角30文字 |
| 14 | 競走名カッコ内 | Kakko | varchar | 60 | | 全角30文字 |
| 15 | 競走名本題欧字 | HondaiEng | varchar | 120 | | 半角120文字 |
| 16 | 競走名副題欧字 | FukudaiEng | varchar | 120 | | 半角120文字 |
| 17 | 競走名カッコ内欧字 | KakkoEng | varchar | 120 | | 半角120文字 |
| 18 | 競走名略称１０字 | Ryakusyo10 | varchar | 20 | | 全角10文字 |
| 19 | 競走名略称６字 | Ryakusyo6 | varchar | 12 | | 全角6文字 |
| 20 | 競走名略称３字 | Ryakusyo3 | varchar | 6 | | 全角3文字 |
| 21 | 競走名区分 | Kubun | varchar | 1 | | 0:初期値, 1:本題, 2:副題, 3:カッコ内 |
| 22 | 重賞回次 | Nkai | varchar | 3 | | 重賞の累計回次 |
| 23 | グレードコード | GradeCD | varchar | 1 | | コード表2003参照 |
| 24 | 競走種別コード | SyubetuCD | varchar | 2 | | コード表2005参照 |
| 25 | 競走記号コード | KigoCD | varchar | 3 | | コード表2006参照 |
| 26 | 重量種別コード | JyuryoCD | varchar | 1 | | コード表2008参照 |
| 27 | 競走条件コード 2歳 | JyokenCD1 | varchar | 3 | | コード表2007参照 |
| 28 | 競走条件コード 3歳 | JyokenCD2 | varchar | 3 | | |
| 29 | 競走条件コード 4歳 | JyokenCD3 | varchar | 3 | | |
| 30 | 競走条件コード 5歳以上 | JyokenCD4 | varchar | 3 | | |
| 31 | 競走条件コード 最若年 | JyokenCD5 | varchar | 3 | | |
| 32 | 距離 | Kyori | varchar | 4 | | メートル単位 |
| 33 | トラックコード | TrackCD | varchar | 2 | | コード表2009参照 |
| 34 | コース区分 | CourseKubunCD | varchar | 2 | | A～E |
| 35 | ハンデ発表日 | HandiDate | varchar | 8 | | yyyymmdd |
| 36 | 登録頭数 | TorokuTosu | varchar | 3 | | |

### 2.2 TOKU（特別登録馬）

特別レースへの登録馬情報。TOKU_RACEの子テーブル（最大300レコード/レース）。

| No | カラム名 | フィールド名 | 型 | 桁 | PK | 内容 |
|----|---------|-------------|---|---|---|------|
| 1 | データ作成年月日 | MakeDate | varchar | 8 | | yyyymmdd |
| 2 | 開催年 | Year | varchar | 4 | ○ | |
| 3 | 開催月日 | MonthDay | varchar | 4 | ○ | |
| 4 | 競馬場コード | JyoCD | varchar | 2 | ○ | |
| 5 | 開催回 | Kaiji | varchar | 2 | ○ | |
| 6 | 開催日目 | Nichiji | varchar | 2 | ○ | |
| 7 | レース番号 | RaceNum | varchar | 2 | ○ | |
| 8 | 連番 | Num | varchar | 3 | ○ | 1～300 |
| 9 | 血統登録番号 | KettoNum | varchar | 10 | | 4桁年+1桁品種+5桁番号 |
| 10 | 馬名 | Bamei | varchar | 36 | | 全角18文字 |
| 11 | 馬記号コード | UmaKigoCD | varchar | 2 | | コード表2204参照 |
| 12 | 性別コード | SexCD | varchar | 1 | | コード表2202参照 |
| 13 | 調教師東西所属コード | TozaiCD | varchar | 1 | | コード表2301参照 |
| 14 | 調教師コード | ChokyosiCode | varchar | 5 | | 調教師マスタとリンク |
| 15 | 調教師名略称 | ChokyosiRyakusyo | varchar | 8 | | 全角4文字 |
| 16 | 負担重量 | Futan | varchar | 3 | | 0.1kg単位 |
| 17 | 交流区分 | Koryu | varchar | 1 | | 0:初期値, 1:地方馬, 2:外国馬 |

### 2.3 RACE（レース詳細）★最重要テーブル

レースの全詳細情報。110項目。

| No | カラム名 | フィールド名 | 型 | 桁 | PK | 内容 |
|----|---------|-------------|---|---|---|------|
| 1 | レコード種別ID | RecordSpec | varchar | 2 | | "RA" |
| 2 | データ区分 | DataKubun | varchar | 1 | | 1:出馬表, 2:枠順確定, 3~6:速報, 7:成績, 9:中止, 0:削除 |
| 3 | データ作成年月日 | MakeDate | varchar | 8 | | yyyymmdd |
| 4 | 開催年 | Year | varchar | 4 | ○ | |
| 5 | 開催月日 | MonthDay | varchar | 4 | ○ | |
| 6 | 競馬場コード | JyoCD | varchar | 2 | ○ | |
| 7 | 開催回 | Kaiji | varchar | 2 | ○ | |
| 8 | 開催日目 | Nichiji | varchar | 2 | ○ | |
| 9 | レース番号 | RaceNum | varchar | 2 | ○ | |
| 10 | 曜日コード | YoubiCD | varchar | 1 | | |
| 11 | 特別競走番号 | TokuNum | varchar | 4 | | |
| 12 | 競走名本題 | Hondai | varchar | 60 | | |
| 13 | 競走名副題 | Fukudai | varchar | 60 | | |
| 14 | 競走名カッコ内 | Kakko | varchar | 60 | | |
| 15-17 | 競走名欧字 | HondaiEng/FukudaiEng/KakkoEng | varchar | 120 | | |
| 18-20 | 競走名略称 | Ryakusyo10/6/3 | varchar | 20/12/6 | | |
| 21 | 競走名区分 | Kubun | varchar | 1 | | |
| 22 | 重賞回次 | Nkai | varchar | 3 | | |
| 23 | グレードコード | GradeCD | varchar | 1 | | |
| 24 | 変更前グレードコード | GradeCDBefore | varchar | 1 | | |
| 25 | 競走種別コード | SyubetuCD | varchar | 2 | | |
| 26 | 競走記号コード | KigoCD | varchar | 3 | | |
| 27 | 重量種別コード | JyuryoCD | varchar | 1 | | |
| 28-32 | 競走条件コード | JyokenCD1～5 | varchar | 3 | | 2歳/3歳/4歳/5歳以上/最若年 |
| 33 | 競走条件名称 | JyokenName | varchar | 60 | | 地方馬のみ |
| 34 | 距離 | Kyori | varchar | 4 | | メートル単位 |
| 35 | 変更前距離 | KyoriBefore | varchar | 4 | | |
| 36 | トラックコード | TrackCD | varchar | 2 | | |
| 37 | 変更前トラックコード | TrackCDBefore | varchar | 2 | | |
| 38 | コース区分 | CourseKubunCD | varchar | 2 | | |
| 39 | 変更前コース区分 | CourseKubunCDBefore | varchar | 2 | | |
| 40-46 | 本賞金1～7 | Honsyokin1～7 | varchar | 8 | | 100円単位 |
| 47-51 | 変更前本賞金1～5 | HonsyokinBefore1～5 | varchar | 8 | | |
| 52-56 | 付加賞金1～5 | Fukasyokin1～5 | varchar | 8 | | |
| 57-59 | 変更前付加賞金1～3 | FukasyokinBefore1～3 | varchar | 8 | | |
| 60 | 発走時刻 | HassoTime | varchar | 4 | | hhmm形式 |
| 61 | 変更前発走時刻 | HassoTimeBefore | varchar | 4 | | |
| 62 | 登録頭数 | TorokuTosu | varchar | 2 | | |
| 63 | 出走頭数 | SyussoTosu | varchar | 2 | | |
| 64 | 入線頭数 | NyusenTosu | varchar | 2 | | |
| 65 | 天候コード | TenkoCD | varchar | 1 | | コード表2011参照 |
| 66 | 芝馬場状態コード | SibaBabaCD | varchar | 1 | | コード表2010参照 |
| 67 | ダート馬場状態コード | DirtBabaCD | varchar | 1 | | コード表2010参照 |
| 68-87 | ラップタイム1～20 | LapTime1～20 | varchar | 3 | | 99.9秒, 1ハロン(200m)ごと |
| 88 | 障害マイルタイム | SyogaiMileTime | varchar | 4 | | 障害レースのみ |
| 89 | 前３ハロン | HaronTimeS3 | varchar | 3 | | |
| 90 | 前４ハロン | HaronTimeS4 | varchar | 3 | | |
| 91 | 後３ハロン | HaronTimeL3 | varchar | 3 | | |
| 92 | 後４ハロン | HaronTimeL4 | varchar | 3 | | |
| 93-108 | コーナー通過順 | Corner1～4/Syukaisu1～4/Jyuni1～4 | varchar | 各種 | | |
| 109 | レコード更新区分 | RecordUpKubun | varchar | 1 | | 0:初期値, 1:基準タイム, 2:コースレコード |

---

## 3. 馬毎レース情報

### 3.1 UMA_RACE（馬毎レース情報）★最重要テーブル

各馬のレース出走情報・成績。予想の核となるテーブル。

| No | カラム名 | フィールド名 | 型 | 桁 | PK | 内容 |
|----|---------|-------------|---|---|---|------|
| 1 | レコード種別ID | RecordSpec | varchar | 2 | | "SE" |
| 2 | データ区分 | DataKubun | varchar | 1 | | 1:出馬表, 2:枠順, 3-6:速報, 7:成績, 9:中止, 0:削除 |
| 3 | データ作成年月日 | MakeDate | varchar | 8 | | |
| 4 | 開催年 | Year | varchar | 4 | ○ | |
| 5 | 開催月日 | MonthDay | varchar | 4 | ○ | |
| 6 | 競馬場コード | JyoCD | varchar | 2 | ○ | |
| 7 | 開催回 | Kaiji | varchar | 2 | ○ | |
| 8 | 開催日目 | Nichiji | varchar | 2 | ○ | |
| 9 | レース番号 | RaceNum | varchar | 2 | ○ | |
| 10 | 枠番 | Wakuban | varchar | 1 | | |
| 11 | 馬番 | Umaban | varchar | 2 | ○ | |
| 12 | 血統登録番号 | KettoNum | varchar | 10 | ○ | |
| 13 | 馬名 | Bamei | varchar | 36 | | |
| 14 | 馬記号コード | UmaKigoCD | varchar | 2 | | |
| 15 | 性別コード | SexCD | varchar | 1 | | |
| 16 | 品種コード | HinsyuCD | varchar | 1 | | |
| 17 | 毛色コード | KeiroCD | varchar | 2 | | |
| 18 | 馬齢 | Barei | varchar | 2 | | 2001年～は満年齢 |
| 19 | 東西所属コード | TozaiCD | varchar | 1 | | |
| 20 | 調教師コード | ChokyosiCode | varchar | 5 | | |
| 21 | 調教師名略称 | ChokyosiRyakusyo | varchar | 8 | | |
| 22 | 馬主コード | BanusiCode | varchar | 6 | | |
| 23 | 馬主名 | BanusiName | varchar | 64 | | |
| 24 | 服色標示 | Fukusyoku | varchar | 60 | | |
| 25 | 予備1 | reserved1 | varchar | 60 | | |
| 26 | 負担重量 | Futan | varchar | 3 | | 0.1kg単位 |
| 27 | 変更前負担重量 | FutanBefore | varchar | 3 | | |
| 28 | ブリンカー使用区分 | Blinker | varchar | 1 | | 0:不使用, 1:使用 |
| 29 | 予備2 | reserved2 | varchar | 1 | | |
| 30 | 騎手コード | KisyuCode | varchar | 5 | | |
| 31 | 変更前騎手コード | KisyuCodeBefore | varchar | 5 | | |
| 32 | 騎手名略称 | KisyuRyakusyo | varchar | 8 | | |
| 33 | 変更前騎手名略称 | KisyuRyakusyoBefore | varchar | 8 | | |
| 34 | 騎手見習コード | MinaraiCD | varchar | 1 | | |
| 35 | 変更前騎手見習コード | MinaraiCDBefore | varchar | 1 | | |
| 36 | 馬体重 | BaTaijyu | varchar | 3 | | kg, 999:計量不能, 000:出走取消 |
| 37 | 増減符号 | ZogenFugo | varchar | 1 | | +:増, -:減, 空白:その他 |
| 38 | 増減差 | ZogenSa | varchar | 3 | | kg単位 |
| 39 | 異常区分コード | IJyoCD | varchar | 1 | | コード表2101参照 |
| 40 | 入線順位 | NyusenJyuni | varchar | 2 | | 降着・失格前の着順 |
| 41 | 確定着順 | KakuteiJyuni | varchar | 2 | | 最終確定着順 |
| 42 | 同着区分 | DochakuKubun | varchar | 1 | | 0:なし, 1:同着あり |
| 43 | 同着頭数 | DochakuTosu | varchar | 1 | | |
| 44 | 走破タイム | Time | varchar | 4 | | 最大9:59.9 |
| 45 | 着差コード | ChakusaCD | varchar | 3 | | コード表2102参照 |
| 46 | ＋着差コード | ChakusaCDP | varchar | 3 | | 前走馬失格時 |
| 47 | ＋＋着差コード | ChakusaCDPP | varchar | 3 | | 前々走馬失格時 |
| 48 | 1コーナーでの順位 | Jyuni1c | varchar | 2 | | |
| 49 | 2コーナーでの順位 | Jyuni2c | varchar | 2 | | |
| 50 | 3コーナーでの順位 | Jyuni3c | varchar | 2 | | |
| 51 | 4コーナーでの順位 | Jyuni4c | varchar | 2 | | |
| 52 | 単勝オッズ | Odds | varchar | 4 | | 999.9倍 |
| 53 | 単勝人気順 | Ninki | varchar | 2 | | |
| 54 | 獲得本賞金 | Honsyokin | varchar | 8 | | 100円単位 |
| 55 | 獲得付加賞金 | Fukasyokin | varchar | 8 | | 100円単位 |
| 56-57 | 予備3-4 | reserved3-4 | varchar | 3 | | |
| 58 | 後4ハロンタイム | HaronTimeL4 | varchar | 3 | | 99.9秒 |
| 59 | 後3ハロンタイム | HaronTimeL3 | varchar | 3 | | 99.9秒 |
| 60-65 | 相手馬1～3情報 | KettoNum1-3/Bamei1-3 | varchar | 10/36 | | |
| 66 | タイム差 | TimeDIFN | varchar | 4 | | 1着馬とのタイム差 |
| 67 | レコード更新区分 | RecordUpKubun | varchar | 1 | | |
| 68 | マイニング区分 | DMKubun | varchar | 1 | | 1:前日, 2:当日, 3:直前 |
| 69 | マイニング予想走破タイム | DMTime | varchar | 5 | | 9:99.99形式 |
| 70 | マイニング予想誤差＋ | DMGosaP | varchar | 4 | | |
| 71 | マイニング予想誤差－ | DMGosaM | varchar | 4 | | |
| 72 | マイニング予想順位 | DMJyuni | varchar | 2 | | 01～18 |
| 73 | 今回レース脚質判定 | KyakusituKubun | varchar | 1 | | 1:逃げ, 2:先行, 3:差し, 4:追込, 0:初期値 |

---

## 4. 払戻・票数テーブル

### 4.1 HARAI（払戻）

各賭式の払戻情報。199項目の大規模テーブル。

| No | カラム名 | フィールド名 | 型 | 桁 | PK | 内容 |
|----|---------|-------------|---|---|---|------|
| 1 | レコード種別ID | RecordSpec | varchar | 2 | | "HR" |
| 2 | データ区分 | DataKubun | varchar | 1 | | 1:速報(払戻確定), 2:成績, 9:中止, 0:削除 |
| 3 | データ作成年月日 | MakeDate | varchar | 8 | | |
| 4-9 | **レースキー** | Year/MonthDay/JyoCD/Kaiji/Nichiji/RaceNum | varchar | 各種 | ○ | |
| 10 | 登録頭数 | TorokuTosu | varchar | 2 | | |
| 11 | 出走頭数 | SyussoTosu | varchar | 2 | | |
| 12-20 | 不成立フラグ1～9 | FuseirituFlag1-9 | varchar | 1 | | 単勝/複勝/枠連/馬連/ワイド/予備/馬単/3連複/3連単 |
| 21-29 | 特払フラグ1～9 | TokubaraiFlag1-9 | varchar | 1 | | |
| 30-38 | 返還フラグ1～9 | HenkanFlag1-9 | varchar | 1 | | |
| 39-66 | 返還馬番情報 | HenkanUma1-28 | varchar | 1 | | |
| 67-74 | 返還枠番情報 | HenkanWaku1-8 | varchar | 1 | | |
| 75-82 | 返還同枠情報 | HenkanDoWaku1-8 | varchar | 1 | | |
| 83-100 | 単勝払戻(3組) | PayTansyo* | varchar | 各種 | | 馬番/払戻金/人気 |
| 101-115 | 複勝払戻(5組) | PayFukusyo* | varchar | 各種 | | |
| 116-124 | 枠連払戻(3組) | PayWakuren* | varchar | 各種 | | |
| 125-133 | 馬連払戻(3組) | PayUmaren* | varchar | 各種 | | |
| 134-148 | ワイド払戻(5組) | PayWide* | varchar | 各種 | | |
| 149-157 | 馬単払戻(3組) | PayUmatan* | varchar | 各種 | | |
| 158-166 | 3連複払戻(3組) | PaySanren* | varchar | 各種 | | |
| 167-175 | 3連単払戻(3組) | PaySanrentan* | varchar | 各種 | | |

### 4.2 票数テーブル群

#### HYOSU（票数ヘッダ）

| No | カラム名 | フィールド名 | 型 | 桁 | PK | 内容 |
|----|---------|-------------|---|---|---|------|
| 1 | レコード種別ID | RecordSpec | varchar | 2 | | "H1" |
| 2 | データ区分 | DataKubun | varchar | 1 | | 2:前日売最終, 4:確定, 5:確定(月曜), 9:中止, 0:削除 |
| 3-9 | **レースキー** | MakeDate/Year/MonthDay/JyoCD/Kaiji/Nichiji/RaceNum | varchar | 各種 | ○ | |
| 10-11 | 頭数情報 | TorokuTosu/SyussoTosu | varchar | 2 | | |
| 12-18 | 発売フラグ | HatubaiFlag1-7 | varchar | 1 | | 単勝/複勝/枠連/馬連/ワイド/馬単/3連複 |
| 19 | 複勝着払キー | FukuChakuBaraiKey | varchar | 1 | | 0:発売なし, 2:2着まで, 3:3着まで |
| 20-41 | 返還馬番情報 | HenkanUma1-22 | varchar | 1 | | |

#### HYOSU_TANPUKU（票数_単複）

| No | カラム名 | フィールド名 | 型 | 桁 | PK | 内容 |
|----|---------|-------------|---|---|---|------|
| 1-7 | **レースキー** | MakeDate～RaceNum | varchar | 各種 | ○ | |
| 8 | 馬番 | Umaban | varchar | 2 | ○ | |
| 9 | 単勝票数 | TanHyo | varchar | 11 | | 100円単位 |
| 10 | 単勝人気順 | TanNinki | varchar | 2 | | |
| 11 | 複勝票数 | FukuHyo | varchar | 11 | | 100円単位 |
| 12 | 複勝人気順 | FukuNinki | varchar | 2 | | |

#### HYOSU_WAKU（票数_枠連）

| No | カラム名 | フィールド名 | 型 | 桁 | PK | 内容 |
|----|---------|-------------|---|---|---|------|
| 1-7 | **レースキー** | | varchar | 各種 | ○ | |
| 8 | 組番 | Kumi | varchar | 2 | ○ | |
| 9 | 票数 | Hyo | varchar | 11 | | 100円単位 |
| 10 | 人気順 | Ninki | varchar | 2 | | |

#### HYOSU_UMARENWIDE（票数_馬連_ワイド）

| No | カラム名 | フィールド名 | 型 | 桁 | PK | 内容 |
|----|---------|-------------|---|---|---|------|
| 1-7 | **レースキー** | | varchar | 各種 | ○ | |
| 8 | 組番 | Kumi | varchar | 4 | ○ | |
| 9 | 馬連票数 | UmarenHyo | varchar | 11 | | 100円単位 |
| 10 | 馬連人気順 | UmarenNinki | varchar | 3 | | |
| 11 | ワイド票数 | WideHyo | varchar | 11 | | 100円単位 |
| 12 | ワイド人気順 | WideNinki | varchar | 3 | | |

#### HYOSU_UMATAN（票数_馬単）/ HYOSU_SANREN（票数_3連複）/ HYOSU_SANRENTAN（票数_3連単）

共通構造: レースキー + 組番(Kumi) + 票数(Hyo) + 人気順(Ninki)
- 馬単: 組番4桁, 人気3桁
- 3連複: 組番6桁, 人気3桁
- 3連単: 組番6桁, 人気4桁

---

## 5. オッズテーブル

### 共通キー構造
すべてのオッズテーブルのPK: Year + MonthDay + JyoCD + Kaiji + Nichiji + RaceNum

### 5.1 ヘッダテーブル群

| テーブル | レコード種別 | 発売フラグ | 票数合計 |
|---------|-------------|----------|---------|
| ODDS_TANPUKUWAKU_HEAD | O1 | 単勝/複勝/枠連 | 単勝/複勝/枠連 各11桁 |
| ODDS_UMAREN_HEAD | O2 | 馬連 | 馬連 11桁 |
| ODDS_WIDE_HEAD | O3 | ワイド | ワイド 11桁 |
| ODDS_UMATAN_HEAD | O4 | 馬単 | 馬単 11桁 |
| ODDS_SANREN_HEAD | O5 | 3連複 | 3連複 11桁 |
| ODDS_SANRENTAN_HEAD | O6 | 3連単 | 3連単 11桁 |

ヘッダ共通カラム: RecordSpec, DataKubun(1:中間/2:前日売最終/3:最終/4:確定/5:確定月曜/9:中止/0:削除), MakeDate, レースキー, HappyoTime, TorokuTosu, SyussoTosu, 発売フラグ, 票数合計

### 5.2 明細テーブル群

#### ODDS_TANPUKU（オッズ_単複）

| No | カラム名 | フィールド名 | 型 | 桁 | PK | 内容 |
|----|---------|-------------|---|---|---|------|
| 1-7 | **レースキー** | | varchar | 各種 | ○ | |
| 8 | 馬番 | Umaban | varchar | 2 | ○ | |
| 9 | 単勝オッズ | TanOdds | varchar | 4 | | 999.9倍, "9999":999.9以上, "0000":無投票, "----":前取消, "****":後取消 |
| 10 | 単勝人気順 | TanNinki | varchar | 2 | | |
| 11 | 複勝最低オッズ | FukuOddsLow | varchar | 4 | | 999.9倍 |
| 12 | 複勝最高オッズ | FukuOddsHigh | varchar | 4 | | 999.9倍 |
| 13 | 複勝人気順 | FukuNinki | varchar | 2 | | |

#### ODDS_WAKU（オッズ_枠連）

レースキー + 組番(Kumi, 2桁) + オッズ(Odds, 5桁=9999.9倍) + 人気順(Ninki, 2桁)

#### ODDS_UMAREN / ODDS_UMATAN（オッズ_馬連 / 馬単）

レースキー + 組番(Kumi, 4桁) + オッズ(Odds, 6桁=99999.9倍) + 人気順(Ninki, 3桁)

#### ODDS_WIDE（オッズ_ワイド）

レースキー + 組番(Kumi, 4桁) + 最低オッズ(OddsLow, 5桁) + 最高オッズ(OddsHigh, 5桁) + 人気順(Ninki, 3桁)

#### ODDS_SANREN / ODDS_SANRENTAN（オッズ_3連複 / 3連単）

- 3連複: レースキー + 組番(6桁) + オッズ(6桁=99999.9倍) + 人気順(3桁)
- 3連単: レースキー + 組番(6桁) + オッズ(7桁=999999.9倍) + 人気順(4桁)

---

## 6. 時系列オッズテーブル

通常のオッズテーブルと同じ構造に、**発表月日時分(HappyoTime, 8桁)**がPKに追加されます。
これにより、オッズの時間変化を追跡できます。

| テーブル | 対応する通常オッズ | 追加PK |
|---------|-----------------|--------|
| JODDS_TANPUKUWAKU_HEAD | ODDS_TANPUKUWAKU_HEAD | HappyoTime |
| JODDS_TANPUKU | ODDS_TANPUKU | HappyoTime |
| JODDS_WAKU | ODDS_WAKU | HappyoTime |
| JODDS_UMAREN_HEAD | ODDS_UMAREN_HEAD | HappyoTime |
| JODDS_UMAREN | ODDS_UMAREN | HappyoTime |

---

## 7. マスタテーブル

### 7.1 UMA（競走馬マスタ）★最重要

| No | カラム名 | フィールド名 | 型 | 桁 | PK | 内容 |
|----|---------|-------------|---|---|---|------|
| 1 | レコード種別ID | RecordSpec | varchar | 2 | | "UM" |
| 2 | データ区分 | DataKubun | varchar | 1 | | 1:新規, 2:馬名変更, 3:再登録, 4:その他, 9:抹消, 0:削除 |
| 3 | データ作成年月日 | MakeDate | varchar | 8 | | |
| 4 | 血統登録番号 | KettoNum | varchar | 10 | ○ | 4桁年+1桁品種+5桁番号 |
| 5 | 競走馬抹消区分 | DelKubun | varchar | 1 | | 0:現役, 1:抹消 |
| 6 | 競走馬登録年月日 | RegDate | varchar | 8 | | |
| 7 | 競走馬抹消年月日 | DelDate | varchar | 8 | | |
| 8 | 生年月日 | BirthDate | varchar | 8 | | |
| 9 | 馬名 | Bamei | varchar | 36 | | 全角18文字 |
| 10 | 馬名半角ｶﾅ | BameiKana | varchar | 36 | | |
| 11 | 馬名欧字 | BameiEng | varchar | 60 | | |
| 12 | JRA施設在きゅうフラグ | ZaikyuFlag | varchar | 1 | | 0:JRA施設外, 1:JRA施設内 |
| 13 | 予備 | Reserved | varchar | 19 | | |
| 14 | 馬記号コード | UmaKigoCD | varchar | 2 | | |
| 15 | 性別コード | SexCD | varchar | 1 | | |
| 16 | 品種コード | HinsyuCD | varchar | 1 | | |
| 17 | 毛色コード | KeiroCD | varchar | 2 | | |
| 18-29 | 3代血統情報 | Ketto3Info* | varchar | 10/36 | | 父/母/父父/父母/母父/母母の繁殖登録番号と馬名(6組) |
| 30-45 | 4代血統情報（続き） | Ketto3Info* | varchar | 10/36 | | 3代目以降の繁殖登録番号と馬名 |

### 7.2 KISYU（騎手マスタ）

| No | カラム名 | フィールド名 | 型 | 桁 | PK | 内容 |
|----|---------|-------------|---|---|---|------|
| 1 | レコード種別ID | RecordSpec | varchar | 2 | | "KS" |
| 2 | データ区分 | DataKubun | varchar | 1 | | 1:新規, 2:更新, 0:削除 |
| 3 | データ作成年月日 | MakeDate | varchar | 8 | | |
| 4 | 騎手コード | KisyuCode | varchar | 5 | ○ | |
| 5 | 騎手抹消区分 | DelKubun | varchar | 1 | | 0:現役, 1:抹消 |
| 6 | 騎手免許交付年月日 | IssueDate | varchar | 8 | | |
| 7 | 騎手免許抹消年月日 | DelDate | varchar | 8 | | |
| 8 | 生年月日 | BirthDate | varchar | 8 | | |
| 9 | 騎手名 | KisyuName | varchar | 34 | | 全角17文字 |
| 10 | 予備 | reserved | varchar | 34 | | |
| 11 | 騎手名半角ｶﾅ | KisyuNameKana | varchar | 30 | | |
| 12 | 騎手名略称 | KisyuRyakusyo | varchar | 8 | | 全角4文字 |
| 13 | 騎手名欧字 | KisyuNameEng | varchar | 80 | | |
| 14 | 性別区分 | SexCD | varchar | 1 | | 1:男, 2:女 |
| 15 | 騎乗資格コード | SikakuCD | varchar | 1 | | コード表2302参照 |
| 16 | 騎手見習コード | MinaraiCD | varchar | 1 | | コード表2303参照 |
| 17 | 騎手東西所属コード | TozaiCD | varchar | 1 | | |
| 18 | 招待地域名 | Syotai | varchar | 20 | | |
| 19 | 所属調教師コード | ChokyosiCode | varchar | 5 | | 00000:フリー |
| 20 | 所属調教師名略称 | ChokyosiRyakusyo | varchar | 8 | | |
| 21-32 | 初騎乗情報（平地・障害） | HatuKiJyo1*/2* | varchar | 各種 | | |
| 33-45 | 初勝利情報・最近重賞勝利 | HatuSyori*/SaikinJyusyo* | varchar | 各種 | | |

### 7.3 KISYU_SEISEKI（騎手成績）

騎手の年度別・競馬場別成績。

| No | カラム名 | フィールド名 | 型 | 桁 | PK | 内容 |
|----|---------|-------------|---|---|---|------|
| 1 | データ作成年月日 | MakeDate | varchar | 8 | | |
| 2 | 騎手コード | KisyuCode | varchar | 5 | ○ | |
| 3 | 連番 | Num | varchar | 1 | ○ | |
| 4 | 設定年 | SetYear | varchar | 4 | | |
| 5-8 | 賞金合計 | HonSyokin*/FukaSyokin* | varchar | 10 | | 平地/障害 本賞金/付加賞金 |
| 9-20 | 全体着回数 | HeichiChakukaisu1-6/SyogaiChakukaisu1-6 | varchar | 6 | | 平地/障害 1着～6着外 |
| 21+ | 競馬場別着回数 | Jyo*Chakukaisu* | varchar | 6 | | 札幌/函館/福島/新潟/東京/中山/中京/京都/阪神/小倉 各平地・障害 |

### 7.4 CHOKYO（調教師マスタ）

| No | カラム名 | フィールド名 | 型 | 桁 | PK | 内容 |
|----|---------|-------------|---|---|---|------|
| 1 | レコード種別ID | RecordSpec | varchar | 2 | | "CH" |
| 4 | 調教師コード | ChokyosiCode | varchar | 5 | ○ | |
| 5 | 調教師抹消区分 | DelKubun | varchar | 1 | | |
| 6-7 | 免許日付 | IssueDate/DelDate | varchar | 8 | | |
| 8 | 生年月日 | BirthDate | varchar | 8 | | |
| 9 | 調教師名 | ChokyosiName | varchar | 34 | | |
| 10-12 | 名称 | ChokyosiNameKana/Ryakusyo/Eng | varchar | 各種 | | |
| 13 | 性別区分 | SexCD | varchar | 1 | | |
| 14 | 東西所属コード | TozaiCD | varchar | 1 | | |
| 15 | 招待地域名 | Syotai | varchar | 20 | | |
| 16-42 | 最近重賞勝利1～3 | SaikinJyusyo1-3* | varchar | 各種 | | |

### 7.5 CHOKYO_SEISEKI（調教師成績）

KISYU_SEISEKIと同構造。ChokyosiCodeがPK。

### 7.6 SEISAN（生産者マスタ）

| No | カラム名 | フィールド名 | 型 | 桁 | PK | 内容 |
|----|---------|-------------|---|---|---|------|
| 4 | 生産者コード | BreederCode | varchar | 8 | ○ | |
| 5-6 | 生産者名 | BreederName_Co/BreederName | varchar | 72 | | 法人格有無 |
| 7 | 生産者名半角ｶﾅ | BreederNameKana | varchar | 72 | | |
| 8 | 生産者名欧字 | BreederNameEng | varchar | 168 | | |
| 9 | 住所 | Address | varchar | 20 | | |
| 10-18 | 本年成績 | H_SetYear/H_HonSyokinTotal/H_FukaSyokin/H_ChakuKaisu1-6 | varchar | 各種 | | |
| 19-27 | 累計成績 | R_SetYear/R_HonSyokinTotal/R_FukaSyokin/R_ChakuKaisu1-6 | varchar | 各種 | | |

### 7.7 BANUSI（馬主マスタ）

| No | カラム名 | フィールド名 | 型 | 桁 | PK | 内容 |
|----|---------|-------------|---|---|---|------|
| 4 | 馬主コード | BanusiCode | varchar | 6 | ○ | |
| 5-6 | 馬主名 | BanusiName_Co/BanusiName | varchar | 64 | | |
| 7 | 馬主名半角ｶﾅ | BanusiNameKana | varchar | 50 | | |
| 8 | 馬主名欧字 | BanusiNameEng | varchar | 100 | | |
| 9 | 服色標示 | Fukusyoku | varchar | 60 | | |
| 10-18 | 本年成績 | H_* | varchar | 各種 | | |
| 19-27 | 累計成績 | R_* | varchar | 各種 | | |

### 7.8 HANSYOKU（繁殖馬マスタ）

| No | カラム名 | フィールド名 | 型 | 桁 | PK | 内容 |
|----|---------|-------------|---|---|---|------|
| 4 | 繁殖登録番号 | HansyokuNum | varchar | 10 | ○ | 1頭に複数番号あり |
| 6 | 血統登録番号 | KettoNum | varchar | 10 | | |
| 8 | 馬名 | Bamei | varchar | 36 | | |
| 11 | 生年 | BirthYear | varchar | 4 | | |
| 12 | 性別コード | SexCD | varchar | 1 | | |
| 13 | 品種コード | HinsyuCD | varchar | 1 | | |
| 14 | 毛色コード | KeiroCD | varchar | 2 | | |
| 15 | 持込区分 | HansyokuMochiKubun | varchar | 1 | | 0:内国産, 1:持込, 2:輸入, 3:輸入, 9:その他 |
| 18 | 父馬繁殖登録番号 | HansyokuFNum | varchar | 10 | | |
| 19 | 母馬繁殖登録番号 | HansyokuMNum | varchar | 10 | | |

### 7.9 SANKU（産駒マスタ）

| No | カラム名 | フィールド名 | 型 | 桁 | PK | 内容 |
|----|---------|-------------|---|---|---|------|
| 4 | 血統登録番号 | KettoNum | varchar | 10 | ○ | |
| 5 | 生年月日 | BirthDate | varchar | 8 | | |
| 6 | 性別コード | SexCD | varchar | 1 | | |
| 7 | 品種コード | HinsyuCD | varchar | 1 | | |
| 8 | 毛色コード | KeiroCD | varchar | 2 | | |
| 9 | 産駒持込区分 | SankuMochiKubun | varchar | 1 | | |
| 11 | 生産者コード | BreederCode | varchar | 8 | | |
| 12 | 産地名 | SanchiName | varchar | 20 | | |
| 13 | 父繁殖登録番号 | FNum | varchar | 10 | | |
| 14 | 母繁殖登録番号 | MNum | varchar | 10 | | |
| 15-26 | 4代祖先繁殖番号 | FF/FM/MF/MM/FFF/FFM/FMF/FMM/MFF/MFM/MMF/MMMNum | varchar | 10 | | |

---

## 8. 調教データ

### 8.1 HANRO（坂路調教）

| No | カラム名 | フィールド名 | 型 | 桁 | PK | 内容 |
|----|---------|-------------|---|---|---|------|
| 4 | トレセン区分 | TresenKubun | varchar | 1 | ○ | 0:美浦, 1:栗東 |
| 5 | 調教年月日 | ChokyoDate | varchar | 8 | ○ | |
| 6 | 調教時刻 | ChokyoTime | varchar | 4 | ○ | hhmm |
| 7 | 血統登録番号 | KettoNum | varchar | 10 | ○ | |
| 8 | 4ハロンタイム合計 | HaronTime4 | varchar | 4 | | 800M～0M合計(999.9秒) |
| 9 | ラップタイム(800M～600M) | LapTime4 | varchar | 3 | | 99.9秒 |
| 10 | 3ハロンタイム合計 | HaronTime3 | varchar | 4 | | 600M～0M合計 |
| 11 | ラップタイム(600M～400M) | LapTime3 | varchar | 3 | | |
| 12 | 2ハロンタイム合計 | HaronTime2 | varchar | 4 | | 400M～0M合計 |
| 13 | ラップタイム(400M～200M) | LapTime2 | varchar | 3 | | |
| 14 | ラップタイム(200M～0M) | LapTime1 | varchar | 3 | | |

### 8.2 WOOD_CHIP（ウッドチップ調教）

| No | カラム名 | フィールド名 | 型 | 桁 | PK | 内容 |
|----|---------|-------------|---|---|---|------|
| 4 | トレセン区分 | TresenKubun | varchar | 1 | ○ | 0:美浦, 1:栗東 |
| 5 | 調教年月日 | ChokyoDate | varchar | 8 | ○ | |
| 6 | 調教時刻 | ChokyoTime | varchar | 4 | ○ | hhmm |
| 7 | 血統登録番号 | KettoNum | varchar | 10 | ○ | |
| 8 | コース | Course | varchar | 1 | | 0:A, 1:B, 2:C, 3:D, 4:E |
| 9 | 馬場周り | BabaAround | varchar | 1 | | 0:右, 1:左 |
| 11-29 | タイムデータ | HaronTime10～LapTime1 | varchar | 3-4 | | 10ハロン分のタイムとラップ |

---

## 9. その他のテーブル

### 9.1 BATAIJYU（馬体重）

発走前の馬体重発表データ。1レコードに最大8頭分の情報。

| No | カラム名 | 内容 |
|----|---------|------|
| 1-9 | レースキー | Year～RaceNum |
| 10 | 発表月日時分(HappyoTime) | mmddHHmm |
| 11-15 | 馬番1情報 | Umaban1/Bamei1/BaTaijyu1/ZogenFugo1/ZogenSa1 |
| 16-40 | 馬番2～8情報 | 同構造の繰り返し |

### 9.2 TENKO_BABA（天候馬場状態）

| No | カラム名 | フィールド名 | 型 | 桁 | PK | 内容 |
|----|---------|-------------|---|---|---|------|
| 1-8 | レースキー | | varchar | 各種 | ○ | |
| 9 | 発表月日時分 | HappyoTime | varchar | 8 | ○ | |
| 10 | 変更識別 | HenkoID | varchar | 1 | ○ | 1:初期, 2:天候変更, 3:馬場変更 |
| 11 | 現在天候 | AtoTenkoCD | varchar | 1 | | |
| 12 | 現在馬場・芝 | AtoSibaBabaCD | varchar | 1 | | |
| 13 | 現在馬場・ダート | AtoDirtBabaCD | varchar | 1 | | |
| 14-16 | 変更前情報 | Mae* | varchar | 1 | | |

### 9.3 MINING（データマイニング予想）

| No | カラム名 | 内容 |
|----|---------|------|
| 1-9 | レースキー + MakeHM | |
| 10 | データ作成時分 | hhmm |
| 11-46 | 馬番1～9情報 | Umaban/DMTime(予想タイム)/DMGosaP(誤差+)/DMGosaM(誤差-) × 9頭 |

### 9.4 TAISENGATA_MINING（対戦型データマイニング予想）

| No | カラム名 | 内容 |
|----|---------|------|
| 1-10 | レースキー + MakeHM | |
| 11-46 | 馬番1～18情報 | Umaban/TMScore(予想スコア 0.0～100.0) × 18頭 |

### 9.5 その他

| テーブル | PK | 主要カラム |
|---------|---|----------|
| TORIKESI_JYOGAI | レースキー+Umaban | 事由区分(001:疾病, 002:事故, 003:その他) |
| KISYU_CHANGE | レースキー+HappyoTime+Umaban | 変更前後の騎手コード/負担重量/見習コード |
| HASSOU_JIKOKU_CHANGE | レースキー+HappyoTime | 変更前後の時/分 |
| COURSE_CHANGE | レースキー+HappyoTime | 変更前後の距離/トラックコード, 事由(1:強風, 2:台風, 3:降雪, 4:その他) |
| RECORD | RecInfoKubun+レースキー+SyubetuCD+Kyori+TrackCD | レコードタイム/天候/馬場/記録保持馬情報(3頭分) |
| SCHEDULE | レースキー | 曜日/重賞案内(3レース分) |
| SALE | KettoNum+SaleCode+FromDate | 取引価格(円) |
| BAMEIORIGIN | KettoNum | 馬名の意味由来(全角32文字) |
| KEITO | HansyokuNum | 系統ID/系統名/系統説明 |
| COURSE | JyoCD+Kyori+TrackCD | コース改修日/コース説明 |
| JOGAIBA | レースキー+KettoNum | 出走区分/除外状態区分 |

---

## 10. コード表

### 2001. 競馬場コード

**JRA中央競馬場:**

| コード | 名称 |
|--------|------|
| 01 | 札幌 |
| 02 | 函館 |
| 03 | 福島 |
| 04 | 新潟 |
| 05 | 東京 |
| 06 | 中山 |
| 07 | 中京 |
| 08 | 京都 |
| 09 | 阪神 |
| 10 | 小倉 |

**地方競馬場:**

| コード | 名称 | コード | 名称 |
|--------|------|--------|------|
| 30 | 門別 | 46 | 金沢 |
| 31 | 北見 | 47 | 笠松 |
| 32 | 岩見沢 | 48 | 名古屋 |
| 33 | 帯広 | 49 | 紀三井寺 |
| 34 | 旭川 | 50 | 園田 |
| 35 | 盛岡 | 51 | 姫路 |
| 36 | 水沢 | 52 | 益田 |
| 37 | 上山 | 53 | 福山 |
| 38 | 三条 | 54 | 高知 |
| 39 | 足利 | 55 | 佐賀 |
| 40 | 宇都宮 | 56 | 荒尾 |
| 41 | 高崎 | 57 | 中津 |
| 42 | 浦和 | 58 | 札幌(地方) |
| 43 | 船橋 | 59 | 函館(地方) |
| 44 | 大井 | 60 | 新潟(地方) |
| 45 | 川崎 | 61 | 中京(地方) |

**海外:** A0:その他外国, A2:日本, A4:アメリカ, A6:イギリス, A8:フランス, B0:インド, B2:アイルランド, B4:ニュージーランド, B6:オーストラリア, B8:カナダ, C0:イタリア, C2:ドイツ, C7:UAE, D0:スウェーデン, D6:ロシア, E2:アルゼンチン, E4:ブラジル, F0:韓国, F1:中国, G0:香港, H2:南アフリカ, H4:スイス, M0:シンガポール ほか

### 2002. 曜日コード

| コード | 曜日 |
|--------|------|
| 0 | 未設定 |
| 1 | 土曜日 |
| 2 | 日曜日 |
| 3 | 祝日 |
| 4 | 月曜日 |
| 5 | 火曜日 |
| 6 | 水曜日 |
| 7 | 木曜日 |
| 8 | 金曜日 |

### 2003. グレードコード

| コード | 内容 |
|--------|------|
| A | G1（平地） |
| B | G2（平地） |
| C | G3（平地） |
| D | グレードのない重賞 |
| E | 重賞以外の特別競走 |
| F | J・G1（障害） |
| G | J・G2（障害） |
| H | J・G3（障害） |
| （空白） | 一般競走 |

### 2005. 競走種別コード

| コード | 内容 |
|--------|------|
| 11 | サラ系2歳 |
| 12 | サラ系3歳 |
| 13 | サラ系3歳以上 |
| 14 | サラ系4歳以上 |
| 18 | サラ系障害3歳以上 |
| 19 | サラ系障害4歳以上 |

### 2007. 競走条件コード

| コード | 内容 |
|--------|------|
| 001-100 | 収得賞金100万円単位（001=100万以下, ..., 100=1億以下） |
| 701 | 新馬 |
| 702 | 未出走 |
| 703 | 未勝利 |
| 999 | オープン |

### 2008. 重量種別コード

| コード | 内容 |
|--------|------|
| 1 | ハンデ |
| 2 | 別定 |
| 3 | 馬齢 |
| 4 | 定量 |

### 2009. トラックコード

| コード | 内容 |
|--------|------|
| 10 | 芝・直線 |
| 11 | 芝・左 |
| 12 | 芝・左外 |
| 17 | 芝・右 |
| 18 | 芝・右外 |
| 23 | ダート・左 |
| 24 | ダート・右 |
| 29 | ダート・直線 |
| 51-59 | 障害 |

### 2010. 馬場状態コード

| コード | 内容 |
|--------|------|
| 1 | 良 |
| 2 | 稍重 |
| 3 | 重 |
| 4 | 不良 |

### 2011. 天候コード

| コード | 内容 |
|--------|------|
| 1 | 晴 |
| 2 | 曇 |
| 3 | 雨 |
| 4 | 小雨 |
| 5 | 雪 |
| 6 | 小雪 |

### 2101. 異常区分コード

| コード | 内容 |
|--------|------|
| 0 | 異常なし |
| 1 | 出走取消 |
| 2 | 発走除外 |
| 3 | 競走除外 |
| 4 | 競走中止 |
| 5 | 失格 |
| 6 | 落馬再騎乗 |
| 7 | 降着 |

### 2102. 着差コード

| コード | 内容 |
|--------|------|
| H__ | ハナ |
| A__ | アタマ |
| K__ | クビ |
| _12 | 1/2馬身 |
| _34 | 3/4馬身 |
| 1__ | 1馬身 |
| 112 | 1 1/2馬身 |
| 114 | 1 1/4馬身 |
| 134 | 1 3/4馬身 |
| 2__～Z__ | 2～10馬身 |
| T__ | 大差 |
| D__ | 同着 |

### 2201. 品種コード

| コード | 内容 |
|--------|------|
| 1 | サラブレッド |
| 2 | サラブレッド系種 |
| 5 | アングロアラブ |
| 6 | アラブ系種 |
| 7 | アラブ |

### 2202. 性別コード

| コード | 内容 |
|--------|------|
| 1 | 牡 |
| 2 | 牝 |
| 3 | セン |

### 2203. 毛色コード

| コード | 内容 |
|--------|------|
| 01 | 栗毛 |
| 02 | 栃栗毛 |
| 03 | 鹿毛 |
| 04 | 黒鹿毛 |
| 05 | 青鹿毛 |
| 06 | 青毛 |
| 07 | 芦毛 |
| 08 | 栗粕毛 |
| 09 | 鹿粕毛 |
| 10 | 青粕毛 |
| 11 | 白毛 |

### 2204. 馬記号コード

| コード | 内容 |
|--------|------|
| 00 | 記号なし |
| 01 | (抽) |
| 03 | (父) |
| 04 | (市) |
| 05 | (地) |
| 06 | (外) |
| 15 | (招) |
| 21 | [地] |
| 26 | [外] |

### 2301. 東西所属コード

| コード | 内容 |
|--------|------|
| 1 | 関東（美浦） |
| 2 | 関西（栗東） |
| 3 | 地方招待 |
| 4 | 外国招待 |

### 2302. 騎乗資格コード

| コード | 内容 |
|--------|------|
| 1 | 平地・障害 |
| 2 | 平地のみ |
| 3 | 障害のみ |

### 2303. 騎手見習コード

| コード | 内容 | 減量 |
|--------|------|------|
| 1 | ☆ | 1kg減 |
| 2 | △ | 2kg減 |
| 3 | ▲ | 3kg減 |

---

## テーブル間リレーション

### レースキー（共通結合キー）
Year + MonthDay + JyoCD + Kaiji + Nichiji + RaceNum の6カラムで、ほぼすべてのテーブルが結合可能。

### 主要リレーション図

```
UMA（競走馬マスタ）
  └─ KettoNum ──→ UMA_RACE（馬毎レース情報）
  └─ KettoNum ──→ HANRO（坂路調教）
  └─ KettoNum ──→ WOOD_CHIP（ウッドチップ調教）
  └─ KettoNum ──→ SALE（市場取引価格）
  └─ KettoNum ──→ BAMEIORIGIN（馬名由来）
  └─ KettoNum ──→ SANKU（産駒マスタ）

RACE（レース詳細）
  └─ レースキー ──→ UMA_RACE（馬毎レース情報）
  └─ レースキー ──→ HARAI（払戻）
  └─ レースキー ──→ ODDS_*（各オッズテーブル）
  └─ レースキー ──→ HYOSU_*（各票数テーブル）
  └─ レースキー ──→ TENKO_BABA（天候馬場状態）
  └─ レースキー ──→ BATAIJYU（馬体重）
  └─ レースキー ──→ MINING / TAISENGATA_MINING

UMA_RACE（馬毎レース情報）
  └─ KisyuCode ──→ KISYU（騎手マスタ）
  └─ ChokyosiCode ──→ CHOKYO（調教師マスタ）
  └─ BanusiCode ──→ BANUSI（馬主マスタ）

HANSYOKU（繁殖馬マスタ）
  └─ HansyokuNum ──→ UMA.Ketto3Info*（血統情報）
  └─ HansyokuNum ──→ SANKU.*Num（産駒4代祖先）
  └─ HansyokuNum ──→ KEITO（系統情報）
```

### 予想に特に重要なテーブル

1. **RACE** + **UMA_RACE**: レース条件と各馬の出走情報・成績
2. **UMA**: 競走馬の血統情報（父母系統の分析）
3. **ODDS_TANPUKU**: 単複オッズ（人気の把握）
4. **HANRO / WOOD_CHIP**: 調教データ（馬の状態把握）
5. **KISYU / KISYU_SEISEKI**: 騎手の実績
6. **CHOKYO / CHOKYO_SEISEKI**: 調教師の実績
7. **HARAI**: 過去の払戻データ（回収率分析）
8. **TENKO_BABA**: 当日の天候・馬場状態

---

## よく使うSQLサンプル

### 最近のレース結果を取得
```sql
SELECT r.Year, r.MonthDay, r.JyoCD, r.RaceNum, r.Hondai, r.Kyori, r.TrackCD,
       r.TenkoCD, r.SibaBabaCD, r.DirtBabaCD,
       ur.Umaban, ur.Bamei, ur.KakuteiJyuni, ur.Time, ur.Odds, ur.Ninki,
       ur.Futan, ur.KisyuRyakusyo, ur.ChokyosiRyakusyo
FROM n_race r
JOIN n_uma_race ur ON r.Year = ur.Year AND r.MonthDay = ur.MonthDay
  AND r.JyoCD = ur.JyoCD AND r.Kaiji = ur.Kaiji
  AND r.Nichiji = ur.Nichiji AND r.RaceNum = ur.RaceNum
WHERE r.Year = '2025' AND r.DataKubun = '7'
ORDER BY r.MonthDay DESC, r.JyoCD, CAST(r.RaceNum AS integer),
         CAST(ur.KakuteiJyuni AS integer);
```

### 特定の馬の過去成績を取得
```sql
SELECT ur.Year, ur.MonthDay, ur.JyoCD, ur.RaceNum,
       r.Hondai, r.Kyori, r.TrackCD, r.SibaBabaCD, r.DirtBabaCD,
       ur.KakuteiJyuni, ur.Time, ur.HaronTimeL3, ur.Odds, ur.Ninki,
       ur.Futan, ur.BaTaijyu, ur.KisyuRyakusyo, ur.KyakusituKubun
FROM n_uma_race ur
JOIN n_race r ON ur.Year = r.Year AND ur.MonthDay = r.MonthDay
  AND ur.JyoCD = r.JyoCD AND ur.Kaiji = r.Kaiji
  AND ur.Nichiji = r.Nichiji AND ur.RaceNum = r.RaceNum
WHERE ur.KettoNum = '2019100001'  -- 血統登録番号を指定
  AND ur.DataKubun = '7'
ORDER BY ur.Year DESC, ur.MonthDay DESC;
```

### 騎手の直近成績を取得
```sql
SELECT k.KisyuName, ks.SetYear,
       ks.HeichiChakukaisu1 AS 平地1着,
       ks.HeichiChakukaisu2 AS 平地2着,
       ks.HeichiChakukaisu3 AS 平地3着,
       ks.HonSyokinHeichi AS 平地本賞金
FROM n_kisyu k
JOIN n_kisyu_seiseki ks ON k.KisyuCode = ks.KisyuCode
WHERE k.DelKubun = '0'  -- 現役のみ
ORDER BY CAST(ks.SetYear AS integer) DESC;
```

---

*このドキュメントはeverydb2マニュアル（https://everydb.iwinz.net/edb2_manual/）を基に作成されています。*
