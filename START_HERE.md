# Mining Supplement Analysis - START HERE

## What Was Analyzed?

Mining supplement parquet files containing JRA-VAN Data Mining predictions for 517,018 horses across 2015-2025.

**Analysis Scope:**
- `mining_dm_kubun` distribution (prediction timing)
- Missing rates for each feature
- `mining_dm_time` statistics (predicted run times)
- `mining_dm_jyuni` distribution (predicted ranks)
- Correlation with odds (if available)
- Data alignment with main features

---

## Quick Summary

| Item | Result |
|------|--------|
| **Data Leakage Risk** | ✓ MINIMAL - 100% from 直前 (immediate prediction) |
| **Data Completeness** | ✓ GOOD - Primary features 100%, supporting features 74-97% |
| **Temporal Stability** | ✓ EXCELLENT - ±0.3% year-to-year variation |
| **Data Alignment** | ✓ PERFECT - 100% row match with main features |
| **Production Ready** | ✓ YES - High quality, no anomalies detected |

---

## Which File Should I Read?

### I have 5 minutes
**→ Read:** `/sessions/great-confident-shannon/mnt/keiba-2026/ANALYSIS_SUMMARY.txt`

Executive summary with all key findings and recommendations in one place.

### I have 10 minutes
**→ Read:** `/sessions/great-confident-shannon/mnt/keiba-2026/ANALYSIS_README.md`

Quick reference guide with feature meanings, usage patterns, and interpretation notes.

### I have 30 minutes
**→ Read:** `/sessions/great-confident-shannon/mnt/keiba-2026/MINING_ANALYSIS_REPORT.md`

Comprehensive report with 10 detailed sections, tables, and technical interpretation.

### I need the index
**→ Read:** `/sessions/great-confident-shannon/mnt/keiba-2026/FILE_MANIFEST.txt`

Complete manifest of all generated files, their purposes, and reproducibility info.

---

## Key Findings

### 1. Mining_DM_Kubun Distribution
```
100% from kubun=3 (直前)  ← Most recent data, minimal leakage risk
  0% from kubun=1 (前日)   ← None
  0% from kubun=2 (当日)   ← None
```

**Why it matters:** All predictions are made right before the race, minimizing data leakage.

### 2. Data Quality by Feature
```
mining_dm_kubun:   100% complete  ✓ Perfect
mining_dm_jyuni:   100% complete  ✓ Perfect
mining_dm_time:    100% complete  ✓ Perfect
mining_tm_score:    97% complete  ✓ Good
mining_dm_gosa_*:   74% complete  ✓ Acceptable (error estimates)
```

### 3. Mining_DM_Time (Predicted Run Time)
```
Mean:      1505.4 seconds (≈25 minutes)
Range:     548-5119 seconds (reasonable for all race types)
Variation: ±0.3% year-to-year (very stable)
```

### 4. Mining_DM_Jyuni (Predicted Rank)
```
Mean:          7.74
Distribution:  Uniform top-5 (7.18% each)
               Declining for lower ranks
               No bias detected
```

### 5. Data Alignment
```
Mining supplements:  46,752 rows (2024)
Main features:       46,752 rows (2024)
Match rate:          100% perfect alignment
```

### 6. Odds Correlation
```
Not directly available (by design to prevent data leakage)
Use NDCG metric for ranking quality measurement
Can enable with --with-odds flag if needed
```

---

## How to Re-Run the Analysis

```bash
cd /sessions/great-confident-shannon/mnt/keiba-2026

# High-level summary (kubun, missing rates, stats)
python analyze_mining_supplement.py

# Detailed inspection (columns, data quality, samples)
python analyze_mining_detailed.py
```

**Runtime:** ~15 seconds total

---

## How to Use Mining Features in Training

```bash
# Default (recommended) - includes mining features
python run_train.py --supplement mining

# With odds features (data leakage risk)
python run_train.py --supplement mining --with-odds

# Training only (skip feature engineering)
python run_train.py --supplement mining --train-only

# Evaluation only (model assessment)
python run_train.py --supplement mining --eval-only
```

---

## Mining Features Explained

| Feature | Type | Coverage | Range | Meaning |
|---------|------|----------|-------|---------|
| `mining_dm_time` | float | 100% | 548-5119 sec | Predicted run time |
| `mining_dm_jyuni` | int | 100% | 1-18 | Predicted rank |
| `mining_dm_kubun` | int | 100% | 3 | Timing (always 直前) |
| `mining_tm_score` | float | 97% | 62-881 | Confrontation score |
| `mining_dm_gosa_p` | float | 74% | 0-20 sec | Positive error estimate |
| `mining_dm_gosa_m` | float | 74% | 0-20 sec | Negative error estimate |
| `mining_dm_gosa_range` | float | 74% | 0.2-20 sec | Error range width |

---

## Recommendations

### ✓ DO
- Continue using `--supplement mining` as default
- Monitor `mining_dm_jyuni` and `mining_tm_score` feature importance via SHAP
- Use NDCG metric to evaluate ranking quality

### ⚠ CONSIDER
- Error features (`gosa_*`) have ~26% missing
- Could drop if missing rate becomes problematic
- Or engineer derived features like rank confidence

### ✗ DON'T
- Don't use `--with-odds` flag unless comparing against baseline (data leakage risk)
- Don't assume error features are available for all horses

---

## Technical Details

**Data Period:** 2015-2025 (11 years)  
**Total Rows:** 517,018 horses  
**Merge Keys:** _key_year, _key_monthday, _key_jyocd, _key_kaiji, _key_nichiji, _key_racenum, kettonum  
**Missing Value Encoding:** -1.0 (float), -1 (int)  

---

## Generated Files

### Scripts (Executable)
- `analyze_mining_supplement.py` - High-level statistics
- `analyze_mining_detailed.py` - Detailed inspection

### Reports (Documentation)
- `MINING_ANALYSIS_REPORT.md` - Comprehensive 10-section report
- `ANALYSIS_SUMMARY.txt` - Executive summary
- `ANALYSIS_README.md` - Usage guide
- `FILE_MANIFEST.txt` - Complete index
- `START_HERE.md` - This file

### Data Source
- `data/supplements/mining_*.parquet` (11 files, 517,018 total rows)
- `data/features_*.parquet` (11 files, for reference)

---

## Quality Assessment

### Strengths ✓
- Complete primary features (kubun, jyuni, time)
- Optimal timing (100% from 直前 = minimal leakage)
- Stable across all years (no drift)
- Perfect alignment with main features
- Reasonable value ranges

### Considerations ⚠
- Error features have ~26% missing (expected, acceptable)
- Odds correlation not available (by design)
- ~70% of predictions rank 6+ (natural, still valuable)

### Verdict
**✓ HIGH QUALITY - READY FOR PRODUCTION USE**

---

## Next Steps

1. **Read one of the reports** (5-30 minutes depending on depth)
2. **Understand the findings** (all data is high quality, ready to use)
3. **Use in training** with `python run_train.py --supplement mining`
4. **Monitor feature importance** via SHAP analysis on trained models
5. **Track data freshness** (kubun=3 should remain 100%)

---

## Questions?

- **What do the features mean?** → See ANALYSIS_README.md
- **How complete is the data?** → See ANALYSIS_SUMMARY.txt
- **What's the detailed analysis?** → See MINING_ANALYSIS_REPORT.md
- **How to run analysis again?** → See FILE_MANIFEST.txt
- **How to integrate into training?** → See ANALYSIS_README.md usage section

---

**Report Generated:** 2026-02-20  
**Status:** ✓ COMPLETE  
**Recommendation:** ✓ APPROVED FOR PRODUCTION USE

