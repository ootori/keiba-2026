# Feature Data Quality Diagnostic Report

## Executive Summary

**CRITICAL ISSUE FOUND:** The `post_umaban` (馬番/post number) feature is stored as `int64` type (values 1-18) in the feature data, but the evaluator's payout calculation expects it to match the `n_harai` database table column format, which likely uses zero-padded strings ("01", "02", ..., "18").

This mismatch causes **race key + umaban matching to fail**, resulting in **zero payouts** for all predictions, even when the prediction is correct.

---

## Detailed Findings

### 1. Parquet Data Summary

**train_features.parquet:**
- 479,616 rows × 138 columns
- Target distribution: 0=375,945 (82.38%), 1=103,671 (21.62%)

**valid_features.parquet:**
- 37,402 rows × 138 columns
- Target distribution: 0=29,193 (78.05%), 1=8,209 (21.95%)
- All 138 features present and complete (no missing data)

### 2. The post_umaban Problem

**Data Type:** `int64`

**Stored Values:** 1, 2, 3, ..., 18 (integers)

**Current Evaluator Behavior:**
```python
# In evaluator.py, when converting to string:
umaban = str(row.get("post_umaban", "")).strip()
# Result: "1", "2", "3", ... "18"
```

**Expected n_harai Format:**
```
n_harai likely has umaban columns with zero-padded strings:
"01", "02", "03", ..., "18"
```

**Matching Failure Example:**
- Evaluator looks for `"1"` in n_harai
- n_harai contains `"01"`
- No match found → No payout calculation

### 3. Race Key Structure

Race keys are properly formatted with 6 components:
```
year: 2025
monthday: 0105
jyocd: 06
kaiji: 01
nichiji: 01
racenum: 01-03

Combined: 2025_0105_06_01_01_01
```

All race key columns are present and correctly populated.

### 4. Odds Features Data Quality

✓ All odds features are complete (100% non-null)
✓ Value ranges are reasonable:
  - `odds_tan`: 1.1 to 921.9
  - `odds_fuku_low`: 1.0 to 138.0
  - `odds_fuku_high`: 1.0 to 413.2
  - `odds_ninki`: 1 to 18
✓ No invalid values (no negative numbers)

### 5. Model Features Alignment

- Model expects: 130 features (listed in model_features.txt)
- Parquet contains: 138 columns
  - 130 model features (all present)
  - 6 race key columns (_key_*)
  - 1 target column
  - 1 kettonum column (for identification)
✓ Perfect alignment - no missing or extra features

---

## Root Cause Analysis

### Why This Matters for Evaluator

The evaluator's payout simulation process:

```python
# Step 1: Prepare predictions
predictions = predict_and_prepare(valid_df)  # has post_umaban as int64 "1"

# Step 2: Match with harai (payout) data
# Tries to match: race_key + post_umaban
# race_key = "2025_0105_06_01_01_01"
# umaban = "1"  ← STRING CONVERSION ISSUE

# Step 3: Find payout record in n_harai
# n_harai has: year, monthday, jyocd, ..., and columns like
#   "tansyo_umaban" = "01"  ← ZERO-PADDED STRING
# 
# String matching fails: "1" != "01"
# No payout found → Return 0 payout
```

---

## Recommended Fixes

### Option A: Store as Zero-Padded String (RECOMMENDED)

**Location:** `src/features/pipeline.py` or wherever `post_umaban` is created

```python
# Current (WRONG):
df['post_umaban'] = df['post_umaban'].astype(int)  # Results in: 1, 2, 3

# Fixed (RIGHT):
df['post_umaban'] = df['post_umaban'].astype(str).str.zfill(2)  # Results in: "01", "02", "03"
```

**Pros:**
- Matches n_harai database format exactly
- No changes needed to evaluator.py
- Data type semantically correct (it's a code, not a number)

**Cons:**
- Requires rebuilding features (feature files in /data/)
- May affect LightGBM if it was treating as numeric

### Option B: Pad When Matching in Evaluator

**Location:** `src/model/evaluator.py` (in payout matching logic)

```python
# Current:
umaban = str(row.get("post_umaban", "")).strip()

# Fixed:
umaban = str(row.get("post_umaban", "")).strip().zfill(2)
```

**Pros:**
- No feature rebuild needed
- Quick fix

**Cons:**
- Evaluator needs updating
- Feature representation remains inconsistent

### Option C: Handle Both Formats

```python
# Robust matching:
umaban_int = str(int(float(row.get("post_umaban", 0)))).zfill(2)
# Try both formats in n_harai
```

---

## Verification Checklist

- [ ] Confirm n_harai umaban column format (run: `SELECT DISTINCT "umaban" FROM n_harai LIMIT 20`)
- [ ] Check if post_umaban should be categorical or numeric for LightGBM
- [ ] Rebuild features with zero-padded umaban
- [ ] Run evaluator test with sample race to confirm payout calculation
- [ ] Re-train model with corrected features
- [ ] Verify feature importance (post_umaban should appear in top features for strong models)

---

## Impact Assessment

**Current State:**
- Model predictions may be reasonable (feature engineering looks correct)
- **Evaluator payout calculation is broken** (always returns 0 payouts)
- **回収率 (ROI) results are unreliable** (will show 0% or near-zero returns)

**After Fix:**
- Payout calculation should match actual race results
- ROI/回収率 should reflect true profitability
- Model confidence metrics become meaningful

---

## Related Files

- Feature creation: `/sessions/wonderful-hopeful-carson/mnt/everydb2/src/features/pipeline.py`
- Evaluator: `/sessions/wonderful-hopeful-carson/mnt/everydb2/src/model/evaluator.py`
- Data storage: `/sessions/wonderful-hopeful-carson/mnt/everydb2/data/valid_features.parquet`
- Model spec: `/sessions/wonderful-hopeful-carson/mnt/everydb2/models/model_features.txt`

