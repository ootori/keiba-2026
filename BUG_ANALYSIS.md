# Bug Analysis: post_umaban Zero-Padding Issue

## Problem Summary

The evaluator's payout calculation fails to find matching records in the `n_harai` database table because of a data type mismatch in the `post_umaban` (馬番) field.

**Impact:** ALL payouts are calculated as 0, making ROI/回収率 results completely invalid.

---

## Root Cause

### Current Implementation (BROKEN)

**File:** `/sessions/wonderful-hopeful-carson/mnt/everydb2/src/features/race.py`

```python
# post_umaban is stored as integer (1-18)
"post_umaban": umaban,
```

**Result:** When converted to string in evaluator: `"1"`, `"2"`, `"3"`, ..., `"18"`

### Evaluator Matching (FILE: evaluator.py)

```python
# Line ~340-350 (exact location varies)
umaban = str(top1.get("post_umaban", "")).strip()  # Results in: "1"

# Then tries to find in database:
if umaban and umaban in tansho:  # Looking for "1"
    payout = tansho[umaban]      # But tansho keys are "01", "02", etc.
```

**Problem:** String matching fails
- Evaluator sends: `"1"`
- Database has: `"01"`
- Result: `"1" != "01"` → No match → Payout = 0

---

## Verified Test Case

**Input:**
- `post_umaban` from parquet: `np.int64(1)`
- After `str().strip()`: `"1"`
- Expected in n_harai: `"01"` (zero-padded)

**Simulation:**
```python
tansho = {"01": 320, "02": 280, "03": 450}  # Database format

# Current (broken):
umaban = "1"
umaban in tansho  # False → payout = 0

# Fixed:
umaban = "01"
umaban in tansho  # True → payout = 320
```

---

## Recommended Fix (Option A - BEST)

### Step 1: Fix Feature Creation

**File:** `/sessions/wonderful-hopeful-carson/mnt/everydb2/src/features/race.py`

```python
# CHANGE THIS (around line with "post_umaban": umaban,):
# FROM:
"post_umaban": umaban,

# TO:
"post_umaban": str(umaban).zfill(2) if umaban else "",
```

OR in the data preparation section, ensure umaban is zero-padded before this point.

### Step 2: Rebuild Features

```bash
cd /sessions/wonderful-hopeful-carson/mnt/everydb2
python run_train.py --build-features-only
```

### Step 3: Retrain Model

```bash
python run_train.py --train-only
```

### Step 4: Re-evaluate

```bash
python run_train.py --eval-only
```

---

## Alternative Fix (Option B - QUICK FIX)

If you want to avoid rebuilding features, modify the evaluator:

**File:** `/sessions/wonderful-hopeful-carson/mnt/everydb2/src/model/evaluator.py`

Find these lines:
```python
umaban = str(top1.get("post_umaban", "")).strip() if "post_umaban" in group.columns else ""
```

Change to:
```python
umaban_raw = top1.get("post_umaban", "") if "post_umaban" in group.columns else ""
umaban = str(umaban_raw).strip().zfill(2)  # Add zero-padding
```

Repeat for both `_bet_top1_tansho` and `_bet_top1_fukusho` methods (and any other betting methods).

---

## Verification Steps

After applying the fix:

1. **Confirm post_umaban format in parquet:**
   ```bash
   python3 -c "
   import pandas as pd
   df = pd.read_parquet('data/valid_features.parquet')
   print('Sample post_umaban values:')
   for val in df['post_umaban'].unique()[:5]:
       print(f'  {val!r}')
   "
   ```

2. **Run evaluator test:**
   ```bash
   python3 -c "
   from src.model.evaluator import Evaluator
   # Should now find payouts instead of 0
   "
   ```

3. **Check ROI results:**
   - Before fix: ~0% or near-zero
   - After fix: Should reflect actual profitability

---

## Files Affected

| File | Change | Reason |
|------|--------|--------|
| `src/features/race.py` | Zero-pad `post_umaban` | Data consistency with database |
| `src/model/evaluator.py` | Zero-pad when matching (Option B) | String matching format |
| `data/*.parquet` | Rebuild required | Feature data changes |
| `models/*.txt` | No change needed | Feature names unchanged |

---

## Testing Checklist

- [ ] Verify post_umaban dtype is string after fix
- [ ] Check that payouts > 0 are now found in evaluator
- [ ] Confirm ROI/回収率 reflects realistic profitability
- [ ] Run full train pipeline without errors
- [ ] Verify model performance hasn't degraded
- [ ] Compare before/after ROI values

---

## Expected Outcome

**Before Fix:**
- All payouts = 0
- 回収率 = 0%
- Model appears worthless even if predictions are good

**After Fix:**
- Payouts correctly matched to predictions
- 回収率 = realistic percentage (could be positive or negative)
- Model quality assessment becomes meaningful

---

## Questions for Confirmation

1. Does `n_harai` actually use zero-padded umaban strings ("01", "02")?
   - Run: `SELECT DISTINCT "tansyo_umaban" FROM n_harai LIMIT 20`
   - Or similar for actual column name

2. Is `post_umaban` used as a numeric or categorical feature in LightGBM?
   - Check `trainer.py` categorical_feature list
   - If numeric: converting to string might affect model performance

3. Are there other places where post_umaban is used that need updating?
   - Search: `grep -r "post_umaban" src/ models/`

