# Visual Summary: post_umaban Bug

## The Data Flow Problem

```
┌─────────────────────────────────────────────────────────────────────┐
│                  CURRENT BROKEN FLOW                                │
└─────────────────────────────────────────────────────────────────────┘

FEATURE CREATION (src/features/race.py)
│
├─ umaban from DB: 1 (int)
├─ Store in parquet: 1 (int64)
│
↓
EVALUATOR (src/model/evaluator.py)
│
├─ Read from parquet: numpy.int64(1)
├─ Convert: str(1).strip() = "1"
│
↓
DATABASE LOOKUP (n_harai table)
│
├─ Looking for: "1"
├─ Database has: "01"
│
↓
MATCHING RESULT
│
├─ "1" == "01" ? NO ✗
├─ Payout found: No
├─ Return value: 0
│
↓
OUTPUT (report)
│
├─ All payouts: 0
├─ All ROI: 0%
└─ Model appears worthless
```

---

## The Fix (Option A - Recommended)

```
┌─────────────────────────────────────────────────────────────────────┐
│                  FIXED FLOW (Option A)                              │
└─────────────────────────────────────────────────────────────────────┘

FEATURE CREATION (src/features/race.py)
│
├─ umaban from DB: 1 (int)
├─ Convert: str(1).zfill(2) = "01"
├─ Store in parquet: "01" (string)
│
↓
EVALUATOR (src/model/evaluator.py)
│
├─ Read from parquet: "01"
├─ Convert: str("01").strip() = "01"
│
↓
DATABASE LOOKUP (n_harai table)
│
├─ Looking for: "01"
├─ Database has: "01"
│
↓
MATCHING RESULT
│
├─ "01" == "01" ? YES ✓
├─ Payout found: 320
├─ Return value: 320
│
↓
OUTPUT (report)
│
├─ Payout calculated: 320
├─ ROI: realistic value
└─ Model assessment: meaningful
```

---

## The Quick Fix (Option B)

```
┌─────────────────────────────────────────────────────────────────────┐
│                  QUICK FIX (Option B)                               │
└─────────────────────────────────────────────────────────────────────┘

FEATURE CREATION (src/features/race.py)
│
├─ umaban from DB: 1 (int)
├─ Store in parquet: 1 (int64)
│                      ↑ NO CHANGE
│
↓
EVALUATOR (src/model/evaluator.py)  ← FIX APPLIED HERE
│
├─ Read from parquet: numpy.int64(1)
├─ Convert: str(1).strip().zfill(2) = "01"  ← ADD .zfill(2)
│
↓
DATABASE LOOKUP (n_harai table)
│
├─ Looking for: "01"
├─ Database has: "01"
│
↓
MATCHING RESULT
│
├─ "01" == "01" ? YES ✓
├─ Payout found: 320
├─ Return value: 320
│
↓
OUTPUT (report)
│
├─ Payout calculated: 320
├─ ROI: realistic value
└─ Model assessment: meaningful
```

---

## Data Structure Comparison

### Current (Broken) Storage
```
┌─────────────────────────────────────────┐
│  valid_features.parquet                 │
├─────────────────────────────────────────┤
│ post_umaban: int64                      │
│  [1, 2, 3, 4, 5, ..., 18]              │
│  └─ Converts to: "1", "2", "3", ...    │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  n_harai table (PostgreSQL)             │
├─────────────────────────────────────────┤
│ tansyo_umaban: varchar                  │
│  ["01", "02", "03", "04", "05", ...] │
│  └─ Already zero-padded                │
└─────────────────────────────────────────┘

Result: NO MATCH ✗
```

### After Fix (Correct)
```
┌─────────────────────────────────────────┐
│  valid_features.parquet                 │
├─────────────────────────────────────────┤
│ post_umaban: string                     │
│  ["01", "02", "03", "04", "05", ...]   │
│  └─ Same format as database             │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  n_harai table (PostgreSQL)             │
├─────────────────────────────────────────┤
│ tansyo_umaban: varchar                  │
│  ["01", "02", "03", "04", "05", ...]   │
│  └─ Already zero-padded                 │
└─────────────────────────────────────────┘

Result: MATCH ✓
```

---

## Impact on ROI Calculation

### Before Fix
```
Race 1: Prediction correct ✓, but payout = 0 → ROI = 0%
Race 2: Prediction correct ✓, but payout = 0 → ROI = 0%
Race 3: Prediction correct ✓, but payout = 0 → ROI = 0%
...
Race N: Prediction correct ✓, but payout = 0 → ROI = 0%

OVERALL ROI = 0% (All returns = 0)
MODEL EVALUATION = Appears worthless even if predictions are good
```

### After Fix
```
Race 1: Prediction correct ✓, payout = 320 → ROI = 220% ✓
Race 2: Prediction wrong ✗, payout = 0 → ROI = -100% ✓
Race 3: Prediction correct ✓, payout = 280 → ROI = 180% ✓
...
Race N: Prediction correct ✓, payout = 400 → ROI = 300% ✓

OVERALL ROI = realistic value (e.g., +15%, -5%, etc.)
MODEL EVALUATION = Meaningful assessment of profitability
```

---

## Code Change Visualization

### Option A: Fix at Source (Feature Creation)

```diff
# src/features/race.py, line ~XXX

  features = {
      # ... other features ...
      "post_wakuban": wakuban,
-     "post_umaban": umaban,  # ← int64: 1, 2, 3, ...
+     "post_umaban": str(umaban).zfill(2) if umaban else "",  # ← string: "01", "02", "03", ...
      "post_umaban_norm": (umaban / tosu if tosu > 0 and umaban > 0 else -1.0),
      # ... other features ...
  }
```

### Option B: Fix at Matching (Evaluator)

```diff
# src/model/evaluator.py, _bet_top1_tansho method

  def _bet_top1_tansho(self, group, harai_data, race_key):
      """予測1位の単勝を購入する."""
      top1 = group.nlargest(1, "pred_prob").iloc[0]
-     umaban = str(top1.get("post_umaban", "")).strip() if "post_umaban" in group.columns else ""
+     umaban = str(top1.get("post_umaban", "")).strip().zfill(2) if "post_umaban" in group.columns else ""
      
      # ... rest of method ...
```

---

## Timeline to Fix

```
Option A (Proper Fix):
┌──────────┬──────────┬────────┬──────────┬──────────┐
│ Fix Code │ Rebuild  │ Retrain│ Re-eval  │ Validate │
│ 10 min   │ 30 min   │ 30 min │ 10 min   │ 10 min   │
└──────────┴──────────┴────────┴──────────┴──────────┘
Total: ~90 minutes

Option B (Quick Fix):
┌──────────┬──────────┐
│ Fix Code │ Validate │
│ 5 min    │ 5 min    │
└──────────┴──────────┘
Total: ~10 minutes
```

---

## Dependency Chain

```
                    ┌─ model_features.txt (unchanged)
                    │
      FEATURE DATA  ├─ train_features.parquet (needs rebuild if Option A)
            (parquet)├─ valid_features.parquet (needs rebuild if Option A)
                    │
                    └─ post_umaban storage format
                           ↓
                    (Option A: change here)
                           ↓
                    ┌─────────────────────┐
                    │  src/features/*.py  │
                    │  (where umaban is   │
                    │   extracted from DB)│
                    └─────────────────────┘
                           ↓
                    (Option B: change here)
                           ↓
                    ┌─────────────────────┐
                    │  src/model/         │
                    │  evaluator.py       │
                    │  (where matching    │
                    │   happens)          │
                    └─────────────────────┘
                           ↓
                    ┌─────────────────────┐
                    │  Payout Calculation │
                    │  (works only if     │
                    │   matching succeeds) │
                    └─────────────────────┘
                           ↓
                    ┌─────────────────────┐
                    │  ROI/回収率 Results │
                    │  (meaningful only if│
                    │   payouts work)     │
                    └─────────────────────┘
```

---

Generated: 2025-02-15
