# Data Quality Audit - Complete Report Index

## Quick Links

1. **EXECUTIVE_SUMMARY.txt** ← Start here
   - Critical findings at a glance
   - Recommended actions
   - Questions to verify

2. **DIAGNOSTIC_REPORT.md**
   - Detailed technical analysis
   - Data quality metrics (all features)
   - Root cause explanation
   - 3 fix options with pros/cons
   - Impact assessment

3. **BUG_ANALYSIS.md**
   - Bug reproduction with code
   - Exact code locations
   - Step-by-step fix instructions
   - Testing checklist
   - Expected outcomes

---

## The Issue in 30 Seconds

**Problem:** `post_umaban` is stored as int64 (1-18), but database expects zero-padded strings ("01"-"18").

**Impact:** Payout matching fails → All ROI/回収率 results are 0 (broken).

**Fix:** Change how `post_umaban` is stored or formatted during matching.

**Time to fix:** 15 minutes (quick) or 1-2 hours (proper).

---

## Files Affected

- **src/features/race.py** — Where post_umaban is created
- **src/model/evaluator.py** — Where post_umaban is matched with payouts
- **data/*.parquet** — Feature files (may need rebuild)

---

## Quick Reference: The Bug

```python
# CURRENT (BROKEN):
post_umaban = 1  # int64
umaban_str = str(1).strip()  # "1"
if "1" in {"01": 320, "02": 280}:  # False! → payout = 0

# FIXED:
post_umaban = "01"  # string
umaban_str = str("01").strip()  # "01"
if "01" in {"01": 320, "02": 280}:  # True! → payout = 320
```

---

## Audit Results Summary

| Category | Status | Details |
|----------|--------|---------|
| Parquet integrity | ✓ PASS | 479K + 37K rows, all complete |
| Feature completeness | ✓ PASS | 130/130 features present |
| Feature alignment | ✓ PASS | No missing/extra features |
| Data types | ✓ PASS | Correct except post_umaban |
| Race keys | ✓ PASS | Properly formatted, unique |
| Odds data | ✓ PASS | Realistic value ranges |
| **Payout matching** | **✗ FAIL** | **post_umaban format mismatch** |
| **ROI calculation** | **✗ BROKEN** | **All payouts = 0** |

---

## Next Action

1. Read EXECUTIVE_SUMMARY.txt
2. Read DIAGNOSTIC_REPORT.md or BUG_ANALYSIS.md
3. Decide: Option A (proper fix) or Option B (quick fix)
4. Apply the fix
5. Test with verification steps

---

Generated: 2025-02-15
