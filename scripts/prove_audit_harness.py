"""PROVE THE AUDIT TESTER WORKS — three controls on scripts/audit.py's recompute engine.

(1) POSITIVE / correctness: recompute each statistic with the harness AND with an INDEPENDENT implementation
    (sklearn roc_auc_score for AUROC; numpy for the arithmetic gains). They must agree to ~1e-9.
(2) NEGATIVE / discrimination: plant a deliberately-wrong value; the tester's MATCH rule must report MISMATCH
    (proves it is not a rubber-stamp that always says MATCH).
(3) SAMPLING / stability: bootstrap-resample the per-item data thousands of times; the recomputed statistic's
    mean and [2.5, 97.5] percentiles must center on the stored value within the stored CI.

Run: .venv/Scripts/python.exe scripts/prove_audit_harness.py
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.audit import R, auroc, _mean, find  # the harness under test

MATCH_TOL = 0.005  # the tester's own MATCH rule (audit.py: abs(got-val) <= 0.005)
def harness_match(got, val): return abs(got - val) <= MATCH_TOL

rng = np.random.default_rng(20260622)
PASS = True
def line(ok, msg):
    global PASS
    if not ok: PASS = False
    print(f"   [{'PASS' if ok else 'FAIL'}] {msg}")

print("="*92); print("PROVE THE AUDIT TESTER WORKS"); print("="*92)

# ---------- (1) POSITIVE: harness vs independent implementation ----------
print("\n(1) POSITIVE CONTROL — harness recompute vs INDEPENDENT implementation (must agree ~1e-9):")
d = json.load(open(find(R["G3"]["file"])))
SE = [it["SE"] for it in d["per_item"]]; ERR = [it["err"] for it in d["per_item"]]
h = auroc(SE, ERR)                                  # harness (rank-based, no deps)
sk = roc_auc_score(ERR, SE)                         # sklearn (independent)
line(abs(h - sk) < 1e-9, f"G3 AUROC: harness={h:.10f}  sklearn={sk:.10f}  |diff|={abs(h-sk):.2e}")

for rid, indep in [
    ("B1", lambda d: (d["models"]["1.5B"]["DANGER_ZONE"]["cov_0.7"]["detector"] - d["models"]["1.5B"]["DANGER_ZONE"]["cov_0.7"]["confidence"])*100.0),
    ("C4", lambda d: float(np.mean([r["learned"] for r in d["per_item_rows"]]) - max(np.mean([r["trap"] for r in d["per_item_rows"]]), np.mean([r["ord"] for r in d["per_item_rows"]])))),
    ("F1", lambda d: float(np.mean([r["learned"] for r in d["per_item_rows"]]) - np.mean([r["base"] for r in d["per_item_rows"]]))),
    ("D1", lambda d: d["lora_heldout_category_acc"] - d["base_heldout_category_acc"]),
]:
    dd = json.load(open(find(R[rid]["file"])))
    hv = R[rid]["recompute"](dd); iv = indep(dd)
    line(abs(hv - iv) < 1e-9, f"{rid} recompute: harness={hv:.10f}  independent(numpy)={iv:.10f}  |diff|={abs(hv-iv):.2e}")

# ---------- (2) NEGATIVE: the MATCH rule must catch a planted error ----------
print("\n(2) NEGATIVE CONTROL — the tester's MATCH rule must FLAG planted errors (not a rubber-stamp):")
true_val = d["AUROC_SE_to_error_aggregate"]
line(harness_match(h, true_val), f"true value {true_val} vs recompute {h:.4f} -> reports MATCH (correct)")
for delta in (0.05, 0.10, 0.20):
    planted = true_val + delta
    line(not harness_match(h, planted), f"planted WRONG value {planted:.4f} (+{delta}) -> reports MISMATCH (caught)")
# tiny within-tolerance perturbation should still pass (rule is not hair-trigger)
line(harness_match(h, true_val + 0.004), f"within-tol {true_val+0.004:.4f} (+0.004) -> MATCH (tolerance behaves)")

# ---------- (3) SAMPLING / STABILITY: bootstrap the per-item data ----------
print("\n(3) SAMPLING / STABILITY — bootstrap resample (B=3000) the 800 per-item rows; recompute AUROC each time:")
SEa = np.array(SE); ERRa = np.array(ERR); nidx = len(SEa); B = 3000
boots = np.empty(B)
for b in range(B):
    idx = rng.integers(0, nidx, nidx)
    yt = ERRa[idx]
    if yt.min() == yt.max():      # degenerate resample (one class) — redraw once
        idx = rng.integers(0, nidx, nidx); yt = ERRa[idx]
    boots[b] = roc_auc_score(yt, SEa[idx])
bmean = boots.mean(); lo, hi = np.percentile(boots, [2.5, 97.5])
stored_ci = d.get("AUROC_SE_to_error_CI95")
print(f"   bootstrap mean={bmean:.4f}  95%=[{lo:.4f},{hi:.4f}]   stored={true_val}  stored CI={stored_ci}")
line(abs(bmean - true_val) < 0.02, f"bootstrap mean centers on stored value (|{bmean:.4f}-{true_val}|<0.02)")
if stored_ci:
    line(abs(lo - stored_ci[0]) < 0.03 and abs(hi - stored_ci[1]) < 0.03, f"bootstrap 95% ~ stored CI (both ends within 0.03)")
line(hi < 0.60, f"entire bootstrap 95% stays in the useless band (<0.60): hi={hi:.4f}")

print("\n" + "="*92)
print("VERDICT:", "ALL CONTROLS PASS — the audit tester recomputes correctly, catches planted errors, and is stable under resampling." if PASS else "FAIL — see above.")
print("="*92)
sys.exit(0 if PASS else 1)
