"""SIMULATION that verifies the theorem: PR (participation-ratio) input-invariance => an unsupervised endpoint /
dispersion readout (and semantic entropy) is BLIND to confident error, while a supervised DIRECTION readout separates.

Construction (no real model needed - this tests the MATH):
  * Two classes ("confident-correct" y=0 / "confident-wrong" y=1) share the SAME within-class covariance Sigma
    (=> the same participation ratio / spectral geometry => PR is input-invariant by construction).
  * The classes differ ONLY by a mean shift along one direction u (the "truth direction").
  * DISPERSION readout (semantic-entropy analog): the spread of K samples per item. Same Sigma => same spread =>
    it CANNOT separate the classes  -> AUROC ~ 0.5 (the theorem's prediction: blind).
  * DIRECTION readout (the escape hatch): the supervised projection onto u  -> AUROC ~ 1.0 (separates).
  * CONTROL: break the invariance - give the two classes DIFFERENT within-class covariance (different PR). Now the
    dispersion readout DOES separate (AUROC > 0.5), proving the blindness was caused by the PR-invariance, not an artifact.

Run: .venv/Scripts/python.exe scripts/prove_theorem_sim.py
"""
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path

OUT_PNG = Path(__file__).resolve().parents[1] / "process/papers/figs/theorem_sim.png"
_ap = argparse.ArgumentParser(); _ap.add_argument("--out", default=str(OUT_PNG)); _ARGS, _ = _ap.parse_known_args()

rng = np.random.default_rng(2026)
d, n_items, K, k = 64, 1500, 8, 8   # k = latent rank << d=64 : the theorem's o(d) LOW-RANK premise

def pr(X):
    """participation ratio of the covariance spectrum: (sum lambda)^2 / sum(lambda^2) ~ the effective rank."""
    C = np.cov(X.T); ev = np.linalg.eigvalsh(C); ev = ev[ev > 1e-9]
    return float((ev.sum() ** 2) / (ev ** 2).sum())

# shared LOW-RANK loading B: the representations occupy a rank-k subspace (k<<d), so PR ~ k = o(d) and the norm
# is bounded -- exactly the theorem's premise. The two classes differ ONLY by a mean shift along direction u.
B = rng.standard_normal((d, k))
u = rng.standard_normal(d); u /= np.linalg.norm(u)
y = rng.integers(0, 2, n_items); delta = 2.0

disp = np.empty(n_items); proj = np.empty(n_items); pooled = {0: [], 1: []}
for i in range(n_items):
    mean = (y[i] - 0.5) * 2 * delta * u                 # class mean shifted ALONG u only
    S = mean + rng.standard_normal((K, k)) @ B.T          # K samples inside the SHARED rank-k subspace
    disp[i] = float(np.mean(np.linalg.norm(S - S.mean(0), axis=1)))   # dispersion = SE/endpoint analog
    proj[i] = float(S.mean(0) @ u)                        # supervised direction readout
    pooled[y[i]].append(S.mean(0))

pr0, pr1 = pr(np.array(pooled[0])), pr(np.array(pooled[1]))
au_disp = roc_auc_score(y, disp)
au_dir = roc_auc_score(y, proj)

# ---- control: break the invariance by giving the classes DIFFERENT rank (so PR genuinely differs) ----
n2 = n_items // 2; k2 = 28
B0 = rng.standard_normal((d, k)); B1 = rng.standard_normal((d, k2))
Xc0 = rng.standard_normal((n2, k)) @ B0.T
Xc1 = rng.standard_normal((n2, k2)) @ B1.T               # higher rank => higher PR and dispersion
ctrl_disp = np.r_[np.linalg.norm(Xc0 - Xc0.mean(0), axis=1), np.linalg.norm(Xc1 - Xc1.mean(0), axis=1)]
yc = np.r_[np.zeros(n2), np.ones(n2)]
au_ctrl = roc_auc_score(yc, ctrl_disp)
prc0, prc1 = pr(Xc0), pr(Xc1)

# ---- visualization: overlapping endpoint distributions (blind) vs separated direction (escape) vs control ----
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
    panels = [(ax[0], disp, y, f"Endpoint / dispersion readout\n(PR-invariant) — BLIND, AUROC {au_disp:.2f}"),
              (ax[1], proj, y, f"Direction readout (escape hatch)\n— SEPARATES, AUROC {au_dir:.2f}"),
              (ax[2], ctrl_disp, yc, f"Control: PR-invariance BROKEN\nendpoint now DETECTS, AUROC {au_ctrl:.2f}")]
    for a, sc, lab, title in panels:
        lab = np.asarray(lab)
        a.hist(sc[lab == 0], bins=40, alpha=0.6, color="#1a7f37", density=True, label="confident-correct")
        a.hist(sc[lab == 1], bins=40, alpha=0.6, color="#cf222e", density=True, label="confident-wrong")
        a.set_title(title, fontsize=10); a.legend(fontsize=7, loc="upper right"); a.set_yticks([])
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Theorem simulation: PR input-invariance ⇒ an endpoint readout is blind to confident error "
                 "(a direction readout escapes; breaking the invariance restores detection)", fontsize=10.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    Path(_ARGS.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_ARGS.out, dpi=130); print(f"[viz] saved {_ARGS.out}")
except Exception as e:
    print("[viz] skipped:", str(e)[:90])

print("=" * 88)
print("THEOREM SIMULATION  -  PR input-invariance => endpoint/dispersion readout is blind")
print("=" * 88)
print(f"\n[invariant regime, theorem premise] participation ratio: class0 PR={pr0:.1f}  vs  class1 PR={pr1:.1f}"
      f"  (both ~ rank {k} = o(d) << d={d}, bounded norm; input-INVARIANT)")
print(f"  DISPERSION / endpoint readout (semantic-entropy analog):  AUROC = {au_disp:.3f}   "
      f"-> {'BLIND (~0.5) — as predicted' if abs(au_disp-0.5) < 0.06 else 'UNEXPECTED'}")
print(f"  DIRECTION readout (the supervised escape hatch):          AUROC = {au_dir:.3f}   "
      f"-> {'SEPARATES (~1.0) — escape hatch works' if au_dir > 0.9 else 'UNEXPECTED'}")
print(f"\n[control: invariance broken] class0 PR={prc0:.1f} vs class1 PR={prc1:.1f} (different RANK by construction)")
print(f"  same DISPERSION readout now:                              AUROC = {au_ctrl:.3f}   "
      f"-> {'SEPARATES (>0.5) — the blindness was caused by the shared low-rank geometry' if au_ctrl > 0.6 else 'UNEXPECTED'}")
ok = abs(au_disp - 0.5) < 0.06 and au_dir > 0.9 and au_ctrl > 0.6
print("\n" + "=" * 88)
print("VERDICT:", "PREDICTED BEHAVIOR HOLDS IN SIMULATION (a demonstration, not a formal proof) — under the theorem's"
      " low-rank/bounded-norm premise the endpoint readout is blind, a direction readout separates, and breaking the"
      " invariance restores detection." if ok else "CHECK — numbers off.")
print("=" * 88)
import sys; sys.exit(0 if ok else 1)
