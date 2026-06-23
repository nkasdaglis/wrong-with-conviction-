"""MANUAL-AUDIT HARNESS — confirm each load-bearing Paper-A result against its source data, intuitively.

PI runs:  .venv/Scripts/python.exe scripts/audit.py <RESULT_ID>     (e.g. G3)
          .venv/Scripts/python.exe scripts/audit.py --all
          .venv/Scripts/python.exe scripts/audit.py --list

For each result it prints a 6-block CARD: (1) CLAIM as stated in the paper, (2) SOURCE file (path + sha16),
(3) DATA (n, fields), (4) STATS + sanity/degeneracy checks, (5) RESULT (stored), (6) VERIFY — recompute the statistic
from the stored data where raw per-item arrays exist (tier-2), else confirm paper==file (tier-1) — and print MATCH/MISMATCH.
$0, read-only, recomputes the stat from stored source data only (never re-runs an experiment).
"""
import json, glob, hashlib, sys, math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def sha16(p):
    try: return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
    except Exception: return "??"

def find(pat):
    g = glob.glob(str(REPO / pat))
    if g:
        return g[0]
    alt = REPO / "docs" / "data" / Path(pat).name   # public-repo layout: result files live in docs/data/
    return str(alt) if alt.exists() else None

def auroc(scores, labels):
    """Rank-based AUROC (Mann-Whitney); labels: 1=positive(error), 0=negative. No deps."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    ranks = [0.0]*len(pairs); i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]: j += 1
        r = (i + j - 1)/2.0 + 1
        for k in range(i, j): ranks[k] = r
        i = j
    npos = sum(l for _, l in pairs); nneg = len(pairs) - npos
    if npos == 0 or nneg == 0: return float("nan")
    sum_pos = sum(rk for rk, (_, l) in zip(ranks, pairs) if l == 1)
    return (sum_pos - npos*(npos+1)/2.0) / (npos*nneg)

def _mean(rows, col): return sum(r[col] for r in rows)/len(rows)

def _median(xs):
    s = sorted(xs); n = len(s)
    return s[n//2] if n % 2 else (s[n//2-1] + s[n//2]) / 2.0

def _recompute_g4(_d):
    """G4 cross-task inversion: join e25 open-gen SE with the independent forced-choice label y
    (layerwise_features.npz), by item_index, then AUROC(SE | fc-wrong vs fc-correct). Mirrors se_kinematic_join.py."""
    import numpy as np
    e = json.load(open(find("experiments/epsilon_qwen_2026_06_08/outputs/e25_se_capture_result.json")))["per_item"]
    fc = np.load(find("experiments/epsilon_qwen_2026_06_08/outputs/layerwise_features.npz"), allow_pickle=True)["y"].astype(int)
    se = [r["SE"] for r in e]; idx = [r["item_index"] for r in e]
    return auroc(se, [int(fc[i]) for i in idx])

# --- result registry: id -> spec ---
R = {
 "G3": {
   "claim": "Semantic entropy is BLIND to confident error (double dissociation). Gemma-2-27B SE->error AUROC = 0.527, CI [0.489,0.565], n=800; all 12 models bounded <0.60. Paper: Table I + IV.B.",
   "file": "experiments/epsilon_qwen_2026_06_08/outputs/semantic_entropy_nli_gemma27b_result.json",
   "stored": lambda d: ("AUROC_SE_to_error_aggregate", d["AUROC_SE_to_error_aggregate"], d.get("AUROC_SE_to_error_CI95"), d.get("n")),
   "recompute": lambda d: auroc([it["SE"] for it in d["per_item"]], [it["err"] for it in d["per_item"]]),
   "checks": lambda d: [
       ("n matches", d["n"] == len(d["per_item"])),
       ("AUROC in useless band (<0.60)", d["AUROC_SE_to_error_aggregate"] < 0.60),
       ("both classes present (non-degenerate)", 0 < sum(it["err"] for it in d["per_item"]) < d["n"]),
       ("SE has variance (not all-equal)", len(set(it["SE"] for it in d["per_item"])) > 5),
   ]},
 "A1": {
   "claim": "Supervised internal-state probe DETECTS confident error. Qwen-0.5B AUROC = 0.731, CI [0.715,0.746], perm p=0.001; clears confound floor 0.566; beats confidence baseline 0.633. Paper: Sec VI + Fig 2.",
   "file": "experiments/epsilon_qwen_2026_06_08/outputs/c1_detector_Qwen2p5-0p5B-Instruct_result.json",
   "stored": lambda d: ("H1_detector.auroc", d["H1_detector"]["auroc"], d["H1_detector"]["ci95"], d.get("confident_WRONG_n")),
   "recompute": None,
   "checks": lambda d: [
       ("clears confound floor", d["H1_detector"]["clears_floor"] and d["H1_detector"]["auroc"] > d["H1_detector"]["confound_floor"]),
       ("permutation significant (p<0.05)", d["H1_detector"]["perm_p"] < 0.05),
       ("beats confidence baseline", d["H1_detector"]["auroc"] > d["H1_detector"]["confidence_baseline_auroc"]),
       ("prereg PASS", d["H1_detector"]["PASS_preregistered"]),
   ]},
 "G4": {
   "claim": "Cross-task SE INVERSION: open-gen SE vs forced-choice error AUROC = 0.443, CI excludes high; n=250; 56.5%% of fc-wrong in low-SE cluster. Paper: Sec (inversion).",
   "file": "experiments/epsilon_qwen_2026_06_08/outputs/se_kinematic_join_result.json",
   "stored": lambda d: ("AUROC_SE_fc...", d["AUROC_SE_forcedchoiceWRONG_vs_CORRECT_EXPECT_0.5"], d.get("CI95"), d.get("n")),
   "recompute": _recompute_g4,
   "checks": lambda d: [
       ("n = fc_wrong + fc_correct", d["n"] == d["n_fc_wrong"] + d["n_fc_correct"]),
       ("non-circular (independent task)", "NONCIRCULAR" in d),
       ("inversion direction (<0.5 or near)", d["AUROC_SE_forcedchoiceWRONG_vs_CORRECT_EXPECT_0.5"] < 0.55),
   ]},
 "B1": {
   "claim": "Internal refusal gate beats calibrated confidence at matched coverage: +5.7 (cov 0.9) to +6.4 (cov 0.7) pts in the danger zone, Qwen-1.5B. Paper: Sec (refusal payoff).",
   "file": "experiments/epsilon_qwen_2026_06_08/outputs/r2_cross_vendor_risk_coverage_result.json",
   "stored": lambda d: ("DANGER_ZONE det_minus_conf_pp @cov_0.7", d["models"]["1.5B"]["DANGER_ZONE"]["cov_0.7"]["det_minus_conf_pp"], None, d["models"]["1.5B"]["danger_zone_n"]),
   "recompute": lambda d: (d["models"]["1.5B"]["DANGER_ZONE"]["cov_0.7"]["detector"] - d["models"]["1.5B"]["DANGER_ZONE"]["cov_0.7"]["confidence"])*100,
   "checks": lambda d: [
       ("det>conf @cov_0.9 (gate beats calibrated confidence)", d["models"]["1.5B"]["DANGER_ZONE"]["cov_0.9"]["det_minus_conf_pp"] > 0),
       ("det_minus_conf @cov_0.9 ~5.7", abs(d["models"]["1.5B"]["DANGER_ZONE"]["cov_0.9"]["det_minus_conf_pp"] - 5.7) < 0.5),
       ("det_minus_conf @cov_0.7 ~6.4", abs(d["models"]["1.5B"]["DANGER_ZONE"]["cov_0.7"]["det_minus_conf_pp"] - 6.4) < 0.5),
       ("entity-grouped detector AUROC > 0.8", d["models"]["1.5B"]["detector_AUROC_entitygrouped"] > 0.8),
   ]},
 "C4": {
   "claim": "Router (detect->route->correct->refuse) beats best single fix by +7.1pts on Gemma-2-2B (LEARNED 0.802 vs best-single 0.731); McNemar p<0.001; boot CI excludes 0; n=1013. Paper: Sec VIII.",
   "file": "experiments/epsilon_qwen_2026_06_08/outputs/regime_router_powered_gemma2b_result.json",
   "stored": lambda d: ("gain_over_best_single_fix", d["gain_over_best_single_fix"], d.get("gain_boot95_ci"), len(d["per_item_rows"])),
   "recompute": lambda d: _mean(d["per_item_rows"], "learned") - max(_mean(d["per_item_rows"], "trap"), _mean(d["per_item_rows"], "ord")),
   "checks": lambda d: [
       ("ordering router>best-single>base", d["ordering_router_gt_bsf_gt_base"]),
       ("gain boot95 CI excludes 0", d["gain_boot95_ci"][0] > 0),
       ("McNemar p<0.001", d["mcnemar"]["p_two_sided"] < 0.001),
       ("leakage-controlled (heldout-source routing present)", "heldout_routed_ordinary_acc" in d["leakage_control"]),
   ]},
 "F1": {
   "claim": "Integrated governor lift base->router: Gemma-2-2B 0.597->0.802 (+20.4pts); headline span +13 to +20.5 across models. Paper: Sec (governor).",
   "file": "experiments/epsilon_qwen_2026_06_08/outputs/regime_router_powered_gemma2b_result.json",
   "stored": lambda d: ("LEARNED_router - BASE", round(d["LEARNED_router"] - d["BASE"], 4), None, len(d["per_item_rows"])),
   "recompute": lambda d: _mean(d["per_item_rows"], "learned") - _mean(d["per_item_rows"], "base"),
   "checks": lambda d: [
       ("router > base", d["LEARNED_router"] > d["BASE"]),
       ("lift in +13..+20.5 headline span", 0.13 <= (d["LEARNED_router"] - d["BASE"]) <= 0.21),
   ]},
 "C3": {
   "claim": "Anti-transfer honestly DOWNGRADED to a style-confound: the within-regime CROSS-DATASET transfer net median = -0.752 (every cross-dataset axis is negative), so the axes are dataset/style-specific, not two opposite causal regimes. EXPLORATORY. Paper: Limitations.",
   "file": "experiments/epsilon_qwen_2026_06_08/outputs/anti_transfer_matrix_qwen0.5b_result.json",
   "stored": lambda d: ("cross-dataset transfer net median (paper -0.752)", -0.752, None, len(d["diagnostics"]["ordinary_to_ordinary_nets"])),
   "recompute": lambda d: _median(d["diagnostics"]["ordinary_to_ordinary_nets"]),
   "checks": lambda d: [
       ("all cross-dataset transfers negative (style-specific)", all(x < 0 for x in d["diagnostics"]["ordinary_to_ordinary_nets"])),
       ("honest downgrade recorded (STYLE-CONFOUND)", "STYLE-CONFOUND" in d["verdict_tag"]),
       ("flagged EXPLORATORY", d.get("EXPLORATORY") is True),
   ]},
 "D1": {
   "claim": "Regime-matched LoRA corrector: base 0.561 -> LoRA 0.852 on HELD-OUT categories (+0.291). Paper: Sec (correctors).",
   "file": "experiments/epsilon_qwen_2026_06_08/outputs/governance_lora_result.json",
   "stored": lambda d: ("lift (held-out category)", d["lift"], None, d.get("n_test_heldout_category")),
   "recompute": lambda d: d["lora_heldout_category_acc"] - d["base_heldout_category_acc"],
   "checks": lambda d: [
       ("base ~0.561", abs(d["base_heldout_category_acc"] - 0.561) < 0.01),
       ("LoRA ~0.852", abs(d["lora_heldout_category_acc"] - 0.852) < 0.01),
       ("lift positive + held-out (not in-sample)", d["lift"] > 0),
   ]},
 # ----- SUPPORTING (non-load-bearing) claims -----
 "SAE": {
   "group": "supporting",
   "claim": "An unsupervised Gemma-Scope SAE's reconstruction error is also BLIND to confident error (mirrors semantic entropy). Gemma-2-2B, n=817: recon-error AUROC ~0.497 [.448,.545], perm p=0.56; mean recon-error confident-wrong ~ correct. Paper: Sec II/IV (SAE).",
   "file": "experiments/epsilon_qwen_2026_06_08/outputs/sae_confident_error_gemma2b_result.json",
   "stored": lambda d: ("D3 recon-err mean: confident-wrong vs correct (L12)",
       "{:.2f} vs {:.2f}".format(d["per_sae_layer"]["pt_commitband_L12"]["D3_unsupervised_sae_recon_error"]["mean_recon_err_confidentwrong"],
                                 d["per_sae_layer"]["pt_commitband_L12"]["D3_unsupervised_sae_recon_error"]["mean_recon_err_correct"]), None, d.get("n")),
   "recompute": None,
   "checks": lambda d: [
       ("recon-err near-identical -> no separation (blind)", abs(d["per_sae_layer"]["pt_commitband_L12"]["D3_unsupervised_sae_recon_error"]["mean_recon_err_confidentwrong"] - d["per_sae_layer"]["pt_commitband_L12"]["D3_unsupervised_sae_recon_error"]["mean_recon_err_correct"]) < 1.0),
       ("supervised raw-residual probe still reads it (D1>0.5)", d["per_sae_layer"]["pt_commitband_L12"]["D1_supervised_probe_raw_resid"]["AUROC_oof"] > 0.5),
       ("n=817 (powered)", d.get("n") == 817),
   ]},
 "DET8B": {
   "group": "supporting",
   "claim": "The supervised detector GENERALIZES across model families at scale. Llama-3.1-8B (Meta) AUROC 0.849 [.83,.866], perm p=0.001, clears confound floor 0.698 -- the strongest in the four-family set (Gemma-2-27B 0.775, Qwen-14B 0.802, Qwen-32B 0.821, Mistral-24B 0.727 marginal). Same detector protocol, cloud-run. Paper: Table II.",
   "file": "experiments/epsilon_qwen_2026_06_08/outputs/c1_detector_bigmodels_folded_from_s3/c1_detector_Llama-3p1-8B-Instruct_result.json",
   "stored": lambda d: ("H1_detector.auroc (Llama-3.1-8B)", d["H1_detector"]["auroc"], d["H1_detector"]["ci95"], d.get("confident_WRONG_n")),
   "recompute": None,
   "checks": lambda d: [
       ("clears confound floor", d["H1_detector"]["clears_floor"] and d["H1_detector"]["auroc"] > d["H1_detector"]["confound_floor"]),
       ("permutation significant (p<0.05)", d["H1_detector"]["perm_p"] < 0.05),
       ("cross-family (Meta Llama, not Qwen)", "Llama" in d.get("model_id", "")),
   ]},
 "RX3B": {
   "group": "supporting",
   "claim": "The router (detect->route->correct->refuse) GENERALIZES across models. Gain over the best single fix: Qwen-1.5B +4.7, Qwen-3B +6.2, Gemma-2-2B +7.1, Llama-3.2-3B +5.3 -- all McNemar p<0.001, n=1013/model. Card shows Qwen-3B (+6.2). Paper: Table (router).",
   "file": "experiments/epsilon_qwen_2026_06_08/outputs/regime_router_powered_qwen3b_result.json",
   "stored": lambda d: ("gain_over_best_single_fix (Qwen-3B)", d["gain_over_best_single_fix"], d.get("gain_boot95_ci"), len(d["per_item_rows"])),
   "recompute": lambda d: _mean(d["per_item_rows"], "learned") - max(_mean(d["per_item_rows"], "trap"), _mean(d["per_item_rows"], "ord")),
   "checks": lambda d: [
       ("McNemar p<0.001", d["mcnemar"]["p_two_sided"] < 0.001),
       ("router > base", d["LEARNED_router"] > d["BASE"]),
       ("n=1013", len(d["per_item_rows"]) == 1013),
   ]},
}

def _extra_card(rid):
    """Confirm an inventory claim (no per-item recompute): read the cited field from its result file."""
    try:
        from scripts.build_audit_site import CLAIMS_EXTRA, EXTRA_LIVE
    except Exception:
        from build_audit_site import CLAIMS_EXTRA, EXTRA_LIVE
    c = CLAIMS_EXTRA.get(rid)
    if not c:
        print(f"unknown id {rid}; --list for ids"); return
    f = find("experiments/**/outputs/" + c["file"]) or find(c["file"])
    print("=" * 92); print(f"AUDIT CARD — {rid}"); print("-" * 92)
    print("1. CLAIM:\n   " + c["title"])
    if not f:
        print(f"\n   !! SOURCE FILE NOT FOUND: {c['file']}"); print("=" * 92); return
    print(f"\n2. SOURCE: {Path(f).name}\n   sha16: {sha16(f)}")
    d = json.load(open(f, encoding="utf-8"))
    fp = EXTRA_LIVE.get(rid, ("", ""))[0]; cur = d
    try:
        for part in fp.split("."): cur = cur[part]
    except Exception:
        cur = None
    print(f"\n3-5. PUBLISHED: {c['num']}")
    print("\n6. VERIFY:")
    print(f"     read '{fp}' from the result file = {repr(cur)}")
    print("     -> confirm this is the value quoted in the published claim above.")
    print("=" * 92)

def card(rid):
    spec = R.get(rid)
    if not spec: return _extra_card(rid)
    f = find(spec["file"])
    print("="*92); print(f"AUDIT CARD — {rid}")
    print("-"*92)
    print("1. CLAIM:\n   " + spec["claim"])
    if not f: print(f"\n   !! SOURCE FILE NOT FOUND: {spec['file']}"); return
    rel = str(Path(f).relative_to(REPO)); print(f"\n2. SOURCE: {rel}\n   sha16: {sha16(f)}")
    d = json.load(open(f))
    if spec.get("display_only"):
        nums = {k: v for k, v in d.items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
        print("\n3-5. STORED VALUES (confirm against the paper number):")
        for k, v in list(nums.items())[:24]: print(f"     {k}: {repr(v)[:80]}")
        print("\n6. VERIFY: tier-1 (display) — confirm the paper's quoted number appears above. (recompute wiring pending for this id.)")
        print("="*92); return
    key, val, ci, n = spec["stored"](d)
    print(f"\n3. DATA: n={n}" + (f", per_item={len(d['per_item'])} raw records" if 'per_item' in d else ""))
    print("\n4. STATS + SANITY/DEGENERACY CHECKS:")
    for label, ok in spec["checks"](d): print(f"     [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"\n5. RESULT (stored): {key} = {val}  CI={ci}  n={n}")
    rec = spec.get("recompute")
    print("\n6. VERIFY:")
    if rec:
        got = rec(d)
        match = (not math.isnan(got)) and abs(got - val) <= 0.005
        print(f"     RECOMPUTED from stored raw data: {got:.4f}")
        print(f"     PAPER/FILE says: {val}")
        print(f"     -> {'MATCH (within 0.005)' if match else 'MISMATCH — INVESTIGATE'}")
    else:
        print(f"     tier-1: stored file value = {val}; confirm it equals the paper's quoted number.")
    print("="*92)

def main():
    a = sys.argv[1:]
    extra_ids = []
    try:
        from scripts.build_audit_site import CLAIMS_EXTRA as _CE
        extra_ids = list(_CE.keys())
    except Exception:
        try:
            from build_audit_site import CLAIMS_EXTRA as _CE; extra_ids = list(_CE.keys())
        except Exception:
            pass
    if not a or a[0] in ("--list", "-l"):
        print("RESULT IDs:"); [print(f"  {k}: {R[k]['claim'][:80]}") for k in R]
        for k in extra_ids: print(f"  {k}")
        return 0
    ids = (list(R.keys()) + extra_ids) if a[0] == "--all" else a
    for rid in ids: card(rid); print()
    return 0

if __name__ == "__main__":
    sys.exit(main())
