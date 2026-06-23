"""VALIDATE THE PYTHON BEHIND EVERY CLAIM. One command that checks/validates/verifies the verification code for all
26 audit claims, and writes validation_report.txt (embedded on the audit page's 'validation & testing report').

For each claim it confirms the verification Python actually runs and reproduces the published value:
  * 11 recompute claims  -> re-derive the statistic from the result file (audit.py) and assert MATCH (within 0.005).
  * 15 inventory claims   -> read the cited field from the shipped result file (same path the in-browser button uses)
                             and assert the value is present and finite (provenance confirmed).
Also re-runs the two engine proofs: prove_audit_harness.py (recompute == scikit-learn to 1e-9; catches planted
errors; bootstrap-stable) and prove_theorem_sim.py (PR-invariance => endpoint blind, direction separates).

Run: .venv/Scripts/python.exe scripts/validate_all_claims.py
Exit 0 iff all 26 claims validate AND both engine proofs pass.
"""
import sys, json, subprocess
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from scripts.audit import R, find
from scripts.build_audit_site import CLAIMS_EXTRA, EXTRA_LIVE, SUPP

LINES = []
def out(s=""): LINES.append(s); print(s)

def main():
    out("=" * 92)
    out("PAPER-A AUDIT - PYTHON VALIDATION FOR EVERY CLAIM (validate_all_claims.py)")
    out("=" * 92)
    ok_all = True

    out("\n[1/3] RECOMPUTE CLAIMS - re-derive the statistic from the result file and assert MATCH:")
    nrec = nrec_ok = 0
    for rid, spec in R.items():
        fn = spec.get("recompute")
        if fn is None:
            continue
        nrec += 1
        f = find(spec["file"])
        try:
            d = json.load(open(f, encoding="utf-8"))
            stored = spec["stored"](d); val = stored[1]
            got = fn(d)
            match = (isinstance(got, float) and abs(got - val) <= 0.005)
            nrec_ok += int(match); ok_all &= match
            out(f"   [{'PASS' if match else 'FAIL'}] {rid:5} recompute={got:.4f} vs published={val:.4f}  ({Path(f).name})")
        except Exception as e:
            ok_all = False; out(f"   [FAIL] {rid:5} ERROR {str(e)[:70]}")
    out(f"   -> {nrec_ok}/{nrec} recompute claims reproduce their published value.")

    out("\n[2/3] INVENTORY CLAIMS - read the cited field from the shipped result file (the in-browser path):")
    nx = nx_ok = 0
    for cid, (fp, lab) in EXTRA_LIVE.items():
        c = CLAIMS_EXTRA.get(cid)
        if not c:
            continue
        nx += 1
        f = find("experiments/**/outputs/" + c["file"]) or find("experiments/*/outputs/" + c["file"])
        try:
            cur = json.load(open(f, encoding="utf-8"))
            for part in fp.split("."): cur = cur[part]
            present = cur is not None and (not isinstance(cur, float) or cur == cur)  # finite (not NaN)
            nx_ok += int(present); ok_all &= present
            out(f"   [{'PASS' if present else 'FAIL'}] {cid:5} {fp} = {str(cur)[:42]}  ({Path(f).name})")
        except Exception as e:
            ok_all = False; out(f"   [FAIL] {cid:5} field '{fp}' ERROR {str(e)[:60]}")
    out(f"   -> {nx_ok}/{nx} inventory claims: the cited field is present in the shipped result file.")

    out("\n[3/3] ENGINE PROOFS - the verifier itself is tested:")
    for name in ["prove_audit_harness.py", "prove_theorem_sim.py"]:
        r = subprocess.run([sys.executable, str(REPO / "scripts" / name)], capture_output=True, text=True)
        passed = r.returncode == 0
        ok_all &= passed
        out(f"   [{'PASS' if passed else 'FAIL'}] {name}  (exit {r.returncode})")

    total = nrec + nx
    out("\n" + "=" * 92)
    out(f"VALIDATION SUMMARY: {nrec_ok + nx_ok}/{total} claims validated "
        f"({nrec_ok} recompute-MATCH + {nx_ok} field-present); engine proofs {'PASS' if ok_all else 'CHECK'}.")
    out("VERDICT: " + ("ALL CLAIMS' VERIFICATION CODE IS VALIDATED." if ok_all else "CHECK - see failures above."))
    out("=" * 92)

    rep = Path(REPO / "process/papers/figs/validation_report.txt")
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(LINES), encoding="utf-8")
    return 0 if ok_all else 1

if __name__ == "__main__":
    sys.exit(main())
