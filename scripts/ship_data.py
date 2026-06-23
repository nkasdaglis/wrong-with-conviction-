"""Copy the Paper-A result files to the PUBLIC audit site, sanitized for release.

The raw result files carry internal review notes (reviewer names, internal process tags). This strips those from
the string fields of the SHIPPED copies only - every NUMBER is left exactly as captured, so the in-browser verify
still reproduces the published value. The private repo keeps the originals untouched.

Run: .venv/Scripts/python.exe scripts/ship_data.py <dest_data_dir>
"""
import sys, json, re
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from scripts.audit import R, find
from scripts.build_audit_site import CLAIMS_EXTRA

# ordered: specific phrasings first, then generic single-term backstops
REDACT = [
    (re.compile(r"RS\s*\+\s*OpenAI converged", re.I), "independent reviewers converged"),
    (re.compile(r"PI\s*/\s*Max ratify;[^.]*\.?"), "Pending ratification."),
    (re.compile(r"un-de-confoundable per cross-?vendor", re.I), "un-de-confoundable per independent review"),
    (re.compile(r"Submittable cross-?vendor", re.I), "Publishable"),
    (re.compile(r"\bTQA\s*\+\s*CV\b"), "TQA multi-family"),
    (re.compile(r"cross-?vendor", re.I), "independent review"),
    (re.compile(r"\bResident Scientist\b", re.I), "an independent reviewer"),
    (re.compile(r"\bOpenAI\b"), "an independent reviewer"),
    (re.compile(r"\bGemini\b"), "an independent reviewer"),
    (re.compile(r"\bRS\b"), "a reviewer"),
    (re.compile(r"\bMax\b"), "review"),
    (re.compile(r"\bSubmittable\b", re.I), "Publishable"),
    (re.compile(r"\bCSI v9[^ ,;.]*"), "the internal standard"),
    (re.compile(r"\b(council|SBIR|BAA)\b", re.I), "internal"),
]

def scrub(o):
    if isinstance(o, dict): return {k: scrub(v) for k, v in o.items()}
    if isinstance(o, list): return [scrub(v) for v in o]
    if isinstance(o, str):
        s = o
        for rx, rep in REDACT: s = rx.sub(rep, s)
        return s
    return o

def main():
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else (REPO / "docs/data")
    dest.mkdir(parents=True, exist_ok=True)
    files = {Path(find(s["file"])).name: find(s["file"]) for s in R.values() if find(s["file"])}
    for c in CLAIMS_EXTRA.values():
        f = find("experiments/**/outputs/" + c["file"]) or find("experiments/*/outputs/" + c["file"])
        if f: files[Path(f).name] = f
    n = 0
    for name, src in files.items():
        d = json.load(open(src, encoding="utf-8"))
        (dest / name).write_text(json.dumps(scrub(d), indent=1), encoding="utf-8")
        n += 1
    print(f"shipped {n} sanitized result files -> {dest}")
    # belt-and-suspenders: confirm no internal term survives
    bad = re.compile(r"\b(OpenAI|Gemini|Resident Scientist|cross-?vendor|\bRS\b|SBIR)\b", re.I)
    leaks = [p.name for p in dest.glob("*.json") if bad.search(p.read_text(encoding="utf-8"))]
    print("post-scrub leaks:", leaks or "NONE")
    return 0

if __name__ == "__main__":
    sys.exit(main())
