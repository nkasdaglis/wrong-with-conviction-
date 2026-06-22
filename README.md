# Wrong With Conviction — reproducibility & audit artifact

This repository accompanies the paper **"Wrong With Conviction"** (N. Kasdaglis), which shows that
unsupervised endpoint detectors (and semantic entropy) are **structurally blind to confident error**
in large language models, and that a **supervised, internal-state Governor** can read the error regime,
route a matching correction, and **refuse** when residual risk is high.

The purpose of this repo is **verification**: every load-bearing number in the paper can be checked,
intuitively, against its source — code, data, and statistics, side by side.

## What's here

Each load-bearing result gets one **audit card** with six blocks:

1. **Claim** — the result as stated (value, CI, *n*, and where it appears in the paper).
2. **Code excerpt** — the actual lines that produced the number (data-selection *and* the analysis).
3. **Data** — the source rows (inline if small, else a pointer), with row counts and the columns that matter.
4. **Descriptive stats + sanity checks** — *n*, estimate, CI, and the degeneracy checks for that result type
   (class balance, non-zero variance, non-singular covariance) so the number is not a degenerate artifact.
5. **Result** — the output the analysis produced.
6. **Verify** — a one-command re-run, keyed by result ID, that recomputes the statistic from the stored
   source data and prints the number live.

Two tiers of reproduction:

- **Audit / recompute (this repo, $0):** recompute each reported statistic from the stored source data.
  This is what a reviewer pulls — it confirms the numbers are real and not leakage/degeneracy artifacts.
- **Full reproduction (documented):** re-running the model captures and probe training from the open-weight
  models requires GPU and the capture pipeline; that path is documented separately for the determined replicator.

## What's *not* here (by design)

This is a **results-and-verification** artifact. It deliberately does **not** include the internal
research-process materials (the multi-agent laboratory's governance, logs, or deliberation), which are the
subject of a separate methodological treatment. Nothing here contains proprietary or pre-publication-sensitive
material.

## Status

Scaffold. Per-result audit cards and the verify harness are landing here; this README will index them as they
are added.

## Citation

See the paper. (Citation block to be added on preprint posting.)

## License

See `LICENSE`.
