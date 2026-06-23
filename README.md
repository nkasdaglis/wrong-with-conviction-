# Wrong With Conviction: Why Confident Errors Evade Detection in Language Models
**Nicholas Kasdaglis, Ph.D. — TOPP Interactive Design** · <nicholas@toppsystems.com>

Language models produce some wrong answers with the same internal and output signature as correct ones —
*confident errors*. This work shows that unsupervised endpoint detectors, and semantic entropy, are
**structurally blind** to confident errors, and that a **supervised read of the model's internal state** can
detect the error regime, route a matching correction, and **refuse** when residual risk remains.

## ▶ Interactive result audit

Every claim in the paper is laid out with its experiment, sample size, effect size, statistical test, result,
the reason for its status, and the exact file it came from — and a button to **check the number live in your
browser**:

### → **[Open the interactive audit](https://nkasdaglis.github.io/wrong-with-conviction/audit.html)**

Click any claim to expand it; press **Verify** to recompute the statistic from the raw data, or to read the
value back from its result file, in your browser. The page is read-only — anything you do (verify, mark
reviewed) stays in your own browser.

## What's in this repository

| Path | What it is |
|------|------------|
| [`docs/audit.html`](https://nkasdaglis.github.io/wrong-with-conviction/audit.html) | The interactive audit (the link above) |
| [`docs/data/`](docs/data) | The source result file behind every number |
| [`docs/validation_report.txt`](docs/validation_report.txt) | The verifier's own test results (the recompute engine checked against scikit-learn) |
| [`LICENSE`](LICENSE), [`PATENTS`](PATENTS) | License and patent notice |

## Reproducing the numbers

- **In your browser:** open the audit and press **Verify** on any claim.
- **From the data:** every reported number is recomputed from, or read back from, the result files in
  `docs/data/`. The recompute engine is checked against scikit-learn to within 1e-9, catches planted wrong
  values, and is stable under bootstrap resampling (see `docs/validation_report.txt`).
- **From scratch:** re-running the model captures and probe training from the open-weight models requires a
  GPU; that pipeline is described in the paper.

## Citation

If you use this work, please cite the preprint:

> Kasdaglis, N. (2026). *Wrong With Conviction: Why Confident Errors Evade Detection in Language Models* (Version 1). Zenodo. https://doi.org/10.5281/zenodo.20820157

```bibtex
@misc{kasdaglis2026wrongwithconviction,
  author    = {Kasdaglis, Nicholas},
  title     = {Wrong With Conviction: Why Confident Errors Evade Detection in Language Models},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1},
  doi       = {10.5281/zenodo.20820157},
  url       = {https://doi.org/10.5281/zenodo.20820157}
}
```

## Use & terms

These materials are shared for **peer review, verification, and academic / non-commercial use**. They are
licensed under **Creative Commons Attribution-NonCommercial 4.0** — you may share and adapt them, with
attribution, for non-commercial purposes (`LICENSE`). The methods and system described in the paper are
**patent pending** (`PATENTS`); commercial use, or implementing the methods, requires permission from the
author. Please cite the paper if you use these materials.
