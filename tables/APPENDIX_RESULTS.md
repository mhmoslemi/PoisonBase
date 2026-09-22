# Completed September 21 experiments

The tables match the grouped attack/budget headers, booktabs rules, small-caps
method names, one-decimal ASR, and bold/underlined best/second-best convention
in `abaltion_diff.tex`. All exact ties receive the same formatting. Budgets
in the new tables are **percentages** (0.1, 0.2, 0.5), not fractions.

Use `\input{tables/score_alternatives.tex}` for the five new scoring methods.
Use `\input{tables/tinyimagenet20_gm.tex}` for the small TinyImageNet experiment.
The optional `score_alternatives_with_reference.tex` combines the new results
with the eight ConvNet columns of the existing ablation, clearly separated as
historical values. Use it instead of the new-only score table if desired.
The reference rows were not recomputed or rerun; they must not be described as
matched 8-target × 6-victim results. No significance claim is made across blocks.

Required LaTeX packages: `booktabs`, `graphicx`, and `amsmath`.
`appendix_results_preview.tex` is a standalone compilation wrapper, not a paper
fragment. It produces a two-page preview in `output/pdf/` when compiled from
this repository's root.

## Audit and provenance

Run `python3 tools/build_appendix_result_tables.py` to regenerate all three
fragments and `appendix_results_audit.json`. The generator uses only the Python
standard library and fails if a batch is incomplete or its saved summary
disagrees with the trial results. It does not modify raw results or logs.

- Scoring batch: 40 cells × 48 distinct target/victim pairs = 1,920 trials.
- TinyImageNet batch: 3 methods × 48 distinct pairs = 144 trials.
- Total: 2,064 trials; 48 logs checked (40 scoring and 8 Tiny setup/method logs).
- All 40 scoring logs contain the final 48-evaluation verification marker.
- All 144 Tiny victim runs have a logged epoch-30 completion and matching
  per-trial JSON/CSV results. No searched failure marker was found in these logs.
- Saved ASR means, population SDs across eight targets, and mean post-training
  clean accuracies were independently recomputed and matched.
- Actual target IDs and source paths are recorded per cell in the audit JSON;
  SHA-256 hashes cover every source file read. These are local copied results
  and logs; scheduler accounting was not required or inferred from them.
- Decimal half-up rounding is applied to the exact success fractions. Thus
  15/48 is displayed as 31.3%, not Python's half-even 31.2%. Rank formatting uses
  unrounded values, not rounded display values.

For the new scoring methods, CIFAR-10/ConvNetBN uses the poison class dog and
true target class bird. The code's `fc` sample attack is displayed as BP,
matching the original table. The five rows correspond to classifier-weight
alignment `(r_i^T r_t)(h_i^T h_t)`, feature-normalized classifier alignment
`(r_i^T r_t) cos(h_i,h_t)`, the target-class logit margin, standardized similarity
plus standardized target-class margin, and negative poison-class confidence.
Targets are shared across methods and budgets within each attack; BP and GM
use different pinned target sets, with SAPA sharing the GM target set.

The TinyImageNet result is the **20-class, 4,000-training-image subset**, not
full TinyImageNet or ImageNet-100 from ImageNet-1K. It uses 1,000 validation
images, native 64×64 inputs, a ResNet18 small-image stem, 30 epochs, three shared
clean surrogates, GM with 250 steps and eight restarts, and eight poisons at
0.2% of the subset training size. The fixed poison/target WNIDs are
`n03804744` / `n03837869` respectively. The class list and original sample
indices are retained in the result manifest. The ± values are **population
standard deviations over eight per-target ASRs**, not confidence intervals or
standard errors. Clean accuracy is evaluated after poisoned training; no clean
victim control comparison is inferred.

This TinyImageNet run has 1/48 successes for RAND and 0/48 for both M-only and
BASIS, with mean clean validation accuracies 53.64%, 53.50%, and 53.30%.
It does not demonstrate a BASIS advantage on this subset. The output tables
retain all methods and all trials.
