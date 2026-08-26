# H200 PoisonBase experiment artifacts — 2026-08-26

This directory is a local snapshot of the PoisonBase experiments run overnight on the `H200-1` server. The original repository path was:

`/home/ubuntu/mohammad/PoisonBase`

The snapshot contains logs, scripts, CSV results, summaries, and JSON metadata for two experiment groups:

1. DPP defense evaluation with EPIC and FRIENDS.
2. Cross-architecture DPP selection with `K=1` and `K=3`.

Only optimized perturbation tensors were transferred: files matching `poison_cache/delta_*.pt` inside the relevant run directories. Shared surrogate/victim training caches and model checkpoints were intentionally excluded. This snapshot supports result inspection, table generation, debugging, provenance, and downstream evaluation from the saved perturbations, but it does not contain the trained surrogate/victim cache needed to reproduce every training stage without retraining.

## Start here

- `log_diverese/final_report.txt`: human-readable summary of the defense experiment.
- `log_diverese/final_deadline_summary.tsv`: authoritative state and metrics for all 12 requested defense configurations.
- `log_diverese/cross_k13/final_report.txt`: human-readable cross-architecture summary.
- `log_diverese/cross_k13/final_deadline_summary.tsv`: authoritative state and metrics for all 24 cross-architecture cells.
- `log_diverese/cross_k13/protocol_notes.txt`: protocol exceptions and interpretation notes.

Do not infer that a run is complete merely because its result directory exists. Always check the `state`, `rows`, and `deltas` fields in the appropriate final summary.

## Directory layout

### `log_diverese/`

Run orchestration, per-configuration logs, manifests, deadline reports, monitoring records, and clean final summaries for the defense experiments.

Important files:

- `manifest.tsv`: the 12 requested defense configurations.
- `status.tsv`: chronological scheduler and pipeline events.
- `04_convnet_sapa_dog_bird_b0_02.log`, `08_resnet20_sapa_dog_bird_b0_01.log`, and `11_vgg13_sapa_dog_bird_b0_005.log`: detailed logs for the configurations that reached defense evaluation.
- `resnet20_sapa_b0_01_friends_partial_18.csv`: immutable merged snapshot of the 18 completed ResNet FRIENDS trials at the deadline.
- `cross_k13/`: all orchestration, logs, reports, and protocol notes for the cross-architecture experiments.

The directory name `log_diverese` preserves the spelling used during the run.

### `ours_result_alpha025_live/`

Attack-generation outputs and metadata for the defense workload using DPP selection with `SEL_ALPHA=0.25` and Jacobian scoring disabled. Completed optimized perturbations are stored under each run's `poison_cache/` directory as `delta_<target>.pt`.

### `defense_result/`

EPIC and FRIENDS evaluation results for the completed or partially completed `seldpp0.25` configurations.

- `results.csv` normally represents a fully merged result.
- `results_rank*.csv` are worker-level outputs and may indicate an incomplete run.
- For the partial ResNet FRIENDS run, use the explicit 18-row snapshot in `log_diverese/` rather than treating it as a completed 20-trial cell.

### `ours_result/`

Cross-architecture attack results whose directory names contain `seldpp2_selarch..._K1` or `seldpp2_selarch..._K3`. The victim architecture is the model immediately after `CIFAR10_`; the selector architecture appears after `selarch`. Completed optimized perturbations are included as `poison_cache/delta_<target>.pt`.

## Experiment 1: EPIC and FRIENDS defenses

Protocol:

- DPP selector with `SEL_ALPHA=0.25`.
- Jacobian score disabled: `USE_JACOBIAN_SCORE=0`.
- Five attack targets and four defense victims per configuration, giving 20 defense trials per defense when complete.
- Defenses: EPIC and FRIENDS.
- Physical GPUs 1–3 were used during the authorized seven-hour window.

Final configuration state:

- Complete: 2/12 configurations.
- Partial: 3/12 configurations.
- Not started before the deadline: 7/12 configurations.

Table-ready completed results report `ASR% (CTA%)`:

| Configuration | EPIC | FRIENDS |
|---|---:|---:|
| ConvNetBN, SAPA, dog–bird, budget 0.02 | 40.0 (76.87) | 60.0 (74.74) |
| ResNet20BN, SAPA, dog–bird, budget 0.01 | 55.0 (80.19) | Partial: 33.3 (74.39), 18/20 |
| VGG13BN, SAPA, dog–bird, budget 0.005 | 40.0 (77.59) | 20.0 (80.33) |

The ResNet FRIENDS value is partial and must not be inserted as a final 20-trial table cell without an explicit partial-result annotation.

Additional attack state:

- ConvNetBN FC budget 0.01 completed all five attack deltas and had 100% one-victim verification ASR, but its defenses were not started.
- ResNet20BN GradMatch budget 0.02 began perturbation optimization but produced no completed delta before it was paused.

## Experiment 2: cross-architecture K1/K3

Protocol:

- DPP `alpha=2`.
- Jacobian scoring disabled.
- 20 attack surrogates.
- Five targets and four victims per cell.
- Selector and victim architectures are different (off-diagonal architecture pairs).
- `K=1` and `K=3` settings were requested.

Final cell state:

- Complete: 11/24 cells.
- Partial: 5/24 cells.
- Not started before the deadline: 8/24 cells.

Notable completed K3 result:

- `x13`: ConvNetBN FC with ResNet20BN selector, 20/20 rows, ASR 60.0%, CTA 79.955%.

Partial cells at cutoff:

- `x09`: 16/20 rows.
- `x10`: 16/20 rows.
- `x14`: 13/20 rows.
- `x15`: 5/20 rows.
- `x16`: 0/20 evaluation rows with one crafted target.

Protocol exception: `x05` and `x06` are classified as complete with 16 rows because pinned VGG FC target 2171 was a live-cache baseline free-win and was dropped. See `log_diverese/cross_k13/protocol_notes.txt` before using those cells.

## Metric interpretation

- **ASR**: attack success rate, reported as a percentage.
- **CTA**: clean test accuracy after the evaluated attack/defense condition, reported as a percentage in the human-readable reports. Machine-readable CSV/TSV files may store it as a fraction between 0 and 1.
- **Rows**: completed target–victim evaluations. A standard complete cell has 20 rows unless a documented protocol exception applies.
- **Deltas**: completed crafted perturbations, normally one per target and therefore five for a fully crafted cell.

## Integrity and limitations

- No traceback, CUDA error, OOM, segmentation fault, or disk-space failure was detected in the experiment logs.
- The runs were stopped at their authorized deadlines without terminating unrelated server processes.
- The source `.tex` tables were not updated automatically. Only complete cells should be inserted unless the table explicitly labels partial results.
- Exactly 89 optimized perturbation tensors are included, totaling 350,351,993 bytes. Every transferred `.pt` file matches `poison_cache/delta_*.pt`.
- The 159 shared surrogate/victim training-cache checkpoints (about 2.13 GB on H200-1) were not transferred, and no `cache_alpha025_live` or `cache_cross_k13_live` directory is present locally.
- Snapshot contents after adding this README and the perturbations: 509 files totaling 351,586,284 logical bytes.
