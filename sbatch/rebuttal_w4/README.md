# Reviewer 2 W4 only

Run `sh submit_w4_ghaddar.sh` from the project root. It submits 44 jobs under
`aip-bghaddar`, each capped at five hours with one CPU, 6G RAM, and one L40S.
There are 12 First-m jobs and 32 random-draw shards. Each job belongs to one
table cell. The 44-job layout gives ConvNet 0.1% more headroom than the 42-job
estimate. Nothing is submitted while preparing these files.

Scope: ConvNetBN and ResNet20BN, GM and SAPA, dog-to-bird, budgets 0.1%, 0.2%,
and 0.5%. Every new draw uses the same ten pinned targets and six victim seeds
per target. First-m chooses the first 50/100/250 dog images in CIFAR-10 training
order. Random draws use independent CPU generators; drawing bases never changes
the target-specific crafting seed or the paired victim initialization/order.

The random row has six draws TOTAL: the original Table 1 draw plus five fresh
draws. All twelve original 60-trial grids are frozen in `sbatch/rebuttal_w4/baselines`.
Nine are saved CSVs. The three ConvNet GM grids are recovered from the original
per-victim logs in `results/raw.txt` (lines 5317--5651); their frozen CSVs include
source line numbers, and the exact source log blocks are saved beside them.
Their 60 observed outcomes reproduce 0.0%, 0.0%, and 1.7%. Between-draw sample
SD uses six draw-level ASRs (ddof=1), not 360 individual outcomes. The separately
reported single-draw Rand and BASIS reference rows are retained.

New work totals 120 First-m poison sets/720 victims plus 600 new random poison
sets/3,600 victims: 4,320 new victim trainings. All targets and settings are pinned
in `sbatch/rebuttal_w4/plan.json`. Crafting follows the paper's 250-step setup,
eight restarts, first five full-data 60-epoch crafting surrogates, epsilon
0.0313725, alpha 0.0039216, and the existing DiffAugment strategy. SAPA uses
worst-mode sigma 0.05. The unrelated six-target/750-step ConvNet GM transfer
runs are not reused as Table 1 draws. Victims use the original 50-epoch SGD
recipe: learning rate 0.1, momentum 0.9, batch 125, decay at epoch 40, no weight
decay or augmentation. No surrogates or clean controls are trained.

Existing scientific source files and historical results are unchanged. A frozen
copy of final_update/networks/utils supplies the same models, normalization, and
GM/SAPA crafting. The new worker adds explicit base selection and checkpointed
victim SGD. It checkpoints crafting every 50 steps and victims every ten epochs,
plus on cooperative stop. Model/optimizer/RNG state persist under
`rebuttal_w4_result`; completed victims and poison sets are skipped. A hard kill
can lose at most the latest checkpoint interval. Rerunning the submitter skips
completed/active tasks and resubmits unfinished tasks under Ghaddar.

Jobs pause five minutes before their limit and handle Slurm USR1. Logs print
resumed steps/epochs, new progress, and each completed result. Cell summaries
are written under `rebuttal_w4_result/summaries`; final ASR/SD appears only after
the full cell is complete. Tiny selection and exact SGD/crafting resume checks
run inside each allocation before scientific work. No Python or training runs
on the login node. W5, BASIS reruns, and TeX edits are outside this batch.
