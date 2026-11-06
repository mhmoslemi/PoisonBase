# Run the same reviewer sweep with Wang

On Killarney, use the same project and environment as the older submissions:
`/home/mmoslem3/scratch/attack_if` and `/home/mmoslem3/ENV/bin/activate`.
Copy `reviewer_120_wang_scripts.tar` into that project, then run:

```sh
cd /home/mmoslem3/scratch/attack_if
tar -xf reviewer_120_wang_scripts.tar
sh submit_reviewer_120_boyuwang.sh
```

The launcher also works from the Vulcan project copy when a second Wang sweep
is wanted. It submits 120 configuration arrays plus preparation, clean controls,
and aggregation: 143 submissions / 855 tasks, all on `aip-boyuwang`, each with
1 CPU, 6 GB RAM and 1 L40S. It preserves the original manifest, targets, seeds,
algorithms, array sizes, dependencies and walltimes. Running it again skips
already-recorded submissions; failed jobs are not automatically resubmitted.

Results are in `reviewer_20261106_wang/reviewer_20261106_result/`; job IDs are in
`reviewer_20261106_wang/sbatch/submissions.tsv` and logs in that directory's
`logs/` subdirectory. Existing Lu results and job IDs are not reused. The launcher
checks the current Slurm cluster, Wang association, L40S inventory, exact module
versions, source snapshot and full-data surrogate hashes before submitting.

The small scripts archive does not include datasets, Python environments, or
model checkpoints. Existing inputs required on Killarney:

- CIFAR-10's extracted `cifar-10-batches-py` directory and
  `tiny-imagenet-200.zip` under the project's `data/` or `~/scratch/data/`.
- The identical `cache/surrogates/ConvNetBN_60ep_lr0.1_bs128_seed42/net_0.pt`
  through `net_19.pt`, and the corresponding `ResNet20BN` pool. The 40 exact
  file hashes are recorded in `validated_surrogates.sha256`; copy these pools
  from Vulcan if Killarney's copies are missing or differ (about 93 MB total).
- The existing `~/ENV` with the experiment dependencies; `ENV_ACTIVATE` and
  `DATA_ROOT` can override these locations as in the older submission scripts.

The runtime receipt records the actual passed Vulcan validation; it is not a
claim that Killarney GPU execution was tested. The shell-only Slurm simulation
passed on Vulcan. Remote inspection could not authenticate to Killarney from
this session. This launcher does not transfer files or submit over SSH.
