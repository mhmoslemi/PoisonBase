# appendix.tex run scripts

Matched to the revised appendix.tex. Nothing in it can be filled from existing logs.

## Protocol (from the appendix preamble)

ConvNetBN unless stated, **5 targets** sampled uniformly from the target class with a
fixed seed (**not** by difficulty, which is how every main-table run picked targets),
**5 victims** (seeds 0–4), selector defaults λ=1, α=2, K=20. `pin_targets.py` freezes
the targets before anything runs and prints the row for
`tab:additional-target-indices`.

## Pair notation — read this before launching

appendix.tex writes pairs as **target → adversarial**. `final_update.py` takes
`--class_pair "<adv>-<target>"`, so every pair below is the paper's arrow reversed:

| paper | script |
|---|---|
| cat → dog | `dog-cat` |
| deer → horse | `horse-deer` |
| automobile → truck | `truck-automobile` |
| bird → airplane | `airplane-bird` |
| ship → frog | `frog-ship` |
| truck → cat | `cat-truck` |
| **bird → dog** (the main pair) | `dog-bird` |
| **airplane → frog** (the main pair) | `frog-airplane` |

appendix.tex now states this convention in a Notation paragraph and names the two main
pairs bird→dog and airplane→frog, which is what those runs actually did. The appendix
therefore attacks the same instances as the main sweep. Run dirs do not collide with
the main sweep's, because these use `--target_select random` and carry no `_tgt<N>`
suffix.

## Scripts

Everything is sharded so no single invocation exceeds ~10 h on an L40S. The `ap*-x.sh`
files are thin wrappers that set one environment variable and exec the parent script;
run the wrappers, not the parents.

| shard | table | ~L40S | depends on |
|---|---|---|---|
| `ap6-cost.sh` | computational-cost | 0.2 h | — (**run first**, no attack at all) |
| `ap8-redundancy.sh` | base-redundancy | 4.8 h | — |
| `ap1-a.sh` | broad-cifar (cat→dog, deer→horse) | 6.9 h | — |
| `ap1-b.sh` | broad-cifar (automobile→truck, bird→airplane) | 6.9 h | — |
| `ap1-c.sh` | broad-cifar (ship→frog, truck→cat) | 6.9 h | — |
| `ap1-d.sh` | broad-cifar (bird→dog reference row) | 3.4 h | — |
| `ap2-a.sh` | cross-dataset (CIFAR-10 rows) | 2.7 h | — |
| `ap2-pools.sh` | — (shared TinyImageNet model pools) | 4.3 h | **TinyImageNet must be prepared** |
| `ap2-b.sh` | cross-dataset (TinyImageNet 1–2) | 6.8 h | **ap2-pools** |
| `ap2-c.sh` | cross-dataset (TinyImageNet 3–4) | 6.8 h | **ap2-pools** |
| `ap2-d.sh` | cross-dataset (TinyImageNet 5) | 3.4 h | **ap2-pools** |
| `ap3-a.sh` | matched-architecture (ResNet20BN) | 6.2 h | — |
| `ap3-b.sh` | matched-architecture (ConvNetBN, VGG13BN) | 8.1 h | — |
| `ap4-a.sh` | augmentation-aware (8 crafts) | 5.5 h | — |
| `ap4-b.sh` | augmentation-aware (Crop+Flip replays) | 2.1 h | **ap4-a** |
| `ap4-c.sh` | augmentation-aware (RandAugment, GM) | 5.5 h | **ap4-a** |
| `ap4-d.sh` | augmentation-aware (RandAugment, SAPA) | 5.5 h | **ap4-a** |
| `ap5-a.sh` | utility-defense (calibration sweep) | 2.2 h | — |
| `ap5-b.sh` | utility-defense (EPIC) | 3.8 h | **ap5-a**, then `EPIC_KEEP=<v>` |
| `ap5-c.sh` | utility-defense (FRIENDS) | 3.4 h | **ap5-a**, then `NOISE_EPS_SET=<v>` |

Everything without a dependency can run in parallel. Total ~90 h of GPU time.

`pin_targets.py` freezes
the targets before anything runs and prints the row for
`tab:additional-target-indices`.

## Pair notation — read this before launching

appendix.tex writes pairs as **target → adversarial**. `final_update.py` takes
`--class_pair "<adv>-<target>"`, so every pair below is the paper's arrow reversed:

| paper | script |
|---|---|
| cat → dog | `dog-cat` |
| deer → horse | `horse-deer` |
| automobile → truck | `truck-automobile` |
| bird → airplane | `airplane-bird` |
| ship → frog | `frog-ship` |
| truck → cat | `cat-truck` |
| **bird → dog** (the main pair) | `dog-bird` |
| **airplane → frog** (the main pair) | `frog-airplane` |

appendix.tex now states this convention in a Notation paragraph and names the two main
pairs bird→dog and airplane→frog, which is what those runs actually did. The appendix
therefore attacks the same instances as the main sweep. Run dirs do not collide with
the main sweep's, because these use `--target_select random` and carry no `_tgt<N>`
suffix.

## Scripts

| script | table | runs | notes |
|---|---|---|---|
| `ap6-cost.sh` | tab:computational-cost | 0 | profiling only — no attack, no victims. **Run first.** |
| `ap8-redundancy.sh` | tab:base-redundancy | 7 | Random, Top-1/2/5/10, Greedy, DPP |
| `ap3-matched.sh` | tab:matched-architecture | 12 | one target set, 3 architectures |
| `ap4-augaware.sh` | tab:augmentation-aware | 8 crafts + replay | 12 of 16 cells, see gap 1 |
| `ap5-utilmatch.sh` | tab:utility-defense | calibration + 16 | two-stage, see gap 2 |
| `ap1-broad.sh` | tab:broad-cifar | 42 | 6 new pairs + the main pair × 3 attacks × 2 selections |
| `ap2-crossdata.sh` | tab:cross-dataset | 24 | 4 CIFAR-10 at 5e-3 + 20 TinyImageNet at **1e-3** |

`pin_targets.py` freezes and reports target sets; `profile_selection.py` times the
selector alone; `end_to_end_cost.py` builds the lower half of the cost table from
craft times already in the logs.

## Two gaps that need a decision, not a script

1. **Matched / RandAugment** — resolved in the draft rather than in code. Matched
   optimization needs a differentiable augmentation; RandAugment's discrete photometric
   ops have no gradient, so a matched attacker is undefined for it under this threat
   model. appendix.tex now says so and the table marks those entries n/a. `ap4` runs
   the other 12 cells.
2. **Defense strength** — "within two percentage points" is a judgement call, so `ap5`
   sweeps strengths on clean data, prints each one's clean CTA, and waits for you to
   pass the choice back as `EPIC_KEEP` / `NOISE_EPS_SET`.

Two smaller mismatches with the prose, both cosmetic:

- The appendix says surrogate seeds are {0,…,19}; the code uses `seed + 1000 + i`.
  Changing it would invalidate the whole cached ensemble, so the prose is the cheaper fix.
- `TINY_PAIRS` in `ap2` is a placeholder list of five WordNet ids. The appendix says they
  are drawn uniformly with a fixed seed — commit to that list and edit the script.

## Code changes these needed

- `final_update.py`: `ResNet18`/`ResNet18BN` in `SUPPORTED_MODELS`; `--base_topr`
  (top-r bases replicated to the poison budget, each copy still optimized
  independently, run dirs suffixed `_top<r>`).
- `defense.sh`: `EPIC_SUBSET`, `NOISE_EPS`, `NUM_TARGETS` knobs. Empty keeps
  `defense.py`'s defaults, so existing callers are unaffected.

## Fixes applied after the first review of appendix.tex

- **TinyImageNet budget 5e-3 → 1e-3.** At 5e-3 the attack needs 500 poisons from a
  500-image class: the pool is exhausted and Random, Greedy and DPP return the identical
  set, so the experiment could not show anything. At 1e-3 it selects 100 of 500.
- **`--base_topr` redesigned.** The draft asked for the top r bases replicated to fill
  the budget. That is not expressible here: a poison replaces a specific training
  example, so m copies of one index collapse to a single poisoned image and the budget
  silently drops from m to r (verified). It now concentrates the budget in r
  *neighbourhoods* — r best-scoring seeds plus their nearest neighbours in surrogate
  representation — giving m distinct poisons for every r. appendix.tex describes this.
- **Pair direction** aligned with the main sweep, and a Notation paragraph added.
- **Augmentation claim corrected**: the main augmentation table's poisons were crafted
  *with* the default differentiable augmentation, not without it.
- **Seed wording**, target-eligibility filter, the non-determinism of victim training,
  the reduced 5×5 protocol, and the attack-relative nature of the overhead column are
  now stated in the draft.

## TinyImageNet has to be prepared once

`utils.get_dataset` loads `$DATA_PATH/tinyimagenet.pt`, and nothing in this repository
ever created it — only CIFAR-10 is on disk. Before any `ap2-b/c/d` shard:

    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip        # login node: compute nodes have no internet
    python appendix/prep_tinyimagenet.py --src tiny-imagenet-200.zip

It writes uint8 tensors (~1.2 GB); `get_dataset` does the scaling and normalisation.
Class names are WordNet ids, which is what `--class_pair` takes
(`n01443537-n01629819` = adversarial-target).

Two things to watch on the TinyImageNet shards:

- **CPU RAM.** `get_dataset` converts the whole train split to float32 — about 4.9 GB,
  on top of the 1.2 GB uint8 source. The `--mem=7G` allocations used so far are likely
  to OOM; ask for 32 G.
- **GPU memory.** `build_context` stacks the training set on the device, so ~4.9 GB of
  the GPU is gone before any model loads. Fine on an L40S, tight on smaller cards.

`ap2-a.sh` (the CIFAR-10 reference rows) needs none of this and can run now.
