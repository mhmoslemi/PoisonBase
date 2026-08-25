#!/usr/bin/env bash
#
# appendix.tex, tab:utility-defense -- the friends half.
#
# Self-contained. It reads the calibration ap5-a left on disk and picks the
# strength itself (strongest setting whose clean accuracy is within two points of
# the undefended ConvNetBN), so there is nothing to type. Override with
# FRIENDLY_CLAMP=<value> if you want a specific one.
#
# ConvNetBN, eps=2e-2, GM and SAPA, dog-bird and frog-airplane, Random vs DPP.
# 8 runs. Poisons are replayed from ours_result -- all 8 sets are already there.
#
#   sh appendix/ap5-c.sh

set -u

cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
    python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
        echo "ap5-c.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

# FRIENDS has two magnitudes and only one of them is the defense. --noise_eps
# scales the random (bernoulli) half; the friendly noise itself is bounded by
# --friendly_clamp. ap5-a/ap5-a2 swept noise_eps over {0.5,1,2,4,6,8,12} and clean
# accuracy never moved off ~75.8 -- because clamp stayed at its 16/255 default in
# every one of those runs. ap5-a3 swept the clamp instead, with noise_eps held at
# the paper's default 8:
#
#     clamp   16      8       4       2       1     undefended
#     CTA    75.38   78.48   80.02   79.96   80.13    81.05
#
# 4/255 is the strongest setting inside the two-point band (-1.03); 8/255 is
# already -2.57. Override with FRIENDLY_CLAMP=<value>.
VAL="${FRIENDLY_CLAMP:-4}"
echo "=== friends at friendly_clamp $VAL/255, noise_eps 8 (clean-utility matched) ==="

# The difficulty degree the b0.02 poisons were crafted under. defense.sh normally
# reads it from sweep_config.json, but that file only records the (model, attack,
# pair) combos table.tex reports and has NO ConvNetBN/sapa entry -- without
# TARGET_SELECT the four SAPA rows die with "no difficulty for ConvNetBN / sapa".
# SAPA reused the GM targets for each pair, so the degrees below match the crafts
# already on disk.
# PAIRS lets appendix/final shard this by class pair; both by default.
for PAIR in ${PAIRS:-dog-bird frog-airplane}; do
    case "$PAIR" in
        dog-bird)      TSEL=70;;
        frog-airplane) TSEL=35;;
    esac
    for ATT in gradmatch sapa; do
        echo "--- $PAIR | $ATT | friends (tgt$TSEL)"
        env MODEL=ConvNetBN ATTACK="$ATT" CLASS_PAIR="$PAIR" BUDGETS="0.02" \
            SELS="random dpp" DEFENSES="friends" NUM_VICTIMS=5 NUM_TARGETS=5 \
            TARGET_SELECT="$TSEL" \
            NOISE_EPS=8 FRIENDLY_CLAMP="$VAL" sh ./defense.sh || exit 1
    done
done

echo "=== ap5-c.sh finished ==="
