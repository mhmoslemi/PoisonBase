
DATASET=CIFAR10
DATA_PATH=./data
SEED=42

# MODELS=(ConvNetBN VGG13BN ResNet20BN)
# ATTACKS=(FC GRAD)
# CLASS_PAIRS=(dog-bird frog-airplane)


EPSILON=0.0313          # 8/255

# Target easiness on a single 0..1 spectrum, under the clean reference ensemble.
#
# Candidate pool = EVERY test image whose true label is not the adversarial class (POOL=all;
# POOL=pair restricts it to the pair's target class, the old behaviour). Images a clean
# model ALREADY classifies as the adversarial class are excluded outright -- nothing left
# to flip, they'd be free wins, not attacks. The rest are ranked by
#   easiness = ensemble p_adv = softmax(mean logits)[y_adv]
# i.e. how close the clean model already is to calling it the adversarial class.
# EASE picks a POSITION in that ranking: 1 = the EASIEST usable targets (adv class nearly
# winning already), 0 = the hardest (adv class nowhere), 0.5 = the middle band.
# NOTE the scale is REVERSED versus the old DIFF knob (which had 0 = easiest).
#
# EVERY clean model in the run votes on that exclusion -- ELIG_ENS reference models, the
# surrogates, and the clean victim baselines -- and one vote for the adversarial class
# vetoes the candidate. MIN_MARGIN then adds the safety strip: a candidate is rejected
# unless every one of those models keeps the adversarial class at least MIN_MARGIN behind
# its own top class. Without it, a candidate that is merely SECOND by 0.001 survives, and
# a victim trained on another seed flips it with no help from the poisons -- a free win.
# Raise MIN_MARGIN if free wins still show up in the audit line; lower it if the easy end
# of the pool empties out. Note the veto makes the target set depend on --model.
REF_MODEL=ResNet20BN
EASE=1.0                # 1 = EASIEST block. Do NOT carry the old DIFF=1e-7 over: on this
                        # scale that is the HARDEST block (p_adv ~ 0) and ASR will be 0.
POOL=all
ELIG_ENS=3
MIN_MARGIN=0.05

# Fraction of the TRAINING set to use: 1.0 = full, 0.5 = class-balanced half, etc.
# Test set is always kept full so CTA stays comparable. --budget is a fraction of the
# SUBSAMPLED train set, so the poison ratio the victim sees is unchanged by FRAC.
FRAC=1

# 'ours' base-selection score s_i = combine(B_i, R_i)   [ignored when --base random]
#   REL     : cosine | l2      -- relevance term R (l2 = raw distance, magnitude-sensitive)
#   COMBINE : product | sum | rank_product
#             product      = B*R raw (B's much wider spread dominates, coef unused)
#             sum          = B + coef*R raw (paper Eq. 8)
#             rank_product = rank(B)*rank(R)^coef, percentile-normalized -> scale-free,
#                            and the only mode where coef really balances the two terms
REL=l2
COMBINE=sum
source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if
# 0.05 0.02 0.01 0.005 0.002 0.001

# for bug in 0.05 0.02 0.01 0.005 0.002 0.001; do
for bug in 0.05 0.02 0.01; do
# for bug in 0.005 0.002 0.001; do

python gemini.py --model "$REF_MODEL" --attack FC --base random \
    --class_pair dog-bird --budget "$bug" \
    --dataset CIFAR10 --seed 42 --num_victims 6 --num_baselines 2 --num_targets 10 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 --victim_decay 40 --no_augmentation \
    --num_surrogate 5 --coef 10.0 --epsilon 0.06274 --craft_steps 250 --craft_lr 0.01 \
    --craft_ensemble 5 --data_frac "$FRAC" \
    --rel_metric "$REL" --score_combine "$COMBINE" \
    --ref_model "$REF_MODEL" --target_easiness "$EASE" --target_pool "$POOL" \
    --target_eligibility_ensemble "$ELIG_ENS" --target_min_margin "$MIN_MARGIN" --no_resume \
    --cache_dir ./cache --data_path /home/mmoslem3/scratch/data --out_dir ours_result --recompute_deltas \
    --fast_gradmatch --restarts 8

done
# resnet fc
# diff 0.2 bug 0.005 -> 11%



# for CLASS_PAIR in "${CLASS_PAIRS[@]}"; do
#   for MODEL in "${MODELS[@]}"; do
#     for ATTACK in "${ATTACKS[@]}"; do
#       for BASE in "${BASES[@]}"; do

#         echo "=== ${CLASS_PAIR} | ${MODEL} | ${ATTACK} | base=${BASE} ==="

#         python main.py \
#           --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
#           --model "$MODEL" --attack "$ATTACK" --base "$BASE" --class_pair "$CLASS_PAIR" \
#           --num_surrogate "$NUM_SURROGATE" --coef "$COEF" \
#           --num_poisons "$NUM_POISONS" --epsilon "$EPSILON" \
#           --craft_steps "$CRAFT_STEPS" --craft_lr "$CRAFT_LR" \
#           --num_targets "$NUM_TARGETS" --num_victims "$NUM_VICTIMS" \
#           --victim_epochs "$VICTIM_EPOCHS" --victim_lr "$VICTIM_LR" --victim_bs "$VICTIM_BS" \
#           --victim_decay $VICTIM_DECAY --no_augmentation \
#           --ref_model "$REF_MODEL" \
#           --target_margin_low "$TARGET_MARGIN_LOW" --target_margin_high "$TARGET_MARGIN_HIGH" \
#           --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR"

#       done
#     done
#   done
# done


