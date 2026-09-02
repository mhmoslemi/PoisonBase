#!/usr/bin/env bash
# EDIT THESE FC CRAFTING HYPERPARAMETERS BEFORE SUBMITTING.
#
# CRAFT_STEPS: number of projected-gradient crafting iterations.
# CRAFT_ALPHA: sign-PGD step size in the [0,1] image scale (1/255 below).
# FC_RESTARTS: independent FC initializations; the lowest-loss result is kept.
# Increasing CRAFT_STEPS or FC_RESTARTS increases job time.

export CRAFT_STEPS=2500
export CRAFT_ALPHA=0.0019608
# export CRAFT_ALPHA=0.0039216
export FC_RESTARTS=1
