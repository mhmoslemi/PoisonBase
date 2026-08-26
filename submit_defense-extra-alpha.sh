#!/bin/sh
# Submit every unfinished alpha=0.25 cell, then all alpha=0.1 cells.
# Each sbatch is one table cell. Defense cells depend on their matching
# poison-generation cell unless that alpha=0.25 attack is already complete.
set -eu

ROOT="/home/mmoslem3/scratch/attack_if"
JOB_DIR="$ROOT/sbatch/defense_extra_alpha"
mkdir -p "$ROOT/sbatch/logs"

submit_attack() {
    label="$1"
    file="$2"
    raw=$(sbatch --parsable "$JOB_DIR/$file")
    id=${raw%%;*}
    printf "Submitted %s as %s\n" "$label" "$id" >&2
    printf "%s\n" "$id"
}

submit_defense() {
    label="$1"
    dependency="$2"
    file="$3"
    if [ -n "$dependency" ]; then
        raw=$(sbatch --parsable --dependency="afterok:$dependency" "$JOB_DIR/$file")
    else
        raw=$(sbatch --parsable "$JOB_DIR/$file")
    fi
    id=${raw%%;*}
    printf "Submitted %s as %s\n" "$label" "$id"
}

printf '\nSubmitting defense-extra alpha=0.25\n'
submit_defense '0.25 config 01 EPIC' "" 'defextra_a025_001_convnet_bp_b001_epic.sh'
sleep 1
submit_defense '0.25 config 01 FRIENDS' "" 'defextra_a025_002_convnet_bp_b001_friends.sh'
sleep 1
attack_a025_02=$(submit_attack '0.25 config 02 No Defense' 'defextra_a025_003_convnet_bp_b002_nodef.sh')
sleep 1
submit_defense '0.25 config 02 EPIC' "$attack_a025_02" 'defextra_a025_004_convnet_bp_b002_epic.sh'
sleep 1
submit_defense '0.25 config 02 FRIENDS' "$attack_a025_02" 'defextra_a025_005_convnet_bp_b002_friends.sh'
sleep 1
attack_a025_03=$(submit_attack '0.25 config 03 No Defense' 'defextra_a025_006_convnet_gm_b002_nodef.sh')
sleep 1
submit_defense '0.25 config 03 EPIC' "$attack_a025_03" 'defextra_a025_007_convnet_gm_b002_epic.sh'
sleep 1
submit_defense '0.25 config 03 FRIENDS' "$attack_a025_03" 'defextra_a025_008_convnet_gm_b002_friends.sh'
sleep 1
attack_a025_05=$(submit_attack '0.25 config 05 No Defense' 'defextra_a025_009_resnet20_bp_b001_nodef.sh')
sleep 1
submit_defense '0.25 config 05 EPIC' "$attack_a025_05" 'defextra_a025_010_resnet20_bp_b001_epic.sh'
sleep 1
submit_defense '0.25 config 05 FRIENDS' "$attack_a025_05" 'defextra_a025_011_resnet20_bp_b001_friends.sh'
sleep 1
attack_a025_06=$(submit_attack '0.25 config 06 No Defense' 'defextra_a025_012_resnet20_gm_b0002_nodef.sh')
sleep 1
submit_defense '0.25 config 06 EPIC' "$attack_a025_06" 'defextra_a025_013_resnet20_gm_b0002_epic.sh'
sleep 1
submit_defense '0.25 config 06 FRIENDS' "$attack_a025_06" 'defextra_a025_014_resnet20_gm_b0002_friends.sh'
sleep 1
attack_a025_07=$(submit_attack '0.25 config 07 No Defense' 'defextra_a025_015_resnet20_gm_b001_nodef.sh')
sleep 1
submit_defense '0.25 config 07 EPIC' "$attack_a025_07" 'defextra_a025_016_resnet20_gm_b001_epic.sh'
sleep 1
submit_defense '0.25 config 07 FRIENDS' "$attack_a025_07" 'defextra_a025_017_resnet20_gm_b001_friends.sh'
sleep 1
submit_defense '0.25 config 08 FRIENDS' "" 'defextra_a025_018_resnet20_sapa_b001_friends.sh'
sleep 1
attack_a025_09=$(submit_attack '0.25 config 09 No Defense' 'defextra_a025_019_vgg13_gm_b0005_nodef.sh')
sleep 1
submit_defense '0.25 config 09 EPIC' "$attack_a025_09" 'defextra_a025_020_vgg13_gm_b0005_epic.sh'
sleep 1
submit_defense '0.25 config 09 FRIENDS' "$attack_a025_09" 'defextra_a025_021_vgg13_gm_b0005_friends.sh'
sleep 1
attack_a025_10=$(submit_attack '0.25 config 10 No Defense' 'defextra_a025_022_vgg13_gm_b001_nodef.sh')
sleep 1
submit_defense '0.25 config 10 EPIC' "$attack_a025_10" 'defextra_a025_023_vgg13_gm_b001_epic.sh'
sleep 1
submit_defense '0.25 config 10 FRIENDS' "$attack_a025_10" 'defextra_a025_024_vgg13_gm_b001_friends.sh'
sleep 1
attack_a025_12=$(submit_attack '0.25 config 12 No Defense' 'defextra_a025_025_vgg13_sapa_b001_nodef.sh')
sleep 1
submit_defense '0.25 config 12 EPIC' "$attack_a025_12" 'defextra_a025_026_vgg13_sapa_b001_epic.sh'
sleep 1
submit_defense '0.25 config 12 FRIENDS' "$attack_a025_12" 'defextra_a025_027_vgg13_sapa_b001_friends.sh'
sleep 1
printf '\nSubmitting defense-extra alpha=0.1\n'
attack_a01_01=$(submit_attack '0.1 config 01 No Defense' 'defextra_a01_028_convnet_bp_b001_nodef.sh')
sleep 1
submit_defense '0.1 config 01 EPIC' "$attack_a01_01" 'defextra_a01_029_convnet_bp_b001_epic.sh'
sleep 1
submit_defense '0.1 config 01 FRIENDS' "$attack_a01_01" 'defextra_a01_030_convnet_bp_b001_friends.sh'
sleep 1
attack_a01_02=$(submit_attack '0.1 config 02 No Defense' 'defextra_a01_031_convnet_bp_b002_nodef.sh')
sleep 1
submit_defense '0.1 config 02 EPIC' "$attack_a01_02" 'defextra_a01_032_convnet_bp_b002_epic.sh'
sleep 1
submit_defense '0.1 config 02 FRIENDS' "$attack_a01_02" 'defextra_a01_033_convnet_bp_b002_friends.sh'
sleep 1
attack_a01_03=$(submit_attack '0.1 config 03 No Defense' 'defextra_a01_034_convnet_gm_b002_nodef.sh')
sleep 1
submit_defense '0.1 config 03 EPIC' "$attack_a01_03" 'defextra_a01_035_convnet_gm_b002_epic.sh'
sleep 1
submit_defense '0.1 config 03 FRIENDS' "$attack_a01_03" 'defextra_a01_036_convnet_gm_b002_friends.sh'
sleep 1
attack_a01_04=$(submit_attack '0.1 config 04 No Defense' 'defextra_a01_037_convnet_sapa_b002_nodef.sh')
sleep 1
submit_defense '0.1 config 04 EPIC' "$attack_a01_04" 'defextra_a01_038_convnet_sapa_b002_epic.sh'
sleep 1
submit_defense '0.1 config 04 FRIENDS' "$attack_a01_04" 'defextra_a01_039_convnet_sapa_b002_friends.sh'
sleep 1
attack_a01_05=$(submit_attack '0.1 config 05 No Defense' 'defextra_a01_040_resnet20_bp_b001_nodef.sh')
sleep 1
submit_defense '0.1 config 05 EPIC' "$attack_a01_05" 'defextra_a01_041_resnet20_bp_b001_epic.sh'
sleep 1
submit_defense '0.1 config 05 FRIENDS' "$attack_a01_05" 'defextra_a01_042_resnet20_bp_b001_friends.sh'
sleep 1
attack_a01_06=$(submit_attack '0.1 config 06 No Defense' 'defextra_a01_043_resnet20_gm_b0002_nodef.sh')
sleep 1
submit_defense '0.1 config 06 EPIC' "$attack_a01_06" 'defextra_a01_044_resnet20_gm_b0002_epic.sh'
sleep 1
submit_defense '0.1 config 06 FRIENDS' "$attack_a01_06" 'defextra_a01_045_resnet20_gm_b0002_friends.sh'
sleep 1
attack_a01_07=$(submit_attack '0.1 config 07 No Defense' 'defextra_a01_046_resnet20_gm_b001_nodef.sh')
sleep 1
submit_defense '0.1 config 07 EPIC' "$attack_a01_07" 'defextra_a01_047_resnet20_gm_b001_epic.sh'
sleep 1
submit_defense '0.1 config 07 FRIENDS' "$attack_a01_07" 'defextra_a01_048_resnet20_gm_b001_friends.sh'
sleep 1
attack_a01_08=$(submit_attack '0.1 config 08 No Defense' 'defextra_a01_049_resnet20_sapa_b001_nodef.sh')
sleep 1
submit_defense '0.1 config 08 EPIC' "$attack_a01_08" 'defextra_a01_050_resnet20_sapa_b001_epic.sh'
sleep 1
submit_defense '0.1 config 08 FRIENDS' "$attack_a01_08" 'defextra_a01_051_resnet20_sapa_b001_friends.sh'
sleep 1
attack_a01_09=$(submit_attack '0.1 config 09 No Defense' 'defextra_a01_052_vgg13_gm_b0005_nodef.sh')
sleep 1
submit_defense '0.1 config 09 EPIC' "$attack_a01_09" 'defextra_a01_053_vgg13_gm_b0005_epic.sh'
sleep 1
submit_defense '0.1 config 09 FRIENDS' "$attack_a01_09" 'defextra_a01_054_vgg13_gm_b0005_friends.sh'
sleep 1
attack_a01_10=$(submit_attack '0.1 config 10 No Defense' 'defextra_a01_055_vgg13_gm_b001_nodef.sh')
sleep 1
submit_defense '0.1 config 10 EPIC' "$attack_a01_10" 'defextra_a01_056_vgg13_gm_b001_epic.sh'
sleep 1
submit_defense '0.1 config 10 FRIENDS' "$attack_a01_10" 'defextra_a01_057_vgg13_gm_b001_friends.sh'
sleep 1
attack_a01_11=$(submit_attack '0.1 config 11 No Defense' 'defextra_a01_058_vgg13_sapa_b0005_nodef.sh')
sleep 1
submit_defense '0.1 config 11 EPIC' "$attack_a01_11" 'defextra_a01_059_vgg13_sapa_b0005_epic.sh'
sleep 1
submit_defense '0.1 config 11 FRIENDS' "$attack_a01_11" 'defextra_a01_060_vgg13_sapa_b0005_friends.sh'
sleep 1
attack_a01_12=$(submit_attack '0.1 config 12 No Defense' 'defextra_a01_061_vgg13_sapa_b001_nodef.sh')
sleep 1
submit_defense '0.1 config 12 EPIC' "$attack_a01_12" 'defextra_a01_062_vgg13_sapa_b001_epic.sh'
sleep 1
submit_defense '0.1 config 12 FRIENDS' "$attack_a01_12" 'defextra_a01_063_vgg13_sapa_b001_friends.sh'
sleep 1

