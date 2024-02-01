#!/bin/bash

readonly CONFIGS_TO_CONSIDER="2d 3d_fullres 3d_lowres 3d_cascade_fullres"

for training_set in {150..154}
do
    nnUNetv2_find_best_configuration $training_set -tr nnUNetTrainerDA5_60epochs -c $CONFIGS_TO_CONSIDER
done
