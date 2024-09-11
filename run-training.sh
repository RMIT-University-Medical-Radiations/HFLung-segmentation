#!/bin/bash

set -e

for training_set in 160
do
    for config in 3d_fullres
    do
        for fold in {1..4}
        do
                echo training set: $training_set, config: $config, fold $fold
                nnUNetv2_train -tr nnUNetTrainerDA5_60epochs -device cuda --npz -p nnUNetResEncUNetPlans_48G $training_set $config $fold
                echo "-----------------------"
        done
    done
done
