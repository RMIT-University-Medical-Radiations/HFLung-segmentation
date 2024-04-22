#!/bin/bash

set -e

for training_set in 161
do
    for config in 3d_fullres
    do
        for fold in {1..4}
        do
                echo training set: $training_set, config: $config, fold $fold
                nnUNetv2_train -tr nnUNetTrainerDA5_60epochs $training_set $config $fold -device cuda --npz
                echo "-----------------------"
        done
    done
done

for training_set in 162
do
    for config in 3d_fullres
    do
        for fold in {0..4}
        do
                echo training set: $training_set, config: $config, fold $fold
                nnUNetv2_train -tr nnUNetTrainerDA5_60epochs $training_set $config $fold -device cuda --npz
                echo "-----------------------"
        done
    done
done
