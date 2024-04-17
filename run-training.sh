#!/bin/bash

for training_set in {160..162}
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
