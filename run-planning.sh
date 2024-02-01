#!/bin/bash

for training_set in {150..154}
do
    for config in 2d 3d_fullres 3d_lowres 3d_cascade_fullres
    do
        for fold in {0..4}
        do
                echo training set: $training_set, config: $config, fold $fold
                nnUNetv2_train -tr nnUNetTrainerDA5_60epochs $training_set $config $fold -device cuda --npz
                echo "-----------------------"
        done
    done
done
