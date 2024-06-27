#!/bin/bash

set -e

for training_set in {160..162}
do
    echo training set: $training_set
    nnUNetv2_find_best_configuration $training_set -tr nnUNetTrainerDA5_60epochs -p nnUNetResEncUNetPlans_48G
    echo "-----------------------"
done
