#!/bin/bash

readonly GPU_MEMORY=48

for training_set in {160..164}
do
    echo training set: $training_set
    nnUNetv2_plan_and_preprocess -d $training_set --verify_dataset_integrity -gpu_memory_target $GPU_MEMORY
    echo "-----------------------"
done
