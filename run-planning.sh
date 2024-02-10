#!/bin/bash

readonly GPU_MEMORY=48

for training_set in {160..164}
do
    nnUNetv2_plan_and_preprocess -d $training_set --verify_dataset_integrity -gpu_memory_target $GPU_MEMORY
done
