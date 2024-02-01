#!/bin/bash

for training_set in {150..154}
do
    nnUNetv2_plan_and_preprocess -d $training_set --verify_dataset_integrity -gpu_memory_target 48
done
