#!/bin/bash

set -e

for training_set in {160..162}
do
    for config in 3d_fullres
    do
        echo training set: $training_set, config: $config, fold $fold
        nnUNetv2_predict -d Dataset$training_set\_RNSH_HFlung -i /datasets/RNSH_HFlung/nnU-Net-processing/nnUNet_raw/Dataset$training_set\_RNSH_HFlung/imagesTs -o /datasets/RNSH_HFlung/nnU-Net-processing/nnUNet_predictions/Dataset$training_set\_RNSH_HFlung/best -f  0 1 2 3 4 -tr nnUNetTrainerDA5_60epochs -c 3d_fullres -p nnUNetResEncUNetPlans_48G -chk checkpoint_best.pth -device cuda
        echo "-----------------------"
    done
done
