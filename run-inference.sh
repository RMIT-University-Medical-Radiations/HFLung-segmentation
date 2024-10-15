#!/bin/bash

set -e

config='2d'
for training_set in {160..162}
do
    echo training set: $training_set, config: $config
    nnUNetv2_predict -d Dataset$training_set\_RNSH_HFlung -i /datasets/RNSH_HFlung/nnU-Net-processing/nnUNet_raw/Dataset$training_set\_RNSH_HFlung/imagesTs -o /datasets/RNSH_HFlung/nnU-Net-processing/nnUNet_predictions/Dataset$training_set\_RNSH_HFlung/best/2d -f  0 1 2 3 4 -tr nnUNetTrainerDA5_60epochs -c $config -p nnUNetResEncUNetPlans_48G -chk checkpoint_best.pth -device cuda
    echo "-----------------------"
done

config='3d_fullres'
for training_set in {160..162}
do
    echo training set: $training_set, config: $config
    nnUNetv2_predict -d Dataset$training_set\_RNSH_HFlung -i /datasets/RNSH_HFlung/nnU-Net-processing/nnUNet_raw/Dataset$training_set\_RNSH_HFlung/imagesTs -o /datasets/RNSH_HFlung/nnU-Net-processing/nnUNet_predictions/Dataset$training_set\_RNSH_HFlung/best -f  0 1 2 3 4 -tr nnUNetTrainerDA5_60epochs -c $config -p nnUNetResEncUNetPlans_48G -chk checkpoint_best.pth -device cuda
    echo "-----------------------"
done
