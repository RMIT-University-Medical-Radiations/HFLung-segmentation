#!/bin/bash

readonly CONFIGS_TO_CONSIDER="2d 3d_fullres 3d_lowres"

while getopts t: flag
do
    case "${flag}" in
        t) training_set=${OPTARG};;
    esac
done

nnUNetv2_find_best_configuration $training_set -tr nnUNetTrainerDA5_60epochs -c $CONFIGS_TO_CONSIDER
