#!/bin/bash

source_host='daryl@krypton.eres.rmit.edu.au'
source_base='/mnt/data/datasets/RNSH_HFlung/nnU-Net-processing'
destination_base=~'/Downloads/HFlung_figures'
source_figures='~/figures'

for training_set in {160..162}
do
    echo training set: $training_set
    destination=$destination_base/dataset_$training_set

    if [ ! -d $destination ]; then
        mkdir -p $destination
    fi

    echo copying predicted images
    scp $source_host:$source_base/nnUNet_predictions/Dataset$training_set\_RNSH_HFlung/best/RNSH_HFlung_\*.nii.gz $destination/predictions

    echo copying labels
    scp $source_host:$source_base/nnUNet_raw/Dataset$training_set\_RNSH_HFlung/labelsTs/\* $destination/labels

    echo copying test images
    scp $source_host:$source_base/nnUNet_raw/Dataset$training_set\_RNSH_HFlung/imagesTs/\* $destination/test_images

    echo "-----------------------"
done

echo copying figures
scp $source_host:$source_figures/\* $destination_base/figures
