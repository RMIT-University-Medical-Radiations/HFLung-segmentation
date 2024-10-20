# High-Function Lung Segmentation CTVI
This is a repository for the pipeline we developed to generate quantised ventilation maps from breath-hold CT images using a 3D neural network based on nnU-Net. It accompanies our paper "CT Ventilation Images Produced by a 3D Neural Network Show Improvement Over the Jacobian and HU DIR-Based Methods to Predict Quantised Lung Function". It will generate the results and figures for the paper, but is also provided as a foundation for high-function lung segmentation work on other data sets.

These instructions assume the pipeline is executed on a remote GPU compute machine running a Linux distributuon OS. We use Ubuntu 22.04.

The base directory for processing on the remote machine:
- `/mnt/data/datasets/RNSH_HFlung`

with the following subdirectories
* `TCIA CTVI` - where the TCIA manifest was downloaded and expanded
* `pre-processed-plastimatch` - parent directory for DIR processing
* `nnU-Net-processing` - parent directory for nnU-Net training and inference

The mount points may differ in your setup, but hopefully these instructions will help. Check the scripts for directory names and output file names to make sure they suit your needs.

# Setup
**install miniconda if you don't alrady have it**  
https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html

**set up the new environment**  
```
conda create --name <name> python=3.10
conda activate <name>
```

**install the requirements**  
`pip install -r requirements.txt`

# Image pre-processing
**prepare the raw images for processing with DIR and for nnU-Net**  
`python ./repos/HFLung-segmentation/batch-preprocessing.py`

This script executes the `data pre-processing 3D plastimatch with post-DIR resampling.ipynb` notebook for each patient.

# Deformable image registration

Plastimatch is executed in a Docker container using the `pypla` image.

**download the Plastimatch image**  
https://pypi.org/project/pyplastimatch/

**run the container**  
`docker run -u $(id -u):$(id -g) --volume="/etc/group:/etc/group:ro" --volume="/etc/passwd:/etc/passwd:ro" --volume="/etc/shadow:/etc/shadow:ro" --rm -it -v /mnt/data/datasets:/datasets -v ./repos/HFLung-segmentation:/HFLung-segmentation -v ~:/daryl --entrypoint bash pypla_22.04`

**inside the container**  
*generate the vector field from DIR*  
`plastimatch register /HFLung-segmentation/register-commands.txt`

*compute the Jacobian determinant of a vector field*  
`plastimatch jacobian --input dvf.mha --output-img vf_jac.mha`

# Compare the DIR-based methods with the PET ground truth
**compare each patient separately**  
`python ./repos/HFLung-segmentation/batch-compare-unquantised-ctvi.py`

This script will execute the `compare patient unquantised CTVI with PET.ipynb` notebook for each patient.

**consolidate the results**  
run all the cells in the `compare all unquantised CTVI with PET.ipynb` notebook

# nnU-Net model training and inference
The nnU-Net processing is done inside a Docker container, because it's the easiest way to manage CUDA installs. Model training requires a CUDA-compatible GPU with 48GB of memory. It may be possible with less memory, but this has not been tested.

**install Docker if you don't already have it**  
https://docs.docker.com/engine/install/

**build the Docker image**  
`docker build . -t daryl/nnunet:0.6 -f ~/repos/HFLung-segmentation/nnunet.dockerfile`

The tag here (`daryl/nnunet:0.6`) is what I use; of course you should change it to be meaingful in your environment.

**run the container**  
`docker run -u $(id -u):$(id -g) --volume="/etc/group:/etc/group:ro" --volume="/etc/passwd:/etc/passwd:ro" --volume="/etc/shadow:/etc/shadow:ro" --rm -it --gpus '"device=3"' --name='ctvi-162-2' --ipc=host -v /mnt/data/datasets:/datasets -v ./repos/HFLung-segmentation:/HFLung-segmentation -v ~:/daryl daryl/nnunet:0.6 /bin/bash`

**inside the container…**  

*convert the pre-processed images to thge format required by nnU-Net*  
`python /HFLung-segmentation/convert-preprocessed-to-nnunet.py`

*run the nnU-Net planning*  
`. /HFLung-segmentation/run-planning.sh`

**model training**  

*batch*  
`. /HFLung-segmentation/run-training.sh`

*or individually*  
`nnUNetv2_train -tr nnUNetTrainerDA5_60epochs -device cuda --npz -p nnUNetResEncUNetPlans_48G $training_set $config $fold`

**check the log for training progress**  
`tail -50 /mnt/data/datasets/RNSH_HFlung/nnU-Net-processing/nnUNet_results/Dataset160_RNSH_HFlung/nnUNetTrainerDA5_60epochs__nnUNetResEncUNetPlans_48G__3d_fullres/fold_2/training_log_`

**copying training progress charts**  
`scp daryl@krypton.eres.rmit.edu.au:/mnt/data/datasets/RNSH_HFlung/nnU-Net-processing/nnUNet_results/Dataset162_RNSH_HFlung/nnUNetTrainerDA5_60epochs__nnUNetResEncUNetPlans_48G__3d_fullres/fold_0/progress.png ~/Downloads/progress-162-0.png`

**finding the best configuration**  
`nnUNetv2_find_best_configuration 160 -tr nnUNetTrainerDA5_60epochs -p nnUNetResEncUNetPlans_48G -c 2d 3d_fullres`

**predict the test set**  

*batch*  
`. /HFLung-segmentation/run-inference.sh`

*or individually*  
`nnUNetv2_predict -d Dataset162_RNSH_HFlung -i /datasets/RNSH_HFlung/nnU-Net-processing/nnUNet_raw/Dataset162_RNSH_HFlung/imagesTs -o /datasets/RNSH_HFlung/nnU-Net-processing/nnUNet_predictions/Dataset162_RNSH_HFlung/best -f  0 1 2 3 4 -tr nnUNetTrainerDA5_60epochs -c 3d_fullres -p nnUNetResEncUNetPlans_48G -chk checkpoint_best.pth -device cuda`

# Compare the nnU-Net results with the DIR-based methods against the PET ground truth

**compare each patient separately**  
`python ./repos/batch-compare-quantised-ctvi.py`

This script will execute the `compare patient nnunet predictions with quantised PET and CTVI-DIR.ipynb` notebook for each patient.

**consolidate the results**  
run all cells in the `compare all nnunet predictions with quantised PET and CTVI-DIR.ipynb` notebook

**download the figures to your local machine**  
`./repos/HFLung-segmentation/download-figures.sh`

# How to cite
If you find this pipeline useful, please cite our paper.