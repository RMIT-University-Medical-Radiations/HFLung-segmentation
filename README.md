# Predicting Quantised Lung Function From BHCT With a 3D Neural Network
This is a repository for the pipeline we developed to generate quantised ventilation maps from breath-hold CT images using a 3D neural network based on nnU-Net. It accompanies our paper "CT Ventilation Images Produced by a 3D Neural Network Show Improvement Over the Jacobian and HU DIR-Based Methods to Predict Quantised Lung Function". It will generate the results and figures for the paper, but is also provided as a foundation for high-function lung segmentation work on other data sets.

![CTVI workflow diagram.](/documentation/workflow.png)

These instructions assume the pipeline is executed on a remote GPU compute machine running a Linux distributuon OS. We use Ubuntu 22.04.

A base directory is required for processing (i.e. raw data, interim results, outputs). In the code this location is:
- `/mnt/data/datasets/RNSH_HFlung`

You may of course change references to this directory and point them to a location that suits your environment.

The base directory must have the following subdirectories:
* `TCIA CTVI` - where the TCIA manifest was downloaded and expanded
* `pre-processed-plastimatch` - parent directory for DIR processing
* `nnU-Net-processing` - parent directory for nnU-Net training and inference

The repository should be cloned to your home directory on the remote machine:  
`git clone git@github.com:RMIT-University-Medical-Radiations/HFLung-segmentation.git`

The Jupyter notebooks will generate the figures in a directory called `figures` which will be created in your home directory.

The mount points referenced throughout may differ in your setup, but hopefully these instructions will help. Check the scripts for directory names and output file names to make sure they suit your needs. They are defined at the start of each script or notebook.

# Environment setup
**install miniconda if you don't already have it**  
https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html

**set up the new environment**  
```
conda create --name <name> python=3.10
conda activate <name>
```

**install the requirements**  
`pip install -r requirements.txt`

# Download the RNSH dataset from TCIA

**install the NBIA data retriever**
```
wget https://cbiit-download.nci.nih.gov/nbia/releases/ForTCIA/NBIADataRetriever_4.4.1/nbia-data-retriever-4.4.1.deb
sudo -S dpkg -i nbia-data-retriever-4.4.1.deb
```

**download the data**
- go to https://nbia.cancerimagingarchive.net/nbia-search/
- enter the collection ID: “CT-vs-PET-Ventilation-Imaging” to filter, and press the cart button
- download the manifest file
- open it with the NBIA data retriever
- use descriptive directory names

# Image pre-processing
**prepare the raw images for processing with DIR and for nnU-Net**  
`python ./repos/HFLung-segmentation/batch-preprocessing.py`

This script executes the `data pre-processing 3D plastimatch with post-DIR resampling.ipynb` notebook for each patient.

# Deformable Image Registration (DIR) processing

We use Plastimatch for the DIR-based methods, which is executed in a Docker container using the `pypla` image.

**download the Plastimatch image**  
https://pypi.org/project/pyplastimatch/

**run the container**  
`docker run -u $(id -u):$(id -g) --volume="/etc/group:/etc/group:ro" --volume="/etc/passwd:/etc/passwd:ro" --volume="/etc/shadow:/etc/shadow:ro" --rm -it -v /mnt/data/datasets:/datasets -v ./repos/HFLung-segmentation:/HFLung-segmentation -v ~:/daryl --entrypoint bash pypla_22.04`

**inside the container...**  
*generate the vector field from DIR*  
`plastimatch register /HFLung-segmentation/register-commands.txt`

*compute the Jacobian determinant of a vector field*  
`plastimatch jacobian --input dvf.mha --output-img vf_jac.mha`

# Compare the DIR-based CTVIs with the PET ground truth
**compare each patient separately**  
`python ./repos/HFLung-segmentation/batch-compare-unquantised-ctvi.py`

This script will execute the `compare patient unquantised CTVI with PET.ipynb` notebook for each patient.

**consolidate the results**  
run all the cells in the `compare all unquantised CTVI with PET.ipynb` notebook

# nnU-Net model training and inference
The nnU-Net processing is done inside a Docker container because it's the easiest way to manage CUDA installation and dependencies. Model training requires a CUDA-compatible GPU with 48GB of memory. It is possible to train with less GPU memory, but results may vary because patch sizes will be different.

**install Docker if you don't already have it**  
https://docs.docker.com/engine/install/

**build the Docker image**  
`docker build . -t daryl/nnunet:0.6 -f ~/repos/HFLung-segmentation/nnunet.dockerfile`

The tag here (`daryl/nnunet:0.6`) is what I use; of course you should change it to be meaningful in your environment.

**run the container**  
`docker run -u $(id -u):$(id -g) --volume="/etc/group:/etc/group:ro" --volume="/etc/passwd:/etc/passwd:ro" --volume="/etc/shadow:/etc/shadow:ro" --rm -it --gpus '"device=3"' --name='ctvi-162-2' --ipc=host -v /mnt/data/datasets:/datasets -v ./repos/HFLung-segmentation:/HFLung-segmentation -v ~:/daryl daryl/nnunet:0.6 /bin/bash`

**inside the container...**  

*convert the pre-processed images to the format required by nnU-Net*  
`python /HFLung-segmentation/convert-preprocessed-to-nnunet.py`

*run the nnU-Net planning*  
`. /HFLung-segmentation/run-planning.sh`

## Model training

### Using pre-trained weights

If you don't want to train the models yourself, the pre-trained weights are _here_. It's about 90 GB. To use them in the inference steps, download and reference them where the inference steps mention the `results` directory.

### Training the models from scratch

These steps will perform five-fold cross validation on the three training sets for two model configurations (3D and 2D) on one GPU. That's 30 models in total, so be prepared for the whole process to take several weeks. There are no dependencies between training sets, folds or configurations, so if you have more than one GPU, a fold can be trained on a GPU in parallel up to the number of GPUs you have available.

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

## Model inference on the test set

*batch*  
`. /HFLung-segmentation/run-inference.sh`

*or individually*  
`nnUNetv2_predict -d Dataset162_RNSH_HFlung -i /datasets/RNSH_HFlung/nnU-Net-processing/nnUNet_raw/Dataset162_RNSH_HFlung/imagesTs -o /datasets/RNSH_HFlung/nnU-Net-processing/nnUNet_predictions/Dataset162_RNSH_HFlung/best -f  0 1 2 3 4 -tr nnUNetTrainerDA5_60epochs -c 3d_fullres -p nnUNetResEncUNetPlans_48G -chk checkpoint_best.pth -device cuda`

# Compare the nnU-Net results and the DIR-based methods with the ground truth

**compare each patient**  
`python ./repos/batch-compare-quantised-ctvi.py`

This script will execute the `compare patient nnunet predictions with quantised PET and CTVI-DIR.ipynb` notebook for each patient.

**consolidate the results**  
run all cells in the `compare all nnunet predictions with quantised PET and CTVI-DIR.ipynb` notebook

**download the figures to your local machine**  
`./repos/HFLung-segmentation/download-figures.sh`

# How to cite
If you find this repository useful, please cite our paper.

Wilding-McBride D, Lim J, Byrne H, O'Brien R. CT ventilation images produced by a 3D neural network show improvement over the Jacobian and HU DIR-based methods to predict quantized lung function. Med Phys. 2024; 1-10. https://doi.org/10.1002/mp.17532

