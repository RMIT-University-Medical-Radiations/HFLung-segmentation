# this is based on https://hub.docker.com/r/nbutter/nnunet

# pull base image
FROM nvidia/cuda:12.2.0-base-ubuntu22.04
MAINTAINER Daryl Wilding-McBride

# install Ubuntu libraries and packages
RUN apt-get update -y && \
apt-get install git curl -y && \
rm -rf /var/lib/apt/lists/*

# set some environemnt variables
ENV PATH="/home/miniconda3/bin:${PATH}"
ARG PATH="/home/miniconda3/bin:${PATH}"

WORKDIR /home

# install Python
RUN curl -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/miniconda3

# update pip
RUN conda install pip
RUN pip install --upgrade pip

# install nnU-Net
RUN git clone https://github.com/MIC-DKFZ/nnUNet.git
WORKDIR /home/nnUNet
RUN pip install -e .

WORKDIR /home

# set environemnt variables for nnU-Net
ENV nnUNet_raw=/datasets/RNSH_HFlung/nnU-Net-processing/nnUNet_raw
ENV nnUNet_preprocessed=/datasets/RNSH_HFlung/nnU-Net-processing/nnUNet_preprocessed
ENV nnUNet_results=/datasets/RNSH_HFlung/nnU-Net-processing/nnUNet_results

# train the model with 5-fold cross-validation
CMD nnUNetv2_plan_and_preprocess -d 138 --verify_dataset_integrity && \
    nnUNetv2_train 138 3d_fullres 0 -device cuda --npz; \
    nnUNetv2_train 138 3d_fullres 1 -device cuda --npz; \
    nnUNetv2_train 138 3d_fullres 2 -device cuda --npz; \
    nnUNetv2_train 138 3d_fullres 3 -device cuda --npz; \
    nnUNetv2_train 138 3d_fullres 4 -device cuda --npz
