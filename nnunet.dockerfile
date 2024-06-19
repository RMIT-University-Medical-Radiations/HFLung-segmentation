# this is based on https://hub.docker.com/r/nbutter/nnunet

# pull base image
FROM nvidia/cuda:12.2.0-base-ubuntu22.04
MAINTAINER Daryl Wilding-McBride

# install Ubuntu libraries and packages
RUN apt-get update -y && \
apt-get install git curl -y && \
rm -rf /var/lib/apt/lists/*

# set the time zone
ENV TZ=Australia/Melbourne
ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get update && apt-get install -y tzdata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# set some environment variables
ENV PATH="/home/miniconda3/bin:${PATH}"
ARG PATH="/home/miniconda3/bin:${PATH}"

WORKDIR /home

# install Python
# RUN curl -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/miniconda3
# PyTorch isn't happy with Python 3.12 yet, so we'll install 3.11
RUN curl -O  https://repo.anaconda.com/miniconda/Miniconda3-py311_24.4.0-0-Linux-x86_64.sh && bash Miniconda3-py311_24.4.0-0-Linux-x86_64.sh -b -p /home/miniconda3

# update pip
RUN conda install pip
RUN pip install --upgrade pip

# install nnU-Net
RUN git clone https://github.com/MIC-DKFZ/nnUNet.git
WORKDIR /home/nnUNet
RUN pip install -e .

WORKDIR /home

# set environment variables for nnU-Net
ENV nnUNet_raw=/datasets/RNSH_HFlung/nnU-Net-processing/nnUNet_raw
ENV nnUNet_preprocessed=/datasets/RNSH_HFlung/nnU-Net-processing/nnUNet_preprocessed
ENV nnUNet_results=/datasets/RNSH_HFlung/nnU-Net-processing/nnUNet_results
ENV MPLCONFIGDIR=/daryl/.config/matplotlib

# set up the custom trainer
RUN ln -s /HFLung-segmentation/nnUNetTrainerDA5_100epochs.py /home/nnUNet/nnunetv2/training/nnUNetTrainer
RUN ln -s /HFLung-segmentation/nnUNetTrainerDA5_60epochs.py /home/nnUNet/nnunetv2/training/nnUNetTrainer
