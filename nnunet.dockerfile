# this is based on https://hub.docker.com/r/nbutter/nnunet

# pull base image
FROM nvidia/cuda:12.2.0-base-ubuntu22.04
MAINTAINER Daryl Wilding-McBride

# create some directories to work with
RUN mkdir /project && mkdir /scratch && mkdir /dataset
RUN mkdir /dataset/nnUNet_raw && mkdir /dataset/nnUNet_preprocessed && mkdir /dataset/nnUNet_results

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

# set environemnt variables for nnU-Net
ENV nnUNet_raw="/dataset/nnUNet_raw"
ENV nnUNet_preprocessed="/dataset/nnUNet_preprocessed"
ENV nnUNet_results="/dataset/nnUNet_results"
