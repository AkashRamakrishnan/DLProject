#!/bin/bash -i

#You can just call this using terminal (eg. ./create_env.sh)
echo "Creating DL environment.."
conda create --quiet --yes -n dl python="3.9.15"
echo "Creating environment done."
echo
echo "Installing required conda packages.."
conda install -n dl ipykernel nb_conda_kernels pytorch-gpu pytorch-lightning h5py pandas nbformat tqdm Pillow scikit-image scikit-learn scipy tqdm timm ipywidgets matplotlib seaborn torchinfo --yes
conda update -n base -c defaults conda --yes
conda update --all --yes
echo "Installing required conda packages done."