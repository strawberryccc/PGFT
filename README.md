# PGFT
# Requirements
Pytorch == 1.12.1 torchvision == 0.13.1 opencv-python tensorboardX einops skimage numpy
# Train Data Preparation
1.1 Download the DIV2K dataset and the Flickr2K dataset.   
1.2 Combine the HR images from these two datasets in your_data_path/DF2K/HR to build the DF2K dataset
# Test Data Preparation
Download benchmark datasets (e.g., Set5, Set14 and other test sets) and prepare HR/LR images in your_data_path/benchmark.
