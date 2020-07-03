# Learning Graph-Convolutional Representations for Point Cloud Denoising (ECCV 2020)
Bibtex entry:
```
@inproceedings{pistilli2020learning,
  title={Learning Graph-Convolutional Representationsfor Point Cloud Denoising},
  author={Pistilli, Francesca and Fracastoro, Giulia and Valsesia, Diego and Magli, Enrico},
  booktitle={The European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

# Requirements
- Python 2.7
- Tensorflow 1.12 with CUDA 9.0 (warning: later versions of tensorflow/cuda might have numerical issues)
- point_cloud_utils (https://github.com/fwilliams/point-cloud-utils)
- h5py
- numpy
- scipy

# Code structure
Code/ : Python source code for training/testing and the network model
Dataset/ : Shapenet training and testing data
log_dir/ : log files and tensorboard events
Results/ : saved checkpoints and denoised point clouds 

# Dataset
Download the Dataset directory from: https://www.dropbox.com/sh/nwdlzgnt987yjma/AAAi8q0E6yioxk5I_BkUpZE5a?dl=0

# Pretrained models
Pretrained models are included for the MSE-SP loss and 16-NN at standard deviations 0.01,0.015,0.02. Results might be slightly different from the ones in the paper.

# Test
```
./launcher_test.sh
```
Denoised point clouds and C2C metrics will be written to Results directory. 

# Train
```
./launcher_train.sh
```
Checkpoints will be written to the Results directory.

# Notes
Code is partially based on this project (https://github.com/diegovalsesia/gcdn). Check out its documetation for more details on the parameters in config.py.
Training and tested was run on
```
CPU: AMD Ryzen 1 1700
RAM: 32 GB
GPUs: 1x Nvidia Quadro P6000 (24 GB)
```