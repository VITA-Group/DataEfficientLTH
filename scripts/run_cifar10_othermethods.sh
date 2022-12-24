#!/bin/bash
set -e

# cosine
python3 train.py --dset cifar10 --data_root data/ --data_size 0.01 --mode imp --out_dir resnet18_coslinear_cifar10_0.01 --rand_aug --cutout --cos_linear > resnet18_coslinear_cifar10_0.01.out &

python3 train.py --dset cifar10 --data_root data/ --data_size 0.02 --mode imp --out_dir resnet18_coslinear_cifar10_0.02 --auto_aug --cutout --cos_linear > resnet18_coslinear_cifar10_0.02.out 

# TVMF
python3 train.py --dset cifar10 --data_root data/ --data_size 0.01 --mode imp --out_dir resnet18_tvmflinear_cifar10_0.01 --rand_aug --cutout --tvmf_linear > resnet18_tvmflinear_cifar10_0.01.out

python3 train.py --dset cifar10 --data_root data/ --data_size 0.02 --mode imp --out_dir resnet18_tvmflinear_cifar10_0.02 --auto_aug --cutout --tvmf_linear > resnet18_tvmflinear_cifar10_0.02.out

# Harmonic
python3 train.py --dset cifar10 --data_root data/ --data_size 0.01 --mode imp --out_dir hresnet18_cifar10_0.01 --rand_aug --cutout --net hresnet18 > hresnet18_cifar10_0.01.out

python3 train.py --dset cifar10 --data_root data/ --data_size 0.02 --mode imp --out_dir hresnet18_cifar10_0.02 --auto_aug --cutout --net hresnet18 > hresnet18_cifar10_0.02.out

# Full Conv
python3 train.py --dset cifar10 --data_root data/ --data_size 0.01 --mode imp --out_dir fresnet18_cifar10_0.01 --rand_aug --cutout --net fresnet18

python3 train.py --dset cifar10 --data_root data/ --data_size 0.02 --mode imp --out_dir fresnet18_cifar10_0.02 --auto_aug --cutout --net fresnet18

# Pretrained (ImageNet)
python3 train.py --dset cifar10 --data_root data/ --data_size 0.01 --mode imp --out_dir preresnet18_cifar10_0.01 --rand_aug --cutout --lr 0.001 --pretrained

python3 train.py --dset cifar10 --data_root data/ --data_size 0.02 --mode imp --out_dir preresnet18_cifar10_0.02 --auto_aug --cutout --lr 0.001 --pretrained