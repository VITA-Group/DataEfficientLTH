#!/bin/bash
set -e

DSET=$1
ROOT=$2

python3 train.py --dset $DSET --data_root $ROOT --mode imp --out_dir ${DSET}_resnet --batch_size 32 --lr 0.001 > "${DSET}_resnet.out"
python3 train.py --dset $DSET --data_root $ROOT --mode imp --out_dir ${DSET}_resnet_pretrained --batch_size 32 --lr 0.001 --pretrained > "${DSET}_resnet_pretrained.out"
python3 train.py --dset $DSET --data_root $ROOT --mode imp --out_dir ${DSET}_hresnet --batch_size 32 --lr 0.001 --net hresnet18 > "${DSET}_hresnet.out"
python3 train.py --dset $DSET --data_root $ROOT --mode imp --out_dir ${DSET}_fresnet --batch_size 32 --lr 0.001 --net fresnet18 > "${DSET}_fresnet.out"
python3 train.py --dset $DSET --data_root $ROOT --mode imp --out_dir ${DSET}_resnet_coslinear --batch_size 32 --lr 0.001 --cos_linear > "${DSET}_resnet_coslinear.out"
python3 train.py --dset $DSET --data_root $ROOT --mode imp --out_dir ${DSET}_resnet_tvmflinear --batch_size 32 --lr 0.001 --tvmf_linear > "${DSET}_resnet_tvmflinear.out"