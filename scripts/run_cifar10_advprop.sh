#!/bin/bash
set -e

echo "train.py autoaug to sparse_advprop_1_autoaug"
python3 train.py --dset cifar10 --data_root data/ --data_size 1 --mode imp --out_dir "sparse_advprop_1_autoaug" --auto_aug --cutout --adv_prop > "sparse_advprop_1_autoaug.out"
grep 'epoch:\|pruning state:\|remaining weight =' "sparse_advprop_1_autoaug.out" > "sparse_advprop_1_autoaug/logs.txt"

echo "train.py autoaug to sparse_advprop_0.5_autoaug"
python3 train.py --dset cifar10 --data_root data/ --data_size 0.5 --mode imp --out_dir "sparse_advprop_0.5_autoaug" --auto_aug --cutout --adv_prop > "sparse_advprop_0.5_autoaug.out"
grep 'epoch:\|pruning state:\|remaining weight =' "sparse_advprop_0.5_autoaug.out" > "sparse_advprop_0.5_autoaug/logs.txt"

echo "train.py autoaug to sparse_advprop_0.2_autoaug"
python3 train.py --dset cifar10 --data_root data/ --data_size 0.2 --mode imp --out_dir "sparse_advprop_0.2_autoaug" --auto_aug --cutout --adv_prop > "sparse_advprop_0.2_autoaug.out"
grep 'epoch:\|pruning state:\|remaining weight =' "sparse_advprop_0.2_autoaug.out" > "sparse_advprop_0.2_autoaug/logs.txt"

echo "train.py autoaug to sparse_advprop_0.1_autoaug"
python3 train.py --dset cifar10 --data_root data/ --data_size 0.1 --mode imp --out_dir "sparse_advprop_0.1_autoaug" --auto_aug --cutout --adv_prop > "sparse_advprop_0.1_autoaug.out"
grep 'epoch:\|pruning state:\|remaining weight =' "sparse_advprop_0.1_autoaug.out" > "sparse_advprop_0.1_autoaug/logs.txt"

echo "train.py autoaug to sparse_advprop_0.02_autoaug"
python3 train.py --dset cifar10 --data_root data/ --data_size 0.02 --mode imp --out_dir "sparse_advprop_0.02_autoaug" --auto_aug --cutout --adv_prop > "sparse_advprop_0.02_autoaug.out"
grep 'epoch:\|pruning state:\|remaining weight =' "sparse_advprop_0.02_autoaug.out" > "sparse_advprop_0.02_autoaug/logs.txt"

echo "train.py randaug to sparse_advprop_0.01_randaug"
python3 train.py --dset cifar10 --data_root data/ --data_size 0.01 --mode imp --out_dir "sparse_advprop_0.01_randaug" --rand_aug --cutout --adv_prop > "sparse_advprop_0.01_randaug.out"
grep 'epoch:\|pruning state:\|remaining weight =' "sparse_advprop_0.01_randaug.out" > "sparse_advprop_0.01_randaug/logs.txt"