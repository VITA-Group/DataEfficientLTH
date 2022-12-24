#!/bin/bash
set -e

echo "train.py autoaug to cifar100_longtailed_sparse_0.1_autoaug"
python3 train.py --dset cifar100 --data_root /home/sneezygiraffe/data/cifar100 --mode imp --out_dir "cifar100_longtailed_sparse_0.1_autoaug" --auto_aug --cutout --long_tailed --long_tailed_factor 0.1 > "cifar100_longtailed_sparse_0.1_autoaug.out"
grep 'epoch:\|pruning state:\|remaining weight =' "cifar100_longtailed_sparse_0.1_autoaug.out" > "cifar100_longtailed_sparse_0.1_autoaug/logs.txt"

echo "train.py autoaug to cifar100_longtailed_sparse_0.05_autoaug"
python3 train.py --dset cifar100 --data_root /home/sneezygiraffe/data/cifar100 --mode imp --out_dir "cifar100_longtailed_sparse_0.05_autoaug" --auto_aug --cutout --long_tailed --long_tailed_factor 0.05 > "cifar100_longtailed_sparse_0.05_autoaug.out"
grep 'epoch:\|pruning state:\|remaining weight =' "cifar100_longtailed_sparse_0.05_autoaug.out" > "cifar100_longtailed_sparse_0.05_autoaug/logs.txt"

echo "train.py autoaug to cifar100_longtailed_sparse_0.02_autoaug"
python3 train.py --dset cifar100 --data_root /home/sneezygiraffe/data/cifar100 --mode imp --out_dir "cifar100_longtailed_sparse_0.02_autoaug" --auto_aug --cutout --long_tailed --long_tailed_factor 0.02 > "cifar100_longtailed_sparse_0.02_autoaug.out"
grep 'epoch:\|pruning state:\|remaining weight =' "cifar100_longtailed_sparse_0.02_autoaug.out" > "cifar100_longtailed_sparse_0.02_autoaug/logs.txt"

echo "train.py autoaug to cifar100_longtailed_sparse_0.01_autoaug"
python3 train.py --dset cifar100 --data_root /home/sneezygiraffe/data/cifar100 --mode imp --out_dir "cifar100_longtailed_sparse_0.01_autoaug" --auto_aug --cutout --long_tailed --long_tailed_factor 0.01 > "cifar100_longtailed_sparse_0.01_autoaug.out"
grep 'epoch:\|pruning state:\|remaining weight =' "cifar100_longtailed_sparse_0.01_autoaug.out" > "cifar100_longtailed_sparse_0.01_autoaug/logs.txt"