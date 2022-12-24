#!/bin/bash
set -e

echo "train.py autoaug to small_1_autoaug"
python3 train.py --dset cifar10 --data_root data/ --data_size 1 --mode train --out_dir "small_1_autoaug" --auto_aug --cutout --in_planes 52 > "small_1_autoaug.out"
grep 'epoch:\|pruning state:\|remaining weight =' "small_1_autoaug.out" > "small_1_autoaug/logs.txt"

echo "train.py autoaug to small_0.5_autoaug"
python3 train.py --dset cifar10 --data_root data/ --data_size 0.5 --mode train --out_dir "small_0.5_autoaug" --auto_aug --cutout --in_planes 58 > "small_0.5_autoaug.out"
grep 'epoch:\|pruning state:\|remaining weight =' "small_0.5_autoaug.out" > "small_0.5_autoaug/logs.txt"

echo "train.py autoaug to small_0.2_autoaug"
python3 train.py --dset cifar10 --data_root data/ --data_size 0.2 --mode train --out_dir "small_0.2_autoaug" --auto_aug --cutout --in_planes 52 > "small_0.2_autoaug.out"
grep 'epoch:\|pruning state:\|remaining weight =' "small_0.2_autoaug.out" > "small_0.2_autoaug/logs.txt"

echo "train.py autoaug to small_0.1_autoaug"
python3 train.py --dset cifar10 --data_root data/ --data_size 0.1 --mode train --out_dir "small_0.1_autoaug" --auto_aug --cutout --in_planes 40 > "small_0.1_autoaug.out"
grep 'epoch:\|pruning state:\|remaining weight =' "small_0.1_autoaug.out" > "small_0.1_autoaug/logs.txt"

echo "train.py autoaug to small_0.02_autoaug"
python3 train.py --dset cifar10 --data_root data/ --data_size 0.02 --mode train --out_dir "small_0.02_autoaug" --auto_aug --cutout --in_planes 12 > "small_0.02_autoaug.out"
grep 'epoch:\|pruning state:\|remaining weight =' "small_0.02_autoaug.out" > "small_0.02_autoaug/logs.txt"

echo "train.py randaug to small_0.01_randaug"
python3 train.py --dset cifar10 --data_root data/ --data_size 0.01 --mode train --out_dir "small_0.01_randaug" --rand_aug --cutout --in_planes 12 > "small_0.01_randaug.out"
grep 'epoch:\|pruning state:\|remaining weight =' "small_0.01_randaug.out" > "small_0.01_randaug/logs.txt"