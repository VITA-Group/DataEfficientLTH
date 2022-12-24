#!/bin/bash
set -e

OUT_DIR_PREFIX=$1
DATA_SIZE=$2
MODE=$3

echo "train.py baseaug to ${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_baseaug"
python3 train.py --dset cifar10 --data_root data/ --data_size $DATA_SIZE --mode $MODE --out_dir "${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_baseaug" > "${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_baseaug.out"
grep 'epoch:\|pruning state:\|remaining weight =' "${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_baseaug.out" > "${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_baseaug/logs.txt"

echo "train.py contrastaug to ${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_contrastaug"
python3 train.py --dset cifar10 --data_root data/ --data_size $DATA_SIZE --mode $MODE --out_dir "${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_contrastaug" --contrast_aug > "${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_contrastaug.out"
grep 'epoch:\|pruning state:\|remaining weight =' "${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_contrastaug.out" > "${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_contrastaug/logs.txt"

echo "train.py randaug to ${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_randaug"
python3 train.py --dset cifar10 --data_root data/ --data_size $DATA_SIZE --mode $MODE --out_dir "${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_randaug" --rand_aug --cutout > "${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_randaug.out"
grep 'epoch:\|pruning state:\|remaining weight =' "${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_randaug.out" > "${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_randaug/logs.txt"

echo "train.py autoaug to ${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_autoaug"
python3 train.py --dset cifar10 --data_root data/ --data_size $DATA_SIZE --mode $MODE --out_dir "${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_autoaug" --auto_aug --cutout > "${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_autoaug.out"
grep 'epoch:\|pruning state:\|remaining weight =' "${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_autoaug.out" > "${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_autoaug/logs.txt"

# echo "train.py customaug to ${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_customaug"
# python3 train.py --dset cifar10 --data_root data/ --data_size $DATA_SIZE --mode $MODE --out_dir "${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_customaug" --custom_aug > "${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_customaug.out"
# grep 'epoch:\|pruning state:\|remaining weight =' "${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_customaug.out" > "${OUT_DIR_PREFIX}_cifar10_${DATA_SIZE}_customaug/logs.txt"

echo "completed expts ${OUT_DIR_PREFIX} ${DATA_SIZE} ${MODE}"