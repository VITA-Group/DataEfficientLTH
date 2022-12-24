# Sparse Winning Tickets are Data-Efficient Image Recognizers
[Mukund Varma T]()<sup>1</sup>,
[Xuxi Chen](https://xxchen.site/)<sup>2</sup>,
[Zhenyu Zhang](https://scholar.google.com/citations?user=ZLyJRxoAAAAJ&hl=zh-CN)<sup>2</sup>,
[Tianlong Chen](https://tianlong-chen.github.io/)<sup>2</sup>,
[Subhashini Venugopalan](https://vsubhashini.github.io/)<sup>3</sup>,
[Zhangyang Wang](https://vita-group.github.io/)<sup>2</sup>

<sup>1</sup>Indian Institute of Technology Madras, <sup>2</sup>University of Texas at Austin, <sup>3</sup>Google Research

Accepted at NeurIPS '22 (Featured Paper)

[Paper](https://openreview.net/forum?id=wfKbtSjHA6F), [Slides](https://docs.google.com/presentation/d/1gVNX23VgFRUR9e_4tHvBlMBXLg6wnMQoa_zWOri1rWM/edit?usp=sharing)

## Abstract

Improving performance of deep networks in data limited regimes has warranted much attention. In this work, we empirically show that “winning tickets” (small subnetworks) obtained via magnitude pruning based on the lottery ticket hypothesis, apart from being sparse are also effective recognizers in data limited regimes. Based on extensive experiments, we find that in low data regimes (datasets of 50-100 examples per class), sparse winning tickets substantially outperform the original dense networks. This approach, when combined with augmentations or fine-tuning from a self-supervised backbone network, shows further improvements in performance by as much as 16% (absolute) on low sample datasets and longtailed classification. Further, sparse winning tickets are more robust to synthetic noise and distribution shifts compared to their dense counterparts. Our analysis of winning tickets on small datasets indicates that, though sparse, the networks retain density in the initial layers and their representations are more generalizable.

## Installation

```bash
pip install -r requirements.txt
```

Additional datasets must be downloaded and placed in the appropriate directories - [CIFAR10-C](https://zenodo.org/record/2535967#.Y6a9EdJBw1g), [CIFAR10.2](https://github.com/modestyachts/cifar-10.2), [ImageNet (50 images/class)](https://github.com/VIPriors/vipriors-challenges-toolkit), [EuroSAT (50 images/class)](https://github.com/cvjena/deic), [ISIC 2018 (80 images/class)](https://github.com/cvjena/deic), [CLaMM (50 images/class)](https://github.com/cvjena/deic)

## Usage

### Training

```bash
# to run cifar10 all augmentation strategies, all data sizes
bash run_cifar10.sh sparse 1 imp
bash run_cifar10.sh sparse 0.5 imp
bash run_cifar10.sh sparse 0.2 imp
bash run_cifar10.sh sparse 0.1 imp
bash run_cifar10.sh sparse 0.02 imp
bash run_cifar10.sh sparse 0.01 imp

# run other methods on cifar10 subsets
bash run_cifar10_othermethods.sh

# run cifar100 long_tailed
bash run_cifar100_longtailed.sh

# run on other datasets
bash run_otherdsets.sh eurosat_rgb <path-to-eurosatrgb>
bash run_otherdsets.sh isic <path-to-isic>
bash run_otherdsets.sh clamm <path-to-clamm>
```

Additional scripts can be found [here](scripts/)

### Evaluation

Code to evaluate robustness - synthetic, adversarial, distribution shifts can be found [here](helpers.ipynb)

## Cite this work

If you find our work / code implementation useful for your own research, please cite our paper.

```
@inproceedings{
    t2022sparse,
    title={Sparse Winning Tickets are Data-Efficient Image Recognizers},
    author={Mukund Varma T and Xuxi Chen and Zhenyu Zhang and Tianlong Chen and Subhashini Venugopalan and Zhangyang Wang},
    booktitle={Advances in Neural Information Processing Systems},
    editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
    year={2022},
    url={https://openreview.net/forum?id=wfKbtSjHA6F}
}
```
