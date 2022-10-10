# Sparse Winning Tickets are Data-Efficient Image Recognizers
[Mukund Varma T]()<sup>1</sup>,
[Xuxi Chen](https://xxchen.site/)<sup>2</sup>,
[Zhenyu Zhang](https://scholar.google.com/citations?user=ZLyJRxoAAAAJ&hl=zh-CN)<sup>2</sup>,
[Tianlong Chen](https://tianlong-chen.github.io/)<sup>2</sup>,
[Subhashini Venugopalan](https://vsubhashini.github.io/)<sup>3</sup>,
[Zhangyang Wang](https://vita-group.github.io/)<sup>2</sup>

<sup>1</sup>Indian Institute of Technology Madras, <sup>2</sup>University of Texas at Austin, <sup>3</sup>Google Research

Accepted at NeurIPS '22

[Paper]()

## Introduction

Improving performance of deep networks in data limited regimes has warranted much attention. In this work, we empirically show that “winning tickets” (small subnetworks) obtained via magnitude pruning based on the lottery ticket hypothesis, apart from being sparse are also effective recognizers in data limited regimes. Based on extensive experiments, we find that in low data regimes (datasets of 50-100 examples per class), sparse winning tickets substantially outperform the original dense networks. This approach, when combined with augmentations or fine-tuning from a self-supervised backbone network, shows further improvements in performance by as much as 16% (absolute) on low sample datasets and longtailed classification. Further, sparse winning tickets are more robust to synthetic noise and distribution shifts compared to their dense counterparts. Our analysis of winning tickets on small datasets indicates that, though sparse, the networks retain density in the initial layers and their representations are more generalizable.

## Installation (To be released)

## Usage

### Training and Evaluation (To be released)

## Cite this work (To be released)

If you find our work / code implementation useful for your own research, please cite our paper.

```

```
