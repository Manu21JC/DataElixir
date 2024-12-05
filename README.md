# DataElixir

This is the official implementation of our paper [DataElixir: Purifying Poisoned Dataset to Mitigate Backdoor Attacks via Diffusion Models](https://arxiv.org/abs/2312.11057), accepted by AAAI 2024.

## Abstract
Dataset sanitization is a widely adopted proactive defense against poisoning-based backdoor attacks, aimed at filtering out and removing poisoned samples from training datasets. However, existing methods have shown limited efficacy in countering the ever-evolving trigger functions, and often leading to considerable degradation of benign accuracy. In this paper, we propose DataElixir, a novel sanitization approach tailored to purify poisoned datasets. We leverage diffusion models to eliminate trigger features and restore benign features, thereby turning the poisoned samples into benign ones. Specifically, with multiple iterations of the forward and reverse process, we extract intermediary images and their predicted labels for each sample in the original dataset. Then, we identify anomalous samples in terms of the presence of label transition of the intermediary images, detect the target label by quantifying distribution discrepancy, select their purified images considering pixel and feature distance, and determine their ground-truth labels by training a benign model. Experiments conducted on 9 popular attacks demonstrate that DataElixir effectively mitigates various complex attacks while exerting minimal impact on benign accuracy, surpassing the performance of baseline defense methods.

## Getting Started

### Installation
Install the required dependencies using `conda`:
```bash
conda env create -f environment.yaml
conda activate dataelixir
```

### Usage
To run DataElixir defense on CIFAR10 for BadNet attack:
```bash
python ./attack/badnet.py --yaml_path ../config/badnet/cifar10.yaml --save_folder_name badnet
```
All results are saved in `./record/<save_folder_name>`, including log, attack model, and defense model. To modify parameters for the attack and defense, you can both specify them in command line and in corresponding YAML config file (`./config/badnet/cifar10.yaml`).

## Acknowledgments
Our code is built upon [BackdoorBench](https://github.com/SCLBD/BackdoorBench/tree/v1) and [pytorch_diffusion](https://github.com/pesser/pytorch_diffusion). We greatly appreciate the authors for making their code publicly available.

## Citation
If you find our work useful for your research, please consider citing our paper:
```
@inproceedings{zhou2024dataelixir,
  title={DataElixir: Purifying Poisoned Dataset to Mitigate Backdoor Attacks via Diffusion Models},
  author={Zhou, Jiachen and Lv, Peizhuo and Lan, Yibing and Meng, Guozhu and Chen, Kai and Ma, Hualong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={19},
  pages={21850--21858},
  year={2024}
}
```