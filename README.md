# DataElixir

[![arXiv](https://img.shields.io/badge/cs.CR-arXiv:2312.11057-b31b1b.svg)](https://arxiv.org/abs/2312.11057)

This is the official implementation of our paper [DataElixir: Purifying Poisoned Dataset to Mitigate Backdoor Attacks via Diffusion Models](https://arxiv.org/abs/2312.11057), accepted by AAAI 2024.

## Abstract
Dataset sanitization is a widely adopted proactive defense against poisoning-based backdoor attacks, aimed at filtering out and removing poisoned samples from training datasets. However, existing methods have shown limited efficacy in countering the ever-evolving trigger functions, and often leading to considerable degradation of benign accuracy. In this paper, we propose DataElixir, a novel sanitization approach tailored to purify poisoned datasets. We leverage diffusion models to eliminate trigger features and restore benign features, thereby turning the poisoned samples into benign ones. Specifically, with multiple iterations of the forward and reverse process, we extract intermediary images and their predicted labels for each sample in the original dataset. Then, we identify anomalous samples in terms of the presence of label transition of the intermediary images, detect the target label by quantifying distribution discrepancy, select their purified images considering pixel and feature distance, and determine their ground-truth labels by training a benign model. Experiments conducted on 9 popular attacks demonstrates that DataElixir effectively mitigates various complex attacks while exerting minimal impact on benign accuracy, surpassing the performance of baseline defense methods.

## TODO

- [x] ~~upload paper to arXiv~~

- [ ] release code

## Citation
If you find our work useful for your research, please consider citing our paper:
```
@misc{zhou2023dataelixir,
      title={DataElixir: Purifying Poisoned Dataset to Mitigate Backdoor Attacks via Diffusion Models}, 
      author={Jiachen Zhou and Peizhuo Lv and Yibing Lan and Guozhu Meng and Kai Chen and Hualong Ma},
      year={2023},
      eprint={2312.11057},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
