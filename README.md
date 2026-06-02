<div align="center">

# LiAuto-GeoX LiAuto-GeoX: Efficient Grounded Driving Transformer

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://ljwwwiop.github.io/GeoX/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2605.xxxxx)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Demo-yellow)](https://huggingface.co)
[![Code](https://img.shields.io/badge/Code-Coming%20Soon-blue)](#)

Jiawei Lian<sup>1,2,\*</sup>,
Haoyi Sun<sup>2,\*</sup>,
Yang Wu<sup>1,\*</sup>,
Lifu Mu<sup>2,\*</sup>,
Siyuan Wang<sup>1,2</sup>,
Tao Wei<sup>2</sup>,
Le Hui<sup>3,4,†</sup>,
Ning Mao<sup>2,†,‡</sup>,
Pan Zhou<sup>2</sup>,
Kun Zhan<sup>2</sup>,
Jian Yang<sup>1,†</sup>

<sup>1</sup>Nanjing University of Science and Technology &nbsp;&nbsp;
<sup>2</sup>Li Auto Inc. &nbsp;&nbsp;
<sup>3</sup>Northwestern Polytechnical University &nbsp;&nbsp;
<sup>4</sup>Department of Computing, The Hong Kong Polytechnic University

<sup>*</sup>Equal Contribution &nbsp;&nbsp;
<sup>†</sup>Corresponding Author &nbsp;&nbsp;
<sup>‡</sup>Project Leader

</div>

---

## Release Plan

- [ ] Pretrained Models
  - [ ] LiAuto-GeoX-155M (Compact Student Model)
  - [x] ~~LiAuto-GeoX-Teacher (Large-scale Teacher Model)~~ - Not for release
- [x] ~~Training Pipeline (Complete)~~ - Not for release
- [ ] Inference Instructions

---

## Pretrained Models

Before using the models, please request access to the checkpoints once they are released.  
All released models will be evaluated under the same protocol as reported in the paper.

| Model | Parameters | Input Setting | Download |
|---|---:|---|---|
| LiAuto-GeoX-155M | 155M | Surround-view / Video | Coming soon |
| LiAuto-GeoX-Teacher | 1.1B | Surround-view | - |

---

## Citation

If you find LiAuto-GeoX useful for your work, please cite:

```bibtex
@article{lian2026geox,
  author    = {Lian, Jiawei and Sun, Haoyi and Wu, Yang and Mu, Lifu and Wang, Siyuan and Wei, Tao and Hui, Le and Mao, Ning and Zhou, Pan and Zhan, Kun and Yang, Jian},
  title     = {LiAuto-GeoX: Efficient Grounded Driving Transformer},
  journal   = {arXiv},
  year      = {2026},
}

