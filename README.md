<div align="center">

<img src="./assert/logo.png" alt="LiAuto-GeoX Logo" width="320">

# LiAuto-GeoX: Efficient Grounded Driving Transformer

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://ljwwwiop.github.io/GeoX/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.05774-b31b1b.svg)](https://arxiv.org/pdf/2606.05774)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-ckpt-yellow)](https://huggingface.co/Veiiaj3dad/GeoX/tree/main)

Jiawei Lian<sup>1,2,\*</sup>,
Haoyi Sun<sup>2,\*</sup>,
Yang Wu<sup>1,\*</sup>,
Lifu Mu<sup>2,\*</sup>,
Siyuan Wang<sup>1,2</sup>,  
Le Hui<sup>3,4,†</sup>,
Ning Mao<sup>2,†,‡</sup>,
Tao Wei<sup>2</sup>,
Pan Zhou<sup>2</sup>,
Kun Zhan<sup>2</sup>,
Jian Yang<sup>1,†</sup>

<sup>1</sup>🎓 Nanjing University of Science and Technology &nbsp;&nbsp;
<sup>2</sup>🏢 Li Auto Inc. &nbsp;&nbsp;
<sup>3</sup>🎓 Northwestern Polytechnical University &nbsp;&nbsp;
<br>
<sup>4</sup>🎓 Department of Computing, The Hong Kong Polytechnic University

<sup>\*</sup>Equal Contribution &nbsp;&nbsp;
<sup>†</sup>✉️ Corresponding Author &nbsp;&nbsp;
<sup>‡</sup>Project Leader

</div>

---

## Release Plan

- [✓] [Paper Release](https://arxiv.org/pdf/2606.05774)
- [✓] LiAuto-GeoX Weight
- [✓] Inference Instructions
- [ ] GeoX-Large
- [ ] Data Processing
- [ ] Training Pipeline


---

## Pretrained Models

Before using the models, please request access to the checkpoints once they are released.  
All released models will be evaluated under the same protocol as reported in the paper.

| Model | Parameters | Input Setting | Download |
|---|---:|---|---|
| LiAuto-GeoX | 0.15B | Surround-view / Video | [🤗 Hugging Face](https://huggingface.co/Veiiaj3dad/GeoX/tree/main) |
| LiAuto-GeoX-Teacher | 1.1B | Surround-view | - |

---

## Inference

### Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage Examples

**Single Frame Example** - Basic inference with RGB images:
```bash
CUDA_VISIBLE_DEVICES=2 python inference.py \
    --image_folder /path/to/your/images \
    --port 8082
```

**RGB + Sky Mask Example** - Filter out sky regions for cleaner reconstruction:
```bash
CUDA_VISIBLE_DEVICES=2 python inference.py \
    --image_folder /path/to/your/images \
    --port 8082 \
    --mask_sky
```

**RGB + Pose Example** - Use ground truth camera poses for better accuracy:
```bash
CUDA_VISIBLE_DEVICES=2 python inference.py \
    --image_folder /path/to/your/images \
    --camera_folder /path/to/your/cameras \
    --port 8083
```

After running inference, open your browser and navigate to `http://localhost:PORT` (replace `PORT` with your specified port) to visualize the 3D reconstruction results interactively.

**Additional Options:**
- `--conf_threshold`: Adjust the confidence threshold (default: 10.0) to filter low-confidence points. Lower values show more points, higher values show fewer but more confident points.
- `--mask_black_bg`: Filter out black background pixels
- `--mask_white_bg`: Filter out white background pixels
- `--save_glb`: Export the reconstruction as a GLB file

---

## Acknowledgements

Thanks to these great repositories:
[DINOv2](https://github.com/facebookresearch/dinov2),
[CUT3R](https://github.com/CUT3R/CUT3R),
[VGGT](https://github.com/facebookresearch/vggt),
[DA3](https://github.com/ByteDance-Seed/Depth-Anything-3),
[PI3](https://github.com/yyfz/Pi3/tree/training),
[DVGT](https://github.com/wzzheng/DVGT/tree/main),
[OmniVGGT](https://github.com/Livioni/OmniVGGT-official),
[FastVGGT](https://github.com/mystorm16/FastVGGT),
[LiteVGGT](https://github.com/GarlicBa/LiteVGGT-repo),
[SparseWorld-TC](https://github.com/MrPicklesGG/SparseWorld-TC),
and many other inspiring works in the community.

---

## Citation

If you find LiAuto-GeoX useful for your work, please cite:

```bibtex
@article{lian2026geox,
  author    = {Lian, Jiawei and Sun, Haoyi and Wu, Yang and Mu, Lifu and Wang, Siyuan and Wei, Tao and Hui, Le and Mao, Ning and Zhou, Pan and Zhan, Kun and Yang, Jian},
  title     = {LiAuto-GeoX: Efficient Grounded Driving Transformer},
  journal   = {arXiv:2606.05774},
  year      = {2026},
}

