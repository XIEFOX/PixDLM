# 📄 [CVPR 2026] PixDLM: A Dual-Path Multimodal Language Model for UAV Reasoning Segmentation

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Coming_Soon-blue)](#)
[![Dataset](https://img.shields.io/badge/Dataset-DRSeg-green)](https://huggingface.co/datasets/WhynotHug/DRSeg)
[![Model](https://img.shields.io/badge/Model-PixDLM-orange)](https://huggingface.co/WhynotHug/PixDLM)

</div>

---

## ✨ Authors

**Shuyan Ke**<sup>1</sup>, **Yifan Mei**<sup>1</sup>, **Changli Wu**<sup>1, 2</sup>, **Yonghan Zheng**<sup>1</sup>, **Jiayi Ji**<sup>1</sup>, **Liujuan Cao**<sup>1</sup>, **Rongrong Ji**<sup>1</sup> ✉

<sup>1</sup> Key Laboratory of Multimedia Trusted Perception and Efficient Computing, Ministry of Education of China, Xiamen University, 361005, P.R. China  
<sup>2</sup> Shanghai Innovation Institute  

---

## 📢 News

* **[2026.04]** Pretrained models and inference code are available!
* **[2026.04]** Training code released.
* **[2026.03]** 🧩 [DRSeg dataset](https://huggingface.co/datasets/WhynotHug/DRSeg) released on HuggingFace.
* **[2026.01]** 🎉 PixDLM is accepted by **CVPR 2026**!


---

## 🚀 Overview

Understanding complex aerial scenes requires not only pixel-level perception but also structured reasoning under UAV-specific visual characteristics such as small objects, large viewpoints, and high scene complexity.

In this project, we introduce **PixDLM**, a **Dual-Path Multimodal Language Model** designed for **UAV reasoning segmentation**, a new task that integrates instruction following, reasoning, and fine-grained segmentation.

PixDLM explicitly models the synergy between:
* **Semantic reasoning** (Language-aligned reasoning path)
* **Fine-grained perception** (Pixel-level visual path)

This dual-path design enables robust performance under challenging UAV scenarios.

---

## 🌟 Highlights

* **📌 New Task: UAV Reasoning Segmentation**
  We formalize it as an instruction-driven pixel-level prediction task, highlighting the limitations of existing reasoning models under aerial viewpoints.
* **📊 New Dataset: DRSeg**
  The first large-scale UAV reasoning segmentation benchmark featuring high-resolution aerial imagery and chain-of-thought (CoT) aligned reasoning annotations.
* **🧠 New Model: PixDLM**
  A **Dual-Path pixel-level MLLM** that decouples reasoning and perception, enabling structured reasoning-guided segmentation with strong performance on both UAV and referring segmentation benchmarks.

---

## 🗂️ DRSeg Dataset

DRSeg is a large-scale benchmark designed specifically for UAV reasoning segmentation. 

* **Statistics**: 10,000 high-resolution UAV images | 10,000 instance masks | 10,000 reasoning QA pairs.
* **Reasoning Types**: Spatial reasoning (33.33%), Attribute reasoning (33.34%), Scene-level reasoning (33.33%).
* **UAV-specific Properties**: Multi-altitude distribution (30m, 60m, 100m) and small-object dominance (58.08% of instances occupy < 2% of the image area).

---

## ⚙️ Installation

```bash
git clone https://github.com/your-repo/PixDLM.git
cd PixDLM

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

---

## 📁 Data & Weights Preparation

Please download:

* 🤗 **Pretrained weights** from our HuggingFace model page
* 📊 **DRSeg dataset** from our HuggingFace dataset page

---

### 📂 Directory Structure

Organize your project directory as follows:

```plaintext
PixDLM/
├── PixDLM/
│   └── pytorch_model.bin        <-- Place the pretrained weights here
├── data/
│   └── DRSeg/                   <-- Place the dataset here
├── model/
├── train.sh
├── eval.sh
└── ...
```

> ⚠️ **Important:**
> Make sure the pretrained weight file `pytorch_model.bin` is placed under:
>
> ```
> PixDLM/PixDLM/pytorch_model.bin
> ```
>
> Otherwise, the model will not be loaded correctly.


## 🏋️ Training

### 🔹 Pretrained Initialization

Our model is initialized from:

* `liuhaotian/llava-v1.6-vicuna-7b`
* `openai/clip-vit-large-patch14`
* `sam2.1_hiera_l`

---

### 🔹 Training

```bash
sh train.sh
```

---

## 🔧 Post-processing

### Convert DeepSpeed weights

```bash
cd path_to_ckpt_model
python zero_to_fp32.py . ../pytorch_model.bin
```

---

### Merge LoRA weights

```bash
sh merge.sh
```

---

### Extract model components

```bash
python extract.py
```

Update path in:

```
./model/llava/model/multimodal_encoder/multipath_encoder_wapper.py
```

```python
weights_dict = torch.load("path_to_extracted_model", map_location="cpu")
```

---

## 📊 Evaluation

```bash
sh eval.sh
```

---

## 🔮 Future Work

* Scaling UAV reasoning datasets
* Improving long-chain reasoning consistency
* Real-world UAV deployment

---

## 📌 Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{ke2026pixdlm,
  title={PixDLM: A Dual-Path Multimodal Language Model for UAV Reasoning Segmentation},
  author={Ke, Shuyan and Mei, Yifan and Wu, Changli and Zheng, Yonghan and Ji, Jiayi and Cao, Liujuan and Ji, Rongrong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

---

## ⭐ Acknowledgements

This project is built upon the following open-source works:

* LLaVA
* CLIP
* Segment Anything (SAM)

We sincerely thank the authors for their contributions.


---

⭐ If you find this project useful, please consider giving it a star!
