# 📄 [CVPR 2026] PixDLM: A Dual-Path Multimodal Language Model for UAV Reasoning Segmentation

> **Official PyTorch Implementation of our CVPR 2026 paper**
> *PixDLM: A Dual-Path Multimodal Language Model for UAV Reasoning Segmentation*

---

## ✨ Authors

**Shuyan Ke**<sup>1</sup>, **Yifan Mei**<sup>1</sup>, **Changli Wu**<sup>1, 2</sup>, **Yonghan Zheng**<sup>1</sup>, **Jiayi Ji**<sup>1</sup>, **Liujuan Cao**<sup>1</sup>, **Rongrong Ji**<sup>1</sup> ✉

<sup>1</sup> Key Laboratory of Multimedia Trusted Perception and Efficient Computing, Ministry of Education of China, Xiamen University, 361005, P.R. China  
<sup>2</sup> Shanghai Innovation Institute  

*( ✉ Corresponding author )*

> 📧 **Contact:** `{keshuyan, meiyifan, wuchangli, zhengyonghan}@stu.xmu.edu.cn`, `jjyxmu@gmail.com`, `{caoliujuan, rrji}@xmu.edu.cn`

---

## 🔗 Links

* 📄 Paper: *Coming Soon (arXiv)*
* 🌐 Project Page: *Coming Soon*
* 📊 Pretrained Models: *Coming Soon*
* 🤗 DRSeg Dataset: https://huggingface.co/datasets/WhynotHug/DRSeg

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

* **semantic reasoning (language-aligned reasoning path)**
* **fine-grained perception (pixel-level visual path)**

enabling robust performance under challenging UAV scenarios.

---

## 🌟 Highlights

* **📌 New Task**
  We formalize **UAV Reasoning Segmentation** as an instruction-driven pixel-level prediction task, highlighting the limitations of existing reasoning models under aerial viewpoints.

* **📊 New Dataset: DRSeg**
  We introduce the first large-scale UAV reasoning segmentation benchmark with:

  * high-resolution aerial imagery
  * chain-of-thought (CoT) aligned reasoning annotations

* **🧠 New Model: PixDLM**
  A **Dual-Path pixel-level MLLM** that:

  * decouples reasoning and perception
  * enables structured reasoning-guided segmentation
  * achieves strong performance on both UAV and referring segmentation benchmarks

---

## 🗂️ DRSeg Dataset

DRSeg is a large-scale benchmark designed for UAV reasoning segmentation.

### 📦 Dataset Statistics

* **10,000** high-resolution UAV images
* **10,000** instance masks
* **10,000** reasoning QA pairs

Each image contains a single annotated target for reasoning, while maintaining high object density with multiple distractors.

---

### 🧩 Reasoning Annotation

The dataset provides **chain-of-thought aligned supervision**, covering three reasoning types:

* **Spatial reasoning**: 33.33%
* **Attribute reasoning**: 33.34%
* **Scene-level reasoning**: 33.33%

---

### 📐 UAV-specific Properties

* **Multi-altitude distribution**:

  * 30m: 31.44%
  * 60m: 25.45%
  * 100m: 43.11%

* **Small-object dominance**:

  * 58.08% instances occupy < 2% of image area

These properties make DRSeg particularly challenging for both perception and reasoning.

---

## 🏗️ Model Architecture

PixDLM adopts a **Dual-Path architecture**:

* **Reasoning Path**

  * Language-guided reasoning
  * Instruction understanding
  * Chain-of-thought generation

* **Perception Path**

  * High-resolution visual encoding
  * Pixel-level feature extraction
  * Fine-grained segmentation

The two paths are **jointly optimized** to align reasoning with pixel-level prediction.

---

## ⚙️ Installation

```bash
git clone https://github.com/your-repo/PixDLM.git
cd PixDLM

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

---

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
