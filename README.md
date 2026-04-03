# PixDLM
> #### Shuyan Ke, Yifan Mei, Changli Wu, Yonghan Zheng, Jiayi Ji, Liujuan Cao, and Rongrong Ji<sup>&dagger;</sup><sup>&ddagger;</sup>
> <sup>&dagger;</sup>Corresponding author

> Xiamen University

# Highlight

1.We formalize UAV Reasoning Segmentation as an instruction-driven pixel-level prediction task under UAV-specific visual statistics, clarifying why existing reasoning models fail under aerial viewpoints.

2.We introduce DRSeg, the first large-scale UAV reasoning segmentation benchmark with over 10,000 high-resolution images and CoT-aligned reasoning annotations.

3.We present PixDLM, a Dual-Path pixel-level multimodal language model that establishes a strong baseline for UAV reasoning segmentation and demonstrates competitive performance on standard referring segmentation datasets.

# DRSeg dataset
DRSeg contains a total of 10,000 high-resolution UAV images and 10,000 instance masks, making it one of the largest UAV-view segmentation datasets to date.  
Although each image is annotated with only one target instance for reasoning, aerial scenes typically contain many distractor objects, leading to high effective object density.

For reasoning supervision, DRSeg provides 10,000 reasoning QA pairs, one for each annotated instance, covering all three reasoning dimensions.
Among them, 33.33\% correspond to Spatial reasoning, 33.34\% to Attribute reasoning, and 33.33\% to Scene-level reasoning, demonstrating a balanced distribution that supports diverse reasoning behaviors.

In terms of geometric diversity, the dataset spans three flight altitudes (30\,m, 60\,m, and 100\,m), with 31.44\%, 25.45\%, and 43.11\% of images captured at each altitude, respectively.  
Object scale variation is equally significant: 58.08\% of all instances fall into the small-object regime (area $<\! 2\%$ of the image), highlighting the fine-grained challenges introduced by UAV viewpoints.Additional dataset visualizations are provided in the Supplementary Material.

 ## Installation

```
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Training

### Pre-trained weights

Our training pipeline starts from pretrained weights. Specifically, we initialize the model with `liuhaotian/llava-v1.6-vicuna-7b`, `openai/clip-vit-large-patch14`, and `sam2.1_hiera_l`.

### Training
```
sh train.sh
```

When training is finished, to get the full model weight:
```
cd path_to_ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
```

### Merge LoRA Weight
Merge the LoRA weights of `pytorch_model.bin`, save the resulting model into your desired path in the Hugging Face format:
```
sh merge.sh
```

### Update model parameters
save part of the model structure:
```
python extract.py
```

Please update the path to the extracted model in 
`./model/llava/model/multimodal_encoder/multipath_encoder_wapper.py`:

```
python
weights_dict = torch.load("path_to_extracted_model", map_location="cpu")
```

### Evaluating

```
sh eval.sh

```