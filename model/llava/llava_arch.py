from abc import ABC, abstractmethod

import torch
import torch.nn as nn

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_encoder.custom_clip import _CLIPVisionModel
import torch.nn.functional as F
class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            print("config",config)
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
        self.separate_mm_projector = config.separate_mm_projector
        if config.separate_mm_projector:
            hidden_size = config.hidden_size if config.mm_projector_hidden_dim == 1 else config.hidden_size*2
            out_size = config.hidden_size if config.mm_projector_out_dim == 1 else config.mm_hidden_size
            self.mm_projector = nn.Sequential(nn.Linear(config.mm_hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, config.hidden_size))
            self.out_mm_projector = nn.Sequential(nn.Linear(config.mm_hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, out_size))
    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower
        print("model_arfs:",model_args)
        vision_tower = build_vision_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if not hasattr(self, "mm_projector"):
            self.mm_projector = nn.Linear(
                self.config.mm_hidden_size, self.config.hidden_size
            )

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )
def bipartite_token_merge(x, num_tokens_out=256, key_vectors=None, size=None):      
    """      
    ToMe双向匹配Token合并的函数式实现      
          
    Args:      
        x: [B, N, C] 输入token特征    
        num_tokens_out: 输出token数量    
        key_vectors: [B, N, head_dim] attention key向量(如果提供则使用)  
        size: [B, N, 1] token大小(用于加权平均)  
          
    Returns:      
        compressed_x: [B, M, C] 压缩后的特征  
        new_size: [B, M, 1] 更新后的token大小  
    """      
    import torch      
          
    B, N, C = x.shape      
    r = N - num_tokens_out  
      
    r = min(r, N // 2)  
      
    if r <= 0:  
        return x, size if size is not None else torch.ones(B, N, 1, device=x.device)  
      
    if size is None:  
        size = torch.ones(B, N, 1, device=x.device,dtype=x.dtype)  
          
    if key_vectors is not None:    
        metric = key_vectors / key_vectors.norm(dim=-1, keepdim=True)    
    else:    
        metric = x / x.norm(dim=-1, keepdim=True)    
        
    a, b = metric[..., ::2, :], metric[..., 1::2, :]                 
    scores = a @ b.transpose(-1, -2)                            
          
    node_max, node_idx = scores.max(dim=-1)              
    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]                 
          
    unm_idx = edge_idx[..., r:, :]                  
    src_idx = edge_idx[..., :r, :]                     
    dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)                     
      
    src_tokens, dst_tokens = x[..., ::2, :], x[..., 1::2, :]                 
    src_size, dst_size = size[..., ::2, :], size[..., 1::2, :]                 
      
    n, t1, c = src_tokens.shape  
      
    unm = src_tokens.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))  
    unm_size = src_size.gather(dim=-2, index=unm_idx.expand(n, t1 - r, 1))  
      
    src = src_tokens.gather(dim=-2, index=src_idx.expand(n, r, c))  
    src_s = src_size.gather(dim=-2, index=src_idx.expand(n, r, 1))  
      
    dst_tokens = dst_tokens.scatter_reduce(  
        -2, dst_idx.expand(n, r, c), src * src_s, reduce="sum"  
    )  
    dst_size = dst_size.scatter_reduce(  
        -2, dst_idx.expand(n, r, 1), src_s, reduce="sum"  
    )  
      
    dst_tokens = dst_tokens / dst_size  
      
    result = torch.cat([unm, dst_tokens], dim=1)  
          
    return result[:, :num_tokens_out]
        
def enhance_image_with_text(txt_feat, image_features):
    """
    用文本特征增强图像特征（Text-guided Image Feature Enhancement）

    参数:
        txt_feat: torch.Tensor, [B, T, D]  文本token特征
        image_features: torch.Tensor, [B, P, D] 图像patch特征

    返回:
        enhanced_image_features: torch.Tensor, [B, P, D]
    """

    txt_feat_norm = F.normalize(txt_feat, p=2, dim=-1)                    
    img_feat_norm = F.normalize(image_features, p=2, dim=-1)              

    similarity = torch.bmm(txt_feat_norm, img_feat_norm.transpose(1, 2))             

    attention_weights_img = F.softmax(similarity.transpose(1, 2), dim=-1)             

    context = torch.bmm(attention_weights_img, txt_feat)             

    enhanced_image_features = image_features + context             

    return enhanced_image_features

class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass
    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    def encode_images(self, images, clip_resize_list,txt_feat=None,return_project=False):
        vision_tower = self.get_model().get_vision_tower()
        vit_attention_mask_for_llm = None
        if isinstance(vision_tower.fast_vision_tower.vision_tower, _CLIPVisionModel):
            vit_attention_mask = torch.zeros_like(images[:, 0, :, :])
            for i, size in enumerate(clip_resize_list):
                vit_attention_mask[i, :size[0], :size[1]] = 1
            with torch.no_grad():
                patch_size = 14
                patch_num = vit_attention_mask.shape[-1] // patch_size
                vit_attention_mask = F.interpolate(vit_attention_mask[:, None].float(), size=(patch_num, patch_num), mode="nearest")[:, 0]
                vit_attention_mask = vit_attention_mask.to(images.dtype)
                vit_attention_mask_for_llm = F.interpolate(vit_attention_mask[:, None].float(), size=(16,16), mode="nearest")[:, 0]
                vit_attention_mask_for_llm = vit_attention_mask_for_llm.to(images.dtype)
               
                vit_attention_mask_for_llm = vit_attention_mask_for_llm.flatten(1)
               
                flatten_vit_attention_mask = vit_attention_mask.flatten(1)
                
                flatten_vit_attention_mask = torch.cat((torch.ones(flatten_vit_attention_mask.shape[0], 1, dtype=images.dtype, device=images.device), flatten_vit_attention_mask), dim=-1)
            image_features, pre_image_features= vision_tower(images, attention_mask=flatten_vit_attention_mask,output_keys=False)
        else:
            if hasattr(vision_tower, "fast_vision_tower"):
                if clip_resize_list is not None:
                    vit_attention_mask = torch.zeros_like(images[:, 0, :, :])
                    for i, size in enumerate(clip_resize_list):
                        vit_attention_mask[i, :size[0], :size[1]] = 1
                    with torch.no_grad():
                        patch_size = 14
                        patch_num = vit_attention_mask.shape[-1] // patch_size
                        vit_attention_mask = F.interpolate(
                            vit_attention_mask[:, None].float(),
                            size=(patch_num, patch_num),
                            mode="nearest",
                        )[:, 0]
                        vit_attention_mask = vit_attention_mask.to(images.dtype)
                        flatten_vit_attention_mask = vit_attention_mask.flatten(1)
                        flatten_vit_attention_mask = torch.cat(
                            (
                                torch.ones(
                                    flatten_vit_attention_mask.shape[0],
                                    1,
                                    dtype=images.dtype,
                                    device=images.device,
                                ),
                                flatten_vit_attention_mask,
                            ),
                            dim=-1,
                        )
                    image_features, pre_image_features = vision_tower(
                        images, attention_mask=flatten_vit_attention_mask, output_keys=False
                    )
                else:
                    batch_size = images.shape[0]
                    flatten_vit_attention_mask = torch.ones(
                        batch_size, 1025, dtype=images.dtype, device=images.device
                    )
                    image_features, pre_image_features = vision_tower(
                        images, attention_mask=flatten_vit_attention_mask, output_keys=False
                    )
            else:
                image_features, pre_image_features = self.get_model().get_vision_tower()(images)

        if return_project:
            pre_image_features = [self.get_model().out_mm_projector(f) if self.get_model().separate_mm_projector else self.get_model().mm_projector(f) for f in pre_image_features]
            output_image_features = [self.get_model().out_mm_projector(image_features) if self.get_model().separate_mm_projector else self.get_model().mm_projector(image_features)]
            output_image_features.extend(pre_image_features)
            pre_image_features = output_image_features
        return image_features, vit_attention_mask_for_llm, pre_image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, clip_resize_list,txt_feat
    ):
        vision_tower = self.get_vision_tower()
        vit_attention_mask = None
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (input_ids.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
            return None, input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            assert False
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features, vit_attention_mask, pre_image_features = self.encode_images(images, clip_resize_list)
            attention_mask = torch.ones_like(input_ids).detach() if attention_mask is None else attention_mask
           
            
        pre_image_features = [self.get_model().out_mm_projector(f) if self.get_model().separate_mm_projector else self.get_model().mm_projector(f) for f in pre_image_features]
        output_image_features = [self.get_model().out_mm_projector(image_features) if self.get_model().separate_mm_projector else self.get_model().mm_projector(image_features)]
        output_image_features.extend(pre_image_features)
        n, l, c = image_features.shape
        p_num = int(l ** 0.5)
        image_features = F.interpolate(image_features.permute(0, 2, 1).view(n, c, p_num, p_num).float(), size=(16,16),  mode="bilinear",align_corners=False).to(image_features)
        image_features = image_features.flatten(-2).permute(0, 2, 1)
        image_features = self.get_model().mm_projector(image_features)

        vit_attention_mask = torch.ones_like(image_features[:,:,0]).detach() if vit_attention_mask is None else vit_attention_mask
        self._last_visual_token_num = image_features.shape[1]-1      
        vit_attention_mask = vit_attention_mask.flatten(1)     
        image_features = enhance_image_with_text(txt_feat, image_features)
        
        new_input_embeds = []
        new_attention_mask = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = (
                    cur_input_embeds
                    + (
                        0.0 * self.get_model().mm_projector(vision_tower.dummy_feature)
                    ).sum()
                )
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            cur_new_attention_mask = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                cur_attention_mask = attention_mask[cur_image_idx]
                cur_vit_attention_mask = vit_attention_mask[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False):
                    assert False
                    cur_new_input_embeds.append(
                        self.get_model()
                        .embed_tokens(cur_input_ids[: image_token_start - 1])
                        .detach()
                    )
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start - 1 : image_token_start]
                        )
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start : image_token_start + 1]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_attention_mask.append(cur_attention_mask[:image_token_start])
                    cur_new_attention_mask.append(cur_vit_attention_mask)
                    cur_new_attention_mask.append(cur_attention_mask[image_token_start + 1 : image_token_start + 2])
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start + 1 : image_token_start + 2]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_attention_mask.append(
                        cur_attention_mask[:image_token_start]
                    )
                    cur_new_attention_mask.append(
                        cur_vit_attention_mask
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_labels = cur_labels[image_token_start + 1 :]
                cur_image_idx += 1
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                    cur_attention_mask = cur_attention_mask[image_token_start + 2 :]
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                    cur_attention_mask = cur_attention_mask[image_token_start + 2 :]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1 :]
                    cur_attention_mask = cur_attention_mask[image_token_start + 1 :]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids).detach()
                    )
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids)
                    )
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids)
                    )
                cur_new_attention_mask.append(cur_attention_mask)
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [
                x.to(device=self.device) for x in cur_new_input_embeds
            ]
            cur_new_attention_mask = torch.cat(cur_new_attention_mask, dim=0).bool()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            new_attention_mask.append(cur_new_attention_mask)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):                                    
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_label.dtype,
                                device=cur_new_label.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                    attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat(
                        (
                            new_attn_mask_pad_left,
                            cur_attention_mask,
                            new_attn_mask_pad_right,
                        ),
                        dim=0,
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            new_attention_mask = torch.stack(new_attention_mask, dim=0)
            attention_mask = new_attention_mask
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None and attention_mask.shape[1] < new_input_embeds.shape[1]:
                new_attn_mask_pad_left = torch.full(
                    (
                        attention_mask.shape[0],
                        new_input_embeds.shape[1] - input_ids.shape[1],
                    ),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask), dim=1
                )
                assert attention_mask.shape == new_input_embeds.shape[:2]
        return output_image_features, None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, num_new_tokens):
        if model_args.mm_use_im_start_end:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False


def visualize_attn_mask(mask):
    import cv2
    import numpy as np
    mask = mask[0].squeeze().float()
    fg = mask >= 0
    mask_show = torch.zeros_like(mask)
    mask_show[fg] = 255
    mask_show = mask_show.cpu().numpy()
    cv2.imwrite('test.jpg', mask_show.astype(np.uint8))