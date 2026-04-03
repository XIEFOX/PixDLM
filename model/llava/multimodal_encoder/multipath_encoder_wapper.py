import torch  
import torch.nn as nn  
from torch.utils.checkpoint import checkpoint  
from .clip_encoder import CLIPVisionTower  
import torch.nn.functional as F  
from torch.nn.init import trunc_normal_  
from copy import deepcopy  
import random  
import math  
import sys  
from omegaconf import OmegaConf  
from hydra.utils import instantiate  
from .custom_clip import _expand_mask
  
class MultiPathAlignModule(nn.Module):  
    def __init__(self, fast_vision_dim, slow_vision_dim,pretrained_weights=None, prefix=""):  
        super().__init__()  
  
        self.fast_proj = nn.Linear(fast_vision_dim, fast_vision_dim)  
        self.slow_proj = nn.Linear(slow_vision_dim, fast_vision_dim)  
        self.load_pretrained_weights(pretrained_weights, prefix)

    def load_pretrained_weights(self, weights_dict, prefix=""):
        fast_proj_weight_key = f"{prefix}fast_proj.weight"
        fast_proj_bias_key   = f"{prefix}fast_proj.bias"
        slow_proj_weight_key = f"{prefix}slow_proj.weight"
        slow_proj_bias_key   = f"{prefix}slow_proj.bias"

        if fast_proj_weight_key in weights_dict:
            self.fast_proj.weight.data.copy_(weights_dict[fast_proj_weight_key].to(self.fast_proj.weight.dtype))
            print(f"加载了 {fast_proj_weight_key}")
        if fast_proj_bias_key in weights_dict:
            self.fast_proj.bias.data.copy_(weights_dict[fast_proj_bias_key].to(self.fast_proj.bias.dtype))
            print(f"加载了 {fast_proj_bias_key}")
        if slow_proj_weight_key in weights_dict:
            self.slow_proj.weight.data.copy_(weights_dict[slow_proj_weight_key].to(self.slow_proj.weight.dtype))
            print(f"加载了 {slow_proj_weight_key}")
        if slow_proj_bias_key in weights_dict:
            self.slow_proj.bias.data.copy_(weights_dict[slow_proj_bias_key].to(self.slow_proj.bias.dtype))
            print(f"加载了 {slow_proj_bias_key}")
  
    def forward(self, fast_feat, slow_feat): 
        #修改，这里也写死了
        target_dtype = torch.bfloat16
        if fast_feat.dtype != target_dtype:
            fast_feat = fast_feat.to(target_dtype)
        if slow_feat.dtype != target_dtype:
            slow_feat = slow_feat.to(target_dtype)
    
        if slow_feat.ndim == 4:  
            b, c, h, w = slow_feat.shape  
            slow_feat = slow_feat.view(b, c, -1).transpose(1, 2)  
        assert slow_feat.shape[1] % fast_feat.shape[1] == 0 or fast_feat.shape[1] % slow_feat.shape[1] == 0  
        if slow_feat.shape[1] < fast_feat.shape[1]:  
            # upsample  
            b, l, c = slow_feat.shape  
            src_size = int(math.sqrt(l))  
            dst_size = int(math.sqrt(fast_feat.shape[1]))  
            slow_feat = slow_feat.transpose(1, 2).view(b, c, src_size, src_size)  
            slow_feat = F.interpolate(slow_feat.float(), size=(dst_size, dst_size), mode='bilinear',  
                                      align_corners=True).to(dtype=slow_feat.dtype)  
            slow_feat = slow_feat.view(b, c, -1).transpose(1, 2)  
        elif slow_feat.shape[1] > fast_feat.shape[1]:  
            # pooling  
            b, l, c = slow_feat.shape  
            src_size = int(math.sqrt(l))  
            dst_size = int(math.sqrt(fast_feat.shape[1]))  
            slow_feat = slow_feat.transpose(1, 2).view(b, c, src_size, src_size)  
            slow_feat = F.avg_pool2d(slow_feat, src_size // dst_size, src_size // dst_size)  
            slow_feat = slow_feat.view(b, c, -1).transpose(1, 2)  
        patch_feat = self.fast_proj(fast_feat) + self.slow_proj(slow_feat) 
        # print("patch_feat :",patch_feat.shape) 
        return patch_feat  
  
  
class S2FStitchAlignModuleV2(nn.Module):  
    def __init__(self, fast_vision_dim, slow_vision_dim, zero_init=True):  
        super().__init__()  
  
        self.slow_conv = nn.Conv2d(slow_vision_dim, slow_vision_dim, 1)  
        self.slow_proj = nn.Conv2d(slow_vision_dim, fast_vision_dim, 1)  
  
        self.fast_conv = nn.Conv2d(fast_vision_dim, fast_vision_dim, 7, padding=3, groups=fast_vision_dim)  
        self.fast_proj = nn.Conv2d(fast_vision_dim, fast_vision_dim, 1)  
  
        self.gate = nn.Sequential(  
            nn.Linear(fast_vision_dim*2, fast_vision_dim//2),  
            nn.GELU(),  
            nn.Linear(fast_vision_dim//2, 1) )  
  
        nn.init.xavier_uniform_(self.slow_conv.weight)  
        nn.init.xavier_uniform_(self.fast_conv.weight)  
        nn.init.zeros_(self.slow_conv.bias)  
        nn.init.zeros_(self.fast_conv.bias)  
        if zero_init:  
            nn.init.zeros_(self.slow_proj.weight)  
            nn.init.zeros_(self.fast_proj.weight)  
        else:  
            nn.init.xavier_uniform_(self.slow_proj.weight)  
            nn.init.xavier_uniform_(self.fast_proj.weight)  
        nn.init.zeros_(self.slow_proj.bias)  
        nn.init.zeros_(self.fast_proj.bias)  
    def load_pretrained_weights(self, weights_dict, prefix=""):
        for name, param in self.named_parameters():
            full_key = prefix + name
            if full_key in weights_dict:
                param.data.copy_(weights_dict[full_key].to(param.dtype))
                print(f"加载了 {full_key}")
  
    def src2dst_align(self, src_feat, dst_feat):  
        dst_size = int(math.sqrt(dst_feat.shape[1]))  
        assert src_feat.shape[1] % dst_feat.shape[1] == 0 or dst_feat.shape[1] % src_feat.shape[1] == 0  
        if src_feat.shape[1] < dst_feat.shape[1]:  
            # upsample  
            b, l, c = src_feat.shape  
            src_size = int(math.sqrt(l))  
            dst_size = int(math.sqrt(dst_feat.shape[1]))  
            src_feat = src_feat.transpose(1, 2).view(b, c, src_size, src_size)  
            src_feat = F.interpolate(src_feat.float(), size=(dst_size, dst_size), mode='bilinear',  
                                     align_corners=True).to(dtype=src_feat.dtype)  
            src_feat = src_feat.view(b, c, -1).transpose(1, 2)  
        elif src_feat.shape[1] > dst_feat.shape[1]:  
            # pooling  
            b, l, c = src_feat.shape  
            src_size = int(math.sqrt(l))  
            dst_size = int(math.sqrt(dst_feat.shape[1]))  
            src_feat = src_feat.transpose(1, 2).view(b, c, src_size, src_size)  
            src_feat = F.avg_pool2d(src_feat, src_size // dst_size, src_size // dst_size)  
            src_feat = src_feat.view(b, c, -1).transpose(1, 2)  
        return src_feat, dst_size  
  
    def forward(self, fast_feat, slow_feat):  
        b, c, h, w = slow_feat.shape  
        _, _, d = fast_feat.shape  
        slow_feat = self.slow_proj(F.gelu(self.slow_conv(slow_feat)))  
        slow_feat = slow_feat.view(b, d, -1).transpose(1, 2)  
        slow_feat_align, dst_size = self.src2dst_align(slow_feat, fast_feat)  
        fast_feat = fast_feat.transpose(1, 2).view(b, d, dst_size, dst_size)  
        fast_feat = fast_feat + self.fast_proj(F.gelu(self.fast_conv(fast_feat)))  
        fast_feat = fast_feat.view(b, d, dst_size * dst_size).transpose(1, 2)  
        gate=self.gate(torch.cat([fast_feat,slow_feat_align],-1).mean(1)).unsqueeze(1)  
        fast_feat = fast_feat + slow_feat_align *gate.tanh()  
        return fast_feat  
  
  
class MultiPathCLIPVisionTower(nn.Module):  
    def __init__(self, vision_tower, args, delay_load=False):  
        super().__init__()  
  
        self.is_loaded = False  
  
        # 使用Hydra方式加载SAM2 encoder替换慢速分支  
        sys.path.append("path_to_PixDLM")  
        cfg = OmegaConf.load("path_to_sam2.1_hiera_l.yaml")  
        model = instantiate(cfg.model)  
        ckpt = torch.load("path_to_sam2.1_hiera_large.pt", map_location="cpu") 
        state_dict = ckpt["model"]
        model.load_state_dict(state_dict, strict=False)  #, assign=True
        self.slow_vision_tower = model.image_encoder  
        print("initial fast vision tower")
       
          
        # 快速分支保持CLIP不变  
        args_ = deepcopy(args) 
        # 原来是336 
        args_.input_image_size = 448  
        self.fast_vision_tower = CLIPVisionTower(vision_tower, args_, delay_load=delay_load)  
        print("initial slow vision tower")
      
        self.load_model()  
  
        self.vision_tower_name = vision_tower  
        self.select_layer = args.mm_vision_select_layer  
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')  
        self.splits = self.select_layer // 100 if self.select_layer > 100 else 1  
        self.enable_adapter = not args.freeze_vision  
        print("self.enable_adapter:",self.enable_adapter)
        #暂时没传
        self.image_size = 800  
  
        # SAM2的hidden_size是256（来自neck的d_model）  
        sam2_hidden_size = 256  
  
        if self.enable_adapter:  
            self.align_stages_latent = nn.ModuleList([S2FStitchAlignModuleV2(self.fast_vision_tower.hidden_size,  
                                                                             sam2_hidden_size,  
                                                                             True)  
                                                      for i in range(3)])  
     
        weights_dict = torch.load("path_to_1012.pth", map_location="cpu")
        self.align_stages = nn.ModuleList([MultiPathAlignModule(self.fast_vision_tower.hidden_size,  
                                                                sam2_hidden_size,pretrained_weights=weights_dict,prefix="base_model.model.model.vision_tower.align_stages.0."
                                                                )  
                                           ])  
        for i in range(3): 
            self.align_stages_latent[i].load_pretrained_weights(
                weights_dict,
                prefix=f"base_model.model.model.vision_tower.align_stages_latent.{i}."
            )
    def load_model(self):  
        # SAM2 encoder已经在初始化时加载  
        self.fast_vision_tower.load_model()  
        self.image_processor = self.fast_vision_tower.image_processor  # 使用CLIP的预处理器  
        self.is_loaded = True  
  
    def forward(self, x,attention_mask=None,output_attentions=False,output_keys=False):  #尺寸相同
        # 快速分支预处理  
        fast_image_size = 448
        y = F.interpolate(x.float(), size=(fast_image_size, fast_image_size), mode='bilinear', align_corners=True).to(dtype=x.dtype)  
        y = self.fast_vision_tower.vision_tower.vision_model.embeddings(y)  
        # print("y1:",y.shape)
        y = self.fast_vision_tower.vision_tower.vision_model.pre_layrnorm(y[:, 1:]) 
        # print("y2:",y.shape) 
        
        # SAM2慢速分支处理  
        slow_image_size = 1024  # 或者你想要的其他尺寸  
        x_resized = F.interpolate(x.float(), size=(slow_image_size, slow_image_size),   
                                mode='bilinear', align_corners=True).to(dtype=x.dtype)  
        # print("x_resized:",x_resized.shape)
        with torch.no_grad():  
            sam_backbone_out = self.slow_vision_tower(x_resized)  
        sam_features = sam_backbone_out["vision_features"]  # [B, C, H, W] 
        # print("sam_features:",sam_features.shape) 
        #修改 # 你有 latent
        sam_features = sam_features.to(torch.bfloat16)

        if attention_mask.shape[-1] == 1025:  
            attention_mask = attention_mask[:, 1:]  # 变为 [1, 1024]  
            # 使用 _expand_mask 函数进行维度扩展  
        expanded_mask = _expand_mask(attention_mask, attention_mask.dtype, tgt_len=1024)
        
  
        # 快速分支的分阶段处理  
        fast_blk = self.fast_vision_tower.vision_tower.vision_model.encoder.layers  
        n_blks = len(fast_blk) // 4  
        assert len(fast_blk) == n_blks * 4  
  
        # 第一阶段  
        for blk in fast_blk[:n_blks]:  
            if self.training:  
                y = checkpoint(blk.__call__, y,expanded_mask, None)[0]  
            else:  
                y = blk(y, expanded_mask, None)[0]  
        if self.enable_adapter:  
            y = self.align_stages_latent[0](y, sam_features)  
  
        # 第二阶段  
        for blk in fast_blk[n_blks:2 * n_blks]:  
            if self.training:  
                y = checkpoint(blk.__call__, y, expanded_mask, None)[0]  
            else:  
                y = blk(y, expanded_mask, None)[0]  
        if self.enable_adapter:  
            # print("没有走哦")
            y = self.align_stages_latent[1](y, sam_features)  
  
        # 第三阶段  
        for blk in fast_blk[2 * n_blks:3 * n_blks]:  
            if self.training:  
                y = checkpoint(blk.__call__, y, expanded_mask, None)[0]  
            else:  
                y = blk(y, expanded_mask, None)[0]  
        if self.enable_adapter:  
            y = self.align_stages_latent[2](y, sam_features)  
        
        last_blk_idx = len(fast_blk[3 * n_blks:]) - 1  
        last_attention = None 
        last_keys = None
        # 第四阶段  
        for i, blk in enumerate(fast_blk[3 * n_blks:]):  
            if self.training:  
                if i == last_blk_idx:  
                    # 最后一个 block,获取 attention  
                    # 部分 CLIP layer 不支持 output_keys；仅在可用时请求
                    try:
                        outputs = blk(
                            y, expanded_mask, None, output_attentions=False, output_keys=output_keys
                        )
                    except TypeError:
                        outputs = blk(y, expanded_mask, None, output_attentions=False)
                    y = outputs[0]
                    last_attention = outputs[1] if len(outputs) > 1 else None
                    last_keys = outputs[-1] if output_keys and len(outputs) > 1 else None
                else:  
                    y = checkpoint(blk.__call__, y, expanded_mask, None)[0]  
            else:
                if i == last_blk_idx:  
                    # 最后一个 block,获取 attention  
                    try:
                        outputs = blk(
                            y, expanded_mask, None, output_attentions=False, output_keys=output_keys
                        )
                    except TypeError:
                        outputs = blk(y, expanded_mask, None, output_attentions=False)
                    y = outputs[0]
                    last_attention = outputs[1] if len(outputs) > 1 else None
                    last_keys = outputs[-1] if output_keys and len(outputs) > 1 else None
                else:  
                    y = blk(y, expanded_mask, None)[0]  
  
        # 最终特征融合  
        y = self.align_stages[0](y, sam_features)  
        if last_keys is not None:  
            # 对所有 heads 求平均: [B, num_heads, N, head_dim] -> [B, N, head_dim]  
            last_keys = last_keys.mean(dim=1)  
        #修改
        # return y  
        
        if last_attention is not None:
            last_attention = last_attention.mean(dim=1)
        #修改少返回一点
        if output_keys:
            return y, [y],last_keys
        else:
            return y, [y]
  
    def forward_sam_multilayer_features(self, x):
        """
        专门用于提取 SAM2 encoder 的多层特征，作为 fimg 特征送入下游 decoder。
        最多返回 4 层特征（约对应 256x256, 128x128, 64x64, 32x32），每层通道统一为 256。
        """
        slow_image_size = 1024
        x_resized = F.interpolate(
            x.float(),
            size=(slow_image_size, slow_image_size),
            mode="bilinear",
            align_corners=True,
        ).to(dtype=x.dtype)

        with torch.no_grad():
            sam_backbone_out = self.slow_vision_tower(x_resized)

        backbone_fpn = sam_backbone_out.get("backbone_fpn", None)
        if backbone_fpn is not None and len(backbone_fpn) >= 1:
            # backbone_fpn[0]: (B,  144, 256, 256) - 最高分辨率
            # backbone_fpn[1]: (B,  288, 128, 128)
            # backbone_fpn[2]: (B,  576,  64,  64)
            # backbone_fpn[3]: (B, 1152,  32,  32) - 最低分辨率
            # neck.convs 按通道从低分辨率到高分辨率构建：
            #   convs[0] ← 1152 → backbone_fpn[3]
            #   convs[1] ←  576 → backbone_fpn[2]
            #   convs[2] ←  288 → backbone_fpn[1]
            #   convs[3] ←  144 → backbone_fpn[0]
            neck = self.slow_vision_tower.neck
            max_layers = min(len(backbone_fpn), len(neck.convs))
            selected_backbone_indices = list(range(max_layers))

            processed_features = []
            for backbone_idx in selected_backbone_indices:
                conv_idx = len(neck.convs) - 1 - backbone_idx
                if 0 <= conv_idx < len(neck.convs) and backbone_idx < len(backbone_fpn):
                    backbone_feat = backbone_fpn[backbone_idx]
                    conv_layer = neck.convs[conv_idx]

                    # 如果 conv_layer 内部还有 conv 子模块，检查通道是否匹配
                    if hasattr(conv_layer, "conv"):
                        conv = conv_layer.conv
                        expected_in_channels = conv.in_channels
                        actual_channels = backbone_feat.shape[1]
                        if actual_channels != expected_in_channels:
                            if hasattr(self, "local_rank") and getattr(self, "local_rank", 0) == 0:
                                print(
                                    f"Error: backbone_fpn[{backbone_idx}] has {actual_channels} channels, "
                                    f"but convs[{conv_idx}] expects {expected_in_channels} channels. "
                                    f"backbone_fpn shape: {backbone_feat.shape}"
                                )
                            continue

                    processed_feat = conv_layer(backbone_feat)
                    processed_features.append(processed_feat)

            if len(processed_features) > 0:
                return processed_features

        # 如果无法从 backbone_fpn 中提取到有效多层特征，退回到单层 vision_features，
        # 并复制成若干层，保持与 neck.convs 或 4 层中的较小值一致。
        sam_features = sam_backbone_out["vision_features"]
        fallback_layers = min(len(self.slow_vision_tower.neck.convs), 4)
        return [sam_features for _ in range(fallback_layers)]
  
    def forward_features(self, x):  
        raise NotImplementedError  
  
    @property  
    def dummy_feature(self):  
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)  
  
    @property  
    def dtype(self):  
        return next(self.fast_vision_tower.parameters()).dtype  
  
    @property  
    def device(self):  
        return next(self.fast_vision_tower.parameters()).device  
  
    @property  
    def config(self):  
        raise NotImplementedError  
  
    @property  
    def hidden_size(self):  
        return self.fast_vision_tower.hidden_size  
  
    @property  
    def num_patches(self):  
        return self.fast_vision_tower.num_patches