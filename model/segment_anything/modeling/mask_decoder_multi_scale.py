                                                    
                      

                                                             
                                                         

from typing import List, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F

from .common import LayerNorm2d
from .prompt_encoder import PositionEmbeddingRandom
from copy import deepcopy
class MaskDecoderMultiScale(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        image_feature_scale_num: int = 1,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = nn.ModuleList([deepcopy(transformer) for _ in range(image_feature_scale_num)])

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

                          
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 8, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 8),
            activation(),
        )

        self.upsample_2x = nn.Sequential(
                    nn.ConvTranspose2d(
                        transformer_dim, transformer_dim, kernel_size=2, stride=2),
                        LayerNorm2d(transformer_dim),
                        activation(),)


        self.pe1=PositionEmbeddingRandom(transformer_dim//2)


        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )
        
        self.image_feature_scale_num = image_feature_scale_num
        self.level_embed = nn.Embedding(image_feature_scale_num, transformer_dim)
                                          
                                                
                                                
            


    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        level_num: int,
        previous_masks=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            level_num=level_num,
            previous_masks=previous_masks
        )

                                                     
        if multimask_output:
            mask_slice = slice(0, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

                        
        return masks, iou_pred

class Three_Level_Multi_Scale_Decoder(MaskDecoderMultiScale):  
    """  
    Three-level multi-scale decoder.  
    修复了张量尺寸不匹配和硬编码通道数的问题  
    """  
  
    def __init__(  
        self,  
        *,  
        transformer_dim: int,  
        transformer: nn.Module,  
        num_multimask_outputs: int = 3,  
        activation: Type[nn.Module] = nn.GELU,  
        iou_head_depth: int = 3,  
        iou_head_hidden_dim: int = 256,  
        image_feature_scale_num: int = 3,  
    ) -> None:  
        if image_feature_scale_num != 3:  
            raise ValueError("Three_Level_Multi_Scale_Decoder 只支持恰好3个尺度")  
        super().__init__(  
            transformer_dim=transformer_dim,  
            transformer=transformer,  
            num_multimask_outputs=num_multimask_outputs,  
            activation=activation,  
            iou_head_depth=iou_head_depth,  
            iou_head_hidden_dim=iou_head_hidden_dim,  
            image_feature_scale_num=image_feature_scale_num,  
        )  
  
    def predict_masks(  
        self,  
        image_embeddings: torch.Tensor,  
        image_pe: torch.Tensor,  
        sparse_prompt_embeddings: torch.Tensor,  
        dense_prompt_embeddings: torch.Tensor,  
        level_num: int,  
        previous_masks=None  
    ) -> Tuple[torch.Tensor, torch.Tensor]:  
        """Predicts masks. See 'forward' for more details."""  
                                     
        output_tokens = torch.cat(  
            [self.iou_token.weight, self.mask_tokens.weight], dim=0  
        )  
        output_tokens = output_tokens.unsqueeze(0).expand(  
            sparse_prompt_embeddings.size(0), -1, -1  
        )  
  
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)  
        level = torch.tensor([level_num, ], dtype=torch.long, device=tokens.device).expand((tokens.size(0), 1))  
        level_embed = self.level_embed(level)  
        tokens = tokens + level_embed  
  
                                                                   
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)  
          
                       
        if level_num > 0:  
            src = self.upsample_2x(src)  
            b, c, h, w = src.shape  
              
                                 
            if previous_masks is not None:  
                previous_masks = torch.mean(previous_masks, dim=1)               
                  
                                 
                if previous_masks.dim() == 3:               
                    previous_masks = previous_masks.unsqueeze(1)                  
                  
                                   
                previous_masks = F.interpolate(  
                    previous_masks.float(),   
                    size=(h, w),   
                    mode="bilinear",   
                    align_corners=False  
                ).to(previous_masks)  
                  
                                     
                src = (torch.repeat_interleave(previous_masks, c, dim=1).sigmoid() + 1) * src  
              
                      
            image_pe = self.pe1((h, w)).unsqueeze(0)  
              
                                            
            if dense_prompt_embeddings.shape[-2:] != (h, w):  
                dense_prompt_embeddings = F.interpolate(  
                    dense_prompt_embeddings.float(),   
                    size=(h, w),   
                    mode="bilinear",   
                    align_corners=False  
                ).to(dense_prompt_embeddings)  
          
        src = src + dense_prompt_embeddings  
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)  
        b, c, h, w = src.shape  
  
                               
        hs, src = self.transformer[level_num](src, pos_src, tokens)  
        iou_token_out = hs[:, 0, :]  
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]  
  
                   
        if src.dim() == 3:  
                                       
            src = src.transpose(1, 2)               
            N = src.shape[-1]  
                      
            spatial_size = int(N**0.5)  
            if spatial_size * spatial_size != N:  
                raise ValueError(f"Cannot reshape {src.shape} to 4D spatial format")  
            src = src.view(b, c, spatial_size, spatial_size)  
        elif src.dim() == 4:  
            src = src.transpose(1, 2).view(b, c, h, w)  
        else:  
            raise ValueError(f"Unexpected src dimensions: {src.dim()}")  
  
                                                                           
        upscaled_embedding = self.output_upscaling(src)  
        hyper_in_list: List[torch.Tensor] = []  
        for i in range(self.num_mask_tokens):  
            hyper_in_list.append(  
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])  
            )  
        hyper_in = torch.stack(hyper_in_list, dim=1)  
        b, c, h, w = upscaled_embedding.shape  
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(  
            b, self.num_mask_tokens, h, w  
        )  
  
                                             
        iou_pred = self.iou_prediction_head(iou_token_out)  
  
        return masks, iou_pred


                      
                                                                                                                           
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
