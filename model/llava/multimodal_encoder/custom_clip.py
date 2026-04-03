from transformers import GenerationMixin, GenerationConfig, PretrainedConfig, CLIPVisionModel
from transformers.models.clip.modeling_clip import CLIPVisionTransformer
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from typing import Optional, Tuple, Union
from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
import torch
import torch.nn.functional as F
import math  
from dataclasses import dataclass
from transformers.models.clip.modeling_clip import (  
    CLIPEncoder,   
    CLIPEncoderLayer,  
    CLIPAttention,  
    CLIPMLP  
)
@dataclass
class BaseModelOutputWithPoolingAndKeys(BaseModelOutputWithPooling):
    keys: Optional[Tuple[torch.FloatTensor]] = None

class _CLIPAttention(CLIPAttention):  
    def forward(  
        self,  
        hidden_states: torch.Tensor,  
        attention_mask: Optional[torch.Tensor] = None,  
        causal_attention_mask: Optional[torch.Tensor] = None,  
        output_attentions: Optional[bool] = False,  
        output_keys: Optional[bool] = False,  
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:  
        """自定义 CLIP Attention,支持返回 key 向量"""  
          
        bsz, tgt_len, embed_dim = hidden_states.size()  
  
        query_states = self.q_proj(hidden_states)  
        key_states = self.k_proj(hidden_states)  
        value_states = self.v_proj(hidden_states)  
  
        query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)  
        key_states = key_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)  
        value_states = value_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)  
  
        output_key_states = None  
        if output_keys:  
            output_key_states = key_states.detach()  
  
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))  
          
        if attn_weights.size() != (bsz, self.num_heads, tgt_len, tgt_len):  
            raise ValueError(  
                f"Attention weights should be of size {(bsz, self.num_heads, tgt_len, tgt_len)}, but is"  
                f" {attn_weights.size()}"  
            )  
  
        attn_weights = attn_weights / math.sqrt(self.head_dim)  
  
        if causal_attention_mask is not None:  
            if causal_attention_mask.size() != (bsz, 1, tgt_len, tgt_len):  
                raise ValueError(  
                    f"Attention mask should be of size {(bsz, 1, tgt_len, tgt_len)}, but is"  
                    f" {causal_attention_mask.size()}"  
                )  
            attn_weights = attn_weights + causal_attention_mask  
  
        if attention_mask is not None:  
            if attention_mask.size() != (bsz, 1, tgt_len, tgt_len):  
                raise ValueError(  
                    f"Attention mask should be of size {(bsz, 1, tgt_len, tgt_len)}, but is"  
                    f" {attention_mask.size()}"  
                )  
            attn_weights = attn_weights + attention_mask  
  
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)  
  
        attn_output = torch.matmul(attn_weights, value_states)  
  
        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):  
            raise ValueError(  
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"  
                f" {attn_output.size()}"  
            )  
  
        attn_output = attn_output.transpose(1, 2).contiguous()  
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)  
  
        attn_output = self.out_proj(attn_output)  
  
        return attn_output, attn_weights if output_attentions else None, output_key_states

class _CLIPEncoderLayer(CLIPEncoderLayer):  
    def __init__(self, config):  
        super().__init__(config)  
        self.self_attn = _CLIPAttention(config)  
      
    def forward(  
        self,  
        hidden_states: torch.Tensor,  
        attention_mask: torch.Tensor,  
        causal_attention_mask: torch.Tensor,  
        output_attentions: Optional[bool] = False,  
        output_keys: Optional[bool] = False,  
    ) -> Tuple[torch.FloatTensor]:  
        """  
        Args:  
            hidden_states: [bsz, seq_len, embed_dim]  
            attention_mask: [bsz, 1, tgt_len, src_len]  
            output_attentions: 是否返回 attention 权重  
            output_keys: 是否返回 key 向量  
        """  
        residual = hidden_states  
  
        hidden_states = self.layer_norm1(hidden_states)  
        hidden_states, attn_weights, key_states = self.self_attn(  
            hidden_states=hidden_states,  
            attention_mask=attention_mask,  
            causal_attention_mask=causal_attention_mask,  
            output_attentions=output_attentions,  
            output_keys=output_keys,  
        )  
        hidden_states = residual + hidden_states  
  
        residual = hidden_states  
        hidden_states = self.layer_norm2(hidden_states)  
        hidden_states = self.mlp(hidden_states)  
        hidden_states = residual + hidden_states  
  
        outputs = (hidden_states,)  
  
        if output_attentions:  
            outputs += (attn_weights,)  
  
        if output_keys:  
            outputs += (key_states,)  
  
        return outputs


class _CLIPEncoder(CLIPEncoder):  
    def __init__(self, config):  
        super().__init__(config)  
        self.layers = nn.ModuleList([_CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])  
      
    def forward(  
        self,  
        inputs_embeds,  
        attention_mask: Optional[torch.Tensor] = None,  
        causal_attention_mask: Optional[torch.Tensor] = None,  
        output_attentions: Optional[bool] = None,  
        output_hidden_states: Optional[bool] = None,  
        return_dict: Optional[bool] = None,  
        output_keys: Optional[bool] = False,  
    ):  
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions  
        output_hidden_states = (  
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states  
        )  
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  
  
        encoder_states = () if output_hidden_states else None  
        all_attentions = () if output_attentions else None  
        all_keys = () if output_keys else None  
  
        hidden_states = inputs_embeds  
        for idx, encoder_layer in enumerate(self.layers):  
            if output_hidden_states:  
                encoder_states = encoder_states + (hidden_states,)  
  
            layer_outputs = encoder_layer(  
                hidden_states,  
                attention_mask,  
                causal_attention_mask,  
                output_attentions=output_attentions,  
                output_keys=output_keys,  
            )  
  
            hidden_states = layer_outputs[0]  
  
            if output_attentions:  
                all_attentions = all_attentions + (layer_outputs[1],)  
  
            if output_keys:  
                key_idx = 2 if output_attentions else 1  
                all_keys = all_keys + (layer_outputs[key_idx],)  
  
        if output_hidden_states:  
            encoder_states = encoder_states + (hidden_states,)  
  
        if not return_dict:  
            outputs = (hidden_states,)  
            if output_hidden_states:  
                outputs += (encoder_states,)  
            if output_attentions:  
                outputs += (all_attentions,)  
            if output_keys:  
                outputs += (all_keys,)  
            return outputs  
  
        return {  
            'last_hidden_state': hidden_states,  
            'hidden_states': encoder_states,  
            'attentions': all_attentions,  
            'keys': all_keys,  
        }

CLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def visualize_attn_mask(mask):
    import cv2
    import numpy as np
    mask = mask[0].squeeze().float()
    fg = mask >= 0
    mask_show = torch.zeros_like(mask)
    mask_show[fg] = 255
    mask_show = mask_show.cpu().numpy()
    cv2.imwrite('test.jpg', mask_show.astype(np.uint8))

class _CLIPVisionTransformer(CLIPVisionTransformer):
    def __init__(self, config: CLIPVisionConfig):  
        super().__init__(config)  
        self.encoder = _CLIPEncoder(config)  

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_keys: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)
        encoder_outputs = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            output_keys=output_keys,  
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:  
            outputs = (last_hidden_state, pooled_output)  
            if output_hidden_states:  
                outputs += (encoder_outputs['hidden_states'],)  
            if output_attentions:  
                outputs += (encoder_outputs['attentions'],)  
            if output_keys:  
                outputs += (encoder_outputs['keys'],)  
            return outputs  
        return BaseModelOutputWithPoolingAndKeys(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs['hidden_states'],
            attentions=encoder_outputs['attentions'],
            keys=encoder_outputs['keys'] if output_keys else None
        )

class _CLIPVisionModel(CLIPVisionModel):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = _CLIPVisionTransformer(config)
        self.post_init()
    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_keys: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPVisionModel

        >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attention_mask=attention_mask,
            output_keys=output_keys 
        )
