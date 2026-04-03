CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version="path_to_llava-v1.6-vicuna-7b" \
  --weight="path_to_pytorch_model.bin" \
  --save_path="" \
  --resize_vision_tower \
  --resize_vision_tower_size=448 \
  --vision_tower_for_mask \
  --Three_Level_Multi_Scale_Decoder
