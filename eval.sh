localhost=0
deepspeed --master_port=29501 --include=localhost:$localhost eval.py \
  --version="path_to_your_model" \
  --dataset_dir='path_to_DRSeg' \
  --dataset="custom_seg" \
  --sample_rates="1" \
  --exp_name="your_exp_name" \
  --val_dataset="custom_seg|test" \
  --train_mask_decoder \
  --Three_Level_Multi_Scale_Decoder \
  --vision-tower='path_to_clip-vit-large-patch14' \
  --seg_token_num=3 \
  --num_classes_per_question=3 \
  --batch_size=2 \
  --preprocessor_config='path_to_preprocessor_448.json' \
  --resize_vision_tower \
  --resize_vision_tower_size=448 \
  --vision_tower_for_mask \
  --use_expand_question_list \
  --image_feature_scale_num=3 \
  --conv_type="llava_v1" \
  --is_multipath_encoder \
  --eval_only \
  > v137test.log 2>&1