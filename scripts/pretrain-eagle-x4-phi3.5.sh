#!/bin/bash
NAME="eagle_pretrained"

python -m torch.distributed.run \
    train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --version plain \
    --data_path ./data/pretraining.json \
    --image_folder ./data/images_2 \
    --vision_tower "google/deplot;mPLUG/TinyChart-3B-768-siglip;google/matcha-chart2text-pew" \
    --mm_projector_type cabstractor \
    --resize_type channel_wise_concatenation \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/$NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${NAME}
