#!/bin/bash
NAME="eagle_finetuned"


python -m torch.distributed.run \
    train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --version v1 \
    --data_path /root/data/asd.json \
    --image_folder /root/data/images \
    --vision_tower "google/deplot;mPLUG/TinyChart-3B-768-siglip;google/matcha-chart2text-pew" \
    --pretrain_mm_mlp_adapter /root/custom_eagle/checkpoints/eagle_pretrained/mm_projector.bin \
    --mm_projector_type cabstractor \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/$NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${NAME}  
