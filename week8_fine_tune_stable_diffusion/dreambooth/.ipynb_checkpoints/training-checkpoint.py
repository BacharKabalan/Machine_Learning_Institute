accelerate config default --mixed_precision fp16
import os
export MODEL_NAME = 'CompVis/stable-diffusion-v1-4'
export INSTANCE_DIR = './bash_images'
export OUTPUT_DIR = './dreambooth/sd_aiconos-model-v1-2_400'


accelerate launch ./diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of brk1" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=600