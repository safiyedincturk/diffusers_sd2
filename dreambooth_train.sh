export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="/vol/bitbucket/aa9120/crait/crait_diffusers_sd2/diffusers/crait_sd_example_scripts/photos/dreambooth_renault"
export CLASS_DIR="/vol/bitbucket/aa9120/crait/crait_diffusers_sd2/diffusers/crait_sd_example_scripts/photos/dreambooth_renault/class_dir"
export OUTPUT_DIR="/vol/bitbucket/aa9120/crait/crait_diffusers_sd2/diffusers/crait_sd_example_scripts/models/dreambooth_renault_sd1"

accelerate launch /vol/bitbucket/aa9120/crait/crait_diffusers_sd2/diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of craitrenault car" \
  --class_prompt="a photo of car" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=10 \
  --max_train_steps=800