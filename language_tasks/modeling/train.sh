torchrun --standalone --nproc_per_node=6 \
train_sassha.py \
config/train_gpt2_small_sassha.py \
  --batch_size=10 \
  --gradient_accumulation_steps=8
