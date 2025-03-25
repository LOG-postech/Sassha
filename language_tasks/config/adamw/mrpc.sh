python finetune.py \
    --model_name_or_path squeezebert/squeezebert-uncased \
    --task_name mrpc \
    --max_length 512 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --optimizer adamw \
    --lr_scheduler_type polynomial \
    --lr 1e-4 \
    --weight_decay 1e-7 \
    --eps 1e-4 \
    --seed 0, 1, 2
    