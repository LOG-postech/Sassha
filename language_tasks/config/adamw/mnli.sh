python finetune.py \
    --model_name_or_path squeezebert/squeezebert-uncased \
    --task_name mnli \
    --max_length 512 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --optimizer adamw \
    --lr_scheduler_type polynomial \
    --lr 1e-5 \
    --weight_decay 1e-6 \
    --eps 1e-6 \
    --seed 0, 1, 2
    