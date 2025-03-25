python finetune.py \
    --model_name_or_path squeezebert/squeezebert-uncased \
    --task_name qqp \
    --max_length 512 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --optimizer samadamw \
    --lr_scheduler_type polynomial \
    --lr 1e-4 \
    --weight_decay 1e-6 \
    --rho 1e-2 \
    --eps 1e-6 \
    --seed 0, 1, 2
    