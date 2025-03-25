python finetune.py \
    --model_name_or_path squeezebert/squeezebert-uncased \
    --task_name rte \
    --max_length 512 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --optimizer adamw \
    --lr_scheduler_type polynomial \
    --lr 1e-2 \
    --weight_decay 1e-8 \
    --eps 1e-8 \
    --lazy_hessian 1 \
    --seed 0, 1, 2
    