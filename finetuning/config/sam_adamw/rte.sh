python finetune.py \
    --model_name_or_path squeezebert/squeezebert-uncased \
    --task_name rte \
    --max_length 512 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --optimizer samadamw \
    --lr_scheduler_type polynomial \
    --lr 1e-4 \
    --weight_decay 1e-7 \
    --rho 1e-3 \
    --eps 1e-6 \
    --lazy_hessian 1 \
    --seed 0, 1, 2
    