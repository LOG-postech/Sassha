python finetune.py \
    --model_name_or_path squeezebert/squeezebert-uncased \
    --task_name rte \
    --max_length 512 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --optimizer sassha \
    --lr_scheduler_type polynomial \
    --hessian_power_scheduler constant \
    --lr 1e-2 \
    --weight_decay 1e-4 \
    --rho 1e-2 \
    --eps 1e-4 \
    --lazy_hessian 1 \
    --seed 0, 1, 2
