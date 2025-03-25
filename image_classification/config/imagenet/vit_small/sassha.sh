python train.py --workers 4 --dataset imagenet -a vit_s_32 --epochs 90 -b 1024 \
--LRScheduler cosine --warmup_epochs 8 \
--optimizer sassha \
--hessian_power_scheduler constant \
--lr 0.6 --min_lr 0.0011981 --wd 2e-4 --rho 0.25 --lazy_hessian 10 --eps 1e-6 --seed 0, 1, 2 \
/home/shared/dataset/imagenet