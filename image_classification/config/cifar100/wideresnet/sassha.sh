python train.py --workers 2 --dataset cifar100 -a wideresnet_28_10 --epochs 200 -b 256 \
--LRScheduler multi_step --lr-decay-epoch 60 120 160 --lr-decay 0.2 \
--optimizer sassha \
--hessian_power_scheduler constant \
--lr 0.15 --min_lr 0.0015 --wd 0.0015 --rho 0.2 --lazy_hessian 10 --seed 0, 1, 2