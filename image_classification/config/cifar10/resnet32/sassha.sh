python train.py --workers 2 --dataset cifar10 -a resnet32 --epochs 160 -b 256 \
--LRScheduler multi_step --lr-decay-epoch 80 120 --lr-decay 0.1 \
--optimizer sassha \
--hessian_power_scheduler constant \
--lr 0.15 --min_lr 0.0015 --wd 5e-4 --rho 0.2 --lazy_hessian 10 --seed 0, 1, 2
