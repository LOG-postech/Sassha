python train.py --workers 2 --dataset cifar100 -a resnet32 --epochs 160 -b 256 \
--LRScheduler multi_step --lr-decay-epoch 80 120 --lr-decay 0.1 \
--optimizer sassha \
--hessian_power_scheduler constant \
--lr 0.3 --min_lr 0.003 --wd 1e-3 --rho 0.25 --lazy_hessian 10 --eps 1e-6 --seed 0, 1, 2 \
