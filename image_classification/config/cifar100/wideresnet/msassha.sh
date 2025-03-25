python train.py --workers 2 --dataset cifar100 -a wideresnet_28_10 --epochs 200 -b 256 \
--LRScheduler multi_step --lr-decay-epoch 60 120 160 --lr-decay 0.2 \
--optimizer msassha \
--lr 0.15 --wd 1e-3 --rho 0.25 --lazy_hessian 10 --seed 0, 1, 2 \
