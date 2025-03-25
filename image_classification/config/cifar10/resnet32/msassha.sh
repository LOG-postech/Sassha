python train.py --workers 2 --dataset cifar10 -a resnet32 --epochs 160 -b 256 \
--LRScheduler multi_step --lr-decay-epoch 80 120 --lr-decay 0.1 \
--optimizer msassha \
--lr 0.15 --wd 1e-3 --rho 0.6 --lazy_hessian 10 --seed 0, 1, 2 \
