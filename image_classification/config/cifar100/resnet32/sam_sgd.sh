python train.py --workers 2 --dataset cifar100 -a resnet32 --epochs 160 -b 256 \
--LRScheduler multi_step --lr-decay-epoch 80 120 --lr-decay 0.1 \
--optimizer samsgd \
--lr 0.1 --wd 5e-4 --rho 0.2 --seed 0, 1, 2 \
