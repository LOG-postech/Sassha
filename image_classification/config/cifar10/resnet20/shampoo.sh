python train.py --workers 2 --dataset cifar10 -a resnet20 --epochs 160 -b 256 \
--LRScheduler multi_step --lr-decay-epoch 80 120 --lr-decay 0.1 \
--optimizer shampoo \
--lr 0.8 --wd 5e-3 --eps 1e-4 --seed 0, 1, 2 \
