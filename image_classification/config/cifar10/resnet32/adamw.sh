python train.py --workers 2 --dataset cifar10 -a resnet32 --epochs 160 -b 256 \
--LRScheduler multi_step --lr-decay-epoch 80 120 --lr-decay 0.1 \
--optimizer adamw \
--lr 0.01 --wd 5e-4 --seed 0, 1, 2 \
