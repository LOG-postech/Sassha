python train.py --workers 2 --dataset cifar100 -a wideresnet_28_10 --epochs 200 -b 256 \
--LRScheduler multi_step --lr-decay-epoch 60 120 160 --lr-decay 0.2 \
--optimizer shampoo \
--lr 1 --wd 5e-3 --eps 1e-4 --seed 0, 1, 2 \
