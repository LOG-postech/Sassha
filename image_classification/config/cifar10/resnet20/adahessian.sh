python train.py --workers 2 --dataset cifar10 -a resnet20 --epochs 160 -b 256 \
--LRScheduler multi_step --lr-decay-epoch 80 120 --lr-decay 0.1 \
--optimizer adahessian \
--lr 0.15 --wd 5e-4 --lazy_hessian 1 --seed 0, 1, 2 \
