python train.py --workers 2 --dataset cifar100 -a wideresnet_28_10 --epochs 200 -b 256 \
--LRScheduler multi_step --lr-decay-epoch 60 120 160 --lr-decay 0.2 \
--optimizer adahessian \
--lr 0.3 --wd 5e-4 --lazy_hessian 1 --seed 0, 1, 2 \
