python train.py --workers 2 --dataset cifar10 -a resnet32 --epochs 160 -b 256 \
--LRScheduler multi_step --lr-decay-epoch 80 120 --lr-decay 0.1 \
--optimizer sophiah \
--lr 1e-3 --wd 2e-4 --lazy_hessian 1 --clip_threshold 0.1 --eps 1e-4 --seed 0, 1, 2 \
