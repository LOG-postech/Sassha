python train.py --workers 4 --dataset imagenet -a resnet50 --epochs 90 -b 256 \
--LRScheduler multi_step --lr-decay-epoch 30 60 --lr-decay 0.1 \
--optimizer msassha \
--lr 0.15 --wd 1e-4 --rho 0.1 --lazy_hessian 10 --seed 0, 1, 2 \
/home/shared/dataset/imagenet