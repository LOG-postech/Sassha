python train.py --workers 4 --dataset imagenet -a resnet50 --epochs 90 -b 256 \
--LRScheduler multi_step --lr-decay-epoch 30 60 --lr-decay 0.1 \
--optimizer sassha \
--hessian_power_scheduler constant \
--lr 0.3 --min_lr 0.003 --wd 1e-4 --rho 0.2 --lazy_hessian 10 --seed 0, 1, 2 \
/home/shared/dataset/imagenet