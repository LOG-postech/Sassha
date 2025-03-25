python train.py --workers 4 --dataset imagenet -a resnet50 --epochs 90 -b 256 \
--LRScheduler multi_step --lr-decay-epoch 30 60 --lr-decay 0.1 \
--optimizer sophiah \
--lr 1e-2 --wd 1e-4 --lazy_hessian 1 --clip_threshold 0.1 --eps 1e-4 --seed 0, 1, 2 \
/home/shared/dataset/imagenet