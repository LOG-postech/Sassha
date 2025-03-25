python train.py --workers 4 --dataset imagenet -a resnet50 --epochs 90 -b 256 \
--LRScheduler multi_step --lr-decay-epoch 30 60 --lr-decay 0.1 \
--optimizer adamw \
--lr 1e-3 --wd 1e-4 --seed 0, 1, 2 \
/home/shared/dataset/imagenet