python train.py --workers 4 --dataset imagenet -a vit_s_32 --epochs 90 -b 1024 \
--LRScheduler cosine --warmup_epochs 8 \
--optimizer sgd \
--lr 0.1 --wd 1e-4 --seed 0, 1, 2 \
/home/shared/dataset/imagenet