python train.py --workers 4 --dataset imagenet -a vit_s_32 --epochs 90 -b 1024 \
--LRScheduler cosine --warmup_epochs 8 \
--optimizer sophiah \
--lr 1e-3 --wd 1e-4 --lazy_hessian 1 --clip_threshold 0.01 --eps 1e-4 --seed 0, 1, 2 \
/home/shared/dataset/imagenet