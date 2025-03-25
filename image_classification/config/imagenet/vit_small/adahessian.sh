python train.py --workers 4 --dataset imagenet -a vit_s_32 --epochs 90 -b 1024 \
--LRScheduler cosine --warmup_epochs 8 \
--optimizer adahessian \
--lr 0.15 --wd 1e-4 --lazy_hessian 1 --seed 0, 1, 2 \
/home/shared/dataset/imagenet