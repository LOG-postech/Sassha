python train.py --workers 4 --dataset imagenet -a vit_s_32 --epochs 90 -b 1024 \
--LRScheduler cosine --warmup_epochs 8 \
--optimizer samadamw \
--lr 1e-3 --wd 1e-4 --rho 0.2 --grad_clip 1 --seed 0, 1, 2 \
/home/shared/dataset/imagenet