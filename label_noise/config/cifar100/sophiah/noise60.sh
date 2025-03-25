python train.py --data cifar100 -depth 32 --epochs 160 --batch-size 256 \
--optimizer sophiah \
--noise_level 0.6 \
--lr 1e-5 --wd 5e-4 --clip_threshold 0.1 --eps 1e-4 --lazy_hessian 1 --seed 0, 1, 2 \
