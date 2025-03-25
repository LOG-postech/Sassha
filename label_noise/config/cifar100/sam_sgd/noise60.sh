python train.py --data cifar100 -depth 32 --epochs 160 --batch-size 256 \
--optimizer samsgd \
--noise_level 0.6 \
--lr 0.03 --wd 1e-3 --rho 0.1 --seed 0, 1, 2 \
