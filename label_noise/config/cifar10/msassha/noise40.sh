python train.py --data cifar10 -depth 32 --epochs 160 --batch-size 256 \
--optimizer msassha \
--noise_level 0.4 \
--lr 0.05 --wd 1e-3 --rho 0.25 --lazy_hessian 10 --seed 0, 1, 2 \
