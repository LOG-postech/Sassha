python train.py --data cifar10 -depth 32 --epochs 160 --batch-size 256 \
--optimizer sassha \
--noise_level 0.2 \
--hessian_power_scheduler constant \
--lr 0.1 --min_lr 0.001 --wd 1e-3 --rho 0.1 --lazy_hessian 10 --seed 0, 1, 2 \
