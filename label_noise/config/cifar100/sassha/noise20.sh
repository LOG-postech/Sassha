# we use three seeds 0, 1, 2
python train.py --data cifar100 -depth 32 --epochs 160 --batch-size 256 \
--optimizer sassha \
--noise_level 0.2 \
--hessian_power_scheduler constant \
--lr 0.1 --min_lr 0.001 --wd 5e-4 --rho 0.1 --lazy_hessian 10 --seed 0, 1, 2


