python train.py --data cifar100 -depth 32 --epochs 160 --batch-size 256 \
--optimizer adahessian \
--noise_level 0.4 \
--lr 1e-3 --wd 5e-4 --lazy_hessian 1 --seed 0, 1, 2 \
