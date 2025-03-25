python train.py --workers 4 --dataset imagenet -a resnet50 --epochs 90 -b 256 \
--LRScheduler plateau \
--optimizer adahessian \
--lr 0.15 --wd 1e-4 --lazy_hessian 1 --seed 0, 1, 2 \
/home/shared/dataset/imagenet