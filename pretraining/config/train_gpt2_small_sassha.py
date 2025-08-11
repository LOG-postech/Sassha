wandb_log = True
wandb_project = 'sassha-gpt2'
wandb_run_name='gpt2-small-sassha-50k'

# these make the total batch size be ~0.5M
# 8 batch size * 1024 block size * 6 gradaccum * 10 GPUs = 491,520
batch_size = 8
block_size = 1024
gradient_accumulation_steps = 6

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False

# this makes total number of tokens be 300B
max_iters = 50000 
lr_decay_iters = 50000 

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# optimizer
optimizer_name = 'sassha'
learning_rate = 0.3 # max learning rate
weight_decay = 2e-4
beta1 = 0.965
beta2 = 0.99
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
min_lr = 1.5e-5
rho = 2e-2
lazy_hessian = 10
eps = 1e-4
clipping_threshold = 0.0
hessian_power_scheduler_name = 'custom' # proportion, custom, clin, ccos
hessian_warmup_iters = 2000
rho_warmup_iters= 2000
max_hessian_power = 1
min_hessian_power = 0.8 # or 0.9 better than 0.5 in pretraining
init_hessian_power = 1
hess_clip = False

seed=2099

init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
ckpt = 'ckpt.pt' 
compile = False  # because of computing second-order derivative

out_dir = 'out_small_sassha_100k'
