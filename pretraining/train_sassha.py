import os
import time
import math
import pickle
import contextlib
from contextlib import nullcontext

import numpy as np
import torch
#import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model import GPTConfig, GPT
import torch.autograd as autograd

from hessian_scheduler import ProportionScheduler, CustomConstant, CustomLinear, CustomCosine
from rho_scheduler import Rho_ConstantScheduler

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
#total_bs = 480
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# optimizer
optimizer_name = 'sassha' 
learning_rate = 0.3 # max learning rate of sophiag
max_iters = 600000 # total number of training iterations
weight_decay = 2e-4
beta1 = 0.965
beta2 = 0.99
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
lazy_hessian = 10
eps=1e-4
clipping_threshold = 0.0
rho = 2e-2
hessian_power_scheduler_name = 'custom'
max_hessian_power=1
min_hessian_power=0.8
init_hessian_power=1
hessian_warmup_iters = 2000
rho_warmup_iters=2000
seed=2099
variant = 4
precond = False
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 1.5e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' # 'float32', 'bfloat16'
compile = False # use PyTorch 2.0 to compile the model to be faster checkout
scale_attn_by_inverse_layer_idx = True
ckpt = 'ckpt.pt'
resume_step = False
hess_clip = False
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    ddp_rank = 0                             #ddp_rank is used in get_batch function so this has to be here also when running locally
    master_process = True
    seed_offset = 0
    gradient_accumulation_steps *= 8 # simulate 8 gpus

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(2099)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix_list = []
    # ix = torch.randint(len(data) - block_size, (batch_size,))
    for jj in range(10):
        ix_list.append(torch.randint(len(data) - block_size, (batch_size,)))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix_list[ddp_rank]])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix_list[ddp_rank]])
    # x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    # y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, ckpt)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)


# optimizer
optimizer = model.configure_optimizers(
    optimizer_name, 
    weight_decay, 
    learning_rate, 
    (beta1, beta2), 
    lazy_hessian, 
    eps, 
    clipping_threshold, 
    rho,
    device_type, 
    seed)
    
# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# select a Hessian power scheduler
if hessian_power_scheduler_name == 'proportion':
    hessian_power_scheduler = ProportionScheduler(
        t = iter_num,
        get_lr=get_lr,
        max_lr=learning_rate,
        min_lr=min_lr,
        max_value=max_hessian_power,
        min_value=min_hessian_power)

elif hessian_power_scheduler_name == 'custom':
    hessian_power_scheduler = CustomConstant(
        t = iter_num,
        T_max=max_iters, 
        max_value=max_hessian_power,
        min_value=min_hessian_power,
        init_value=init_hessian_power,
        warmup_steps=hessian_warmup_iters)

elif hessian_power_scheduler_name == 'clin':
    hessian_power_scheduler = CustomLinear(
        t = iter_num,
        T_max=max_iters, 
        max_value=max_hessian_power,
        min_value=min_hessian_power,
        init_value=init_hessian_power,
        warmup_steps=hessian_warmup_iters)

elif hessian_power_scheduler_name == 'ccos':
    hessian_power_scheduler = CustomCosine(
        t = iter_num,
        T_max=max_iters, 
        max_value=max_hessian_power,
        min_value=min_hessian_power,
        init_value=init_hessian_power,
        warmup_steps=hessian_warmup_iters)

# set rho
rho_constant = Rho_ConstantScheduler(
    T_max=max_iters,
    max_value=rho,
    min_value=rho,
    init_value=rho,
    warmup_steps=rho_warmup_iters,
    t=rho_warmup_iters+1,
)

optimizer.hessian_power_scheduler = hessian_power_scheduler
optimizer.rho_scheduler = rho_constant

if init_from == 'resume':
    print("loaded optimizer states")
    optimizer.load_state_dict(checkpoint['optimizer'])
    resume_step = True
    del state_dict
    del checkpoint
# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# no sync
def maybe_no_sync(model):
    if torch.distributed.is_initialized():
        return model.no_sync()
    else:
        return contextlib.ExitStack()

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
stored_batches = []
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
num_param = 1
num_effective = 0
momentum_norm = 0
hessian_norm = 0
hessian_norm2 = 0
clip_time = 0

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }, step=iter_num)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, f'{hessian_power_scheduler_name}-{learning_rate}-{weight_decay}-{eps}-{rho}-{beta1}-{iter_num}.pt'))
        if iter_num % (eval_interval * 5) == 0 or iter_num==warmup_iters:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, f'{hessian_power_scheduler_name}-{learning_rate}-{weight_decay}-{eps}-{rho}-{beta1}-{iter_num}.pt'))
    if iter_num == 0 and eval_only:
        break

    if iter_num % lazy_hessian == 0 or resume_step:
        resume_step = False
        optimizer.zero_hessian()
        with maybe_no_sync(model):
            for micro_step in range(gradient_accumulation_steps): 
                with ctx:
                    logits, loss = model(X, Y)
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                stored_batches.append((X, Y))
                X, Y = get_batch('train')
                # backward pass, with gradient scaling if training in fp16
                # scaler.scale(loss / gradient_accumulation_steps).backward()
                if iter_num > rho_warmup_iters:
                    (loss / gradient_accumulation_steps).backward()
                else:
                    (loss / gradient_accumulation_steps).backward(create_graph=True)
                    # clip the gradient
                    if grad_clip != 0.0 and hess_clip:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.set_hessian()
        
        optimizer._sync_gradient()
        if iter_num > rho_warmup_iters:
            # clip the gradient
            if grad_clip != 0.0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.perturb_weights()  # move to a point which maximizes the loss
            optimizer.zero_grad(set_to_none=True)
                    
            with maybe_no_sync(model):
                while stored_batches:
                    pX, pY = stored_batches.pop()
                    with ctx:
                        _, sam_loss = model(pX, pY)
                    # backward pass, with gradient scaling if training in fp16
                    (sam_loss / gradient_accumulation_steps).backward(create_graph=True)
                    # scaler.scale(loss / gradient_accumulation_steps).backward(create_graph=True)
                    # scaler.unscale_(optimizer)
                    if grad_clip != 0.0 and hess_clip:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.set_hessian()  # accumulate p.hess
                    
                    print(f"\nProcessed a batch, remaining: {len(stored_batches)}, and iter: {iter_num}")
    
            optimizer.unperturb()
            optimizer._sync_gradient()
            
        else:
            stored_batches.clear()
    
        # clip the gradient
        if grad_clip != 0.0:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if total_norm.item() > grad_clip:
                clip_time += 1

        #optimizer.step()
        #scaler.step(optimizer, iter_num)
        #scaler.update()
        optimizer.step(iter_num)
        optimizer.zero_grad(set_to_none=True)

    else:
        with maybe_no_sync(model):
            for micro_step in range(gradient_accumulation_steps): 
                with ctx:
                    logits, loss = model(X, Y)
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                stored_batches.append((X, Y))
                X, Y = get_batch('train')
                # backward pass, with gradient scaling if training in fp16
                (loss / gradient_accumulation_steps).backward()
                # scaler.scale(loss / gradient_accumulation_steps).backward()
            
        optimizer._sync_gradient()
        #scaler.unscale_(optimizer)
        if iter_num > rho_warmup_iters:
            # clip the gradient
            if grad_clip != 0.0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.perturb_weights()
            optimizer.zero_grad(set_to_none=True)
        
            with maybe_no_sync(model):
                while stored_batches:
                    pX, pY = stored_batches.pop()
                    with ctx:
                        _, sam_loss = model(pX, pY)
                    # backward pass, with gradient scaling if training in fp16
                    (sam_loss / gradient_accumulation_steps).backward() 
            
            optimizer.unperturb()
            optimizer._sync_gradient()
        
        else:
            stored_batches.clear()

        # clip the gradient
        if grad_clip != 0.0:
            # scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if total_norm.item() > grad_clip:
                clip_time += 1

        optimizer.step(iter_num)
        optimizer.zero_grad(set_to_none=True)
        
    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() # loss as float. note: this is a CPU-GPU sync point
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        total_param_norm = 0
        momentum_norm = 0
        params = []
        for (name, p) in model.named_parameters():
            params.append(p)
        for p in params:
            param_norm = p.data.norm(2)
            total_param_norm += param_norm.item() ** 2
        total_param_norm = total_param_norm ** 0.5
        momentum_norm = 0
        hessian_norm = 0
        LL = len(optimizer.state_dict()['state'])
        for jj in range(LL):
            momentum_norm += (optimizer.state_dict()['state'][jj]['exp_avg'].detach().norm(2)) ** 2
            hessian_norm += optimizer.state_dict()['state'][jj]['exp_hessian_diag'].detach().norm(2).item() ** 2
        momentum_norm = torch.sqrt(momentum_norm).item()
        hessian_norm = hessian_norm ** 0.5
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
                "rho": optimizer.rho_t if iter_num > rho_warmup_iters else 0,
                "param_norm": total_param_norm,
                "momentum_norm" : momentum_norm,
                "hessian_norm" : hessian_norm,
                "train/clip_rate": clip_time / (iter_num + 1),
                "hessian_power": optimizer.hessian_power_t if optimizer_name == 'sassha' else 0,
            }, step=iter_num)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
