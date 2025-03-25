from optimizers.sam import SAM
from optimizers.msassha import MSASSHA
from optimizers.adahessian import Adahessian
from optimizers.sophiaH import SophiaH
from optimizers.shampoo import Shampoo
from optimizers.sassha import SASSHA
import torch.optim as optim

def configure_sassha(model, args):
    create_graph = True
    two_steps = True
    
    optimizer = SASSHA(
        model.parameters(),
        betas=tuple(args.betas),
        lr=args.lr, 
        weight_decay=args.weight_decay/args.lr,
        rho=args.rho,
        lazy_hessian=args.lazy_hessian,
        eps=args.eps,
        seed=args.seed)

    return optimizer, create_graph, two_steps


def configure_samsgd(model, args):
    create_graph = False
    two_steps = True

    optimizer = SAM(
        model.parameters(), optim.SGD, rho=args.rho, adaptive=args.adaptive,
        momentum=0.9,
        lr=args.lr,
        weight_decay=args.weight_decay)

    return optimizer, create_graph, two_steps


def configure_samadamw(model, args):
    create_graph = False
    two_steps = True

    optimizer = SAM(
        model.parameters(), optim.AdamW, rho=args.rho, adaptive=args.adaptive,
        betas=tuple(args.betas),
        lr=args.lr,
        weight_decay=args.weight_decay/args.lr)

    return optimizer, create_graph, two_steps


def configure_sgd(model, args):
    create_graph = False
    two_steps = False

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)

    return optimizer, create_graph, two_steps


def configure_adamw(model, args):
    create_graph = False
    two_steps = False

    optimizer = optim.AdamW(
        model.parameters(),
        betas=tuple(args.betas),
        lr=args.lr,
        weight_decay=args.weight_decay/args.lr)

    return optimizer, create_graph, two_steps


def configure_adahessian(model, args):
    create_graph = True
    two_steps = False

    optimizer = Adahessian(
        model.parameters(),
        betas=tuple(args.betas),
        lr=args.lr,
        weight_decay=args.weight_decay/args.lr,
        lazy_hessian=args.lazy_hessian,
        eps=args.eps,
        seed=args.seed)

    return optimizer, create_graph, two_steps


def configure_sophiah(model, args):
    create_graph = True
    two_steps = False

    optimizer = SophiaH(
        model.parameters(),
        betas=tuple(args.betas),
        lr=args.lr,
        weight_decay=args.weight_decay / args.lr,
        clip_threshold=args.clip_threshold,
        eps=args.eps,
        lazy_hessian=args.lazy_hessian,
        seed=args.seed)

    return optimizer, create_graph, two_steps


def configure_msassha(model, args):
    create_graph = True
    two_steps = False

    optimizer = MSASSHA(
        model.parameters(),
        lr=args.lr,
        rho=args.rho,
        weight_decay=args.weight_decay / args.lr,
        lazy_hessian=args.lazy_hessian,
        eps=args.eps,
        seed=args.seed)

    return optimizer, create_graph, two_steps


def configure_shampoo(model, args):
    create_graph = False
    two_steps = False

    optimizer = Shampoo(
        params=model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        epsilon=args.eps,
        update_freq=1)

    return optimizer, create_graph, two_steps


def get_optimizer(model, args):
    optimizer_map = {
        'sassha': configure_sassha,
        'samsgd': configure_samsgd,
        'samadamw': configure_samadamw,
        'adahessian': configure_adahessian,
        'adamw': configure_adamw,
        'sgd': configure_sgd,
        'sophiah': configure_sophiah,
        'msassha': configure_msassha,
        'shampoo': configure_shampoo,
    }

    if args.optimizer not in optimizer_map:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
        
    return optimizer_map[args.optimizer](model, args)

