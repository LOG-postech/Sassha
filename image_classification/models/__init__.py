from models.simple_vit import SimpleViT
from models.resnet import *
from models.wide_resnet import *

import re
from torchvision import models

def create_vit_model(arch):
    """Create a Vision Transformer (ViT) model."""
    patch_size = int(arch.split('_')[2])
    if arch.startswith('vit_s_'):
        return SimpleViT(
            image_size=224,
            patch_size=patch_size,
            num_classes=1000,
            dim=384,
            depth=12,
            heads=6,
            mlp_dim=1536
        )
    elif arch.startswith('vit_b_'):
        return SimpleViT(
            image_size=224,
            patch_size=patch_size,
            num_classes=1000,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072
        )
    elif arch.startswith('vit_l_'):
        return SimpleViT(
            image_size=224,
            patch_size=patch_size,
            num_classes=1000,
            dim=1024,
            depth=24,
            heads=16,
            mlp_dim=4096
        )
    else:
        raise ValueError(f"Unknown ViT architecture: {arch}")

def create_resnet_model(arch, dataset):
    """Create a ResNet model."""
    num_classes = 1000 if dataset == 'imagenet' else int(re.findall(r'\d+', dataset)[-1])
    depth = int(re.findall(r'\d+', arch)[-1])
    return resnet(num_classes=num_classes, depth=depth)

def create_wideresnet_model(arch, dataset):
    """Create a Wide ResNet model."""
    num_classes = 1000 if dataset == 'imagenet' else int(re.findall(r'\d+', dataset)[-1])
    parts = arch.split('_')
    depth = int(parts[-2])
    widen_factor = int(parts[-1])
    return Wide_ResNet(depth=depth, widen_factor=widen_factor, dropout_rate=0.0, num_classes=num_classes)

def get_model(args):
    """Main function to create a model based on the architecture."""
    
    if args.arch.startswith('vit_'):
        return create_vit_model(args.arch)
    elif args.arch in ['resnet20', 'resnet32']:
        return create_resnet_model(args.arch, args.dataset)
    elif args.arch.startswith('wideresnet_'):
        return create_wideresnet_model(args.arch, args.dataset)
    else:
        # Default case: PyTorch model loaded directly
        return models.__dict__[args.arch]()
