"""
ResNet18 workload for CIFAR-100.

This module builds a CIFAR-adapted ResNet18.

Why adapt the standard ResNet stem?
-----------------------------------
The original ResNet was designed for much larger ImageNet images.
CIFAR-100 images are only 32x32 pixels.

Therefore we replace:

    7x7 conv, stride=2
    max pooling

with:

    3x3 conv, stride=1
    no max pooling

This preserves more spatial information in the small CIFAR images.
"""

import torch.nn as nn 

from torchvision.models import resnet18 

def build_resnet18_cifar(num_classes: int=100,)->nn.Module:
    """
    num_classes: CIFAR-100 has 100 classes.
    return:
          CIFAR-adapted ResNet18
    """
    model = resnet18(weights=None, num_classes=num_classes,)
    # weights = None, as we want to train from scratch for benchmarking DDP
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    model.maxpool = nn.Identity() # returns simply y=x without modifying the tensor

    return model 