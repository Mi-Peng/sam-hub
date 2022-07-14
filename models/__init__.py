from .build import MODELS_REGISTRY

from .resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)

from .wideresnet import (
    wideresnet28x10,
    wideresnet34x10,
)

from .vit import (
    vit_tiny_patch16_224,
    vit_small_patch16_224,
    vit_base_patch16_224
)