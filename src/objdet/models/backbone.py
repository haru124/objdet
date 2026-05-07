"""
models/backbone.py

Thin wrapper around torchvision's ResNet50-FPN backbone.

torchvision already provides resnet50_fpn_backbone() which builds a
ResNet-50 + Feature Pyramid Network backbone ready to plug into
Faster R-CNN.  We expose it here so the rest of the code imports from
one place, and so backbone construction can be customised easily.

trainable_backbone_layers controls how many ResNet layer groups are
unfrozen during fine-tuning:
    0 → freeze everything (only train the detection head)
    5 → unfreeze everything (full fine-tune)
    3 → default: unfreeze layer3, layer4, and FPN
"""

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def build_resnet50_fpn_backbone(
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
):
    """
    Returns a ResNet-50 + FPN backbone suitable for Faster R-CNN.

    Args:
        pretrained:                Load ImageNet-pretrained ResNet-50 weights.
        trainable_backbone_layers: Number of ResNet stages to fine-tune (0–5).
    """
    # resnet_fpn_backbone is the official torchvision factory.
    # It returns a BackboneWithFPN whose out_channels is 256 by default.
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        # weights parameter replaces the deprecated `pretrained` bool in newer TV
        weights="ResNet50_Weights.IMAGENET1K_V1" if pretrained else None,
        trainable_layers=trainable_backbone_layers,
    )
    return backbone