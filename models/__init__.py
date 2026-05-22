"""
Models module — QPI segmentation model registry.
"""

from models.mobilenet_unet import MobileNetUNet
from models.mobile_sam     import MobileSAMSeg
from models.edge_sam       import EdgeSAMSeg
from models.lora_utils     import inject_lora_into_model, merge_lora_weights


def get_model(model_name: str, config):
    """
    Factory function to instantiate segmentation models.

    model_name options:
        mobilenet_unet   – MobileNetV2-UNet encoder-decoder
        mobile_sam       – MobileSAM with TinyViT encoder
        edge_sam         – EdgeSAM for low-latency deployment

    Each model accepts an optional lora_r in config to trigger
    LoRA injection immediately after construction.
    """
    model_name = model_name.lower()

    num_classes = getattr(config, "num_classes", 1)
    pretrained  = getattr(config, "pretrained",  True)

    if model_name == "mobilenet_unet":
        model = MobileNetUNet(num_classes=num_classes, pretrained=pretrained)

    elif model_name == "mobile_sam":
        image_size = getattr(config, "image_size", 512)
        model = MobileSAMSeg(num_classes=num_classes, pretrained=pretrained,
                             image_size=image_size)

    elif model_name == "edge_sam":
        image_size = getattr(config, "image_size", 1024)
        model = EdgeSAMSeg(num_classes=num_classes, pretrained=pretrained,
                           image_size=image_size)

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: mobilenet_unet, mobile_sam, edge_sam"
        )

    # Inject LoRA if rank is specified in config
    lora_r = getattr(config, "lora_r", None)
    if lora_r is not None:
        strategy     = getattr(config, "insertion_strategy", "attention_blocks")
        lora_alpha   = getattr(config, "lora_alpha",     float(lora_r))
        lora_dropout = getattr(config, "lora_dropout",   0.0)
        model.inject_lora(r=lora_r, lora_alpha=lora_alpha,
                          lora_dropout=lora_dropout, strategy=strategy)

    return model


__all__ = [
    "MobileNetUNet",
    "MobileSAMSeg",
    "EdgeSAMSeg",
    "get_model",
    "inject_lora_into_model",
    "merge_lora_weights",
]