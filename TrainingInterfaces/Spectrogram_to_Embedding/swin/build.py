# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .swin_mlp import SwinMLP


def build_model(config):
    model_type = config["model_type"]
    if model_type == 'swin':
        model = SwinTransformer(img_size=config["img_size"],
                                patch_size=config["patch_size"],
                                in_chans=config["in_chans"],
                                num_classes=config["num_classes"],
                                embed_dim=config["embed_dim"],
                                depths=config["depths"],
                                num_heads=config["num_heads"],
                                window_size=config["window_size"],
                                mlp_ratio=config["mlp_ratio"],
                                qkv_bias=config["qkv_bias"],
                                qk_scale=config["qk_scale"],
                                drop_rate=config["drop_rate"],
                                drop_path_rate=config["drop_path_rate"],
                                ape=config["ape"],
                                patch_norm=config["patch_norm"],
                                use_checkpoint=config["use_checkpoint"])
    elif model_type == 'swin_mlp':
        model = SwinMLP(img_size=config["img_size"],
                        patch_size=config["patch_size"],
                        in_chans=config["in_chans"],
                        num_classes=config["num_classes"],
                        embed_dim=config["embed_dim"],
                        depths=config["depths"],
                        num_heads=config["num_heads"],
                        window_size=config["window_size"],
                        mlp_ratio=config["mlp_ratio"],
                        qkv_bias=config["qkv_bias"],
                        qk_scale=config["qk_scale"],
                        drop_rate=config["drop_rate"],
                        drop_path_rate=config["drop_path_rate"],
                        ape=config["ape"],
                        patch_norm=config["patch_norm"],
                        use_checkpoint=config["use_checkpoint"])
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
