"""Pathology feature-extractor backbones, loaded the official way for each model.

Every backbone is wrapped so that it takes a float tensor in [0, 1] of shape
(B, 3, S, S) and returns a (B, D) embedding. Normalization happens inside the
wrapper, so the exported ONNX / TensorRT engine includes it.

Gated models (UNI, Virchow, CONCH) need a Hugging Face account that has accepted
each model's terms, and a token in the HF_TOKEN environment variable.
"""
import os

import torch
import torch.nn as nn

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
OPENAI_CLIP = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


class Normalized(nn.Module):
    """Normalize a [0, 1] image tensor, then run the feature extractor."""

    def __init__(self, net, mean, std):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


class _ResNetPool(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.body = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        return torch.flatten(self.body(x), 1)


class _HFViTCLS(nn.Module):
    """Hugging Face transformers ViT -> CLS token."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).last_hidden_state[:, 0]


class _VirchowEmbed(nn.Module):
    """Virchow model card: concat(CLS token, mean of patch tokens) -> 2560-d."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        tokens = self.model(x)
        return torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=-1)


class _ConchImage(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.encode_image(x, proj_contrast=False, normalize=False)


# ---------------------------------------------------------------- loaders
def _resnet18():
    from torchvision.models import ResNet18_Weights, resnet18
    return _ResNetPool(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1))


def _hf_vit(model_id):
    def load():
        from transformers import AutoModel
        return _HFViTCLS(AutoModel.from_pretrained(model_id, token=os.environ.get("HF_TOKEN")))
    return load


def _uni():
    import timm
    return timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5,
                             num_classes=0, dynamic_img_size=False)


def _virchow():
    import timm
    from timm.layers import SwiGLUPacked
    return _VirchowEmbed(timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True,
                                           mlp_layer=SwiGLUPacked, act_layer=nn.SiLU))


def _conch():
    try:
        from conch.open_clip_custom import create_model_from_pretrained
    except ImportError as e:
        raise RuntimeError("CONCH needs its package: pip install --no-deps "
                           "git+https://github.com/Mahmoodlab/CONCH.git ftfy regex") from e
    model, _ = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch",
                                            hf_auth_token=os.environ.get("HF_TOKEN"))
    return _ConchImage(model)


# name -> spec. "paper" marks the backbones already reported in the manuscript.
BACKBONES = {
    "resnet18": dict(label="ResNet-18 (ImageNet)", load=_resnet18, size=224,
                     norm=IMAGENET, gated=False, paper=True),
    "vit_tiny": dict(label="ViT-Tiny (Phikon-distil)",
                     load=_hf_vit("1aurent/phikon-distil-vit-tiny-patch16-224-kather2016"),
                     size=224, norm=IMAGENET, gated=False, paper=True),
    "phikon": dict(label="ViT-B (Phikon)", load=_hf_vit("owkin/phikon"), size=224,
                   norm=IMAGENET, gated=False, paper=True),
    "conch": dict(label="CONCH (ViT-B/16, 448px)", load=_conch, size=448,
                  norm=OPENAI_CLIP, gated=True, paper=False),
    "uni": dict(label="UNI (ViT-L/16)", load=_uni, size=224,
                norm=IMAGENET, gated=True, paper=False),
    "virchow": dict(label="Virchow (ViT-H/14)", load=_virchow, size=224,
                    norm=IMAGENET, gated=True, paper=False),
}


def load_backbone(name):
    """Return (wrapped model in eval mode on CPU, spec)."""
    spec = BACKBONES[name]
    net = spec["load"]()
    mean, std = spec["norm"]
    return Normalized(net, mean, std).eval(), spec


def count_params(model):
    """Image-encoder parameters only (CONCH also ships a text tower we never run)."""
    net = model.net
    if isinstance(net, _ConchImage):
        net = net.model.visual
    return sum(p.numel() for p in net.parameters())
