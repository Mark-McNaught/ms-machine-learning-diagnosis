import torch
import torch.nn as nn
from torchvision import models

import numpy as np
import timm

#####################################################################################################
###################################### Model Component Modules ######################################
#####################################################################################################

# ── Classifier Head ───────────────────────────────────────────────────────────
#
# All models in this file use the same binary classification head system.
# The head type is selected via build_classifier_head(in_features, head=...).
#
# Supported values:
#   "mlp"    — two-layer MLP: Linear(in_features->128)->ReLU->Dropout(0.3)->Linear(128->1)  [default]
#   "linear" — single linear layer: Linear(in_features->1)
#
# Pass head= through get_model() to switch all models simultaneously:
#   get_model("cbam_end", head="linear")
#
# Both are compatible with BCEWithLogitsLoss + Adam + the two-phase protocol.
# No changes to trainer.py or criterion are required for either option.

VALID_HEADS = ("mlp", "linear")

def build_classifier_head(in_features: int, num_classes: int = 1, head: str = "mlp") -> nn.Sequential:
    """
    Builds the binary classification head.
    Head designs:
        "mlp" (default) — two-layer MLP with bottleneck and dropout:
            Linear(in_features -> 128) -> ReLU -> Dropout(0.3) -> Linear(128 -> 1)
        "linear" — single linear layer (linear probe):
            Linear(in_features -> 1)
    """
    if head not in VALID_HEADS:
        raise ValueError(f"Unknown head type '{head}'. Choose from: {VALID_HEADS}")

    if head == "linear":
        return nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

    # "mlp" — default
    return nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )

class ChannelAttention(nn.Module):
    """ Channel Attention (like SE) """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """ Spatial Attention"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """ CBAM Block = Channel + Spatial attention """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class BasicBlockCBAMPre(nn.Module):
    """ CBAM applied after block convolutions, before residual addition """
    def __init__(self, block: nn.Module, channels: int):
        super().__init__()
        self.block = block
        self.cbam = CBAM(channels)

    def forward(self, x):
        identity = x

        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        out = self.block.conv2(out)
        out = self.block.bn2(out)

        out = self.cbam(out)

        if self.block.downsample is not None:
            identity = self.block.downsample(x)

        out += identity
        out = self.block.relu(out)

        return out


class BasicBlockCBAMPost(nn.Module):
    """ CBAM applied after residual addition """
    def __init__(self, block: nn.Module, channels: int):
        super().__init__()
        self.block = block
        self.cbam = CBAM(channels)

    def forward(self, x):
        out = self.block(x)
        out = self.cbam(out)
        return out


class SEBlock(nn.Module):
    """
    Canonical SE block (Hu et al., 2018).
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class BasicBlockSEPre(nn.Module):
    """ SE applied after block convolutions, before residual addition """
    def __init__(self, block: nn.Module, channels: int):
        super().__init__()
        self.block = block
        self.se = SEBlock(channels)

    def forward(self, x):
        identity = x

        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        out = self.block.conv2(out)
        out = self.block.bn2(out)

        out = self.se(out)

        if self.block.downsample is not None:
            identity = self.block.downsample(x)

        out += identity
        out = self.block.relu(out)

        return out


class ChannelAttentionSE(nn.Module):
    """
    SE-style channel attention (avg-pool only).
    Identical to SEBlock but without the sigmoid wrapper — used inside CBAMIsolated.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return self.sigmoid(y)


class CBAMIsolated(nn.Module):
    """
    CBAMIsolated for the spatial attention isolation experiment (SRQ1).

    Channel attention: SE-style (avg-pool only, Conv2d) — identical to SEBlock
    Spatial attention: standard CBAM spatial (avg+max, Conv2d 7x7)

    This variant isolates spatial attention's contribution by holding channel
    attention constant (matching SE exactly), so only spatial attention varies.
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttentionSE(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class BasicBlockCBAMIsolatedPre(nn.Module):
    """ CBAMIsolated applied after block convolutions, before residual addition """
    def __init__(self, block: nn.Module, channels: int):
        super().__init__()
        self.block = block
        self.cbam_iso = CBAMIsolated(channels)

    def forward(self, x):
        identity = x

        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        out = self.block.conv2(out)
        out = self.block.bn2(out)

        out = self.cbam_iso(out)

        if self.block.downsample is not None:
            identity = self.block.downsample(x)

        out += identity
        out = self.block.relu(out)

        return out


#####################################################################################################
######################################## ResNet18 Model Classes #####################################
#####################################################################################################

class BaseResNet18(nn.Module):
    """Plain ResNet18 with custom classifier head."""
    def __init__(self, num_classes=1, head="mlp"):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.model.fc.in_features
        self.model.fc = build_classifier_head(num_features, num_classes, head)

    def forward(self, x, return_features=False):
        if return_features:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            return torch.flatten(x, 1)
        return self.model(x)


class CBAMResNet18(nn.Module):
    """ResNet18 with flexible CBAM placement."""
    def __init__(self, num_classes=1, cbam_location="end", head="mlp"):
        super().__init__()
        self.cbam_location = cbam_location
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if cbam_location in ["block_pre", "block_post"]:
            wrapper = BasicBlockCBAMPre if cbam_location == "block_pre" else BasicBlockCBAMPost
            self.model.layer1 = self._wrap_layer(self.model.layer1, 64, wrapper)
            self.model.layer2 = self._wrap_layer(self.model.layer2, 128, wrapper)
            self.model.layer3 = self._wrap_layer(self.model.layer3, 256, wrapper)
            self.model.layer4 = self._wrap_layer(self.model.layer4, 512, wrapper)
        else:
            self.model.avgpool = nn.Sequential(
                CBAM(512),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        num_features = self.model.fc.in_features
        self.model.fc = build_classifier_head(num_features, num_classes, head)

    def _wrap_layer(self, layer, channels, wrapper):
        return nn.Sequential(*[wrapper(block, channels) for block in layer])

    def forward(self, x, return_features=False):
        if return_features:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            return torch.flatten(x, 1)
        return self.model(x)


class SEResNet18(nn.Module):
    """ResNet18 with flexible SE placement."""
    def __init__(self, num_classes=1, se_location="end", head="mlp"):
        super().__init__()
        self.se_location = se_location
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if se_location == "block_pre":
            self.model.layer1 = self._wrap_layer(self.model.layer1, 64, BasicBlockSEPre)
            self.model.layer2 = self._wrap_layer(self.model.layer2, 128, BasicBlockSEPre)
            self.model.layer3 = self._wrap_layer(self.model.layer3, 256, BasicBlockSEPre)
            self.model.layer4 = self._wrap_layer(self.model.layer4, 512, BasicBlockSEPre)
        else:
            self.model.avgpool = nn.Sequential(
                SEBlock(512),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        num_features = self.model.fc.in_features
        self.model.fc = build_classifier_head(num_features, num_classes, head)

    def _wrap_layer(self, layer, channels, wrapper):
        return nn.Sequential(*[wrapper(block, channels) for block in layer])

    def forward(self, x, return_features=False):
        if return_features:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            return torch.flatten(x, 1)
        return self.model(x)


class CBAMIsolatedResNet18(nn.Module):
    """ResNet18 with CBAMIsolated — spatial attention isolation experiment."""
    def __init__(self, num_classes=1, cbam_iso_location="end", head="mlp"):
        super().__init__()
        self.cbam_iso_location = cbam_iso_location
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if cbam_iso_location == "block_pre":
            self.model.layer1 = self._wrap_layer(self.model.layer1, 64)
            self.model.layer2 = self._wrap_layer(self.model.layer2, 128)
            self.model.layer3 = self._wrap_layer(self.model.layer3, 256)
            self.model.layer4 = self._wrap_layer(self.model.layer4, 512)
        else:
            self.model.avgpool = nn.Sequential(
                CBAMIsolated(512),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        num_features = self.model.fc.in_features
        self.model.fc = build_classifier_head(num_features, num_classes, head)

    def _wrap_layer(self, layer, channels):
        return nn.Sequential(*[BasicBlockCBAMIsolatedPre(block, channels) for block in layer])

    def forward(self, x, return_features=False):
        if return_features:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            return torch.flatten(x, 1)
        return self.model(x)


#####################################################################################################
########################################### ViT Model Classes #######################################
#####################################################################################################

class DeiTSmallBinary(nn.Module):
    """
    DeiT-Small/16 for binary MS classification.

    Architecture: 12 transformer layers, 384-dim, 6 attention heads
    Parameters: ~22M (1.9x ResNet18)
    Pretrained: ImageNet-1K with distillation

    Use for: SRQ4 standalone ViT baseline (conservative comparison)

    Head default is "linear" to match the CNN experiments and provide a clean
    SRQ4 comparison. The transformer encoder performs extensive non-linear
    mixing before the head, making an MLP head redundant.
    """
    def __init__(self, num_classes=1, head="linear"):
        super().__init__()
        self.model = timm.create_model(
            'deit_small_patch16_224',
            pretrained=True,
            num_classes=0
        )
        embed_dim = 384
        self.head = build_classifier_head(embed_dim, num_classes, head)

    def forward(self, x):
        features = self.model(x)
        return self.head(features)


class EfficientFormerBinary(nn.Module):
    """
    EfficientFormer-L1 for binary MS classification.

    Architecture: Hybrid MetaFormer with efficient token mixer
    Parameters: ~12M (1.03x ResNet18) — closest parameter match
    Pretrained: ImageNet-1K

    Use for: SRQ4 standalone ViT baseline (fair parameter-matched comparison)

    Head default is "linear" to match the CNN experiments and eliminate head
    capacity as a confound in the SRQ4 CNN vs ViT comparison.
    """
    def __init__(self, num_classes=1, head="linear"):
        super().__init__()
        self.model = timm.create_model(
            'efficientformer_l1',
            pretrained=True,
            num_classes=0
        )
        embed_dim = self.model.num_features
        self.head = build_classifier_head(embed_dim, num_classes, head)

    def forward(self, x):
        features = self.model(x)
        return self.head(features)


# ── CNN-ViT Hybrid ────────────────────────────────────────────────────────────

# All ResNet18 architecture keys that can serve as a CNN backbone.
HYBRID_CNN_ARCHS = frozenset({
    "base",
    "cbam_end", "cbam_block_pre", "cbam_block_post",
    "se_end", "se_block_pre",
    "cbam_isolated_end", "cbam_isolated_block_pre",
})

_DEIT_DIM    = 384   # DeiT-Small embedding dimension
_CNN_OUT_DIM = 512   # ResNet18 layer4 output channels
_N_SPATIAL   = 49    # 7 × 7 spatial tokens (ResNet18 on 224×224 input)


class CNNViTHybrid(nn.Module):
    """
    CNN-ViT Hybrid for SRQ3.

    Architecture (following Dosovitskiy et al., 2021):
        1. CNN backbone  — any ResNet18 variant → 7×7×512 spatial feature map
        2. Tokenisation  — flatten spatial dims → 49 tokens of 512-dim
        3. Projection    — Linear(512 → 384) maps CNN channels to DeiT embedding dim
        4. CLS token     — learnable [CLS] prepended → sequence of 50 × 384
        5. Pos embedding — learnable positional embeddings (50 × 384) added
        6. Transformer   — pretrained DeiT-Small encoder blocks + LayerNorm
        7. Head          — classify from [CLS] token output

    Backbone note on "end"-placed attention variants:
        For cbam_end, se_end, and cbam_isolated_end the attention module lives
        inside self.model.avgpool, which this class intentionally bypasses —
        the spatial map is taken directly from layer4. The attention gate itself
        does not fire in the hybrid forward pass; however, layers 1–4 carry
        representational benefits learned under attention supervision during
        arch-eval training. This is stated explicitly in the methodology.

        For block-level variants (cbam_block_pre, cbam_block_post, se_block_pre,
        cbam_isolated_block_pre), attention modules are embedded inside layer1–4
        and fire normally during the hybrid forward pass.

    Args:
        backbone_arch:   Any ResNet18 architecture key (see HYBRID_CNN_ARCHS).
        head:            "linear" (recommended) or "mlp".
        freeze_backbone: Freeze CNN weights during training (default False).
        num_classes:     Output dimension; 1 for binary BCE classification.

    Usage:
        # From scratch (ImageNet-pretrained backbone, randomly-init projection/head)
        model = CNNViTHybrid(backbone_arch="cbam_end", head="linear")

        # With pre-trained backbone weights from arch-eval
        model = CNNViTHybrid(backbone_arch="cbam_end", head="linear")
        model.load_backbone_weights("arch-eval-results/weights/cbam_end/fold_2.pt")
    """

    def __init__(
        self,
        backbone_arch: str = "cbam_end",
        head: str = "linear",
        freeze_backbone: bool = False,
        num_classes: int = 1,
    ):
        super().__init__()

        if backbone_arch not in HYBRID_CNN_ARCHS:
            raise ValueError(
                f"backbone_arch='{backbone_arch}' is not a valid ResNet18 architecture.\n"
                f"Valid options: {sorted(HYBRID_CNN_ARCHS)}"
            )

        self.backbone_arch = backbone_arch

        # ── 1. CNN backbone ───────────────────────────────────────────────────
        # Instantiate via get_model() so all weight-loading / arch logic is
        # handled consistently. The classifier head is present but never called
        # in this class's forward() — _forward_cnn() stops before avgpool.
        self.backbone = get_model(backbone_arch, head="linear")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ── 2. Token projection: CNN channels → DeiT embedding dim ───────────
        self.token_proj = nn.Linear(_CNN_OUT_DIM, _DEIT_DIM)

        # ── 3. Learnable CLS token and positional embeddings ─────────────────
        # DeiT's pretrained pos_embed covers 14×14+1 = 197 positions.
        # Our 7×7 spatial map yields 49 tokens, so we cannot reuse those
        # embeddings directly. Fresh learnable embeddings are used instead,
        # initialised with truncated normal as per ViT convention.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, _DEIT_DIM))
        self.pos_embed  = nn.Parameter(torch.zeros(1, _N_SPATIAL + 1, _DEIT_DIM))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed,  std=0.02)

        # ── 4. Pretrained DeiT-Small transformer encoder ─────────────────────
        # Load full DeiT-Small, transfer only the transformer blocks and norm.
        # Patch embedding and classification head are discarded.
        _deit = timm.create_model(
            "deit_small_patch16_224",
            pretrained=True,
            num_classes=0,
        )
        self.encoder_blocks = _deit.blocks   # 12 × TransformerBlock (384-dim, 6 heads)
        self.encoder_norm   = _deit.norm     # LayerNorm(384)
        del _deit

        # ── 5. Classification head ────────────────────────────────────────────
        self.head = build_classifier_head(_DEIT_DIM, num_classes, head)

        print(
            f"CNNViTHybrid >>> backbone={backbone_arch!r}  "
            f"freeze_backbone={freeze_backbone}  head={head!r}"
        )
        self._print_param_count()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 7×7×512 spatial feature map from the CNN backbone.

        All ResNet18 variants share self.backbone.model.{conv1,bn1,...,layer4}.
        Execution stops before avgpool so the spatial structure is preserved.
        Output shape: (B, 512, 7, 7).
        """
        m = self.backbone.model
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        return x  # (B, 512, 7, 7)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # 1. CNN backbone → spatial feature map (B, 512, 7, 7)
        spatial = self._forward_cnn(x)

        # 2. Flatten spatial dims → token sequence (B, 49, 512)
        tokens = spatial.flatten(2).transpose(1, 2)

        # 3. Project to DeiT embedding dim → (B, 49, 384)
        tokens = self.token_proj(tokens)

        # 4. Prepend learnable CLS token → (B, 50, 384)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        # 5. Add positional embeddings
        tokens = tokens + self.pos_embed

        # 6. DeiT transformer encoder
        tokens = self.encoder_blocks(tokens)
        tokens = self.encoder_norm(tokens)

        # 7. Classify from CLS token output (B, 384) → (B, 1)
        cls_out = tokens[:, 0]
        return self.head(cls_out)

    # ── Weight loading ────────────────────────────────────────────────────────

    def load_backbone_weights(self, weights_path: str, device=None):
        """
        Load pre-trained CNN backbone weights from an arch-eval .pt checkpoint.

        Only backbone parameters are updated. The token projection, positional
        embeddings, and classification head retain their initialised values —
        these are randomly initialised and must be learned during SRQ3 training.

        Args:
            weights_path: Path to a .pt file saved by utils.save_weights() for
                          the matching backbone architecture.
            device:       Target device. Defaults to the model's current device.

        Returns:
            self (for chaining)
        """
        if device is None:
            device = next(self.parameters()).device

        state = torch.load(weights_path, map_location=device)
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)

        print(f"load_backbone_weights()>>> Loaded from {weights_path}")
        if missing:
            print(f"  Keys in model not in checkpoint (expected — head/proj not saved): {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys in checkpoint: {unexpected}")
        return self

    def _print_param_count(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cnn_total = sum(p.numel() for p in self.backbone.parameters())
        enc_total = sum(p.numel() for p in self.encoder_blocks.parameters())
        print(f"  Total params   : {total:,}")
        print(f"  Trainable      : {trainable:,}  ({trainable/total:.0%})")
        print(f"  ├─ CNN backbone: {cnn_total:,}")
        print(f"  ├─ DeiT encoder: {enc_total:,}")
        proj = sum(p.numel() for p in self.token_proj.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        pos  = self.pos_embed.numel() + self.cls_token.numel()
        print(f"  ├─ Projection  : {proj:,}")
        print(f"  ├─ Pos/CLS     : {pos:,}")
        print(f"  └─ Head        : {head:,}")


#####################################################################################################
########################################### Model Factory ###########################################
#####################################################################################################

def get_model(architecture="base", head="mlp", backbone_arch="cbam_end",
              freeze_backbone=False,
              device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Factory function to instantiate models by name.

    CNN Architecture keys (ResNet18-based):
        "base"                    — plain ResNet18
        "cbam_end"                — ResNet18 + single CBAM before avgpool
        "cbam_block_pre"          — ResNet18 + CBAM inside each block (pre-shortcut)
        "cbam_block_post"         — ResNet18 + CBAM inside each block (post-shortcut)
        "se_end"                  — ResNet18 + single SE before avgpool
        "se_block_pre"            — ResNet18 + SE inside each block (pre-shortcut)
        "cbam_isolated_end"       — ResNet18 + CBAMIsolated (SE channel + spatial) before avgpool
        "cbam_isolated_block_pre" — ResNet18 + CBAMIsolated inside each block

    ViT Architecture keys (SRQ4):
        "deit_small"              — DeiT-Small/16 (~22M params, 1.9x ResNet18)
        "efficientformer"         — EfficientFormer-L1 (~12M params, fair comparison)

    CNN-ViT Hybrid (SRQ3):
        "cnn_vit_hybrid"          — CNNViTHybrid; use backbone_arch to select the
                                    CNN backbone (any ResNet18 architecture key).

        Examples:
            get_model("cnn_vit_hybrid", backbone_arch="cbam_end")
            get_model("cnn_vit_hybrid", backbone_arch="base", freeze_backbone=True)

    Args:
        architecture:    Model architecture key (see above).
        head:            Classifier head type — "linear" (default) or "mlp".
        backbone_arch:   CNN backbone for "cnn_vit_hybrid" only. Ignored for all
                         other architectures. Default "cbam_end".
        freeze_backbone: Freeze CNN backbone weights in "cnn_vit_hybrid" only.
                         Ignored for all other architectures. Default False.
        device:          Target device.

    Note: ViT models require timm (pip install timm).
    """

    # ResNet18-based models
    if architecture == "base":
        model = BaseResNet18(head=head)

    elif architecture == "cbam_end":
        model = CBAMResNet18(cbam_location="end", head=head)
    elif architecture == "cbam_block_pre":
        model = CBAMResNet18(cbam_location="block_pre", head=head)
    elif architecture == "cbam_block_post":
        model = CBAMResNet18(cbam_location="block_post", head=head)

    elif architecture == "se_end":
        model = SEResNet18(se_location="end", head=head)
    elif architecture == "se_block_pre":
        model = SEResNet18(se_location="block_pre", head=head)

    elif architecture == "cbam_isolated_end":
        model = CBAMIsolatedResNet18(cbam_iso_location="end", head=head)
    elif architecture == "cbam_isolated_block_pre":
        model = CBAMIsolatedResNet18(cbam_iso_location="block_pre", head=head)

    # ViT-based models (SRQ4)
    elif architecture == "deit_small":
        model = DeiTSmallBinary(head=head)
    elif architecture == "efficientformer":
        model = EfficientFormerBinary(head=head)

    # CNN-ViT Hybrid (SRQ3)
    elif architecture == "cnn_vit_hybrid":
        model = CNNViTHybrid(
            backbone_arch=backbone_arch,
            head=head,
            freeze_backbone=freeze_backbone,
        )
        print(f"get_model()>>> architecture='cnn_vit_hybrid'  backbone_arch={backbone_arch!r}  head={head!r}")
        return model.to(device)

    else:
        raise ValueError(
            f"Unknown architecture: '{architecture}'. "
            f"Valid CNN options: base, cbam_end, cbam_block_pre, cbam_block_post, "
            f"se_end, se_block_pre, cbam_isolated_end, cbam_isolated_block_pre. "
            f"Valid ViT options: deit_small, efficientformer. "
            f"Valid hybrid option: cnn_vit_hybrid (use backbone_arch= to set the CNN backbone)."
        )

    print(f"get_model()>>> architecture={architecture!r}  head={head!r}")
    return model.to(device)