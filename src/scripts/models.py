import torch
import torch.nn as nn
from torchvision import models
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
#   "linear" — single linear layer: Linear(in_features->1)  [default]
#   "mlp"    — two-layer MLP: Linear(in_features->128)->ReLU->Dropout(0.3)->Linear(128->1)
#
# Pass head= through get_model() to switch all models simultaneously:
#   get_model("cbam_end", head="mlp")
#
# Both are compatible with BCEWithLogitsLoss + Adam + the two-phase protocol.
# No changes to trainer.py or criterion are required for either option.

VALID_HEADS = ("mlp", "linear")


def build_classifier_head(in_features: int, num_classes: int = 1, head: str = "linear") -> nn.Sequential:
    """
    Builds the binary classification head.
    Head designs:
        "linear" (default) — single linear layer (linear probe):
            Linear(in_features -> 1)
        "mlp" — two-layer MLP with bottleneck and dropout:
            Linear(in_features -> 128) -> ReLU -> Dropout(0.3) -> Linear(128 -> 1)
    """
    if head not in VALID_HEADS:
        raise ValueError(f"Unknown head type '{head}'. Choose from: {VALID_HEADS}")

    if head == "mlp":
        return nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    # "linear" — default
    return nn.Sequential(
        nn.Linear(in_features, num_classes)
    )


class ChannelAttention(nn.Module):
    """Channel attention (like SE): avg-pool + max-pool squeezed through shared MLP."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
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
    """Spatial attention: channel-wise avg+max concatenated through a Conv2d."""
    def __init__(self, kernel_size=7):
        super().__init__()
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
    """CBAM block: sequential channel then spatial attention."""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class BasicBlockCBAMPre(nn.Module):
    """CBAM applied after block convolutions, before residual addition."""
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
    """CBAM applied after residual addition."""
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
    """SE applied after block convolutions, before residual addition."""
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
    """CBAMIsolated applied after block convolutions, before residual addition."""
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

class _ResNet18Mixin:
    """
    Shared behaviour for all ResNet18-based model classes.

    Provides two forward helpers and the common _wrap_layer utility

    forward_spatial(x)
        Runs the backbone up to and including layer4, stopping before avgpool.
        Returns (B, 512, 7, 7) — used by CNNMHSAHybrid._forward_cnn to obtain
        the spatial feature map for tokenisation.

    forward(x, return_features=False)
        Standard forward pass. When return_features=True, continues from
        forward_spatial through avgpool and returns a flattened 512-dim vector
        for the NCA-kNN pipeline in trainer.py.

    _wrap_layer(layer, channels, wrapper)
        Replaces every BasicBlock in a ResNet18 layer with a wrapped version.
        Used by CBAMResNet18, SEResNet18, and CBAMIsolatedResNet18.
    """
    def forward_spatial(self, x) -> torch.Tensor:
        """Returns the 7x7x512 spatial feature map, stopping before avgpool."""
        m = self.model
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        return x  # (B, 512, 7, 7)

    def forward(self, x, return_features=False):
        if return_features:
            x = self.forward_spatial(x)
            x = self.model.avgpool(x)
            return torch.flatten(x, 1)  # (B, 512)
        return self.model(x)

    def _wrap_layer(self, layer, channels, wrapper):
        return nn.Sequential(*[wrapper(block, channels) for block in layer])


class BaseResNet18(_ResNet18Mixin, nn.Module):
    """Plain ResNet18 with custom classifier head."""
    def __init__(self, num_classes=1, head="linear"):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.model.fc.in_features
        self.model.fc = build_classifier_head(num_features, num_classes, head)


class CBAMResNet18(_ResNet18Mixin, nn.Module):
    """ResNet18 with flexible CBAM placement."""
    def __init__(self, num_classes=1, cbam_location="end", head="linear"):
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


class SEResNet18(_ResNet18Mixin, nn.Module):
    """ResNet18 with flexible SE placement."""
    def __init__(self, num_classes=1, se_location="end", head="linear"):
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


class CBAMIsolatedResNet18(_ResNet18Mixin, nn.Module):
    """ResNet18 with CBAMIsolated — spatial attention isolation experiment."""
    def __init__(self, num_classes=1, cbam_iso_location="end", head="linear"):
        super().__init__()
        self.cbam_iso_location = cbam_iso_location
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if cbam_iso_location == "block_pre":
            self.model.layer1 = self._wrap_layer(self.model.layer1, 64,  BasicBlockCBAMIsolatedPre)
            self.model.layer2 = self._wrap_layer(self.model.layer2, 128, BasicBlockCBAMIsolatedPre)
            self.model.layer3 = self._wrap_layer(self.model.layer3, 256, BasicBlockCBAMIsolatedPre)
            self.model.layer4 = self._wrap_layer(self.model.layer4, 512, BasicBlockCBAMIsolatedPre)
        else:
            self.model.avgpool = nn.Sequential(
                CBAMIsolated(512),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        num_features = self.model.fc.in_features
        self.model.fc = build_classifier_head(num_features, num_classes, head)



#####################################################################################################
########################################### ViT Model Classes #######################################
#####################################################################################################

class EfficientFormerBinary(nn.Module):
    """
    EfficientFormer-L1 for binary MS classification.

    Architecture: Hybrid MetaFormer with efficient token mixer
    Parameters: ~12M (1.03x ResNet18) — closest parameter match
    Pretrained: ImageNet-1K

    Use for: SRQ4 standalone ViT baseline (fair parameter-matched comparison)
    """
    def __init__(self, num_classes=1, head="linear"):
        super().__init__()
        self.model = timm.create_model('efficientformer_l1', pretrained=True, num_classes=0)
        embed_dim = self.model.num_features
        self.head = build_classifier_head(embed_dim, num_classes, head)

    def forward(self, x):
        features = self.model(x)
        return self.head(features)


#####################################################################################################
######################################### CNN-MHSA Hybrid ###########################################
#####################################################################################################

# All ResNet18 architecture keys that can serve as a CNN backbone.
HYBRID_CNN_ARCHS = frozenset({
    "base",
    "cbam_end", "cbam_block_pre", "cbam_block_post",
    "se_end", "se_block_pre",
    "cbam_isolated_end", "cbam_isolated_block_pre",
})

_CNN_OUT_DIM  = 512   # ResNet18 layer4 output channels
_N_SPATIAL    = 49    # 7 x 7 spatial tokens (ResNet18 on 224x224 input)
_MHSA_DIM     = 256   # Projection dimension for MHSA
_MHSA_HEADS   = 8     # Number of attention heads (head_dim = 32)


class CNNMHSAHybrid(nn.Module):
    """
    CNN + Single Multi-Head Self-Attention Hybrid for SRQ3.

    Positions the architecture as a middle point on the CNN↔ViT complexity
    spectrum: pure CNN → CNN+local attention (CBAM/SE) → CNN+global attention
    (this) → pure ViT. Tests whether global spatial self-attention over CNN
    feature tokens outperforms both local CNN attention variants (SRQ1) and
    the standalone ViT (SRQ4) under data-scarce conditions.

    Architecture:
        1. CNN backbone  — any ResNet18 variant → 7x7x512 spatial feature map
                           (frozen; weights loaded from arch-eval)
        2. Tokenisation  — flatten spatial dims → 49 tokens of 512-dim
        3. Projection    — Linear(512 → 256) + LayerNorm
        4. Pos embedding — learnable positional embeddings (49 x 256)
        5. MHSA          — single nn.MultiheadAttention(256, 8 heads) + residual
        6. Post-norm     — LayerNorm(256)
        7. Pooling       — global average pool over 49 tokens → (B, 256)
        8. Head          — Linear(256 → 1) for binary BCE

    Design rationale:
        A full DeiT encoder (12 blocks, 21M params) overfits severely on
        ~1057 training samples. A single MHSA layer adds only ~400K new
        parameters, is architecturally distinct from CBAM (permutation-aware,
        joint global attention vs. sequential local pooling), and is the
        maximum transformer complexity justified by this dataset scale.

    Backbone note on "end"-placed attention variants:
        For cbam_end, se_end, and cbam_isolated_end the attention module lives
        inside avgpool, which this class bypasses — the spatial map is taken
        directly from layer4. The backbone still carries representational
        benefits from attention-supervised training in arch-eval.

    Args:
        backbone_arch:  Any ResNet18 architecture key (see HYBRID_CNN_ARCHS).
        head:           "linear" (default) or "mlp".
        num_classes:    Output dimension; 1 for binary BCE classification.

    Usage:
        model = CNNMHSAHybrid(backbone_arch="cbam_block_post", head="linear")
        model.load_backbone_weights("arch-eval-results/weights/cbam_block_post/fold_0.pt")
    """

    def __init__(
        self,
        backbone_arch: str = "cbam_block_post",
        head: str = "linear",
        num_classes: int = 1,
    ):
        super().__init__()

        if backbone_arch not in HYBRID_CNN_ARCHS:
            raise ValueError(
                f"backbone_arch='{backbone_arch}' not valid.\n"
                f"Valid options: {sorted(HYBRID_CNN_ARCHS)}"
            )

        self.backbone_arch = backbone_arch

        # ── 1. CNN backbone (backbone frozen after load_backbone_weights) ─────
        self.backbone = get_model(backbone_arch, head="linear")

        # ── 2. Token projection + input normalisation ─────────────────────────
        self.token_proj = nn.Linear(_CNN_OUT_DIM, _MHSA_DIM)
        self.input_norm = nn.LayerNorm(_MHSA_DIM)

        # ── 3. Learnable positional embeddings ────────────────────────────────
        # Initialised with truncated normal (ViT convention, std=0.02).
        # No CLS token — global average pooling is used instead to aggregate
        # the 49 token outputs, keeping the architecture simple and reducing
        # parameter count.
        self.pos_embed = nn.Parameter(torch.zeros(1, _N_SPATIAL, _MHSA_DIM))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── 4. Single Multi-Head Self-Attention layer ─────────────────────────
        # batch_first=True: input shape (B, seq, dim)
        # dropout=0.1 for regularisation
        self.mhsa = nn.MultiheadAttention(
            embed_dim=_MHSA_DIM,
            num_heads=_MHSA_HEADS,
            dropout=0.1,
            batch_first=True,
        )
        self.post_norm = nn.LayerNorm(_MHSA_DIM)

        # ── 5. Classification head ─────────────────────────────────────────────
        self.head = build_classifier_head(_MHSA_DIM, num_classes, head)

        print(f"CNNMHSAHybrid >>> backbone={backbone_arch!r}  head={head!r}")
        self._print_param_count()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 7x7x512 spatial feature map from the CNN backbone.
        Delegates to backbone.forward_spatial() — stops before avgpool so
        spatial structure is preserved for tokenisation.
        Output shape: (B, 512, 7, 7).
        """
        return self.backbone.forward_spatial(x)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. CNN backbone → (B, 512, 7, 7)
        spatial = self._forward_cnn(x)

        # 2. Flatten spatial dims → (B, 49, 512)
        tokens = spatial.flatten(2).transpose(1, 2)

        # 3. Project + normalise → (B, 49, 256)
        tokens = self.input_norm(self.token_proj(tokens))

        # 4. Add positional embeddings
        tokens = tokens + self.pos_embed

        # 5. Single MHSA with residual connection
        attn_out, _ = self.mhsa(tokens, tokens, tokens)
        tokens = self.post_norm(tokens + attn_out)

        # 6. Global average pool over 49 tokens → (B, 256)
        pooled = tokens.mean(dim=1)

        # 7. Classify
        return self.head(pooled)

    # ── Weight loading ────────────────────────────────────────────────────────

    def load_backbone_weights(self, weights_path: str, device=None):
        """
        Load pre-trained CNN backbone weights from an arch-eval checkpoint.
        Only backbone parameters are updated — MHSA components remain
        randomly initialised and are trained during SRQ3 training.
        Backbone is frozen after loading.
        """
        if device is None:
            device = next(self.parameters()).device

        state = torch.load(weights_path, map_location=device)
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)

        # Freeze backbone after loading
        for p in self.backbone.parameters():
            p.requires_grad = False

        print(f"load_backbone_weights()>>> Loaded from {weights_path}")
        print(f"  Backbone frozen — {sum(p.numel() for p in self.backbone.parameters()):,} params frozen")
        if missing:
            print(f"  Missing keys (expected — head not in hybrid): {len(missing)}")
        return self

    def _print_param_count(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cnn       = sum(p.numel() for p in self.backbone.parameters())
        proj      = sum(p.numel() for p in self.token_proj.parameters())
        norm_in   = sum(p.numel() for p in self.input_norm.parameters())
        mhsa      = sum(p.numel() for p in self.mhsa.parameters())
        norm_post = sum(p.numel() for p in self.post_norm.parameters())
        head      = sum(p.numel() for p in self.head.parameters())
        pos       = self.pos_embed.numel()
        print(f"  Total params     : {total:,}")
        print(f"  Trainable        : {trainable:,}  ({trainable/total:.0%})")
        print(f"  ├─ CNN backbone  : {cnn:,}  (frozen after load_backbone_weights)")
        print(f"  ├─ token_proj    : {proj:,}")
        print(f"  ├─ input_norm    : {norm_in:,}")
        print(f"  ├─ pos_embed     : {pos:,}")
        print(f"  ├─ mhsa          : {mhsa:,}")
        print(f"  ├─ post_norm     : {norm_post:,}")
        print(f"  └─ head          : {head:,}")


#####################################################################################################
########################################### Model Factory ###########################################
#####################################################################################################

def get_model(architecture="base", head="linear", backbone_arch="cbam_block_post",
              device=None):
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
        "efficientformer"         — EfficientFormer-L1 (~12M params, fair comparison)

    CNN-MHSA Hybrid (SRQ3):
        "cnn_mhsa_hybrid"         — CNNMHSAHybrid; use backbone_arch to select the
                                    CNN backbone (any ResNet18 architecture key).

        Example:
            get_model("cnn_mhsa_hybrid", backbone_arch="cbam_block_post")

    Args:
        architecture:  Model architecture key (see above).
        head:          Classifier head type — "linear" (default) or "mlp".
        backbone_arch: CNN backbone for "cnn_mhsa_hybrid" only. Ignored for all
                       other architectures. Default "cbam_block_post".
        device:        Target device. Defaults to CUDA if available, else CPU.

    Note: ViT models require timm (pip install timm).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # ViT-based model (SRQ4)
    elif architecture == "efficientformer":
        model = EfficientFormerBinary(head=head)

    # CNN-MHSA Hybrid (SRQ3)
    elif architecture == "cnn_mhsa_hybrid":
        model = CNNMHSAHybrid(backbone_arch=backbone_arch, head=head)
        print(f"get_model()>>> architecture='cnn_mhsa_hybrid'  backbone_arch={backbone_arch!r}  head={head!r}")
        return model.to(device)

    else:
        raise ValueError(
            f"Unknown architecture: '{architecture}'. "
            f"Valid CNN options: base, cbam_end, cbam_block_pre, cbam_block_post, "
            f"se_end, se_block_pre, cbam_isolated_end, cbam_isolated_block_pre. "
            f"Valid ViT options: efficientformer. "
            f"Valid hybrid option: cnn_mhsa_hybrid (use backbone_arch= to set the CNN backbone)."
        )

    print(f"get_model()>>> architecture={architecture!r}  head={head!r}")
    return model.to(device)
