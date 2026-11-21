"""Lightweight local PyTorch architectures optimized for AMD RX 6600 (8 GB VRAM) and fast CPU fallback."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision import models as tv_models
except Exception:  # pragma: no cover
    tv_models = None  # type: ignore[assignment]


class DepthwiseSeparableBlock(nn.Module):
    """Lightweight MobileNet-inspired convolutional block."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _MobileNetV3Encoder(nn.Module):
    """ImageNet-pretrained MobileNetV3-Small multi-scale feature extractor (strides 2, 4, 8, 16)."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = None
        if pretrained and tv_models is not None:
            try:
                weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT
            except Exception:
                weights = None
        try:
            backbone = tv_models.mobilenet_v3_small(weights=weights)
        except Exception:
            # Offline fallback without downloading ImageNet weights
            backbone = tv_models.mobilenet_v3_small(weights=None)
        feats = list(backbone.features.children())
        # Stage splits on MobileNetV3-Small:
        # stage1 (idx 0): stride 2 -> 16 ch
        # stage2 (idx 1): stride 4 -> 16 ch
        # stage3 (idx 2..3): stride 8 -> 24 ch
        # stage4 (idx 4..8): stride 16 -> 48 ch
        self.stage1 = nn.Sequential(*feats[:1])
        self.stage2 = nn.Sequential(*feats[1:2])
        self.stage3 = nn.Sequential(*feats[2:4])
        self.stage4 = nn.Sequential(*feats[4:9])
        self.out_channels = (16, 16, 24, 48)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return c1, c2, c3, c4


class _EfficientNetB0Encoder(nn.Module):
    """ImageNet-pretrained EfficientNet-B0 multi-scale feature extractor (strides 2, 4, 8, 16)."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = None
        if pretrained and tv_models is not None:
            try:
                weights = tv_models.EfficientNet_B0_Weights.DEFAULT
            except Exception:
                weights = None
        try:
            backbone = tv_models.efficientnet_b0(weights=weights)
        except Exception:
            backbone = tv_models.efficientnet_b0(weights=None)
        feats = list(backbone.features.children())
        # EfficientNet-B0 stages:
        # 0..1: stride 2 -> 16 ch
        # 2: stride 4 -> 24 ch
        # 3: stride 8 -> 40 ch
        # 4..5: stride 16 -> 112 ch
        self.stage1 = nn.Sequential(*feats[:2])
        self.stage2 = nn.Sequential(feats[2])
        self.stage3 = nn.Sequential(feats[3])
        self.stage4 = nn.Sequential(*feats[4:6])
        self.out_channels = (16, 24, 40, 112)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return c1, c2, c3, c4


class CloudSegUNet(nn.Module):
    """
    Stage 1: Lightweight UNet for 5-class cloud segmentation:
      0=clear_sky, 1=thin_cloud, 2=dense_cloud, 3=dark_cloud, 4=sun_glare
      (plus static Not_Sky mask ignored via ignore_index=255 during training).
    Supports ImageNet-pretrained MobileNetV3-Small (default), EfficientNet-B0,
    or pure lightweight DepthwiseSeparable encoder, and Phase 1A binary -> Phase 1B 5-class transfer.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 5,
        base_ch: int = 24,
        backbone: str = "mobilenet_v3_small",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone if (tv_models is not None and in_channels == 3) else "lightweight"

        if self.backbone_name == "mobilenet_v3_small":
            self.encoder: nn.Module | None = _MobileNetV3Encoder(pretrained=pretrained)
            c1, c2, c3, c4 = self.encoder.out_channels
        elif self.backbone_name == "efficientnet_b0":
            self.encoder = _EfficientNetB0Encoder(pretrained=pretrained)
            c1, c2, c3, c4 = self.encoder.out_channels
        else:
            self.backbone_name = "lightweight"
            self.encoder = None
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_ch),
                nn.SiLU(inplace=True),
            )
            self.enc1 = DepthwiseSeparableBlock(base_ch, base_ch)
            self.enc2 = DepthwiseSeparableBlock(base_ch, base_ch * 2)
            self.enc3 = DepthwiseSeparableBlock(base_ch * 2, base_ch * 4, dropout=0.1)
            self.bottleneck = DepthwiseSeparableBlock(base_ch * 4, base_ch * 6, dropout=0.15)
            self.pool = nn.MaxPool2d(2)
            c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 6

        self.dec3 = DepthwiseSeparableBlock(c4 + c3, max(32, base_ch * 2), dropout=0.1)
        self.dec2 = DepthwiseSeparableBlock(max(32, base_ch * 2) + c2, max(24, base_ch * 2))
        self.dec1 = DepthwiseSeparableBlock(max(24, base_ch * 2) + c1, base_ch)
        self.head = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def replace_head(self, num_classes: int) -> None:
        """Swap segmentation classification head (e.g. Phase 1A 2-class binary -> Phase 1B 5-class)."""
        in_ch = self.head.in_channels
        self.head = nn.Conv2d(in_ch, num_classes, kernel_size=1)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_hw = x.shape[-2:]
        if self.encoder is not None:
            e1, e2, e3, b = self.encoder(x)
        else:
            s0 = self.stem(x)
            e1 = self.enc1(s0)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))

        d3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        logits = self.head(d1)
        if logits.shape[-2:] != orig_hw:
            logits = F.interpolate(logits, size=orig_hw, mode="bilinear", align_corners=False)
        return logits


class CloudTypeClassifier(nn.Module):
    """
    Stage 2: Compact CNN + Expanded Tabular Fusion Classifier for WMO cloud genera.
    Takes 224x224 RGB image + 18-dim tabular vector:
      - 5 Stage-1 class fractions (clear, thin, dense, dark, sun_glare)
      - solar elevation (normalized)
      - radiometric descriptors (RBR, R-B, luminance, BSI, grayness, saturation)
      - texture descriptors (edge_density, local_contrast, fragmentation, GLCM contrast/homogeneity/entropy)
    """

    def __init__(self, num_classes: int = 11, tabular_dim: int = 18) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.tabular_dim = tabular_dim
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            DepthwiseSeparableBlock(24, 48),
            nn.MaxPool2d(2),
            DepthwiseSeparableBlock(48, 96),
            nn.MaxPool2d(2),
            DepthwiseSeparableBlock(96, 128, dropout=0.15),
            nn.AdaptiveAvgPool2d(1),
        )
        self.tab_net = nn.Sequential(
            nn.Linear(tabular_dim, 48),
            nn.LayerNorm(48),
            nn.SiLU(inplace=True),
            nn.Linear(48, 32),
            nn.LayerNorm(32),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 96),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(96, num_classes),
        )

    def forward(self, img: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        vis = self.backbone(img).flatten(1)
        # Gracefully pad or trim legacy 12-dim feature vectors if passed
        if features.shape[1] < self.tabular_dim:
            pad = torch.zeros(features.shape[0], self.tabular_dim - features.shape[1], device=features.device, dtype=features.dtype)
            features = torch.cat([features, pad], dim=1)
        elif features.shape[1] > self.tabular_dim:
            features = features[:, : self.tabular_dim]
        tab = self.tab_net(features)
        return self.classifier(torch.cat([vis, tab], dim=1))


class TemporalSkyConvGRU(nn.Module):
    """
    Stage 3: Temporal Sequence Nowcasting Network.
    Accepts sequence (B, T, 20) of visual sky features over recent timestamps and
    predicts for H=3 camera-only horizons [5, 15, 20] minutes (or configurable num_horizons):
      [cloud_cover_0_1, clearing_logit, increasing_logit, overcast_logit, darkening_logit, precip_signal_logit]
    """

    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, num_horizons: int = 3) -> None:
        super().__init__()
        self.num_horizons = num_horizons
        self.temporal_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.15)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, num_horizons * 6),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.temporal_conv(seq.transpose(1, 2)).transpose(1, 2))
        out, _ = self.gru(x)
        last = out[:, -1, :]
        preds = self.head(last).view(-1, self.num_horizons, 6)
        return preds


class SkyEventClassifier(nn.Module):
    """
    Legacy/Optional Stage 4 Neural Event Classifier (complemented by deterministic SkyEventEngine).
    Classes: normal, rapid_cloud_development, sky_clearing, cloud_cover_increasing,
             dark_cloud_development, storm_like_structure, camera_obstruction, unusual_change.
    """

    EVENT_CLASSES = [
        "normal",
        "rapid_cloud_development",
        "sky_clearing",
        "cloud_cover_increasing",
        "dark_cloud_development",
        "storm_like_structure",
        "camera_obstruction",
        "unusual_change",
    ]

    def __init__(self, input_dim: int = 20, hidden_dim: int = 48, num_classes: int = 8) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 48),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(48, num_classes),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(seq)
        return self.fc(out[:, -1, :])


def build_model_for_family(
    family: str,
    architecture: str | None = None,
    *,
    num_classes: int | None = None,
    pretrained: bool = True,
    seg_backbone: str = "mobilenet_v3_small",
    num_horizons: int = 3,
) -> tuple[nn.Module, str]:
    """Instantiate a model by family name with hardware-safe defaults."""
    if family == "segmentation":
        arch_l = (architecture or "").lower()
        bb = seg_backbone
        if "efficientnet" in arch_l:
            bb = "efficientnet_b0"
        elif "mobilenetv3" in arch_l or "mobilenet_v3" in arch_l:
            bb = "mobilenet_v3_small"
        elif architecture == "CloudSegUNet-Mobile":
            bb = "lightweight"
        nc = num_classes or 5
        model = CloudSegUNet(in_channels=3, num_classes=nc, base_ch=24, backbone=bb, pretrained=pretrained)
        if architecture:
            arch_label = architecture
        elif model.backbone_name == "mobilenet_v3_small":
            arch_label = "CloudSegUNet-MobileNetV3"
        elif model.backbone_name == "efficientnet_b0":
            arch_label = "CloudSegUNet-EfficientNetB0"
        else:
            arch_label = "CloudSegUNet-Mobile"
        return model, arch_label
    if family == "classification":
        nc = num_classes or 11
        return CloudTypeClassifier(num_classes=nc, tabular_dim=18), architecture or "CloudTypeClassifier-Fusion"
    if family == "temporal":
        return TemporalSkyConvGRU(input_dim=20, hidden_dim=64, num_horizons=num_horizons), architecture or "TemporalSkyConvGRU"
    if family == "event":
        return SkyEventClassifier(input_dim=20, hidden_dim=48, num_classes=8), architecture or "SkyEventClassifier-GRU"
    raise ValueError(f"Unknown model family: {family}")
