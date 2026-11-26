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


class _ChannelSpatialAttention(nn.Module):
    """Concurrent Channel + Spatial Squeeze-and-Excitation (scSE) attention for cloud edge & cirrus refinement."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(8, channels // reduction)
        self.c_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.s_se = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.c_se(x) + x * self.s_se(x)


class _ASPP(nn.Module):
    """Multi-scale Atrous Spatial Pyramid Pooling for simultaneous small cumulus and broad stratus context."""

    def __init__(self, in_ch: int, out_ch: int = 192) -> None:
        super().__init__()
        self.b0 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(16, out_ch),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * 4, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        p = F.interpolate(self.pool(x), size=(h, w), mode="bilinear", align_corners=False)
        return self.proj(torch.cat([self.b0(x), self.b1(x), self.b2(x), p], dim=1))


class _ConvNeXtTinyEncoder(nn.Module):
    """ImageNet-pretrained ConvNeXt-Tiny 4-stage encoder (strides 4, 8, 16, 32 -> 96, 192, 384, 768 ch)."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = None
        if pretrained and tv_models is not None:
            try:
                weights = tv_models.ConvNeXt_Tiny_Weights.DEFAULT
            except Exception:
                weights = None
        try:
            backbone = tv_models.convnext_tiny(weights=weights)
        except Exception:
            backbone = tv_models.convnext_tiny(weights=None)
        feats = list(backbone.features.children())
        # ConvNeXt-Tiny features layout: [stem(0..1), stage1(2..3), stage2(4..5), stage3(6..7)]
        self.stage1 = nn.Sequential(*feats[0:2])  # stride 4 -> 96
        self.stage2 = nn.Sequential(*feats[2:4])  # stride 8 -> 192
        self.stage3 = nn.Sequential(*feats[4:6])  # stride 16 -> 384
        self.stage4 = nn.Sequential(*feats[6:8])  # stride 32 -> 768
        self.out_channels = (96, 192, 384, 768)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return c1, c2, c3, c4


class CloudSegUNet(nn.Module):
    """
    Stage 1: Multi-Scale Cloud Segmentation UNet for 5-class cloud segmentation:
      0=clear_sky, 1=thin_cloud, 2=dense_cloud, 3=dark_cloud, 4=sun_glare
      (plus static Not_Sky mask ignored via ignore_index=255 during training).
    Supports:
      - 'convnext_aspp': ConvNeXt-Tiny + ASPP + scSE Attention (highest accuracy across cloud decks)
      - 'mobilenet_v3_small': Fast ImageNet-pretrained MobileNetV3-Small UNet
      - 'efficientnet_b0': ImageNet-pretrained EfficientNet-B0 UNet
      - 'lightweight': Pure depthwise-separable UNet fallback
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 5,
        base_ch: int = 24,
        backbone: str = "convnext_aspp",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone if (tv_models is not None and in_channels == 3) else "lightweight"
        self.aspp: nn.Module | None = None
        self.attn3: nn.Module | None = None
        self.attn2: nn.Module | None = None
        self.attn1: nn.Module | None = None

        if self.backbone_name == "convnext_aspp":
            self.encoder: nn.Module | None = _ConvNeXtTinyEncoder(pretrained=pretrained)
            c1, c2, c3, c4 = self.encoder.out_channels
            aspp_ch = 192
            self.aspp = _ASPP(c4, out_ch=aspp_ch)
            self.dec3 = DepthwiseSeparableBlock(aspp_ch + c3, 128, dropout=0.1)
            self.attn3 = _ChannelSpatialAttention(128)
            self.dec2 = DepthwiseSeparableBlock(128 + c2, 96)
            self.attn2 = _ChannelSpatialAttention(96)
            self.dec1 = DepthwiseSeparableBlock(96 + c1, 48)
            self.attn1 = _ChannelSpatialAttention(48)
            self.head = nn.Conv2d(48, num_classes, kernel_size=1)
            return

        if self.backbone_name == "mobilenet_v3_small":
            self.encoder = _MobileNetV3Encoder(pretrained=pretrained)
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

        if self.aspp is not None:
            b = self.aspp(b)

        d3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        if self.attn3 is not None:
            d3 = self.attn3(d3)

        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        if self.attn2 is not None:
            d2 = self.attn2(d2)

        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        if self.attn1 is not None:
            d1 = self.attn1(d1)

        logits = self.head(d1)
        if logits.shape[-2:] != orig_hw:
            logits = F.interpolate(logits, size=orig_hw, mode="bilinear", align_corners=False)
        return logits


class DeepLabV3CloudSeg(nn.Module):
    """
    Lightweight daytime-ready DeepLabV3-MobileNetV3-Large segmentation backend
    built directly on official torchvision (BSD-3-Clause).
    Fine-tunable on SWIMSEG/SWINySEG or user sky frames; runs cleanly offline when weights are absent.
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = False) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = "deeplabv3_mobilenet_v3"
        self.net: nn.Module | None = None
        if tv_models is not None and hasattr(tv_models, "segmentation"):
            weights_backbone = None
            if pretrained:
                try:
                    weights_backbone = tv_models.MobileNet_V3_Large_Weights.DEFAULT
                except Exception:
                    weights_backbone = None
            try:
                self.net = tv_models.segmentation.deeplabv3_mobilenet_v3_large(
                    weights=None,
                    weights_backbone=weights_backbone,
                    num_classes=num_classes,
                    aux_loss=False,
                )
            except Exception:
                self.net = tv_models.segmentation.deeplabv3_mobilenet_v3_large(
                    weights=None,
                    weights_backbone=None,
                    num_classes=num_classes,
                    aux_loss=False,
                )
        if self.net is None:
            self.net = CloudSegUNet(in_channels=3, num_classes=num_classes, backbone="lightweight", pretrained=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if isinstance(out, dict):
            return out["out"]
        return out


class _LocalPatchFeatureFallback(nn.Module):
    """
    Deterministic multi-scale patch feature extractor used when optional DINOv3/DINOv2
    weights are not installed locally (keeps offline mode and CPU tests 100% functional).
    """

    def __init__(self, embed_dim: int = 128, patch_size: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=7, stride=4, padding=3, bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
        )
        for p in self.stem.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class DINOv3LinearProbeSeg(nn.Module):
    """
    LenghuSky-8-inspired (Apache-2.0 methodology) frozen Vision Transformer patch encoder +
    trainable 1x1 linear probe for day/night sky segmentation from 50-100 labeled frames.

    Note: Optional DINOv3 backbone weights carry their own separate DINOv3 License and are
    loaded only if explicitly provided/cached locally; otherwise falls back cleanly to a
    frozen local patch encoder without any network downloads.
    """

    def __init__(
        self,
        num_classes: int = 5,
        backbone_id: str = "vit_small_patch14_dinov2.lvd142m",
        weights_path: str = "",
        use_timm_backbone: bool = False,
        fallback_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone_id = backbone_id
        self.backbone_name = "dinov3_linear_probe"
        self.using_timm = False
        self.patch_size = 16
        self.backbone: nn.Module

        loaded_timm = False
        if use_timm_backbone or weights_path:
            try:
                import timm  # type: ignore[import-not-found]

                kwargs: dict[str, object] = {"pretrained": False, "num_classes": 0, "dynamic_img_size": True}
                if weights_path:
                    kwargs["checkpoint_path"] = weights_path
                elif use_timm_backbone:
                    kwargs["pretrained"] = True
                vit = timm.create_model(backbone_id, **kwargs)
                for p in vit.parameters():
                    p.requires_grad = False
                vit.eval()
                self.backbone = vit
                embed_dim = int(getattr(vit, "num_features", 384))
                ps = getattr(getattr(vit, "patch_embed", None), "patch_size", 14)
                self.patch_size = int(ps[0] if isinstance(ps, (tuple, list)) else ps)
                self.using_timm = True
                loaded_timm = True
            except Exception:
                loaded_timm = False

        if not loaded_timm:
            embed_dim = int(fallback_embed_dim)
            self.backbone = _LocalPatchFeatureFallback(embed_dim=embed_dim, patch_size=16)
            self.patch_size = 16

        self.embed_dim = embed_dim
        # Lightweight trainable linear probe head (only these parameters are optimized!)
        self.probe_norm = nn.GroupNorm(8, embed_dim)
        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def train(self, mode: bool = True) -> DINOv3LinearProbeSeg:
        super().train(mode)
        # Keep frozen backbone strictly in eval mode during linear probing
        self.backbone.eval()
        return self

    def extract_patch_grid(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if not self.using_timm:
                return self.backbone(x)
            h, w = x.shape[-2:]
            ps = self.patch_size
            nh = max(ps, (h // ps) * ps)
            nw = max(ps, (w // ps) * ps)
            x_in = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False) if (nh != h or nw != w) else x
            # ImageNet normalization expected by DINOv2/DINOv3 ViTs
            mean = x_in.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = x_in.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            x_norm = (x_in - mean) / std
            feats = self.backbone.forward_features(x_norm)
            if isinstance(feats, dict):
                feats = feats.get("x_norm_patchtokens", next(iter(feats.values())))
            if feats.ndim == 3:
                gh, gw = nh // ps, nw // ps
                n_patches = gh * gw
                patch_tokens = feats[:, -n_patches:, :]
                return patch_tokens.transpose(1, 2).reshape(x.shape[0], self.embed_dim, gh, gw)
            return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_hw = x.shape[-2:]
        grid = self.extract_patch_grid(x)
        logits = self.head(self.probe_norm(grid))
        if logits.shape[-2:] != orig_hw:
            logits = F.interpolate(logits, size=orig_hw, mode="bilinear", align_corners=False)
        return logits


class CloudTypeClassifier(nn.Module):
    """
    Stage 2: Pretrained Visual Backbone + Gated Multimodal Tabular Fusion Classifier for WMO cloud genera.
    Supports 'efficientnet_b0' (default, pretrained ImageNet), 'convnext_tiny', or legacy 'compact' CNN.
    """

    def __init__(
        self,
        num_classes: int = 11,
        tabular_dim: int = 18,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.tabular_dim = tabular_dim
        self.backbone_name = backbone if (tv_models is not None) else "compact"

        if self.backbone_name == "convnext_tiny":
            weights = None
            if pretrained:
                try:
                    weights = tv_models.ConvNeXt_Tiny_Weights.DEFAULT
                except Exception:
                    weights = None
            try:
                m = tv_models.convnext_tiny(weights=weights)
            except Exception:
                m = tv_models.convnext_tiny(weights=None)
            self.vis_encoder = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
            vis_dim = 768
        elif self.backbone_name == "efficientnet_b0":
            weights = None
            if pretrained:
                try:
                    weights = tv_models.EfficientNet_B0_Weights.DEFAULT
                except Exception:
                    weights = None
            try:
                m = tv_models.efficientnet_b0(weights=weights)
            except Exception:
                m = tv_models.efficientnet_b0(weights=None)
            self.vis_encoder = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
            vis_dim = 1280
        else:
            self.backbone_name = "compact"
            self.vis_encoder = None
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
            vis_dim = 128

        self.tab_net = nn.Sequential(
            nn.Linear(tabular_dim, 48),
            nn.LayerNorm(48),
            nn.SiLU(inplace=True),
            nn.Linear(48, 32),
            nn.LayerNorm(32),
            nn.SiLU(inplace=True),
        )
        if self.vis_encoder is not None:
            self.vis_proj = nn.Sequential(
                nn.Linear(vis_dim, 160),
                nn.LayerNorm(160),
                nn.SiLU(inplace=True),
            )
            self.fusion_gate = nn.Sequential(
                nn.Linear(160 + 32, 160),
                nn.Sigmoid(),
            )
            self.head = nn.Sequential(
                nn.Linear(160 + 32, 128),
                nn.SiLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(128, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(128 + 32, 96),
                nn.SiLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(96, num_classes),
            )

    def forward(self, img: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] < self.tabular_dim:
            pad = torch.zeros(
                features.shape[0],
                self.tabular_dim - features.shape[1],
                device=features.device,
                dtype=features.dtype,
            )
            features = torch.cat([features, pad], dim=1)
        elif features.shape[1] > self.tabular_dim:
            features = features[:, : self.tabular_dim]
        tab = self.tab_net(features)

        if self.vis_encoder is not None:
            vis = self.vis_proj(self.vis_encoder(img))
            gate = self.fusion_gate(torch.cat([vis, tab], dim=1))
            fused = vis * gate
            return self.head(torch.cat([fused, tab], dim=1))

        vis = self.backbone(img).flatten(1)
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
    seg_backbone: str = "convnext_aspp",
    cls_backbone: str = "efficientnet_b0",
    num_horizons: int = 3,
) -> tuple[nn.Module, str]:
    """Instantiate a model by family name with hardware-safe defaults."""
    if family == "segmentation":
        arch_l = (architecture or "").lower()
        bb = seg_backbone
        if "dinov3" in arch_l or "dinov2" in arch_l or "linear_probe" in arch_l or "linearprobe" in arch_l:
            bb = "dinov3_linear_probe"
        elif "deeplab" in arch_l:
            bb = "deeplabv3_mobilenet_v3"
        elif "convnext" in arch_l or "aspp" in arch_l:
            bb = "convnext_aspp"
        elif "efficientnet" in arch_l:
            bb = "efficientnet_b0"
        elif "mobilenetv3" in arch_l or "mobilenet_v3" in arch_l:
            bb = "mobilenet_v3_small"
        elif architecture == "CloudSegUNet-Mobile":
            bb = "lightweight"
        nc = num_classes or 5
        if bb == "deeplabv3_mobilenet_v3":
            dl_model = DeepLabV3CloudSeg(num_classes=nc, pretrained=pretrained)
            return dl_model, architecture or "DeepLabV3-MobileNetV3"
        if bb == "dinov3_linear_probe":
            dino_model = DINOv3LinearProbeSeg(num_classes=nc, use_timm_backbone=False)
            return dino_model, architecture or "DINOv3-LinearProbe"
        model = CloudSegUNet(in_channels=3, num_classes=nc, base_ch=24, backbone=bb, pretrained=pretrained)
        if architecture:
            arch_label = architecture
        elif model.backbone_name == "convnext_aspp":
            arch_label = "CloudSegUNet-ConvNeXt-ASPP"
        elif model.backbone_name == "mobilenet_v3_small":
            arch_label = "CloudSegUNet-MobileNetV3"
        elif model.backbone_name == "efficientnet_b0":
            arch_label = "CloudSegUNet-EfficientNetB0"
        else:
            arch_label = "CloudSegUNet-Mobile"
        return model, arch_label
    if family == "classification":
        arch_l = (architecture or "").lower()
        cbb = cls_backbone
        if "convnext" in arch_l:
            cbb = "convnext_tiny"
        elif "efficientnet" in arch_l:
            cbb = "efficientnet_b0"
        elif "compact" in arch_l or architecture == "CloudTypeClassifier-Fusion":
            cbb = "compact"
        nc = num_classes or 11
        cls_model = CloudTypeClassifier(num_classes=nc, tabular_dim=18, backbone=cbb, pretrained=pretrained)
        default_lbl = (
            "CloudTypeClassifier-ConvNeXt"
            if cls_model.backbone_name == "convnext_tiny"
            else "CloudTypeClassifier-EfficientNetB0"
            if cls_model.backbone_name == "efficientnet_b0"
            else "CloudTypeClassifier-Fusion"
        )
        return cls_model, architecture or default_lbl
    if family == "temporal":
        tmp_model = TemporalSkyConvGRU(input_dim=20, hidden_dim=64, num_horizons=num_horizons)
        return tmp_model, architecture or "TemporalSkyConvGRU"
    if family == "event":
        return SkyEventClassifier(input_dim=20, hidden_dim=48, num_classes=8), architecture or "SkyEventClassifier-GRU"
    raise ValueError(f"Unknown model family: {family}")
