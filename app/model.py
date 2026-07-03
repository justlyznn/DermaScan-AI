import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List
class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)

class ResUNet(nn.Module):
    def __init__(self, encoder_name: str = 'resnet34', pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name, pretrained=pretrained,
            features_only=True, out_indices=(0, 1, 2, 3, 4)
        )
        enc_ch: List[int] = self.encoder.feature_info.channels()
        self.decoder4 = DecoderBlock(enc_ch[4], enc_ch[3], 256)
        self.decoder3 = DecoderBlock(256, enc_ch[2], 128)
        self.decoder2 = DecoderBlock(128, enc_ch[1], 64)
        self.decoder1 = DecoderBlock(64, enc_ch[0], 32)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        d4 = self.decoder4(features[4], features[3])
        d3 = self.decoder3(d4, features[2])
        d2 = self.decoder2(d3, features[1])
        d1 = self.decoder1(d2, features[0])
        out = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)
        return torch.sigmoid(self.final_conv(out))

print("ResUNet didefinisikan.")

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        mid_channels = max(in_channels // reduction_ratio, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg_out = self.shared_mlp(F.adaptive_avg_pool2d(x, 1).view(b, c))
        max_out = self.shared_mlp(F.adaptive_max_pool2d(x, 1).view(b, c))
        attention = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        attention = torch.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1)))
        return x * attention

class CBAMBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_attention(self.channel_attention(x))

class CBAMDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAMBlock(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return self.cbam(x)

class CBAMResUNet(nn.Module):
    def __init__(self, encoder_name: str = 'resnet34', pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name, pretrained=pretrained,
            features_only=True, out_indices=(0, 1, 2, 3, 4)
        )
        enc_ch = self.encoder.feature_info.channels()
        
        # CBAM at bottleneck
        self.bottleneck_cbam = CBAMBlock(enc_ch[4])
        
        # Decoders with built-in CBAM
        self.decoder4 = CBAMDecoderBlock(enc_ch[4], enc_ch[3], 256)
        self.decoder3 = CBAMDecoderBlock(256, enc_ch[2], 128)
        self.decoder2 = CBAMDecoderBlock(128, enc_ch[1], 64)
        self.decoder1 = CBAMDecoderBlock(64, enc_ch[0], 32)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        
        # Apply bottleneck CBAM
        bottleneck = self.bottleneck_cbam(features[4])
        
        # Decoder passes with CBAM inside decoder blocks
        d4 = self.decoder4(bottleneck, features[3])
        d3 = self.decoder3(d4, features[2])
        d2 = self.decoder2(d3, features[1])
        d1 = self.decoder1(d2, features[0])
        
        out = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)
        return torch.sigmoid(self.final_conv(out))

print("CBAM-ResUNet didefinisikan.")


