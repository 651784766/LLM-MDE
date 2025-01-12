import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvUnit(nn.Module):
    """Residual convolution module."""
    def __init__(self, features):
        super(ResidualConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()  

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x

class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""
    def __init__(self, features):
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=False)  # align_corners=False 更加安全
        return output

class UpsamplingModel(nn.Module):
    """Model with initial interpolation and four-stage upsampling."""
    def __init__(self, features=256):
        super(UpsamplingModel, self).__init__()
        self.initial_interp = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.feature_fusion1 = FeatureFusionBlock(features)
        self.feature_fusion2 = FeatureFusionBlock(features)
        self.feature_fusion3 = FeatureFusionBlock(features)
        self.feature_fusion4 = FeatureFusionBlock(features)
        self.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        # Initial interpolation to 1,256,30,40
        x = F.interpolate(x, size=(30, 40), mode="bilinear", align_corners=False)
        x = self.initial_interp(x)

        # Four stages of upsampling
        x = self.feature_fusion1(x)
        x = self.feature_fusion2(x)
        x = self.feature_fusion3(x)
        x = self.feature_fusion4(x)

        # Final convolution layers
        x = self.output_conv(x)

        return x

if __name__ == "__main__":
    model = UpsamplingModel(features=256)
    input_tensor = torch.randn(1, 256, 32, 32)
    output_tensor = model(input_tensor)
    print(f"Output shape: {output_tensor.shape}")
