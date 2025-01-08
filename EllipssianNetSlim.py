import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights


class EllipssianNetSlim(nn.Module):
    def __init__(self):
        super(EllipssianNetSlim, self).__init__()

        # --------------------------
        # Encoder (ResNet50)
        # --------------------------
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Remove the fully-connected layer (and AvgPool) from ResNet
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        # Output of self.encoder => [B, 2048, H/32, W/32]

        # --------------------------
        # Single decoder / up-sampling path
        # Adjust channels / number of layers as needed
        # for your image input size.
        # --------------------------
        self.up_decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Optionally one more up step depending on your input resolution
            # nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
        )
        # Output now should match the original input spatial resolution
        # if you had 6 up-sampling steps for a 32× downsample.

        # --------------------------
        # Heads for gradient, center, and covariance
        # --------------------------
        self.gradient_conv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.center_conv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Covariance: produce 4 channels, to be reshaped to [2 x 2]
        self.cov_conv = nn.Conv2d(64, 4, kernel_size=1)

    def forward(self, x, training=True):
        # --------------------------
        # 1) Encode
        # --------------------------
        encoder_output = self.encoder(x)

        # --------------------------
        # 2) Decode / Up-sample
        # --------------------------
        upsampled_feature = self.up_decoder(encoder_output)

        # --------------------------
        # 3) Gradient Head
        # --------------------------
        gradient_output = self.gradient_conv(upsampled_feature)
        # Use gradient map as an attention mask if desired:
        # (If you prefer NOT to use attention, remove this multiplication.)
        attention_mask = gradient_output

        # --------------------------
        # 4) Center Head
        # --------------------------
        center_output = self.center_conv(upsampled_feature * attention_mask)

        # --------------------------
        # 5) Covariance Head
        # Produces 4 channels => reshape to [B, 2, 2, H, W]
        # Then optionally permute to [B, 2, 2, W, H] if needed
        # --------------------------
        cov_output = self.cov_conv(upsampled_feature * attention_mask)
        # Reshape to (B, 2, 2, H, W)
        cov_output = cov_output.view(
            cov_output.size(0), 2, 2, cov_output.size(2), cov_output.size(3)
        )
        # If your pipeline requires [B, 2, 2, W, H], do this:
        cov_output = cov_output.permute(0, 1, 2, 4, 3)

        if training:
            return gradient_output, center_output, cov_output
        else:
            return center_output, cov_output