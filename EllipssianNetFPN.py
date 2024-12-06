
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import torch.nn.functional as F


class EllipssianNetFPN(nn.Module):
    def __init__(self):
        super(EllipssianNetFPN, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Bottom-up layers (ResNet-50 backbone)
        self.conv1 = resnet.conv1  # Output stride: 2
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # Output stride: 4

        self.layer1 = resnet.layer1  # C2, Output stride: 4
        self.layer2 = resnet.layer2  # C3, Output stride: 8
        self.layer3 = resnet.layer3  # C4, Output stride: 16
        self.layer4 = resnet.layer4  # C5, Output stride: 32

        # Lateral layers (1x1 conv to reduce channels)
        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(2048, 256, kernel_size=1),  # C5
            nn.Conv2d(1024, 256, kernel_size=1),  # C4
            nn.Conv2d(512, 256, kernel_size=1),   # C3
            nn.Conv2d(256, 256, kernel_size=1),   # C2
        ])

        # Top-down pathway (upsampling layers)
        self.top_down_layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # P5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # P4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # P3
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # P2
        ])

        # Decoders
        # Gradient Decoder
        self.gradient_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # [240, 320]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # [480, 640]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Center Decoder
        self.center_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # [240, 320]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # [480, 640]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Covariance Decoder
        self.cov_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # [240, 320]
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [480, 640]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, kernel_size=3, padding=1),  # Final 4-channel output
        )

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.shape[2:], mode='nearest') + y

    def forward(self, x, training=True):
        # Bottom-up pathway
        c1 = self.relu(self.bn1(self.conv1(x)))
        c1 = self.maxpool(c1)  # Output stride: 4

        c2 = self.layer1(c1)   # C2
        c3 = self.layer2(c2)   # C3
        c4 = self.layer3(c3)   # C4
        c5 = self.layer4(c4)   # C5

        # Top-down pathway
        p5 = self.lateral_layers[0](c5)  # Lateral connection for C5
        p4 = self._upsample_add(p5, self.lateral_layers[1](c4))  # Merge with C4
        p3 = self._upsample_add(p4, self.lateral_layers[2](c3))  # Merge with C3
        p2 = self._upsample_add(p3, self.lateral_layers[3](c2))  # Merge with C2

        # Apply smoothing
        p5 = self.top_down_layers[0](p5)
        p4 = self.top_down_layers[1](p4)
        p3 = self.top_down_layers[2](p3)
        p2 = self.top_down_layers[3](p2)

        # Upsample and aggregate
        p5 = F.interpolate(p5, size=p2.shape[2:], mode='nearest')
        p4 = F.interpolate(p4, size=p2.shape[2:], mode='nearest')
        p3 = F.interpolate(p3, size=p2.shape[2:], mode='nearest')

        aggregated_feature = p2 + p3 + p4 + p5  # [batch_size, 256, 120, 160]

        # Pass the aggregated feature to the decoders
        gradient_output = self.gradient_decoder(aggregated_feature)

        # Interpolate gradient output to match aggregated feature resolution if needed
        if gradient_output.shape[2:] != aggregated_feature.shape[2:]:
            attention_mask = F.interpolate(
                gradient_output,
                size=aggregated_feature.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        else:
            attention_mask = gradient_output

        modified_aggregated_feature = aggregated_feature * attention_mask

        center_output = self.center_decoder(modified_aggregated_feature)
        cov_output = self.cov_decoder(modified_aggregated_feature)
        cov_output = cov_output.view(cov_output.size(0), 2, 2, cov_output.size(2), cov_output.size(3))
        cov_output = cov_output.permute(0, 1, 2, 4, 3)  # Permute to [batch, 2, 2, W, H]

        if training:
            return gradient_output, center_output, cov_output
        return center_output, cov_output