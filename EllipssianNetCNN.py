
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import torch.nn.functional as F

# Custom Dataset Class
class EllipssianNetCNN(nn.Module):
    def __init__(self):
        super(EllipssianNetCNN, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Encoder: using ResNet's layers without the fully connected part
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        # Decoder: upsampling back to the original image size
        self.gradient_decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),  # Output 1 channels (gray)
            nn.Sigmoid()
        )


        self.center_decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),  # Output 1 channels (gray)
            nn.Sigmoid()
        )

        self.cov_decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            # Upsample by 2x (15x20 -> 30x40)
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            # Upsample by 2x (30x40 -> 60x80)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            # Upsample by 2x (60x80 -> 120x160)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            # Upsample by 2x (120x160 -> 240x320)
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            # Upsample by 2x (240x320 -> 480x640)
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),  # Final 4-channel output
        )

    def forward(self, x, training=True):
        encoder_output = self.encoder(x)

        # Gradient decoder output
        gradient_output = self.gradient_decoder(encoder_output)

        # Upsample gradient output if necessary, then use it as an attention mask
        attention_mask = F.interpolate(gradient_output, size=encoder_output.shape[2:], mode='bilinear',
                                       align_corners=False)

        # Use the attention mask to modulate the encoder output
        modified_encoder_output = encoder_output * attention_mask  # Element-wise multiplication

        # Pass the modified encoder output to the center and covariance decoders
        center_output = self.center_decoder(modified_encoder_output)
        cov_output = self.cov_decoder(modified_encoder_output)
        cov_output = cov_output.view(cov_output.size(0), 2, 2, cov_output.size(2), cov_output.size(3))
        cov_output = cov_output.permute(0, 1, 2, 4, 3)  # Permute to [batch, 2, 2, W, H]

        if training:
            return gradient_output, center_output, cov_output
        return center_output, cov_output


