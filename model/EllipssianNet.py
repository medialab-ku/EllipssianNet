import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
import torch.nn.functional as F

# Custom Dataset Class
class EllipssianNet(nn.Module):
    def __init__(self):
        super(EllipssianNet, self).__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Encoder: using ResNet's layers without the fully connected part
        # Modify ResNet18 to have a downsampling factor of 16 instead of 32
        # This is done by setting the stride of the first block in layer4 to 1
        resnet.layer4[0].conv1.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        # Decoder: upsampling back to the original image size
        self.gradient_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # Center Decoder: Upsamples by factor 16
        self.center_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # Covariance Decoder: Adjusted to upsample by factor 16
        self.cov_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),

            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(True),

            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),

            nn.Conv2d(4, 3, kernel_size=3, padding=1),
        )


    def forward(self, x):
        encoder_output = self.encoder(x)
        # Gradient decoder output
        gradient_output = self.gradient_decoder(encoder_output)

        # Upsample gradient output if necessary, then use it as an attention mask
        attention_mask = F.interpolate(gradient_output, size=encoder_output.shape[2:], mode='bilinear',
                                       align_corners=False)

        # Use the attention mask to modulate the encoder output
        weighted_encoder_output = encoder_output * attention_mask  # Element-wise multiplication

        # Pass the modified encoder output to the center and covariance decoders
        center_output = self.center_decoder(weighted_encoder_output)
        cov_output = self.cov_decoder(weighted_encoder_output)

        return gradient_output, center_output, cov_output
