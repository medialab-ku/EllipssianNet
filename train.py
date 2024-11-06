import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from tqdm import tqdm
from argparse import ArgumentParser
import sys
import numpy as np

import torch.nn.functional as F

# Custom Dataset Class
class ImageDataset(Dataset):
    def __init__(self, input_dir, gradient_dir, center_dir, cov_dir, transform=None):
        self.input_dir = input_dir
        self.gradient_dir = gradient_dir
        self.center_dir = center_dir
        self.cov_dir = cov_dir
        self.input_names = os.listdir(input_dir)  # List of input images
        self.gradient_names = os.listdir(gradient_dir)  # List of target images
        self.center_names = os.listdir(center_dir)  # List of target images
        self.cov_names = os.listdir(cov_dir)  # List of target images
        self.transform = transform

    def __len__(self):
        return len(self.input_names)

    def __getitem__(self, idx):
        input_img_path = os.path.join(self.input_dir, self.input_names[idx])
        gradient_img_path = os.path.join(self.gradient_dir, self.gradient_names[idx])
        center_img_path = os.path.join(self.center_dir, self.center_names[idx])
        cov_np_path = os.path.join(self.cov_dir, self.cov_names[idx])

        # Load the input image and target image
        input_image = Image.open(input_img_path).convert('RGB')
        target_gradient = Image.open(gradient_img_path).convert('L')  # Grayscale for decoder 1
        target_center = Image.open(center_img_path).convert('L')
        target_cov = np.load(cov_np_path)  # Load .npy file for target3


        if self.transform:
            input_image = self.transform(input_image)
            target_gradient = self.transform(target_gradient)
            target_center = self.transform(target_center)
        target_cov = torch.tensor(target_cov, dtype=torch.float32).permute(0, 1, 3, 2)  # Now it’s 2 x 2 x 480 x 640

        return input_image, target_gradient, target_center, target_cov


# Define the model
class EllipssianNet(nn.Module):
    def __init__(self):
        super(EllipssianNet, self).__init__()
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
        encoder_output  = self.encoder(x)

        # Gradient decoder output
        gradient_output = self.gradient_decoder(encoder_output)

        # Upsample encoder output to match gradient output's spatial dimensions
        encoder_output_upsampled = F.interpolate(encoder_output, size=gradient_output.shape[2:], mode='bilinear',
                                                 align_corners=False)

        # Upsample gradient output if necessary, then use it as an attention mask
        attention_mask = F.interpolate(gradient_output, size=encoder_output.shape[2:], mode='bilinear',
                                       align_corners=False)

        # Use the attention mask to modulate the encoder output
        modified_encoder_output = encoder_output * attention_mask  # Element-wise multiplication

        # Pass the modified encoder output to the center and covariance decoders
        center_output = self.center_decoder(modified_encoder_output)
        cov_output = self.cov_decoder(modified_encoder_output)
        cov_output = cov_output.view(cov_output.size(0), 2, 2, cov_output.size(2), cov_output.size(3))

        if training:
                return gradient_output, center_output, cov_output
        return center_output, cov_output



if __name__ == '__main__':
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--dataset_path', type=str, default="")
    args = parser.parse_args(sys.argv[1:])

    dataset_path = args.dataset_path

    # Data Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load Dataset
    train_dataset = ImageDataset(input_dir=dataset_path+'/voronoi', gradient_dir=dataset_path+'/gradient',
                                 center_dir=dataset_path+'/center', cov_dir=dataset_path+'/cov', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


    # Initialize model, loss function, and optimizer
    model = EllipssianNet().cuda()
    criterion_gradient = nn.MSELoss()  # Fully convolutional cross-entropy for gradient
    criterion_center = nn.MSELoss()  # Fully convolutional cross-entropy for center
    encoder_lr = 1e-4
    gradient_decoder_lr = 1e-4
    center_decoder_lr = 1e-4
    cov_decoder_lr = 1e-4




    def weighted_mse_loss(pred, target, weight):
        return torch.mean((pred - target) ** 2)
        return torch.mean(weight * (pred - target) ** 2)

    # Training Encoder, Gradient and Center Decoders
    # Initialize the optimizer with parameter groups
    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': encoder_lr},
        {'params': model.gradient_decoder.parameters(), 'lr': gradient_decoder_lr},
        {'params': model.center_decoder.parameters(), 'lr': center_decoder_lr}
    ])
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Use tqdm to display a progress bar during training
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")

        for inputs, target_gradient, target_center, target_cov in progress_bar:
            inputs = inputs.cuda()
            target_gradient = target_gradient.cuda()
            target_center = target_center.cuda()

            # Forward pass
            output_gradient, output_center, output_cov = model(inputs, training=True)

            # Compute individual losses with weights
            loss_gradient = criterion_gradient(output_gradient, target_gradient) * 1.0
            loss_center = criterion_center(output_center, target_center) * 1.0

            # Combine weighted losses
            loss = loss_gradient + loss_center

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running loss for display
            running_loss += loss.item()
            progress_bar.set_postfix({
                "Total Loss": f"{loss.item():.4f}",
                "Gradient Loss": f"{loss_gradient.item():.4f}",
                "Center Loss": f"{loss_center.item():.4f}"
            })

        print(f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {running_loss / len(train_loader):.4f}')


    # Training Cov Decoder
    # Initialize the optimizer with parameter groups
    optimizer = optim.Adam([
        {'params': model.cov_decoder.parameters(), 'lr': cov_decoder_lr}
    ])

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Use tqdm to display a progress bar during training
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")

        for inputs, target_gradient, target_center, target_cov in progress_bar:
            inputs = inputs.cuda()
            target_gradient = target_gradient.cuda()
            target_cov = target_cov.cuda()

            # Forward pass
            output_gradient, output_center, output_cov = model(inputs, training=True)

            # Compute individual losses with weights
            loss_cov = weighted_mse_loss(output_cov, target_cov, target_center) * 1.0

            # Combine weighted losses
            loss =  loss_cov

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running loss for display
            running_loss += loss.item()
            progress_bar.set_postfix({
                "Cov Loss": f"{loss_cov.item():.4f}"
            })
        print(f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {running_loss / len(train_loader):.4f}')

    torch.save(model.state_dict(), 'EllipssianNet.pth')
    print("Training Complete!")