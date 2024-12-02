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
from torch.optim.lr_scheduler import LambdaLR


from EllipssianNetCNN import EllipssianNetCNN

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
        target_cov = torch.tensor(target_cov, dtype=torch.float32)
        # target_cov = self.transform(target_cov)
        # target_cov = torch.tensor(target_cov, dtype=torch.float32).permute(0, 1, 3, 2)  # Now it’s 2 x 2 x 480 x 640

        return input_image, target_gradient, target_center, target_cov



if __name__ == '__main__':
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--dataset_path', type=str, default="")
    parser.add_argument('--network_type', type=str, default="CNN")
    parser.add_argument('--train_mode', type=str, default="")
    parser.add_argument('--chkpoint_load_path', type=str, default="")
    parser.add_argument('--chkpoint_save_path', type=str, default="")
    args = parser.parse_args(sys.argv[1:])
    dataset_path = args.dataset_path
    network_type = args.network_type
    train_mode = args.train_mode
    chkpoint_load_path = args.chkpoint_load_path
    chkpoint_save_path = args.chkpoint_save_path

    # Save checkpoint function
    def save_checkpoint(epoch, model, optimizer, scheduler, path):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at epoch {epoch} to {path}")


    # Load checkpoint function
    def load_checkpoint(path, model, optimizer, scheduler):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        return start_epoch

    ########################################################################################



    # Data Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load Dataset
    train_dataset = ImageDataset(input_dir=dataset_path+'/voronoi', gradient_dir=dataset_path+'/gradient',
                                 center_dir=dataset_path+'/center', cov_dir=dataset_path+'/cov', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = None
    if network_type == "CNN":
        model = EllipssianNetCNN().cuda()
    criterion_gradient = nn.MSELoss()  # Fully convolutional cross-entropy for gradient
    criterion_center = nn.MSELoss()  # Fully convolutional cross-entropy for center
    criterion_cov = nn.MSELoss()  # Fully convolutional cross-entropy for center
    encoder_lr = 1e-4
    gradient_decoder_lr = 1e-4
    center_decoder_lr = 1e-4
    cov_decoder_lr = 1e-4

    start_lr = 1e-4
    end_lr = 1e-6


    # Custom exponential decay function
    def exponential_decay(epoch, num_epochs, start_lr, end_lr):
        decay_rate = (end_lr / start_lr) ** (1 / num_epochs)
        return decay_rate ** epoch

    # Initialize the optimizer with parameter groups
    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': encoder_lr},
        {'params': model.gradient_decoder.parameters(), 'lr': gradient_decoder_lr},
        {'params': model.center_decoder.parameters(), 'lr': center_decoder_lr},
        {'params': model.cov_decoder.parameters(), 'lr': cov_decoder_lr}
    ])
    start_epoch = 0  # Default to start from scratch
    num_epochs = 100


    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: exponential_decay(epoch, num_epochs, start_lr, end_lr)
    )

    # Uncomment the following line to load a checkpoint and continue training
    if train_mode == "load_chk":
        start_epoch = load_checkpoint(chkpoint_load_path, model, optimizer, scheduler)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        # Use tqdm to display a progress bar during training
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", unit="batch")

        for inputs, target_gradient, target_center, target_cov in progress_bar:
            inputs = inputs.cuda()
            target_gradient = target_gradient.cuda()
            target_center = target_center.cuda()
            target_cov = target_cov.cuda()

            # Forward pass
            output_gradient, output_center, output_cov = model(inputs, training=True)

            # Compute individual losses
            loss_gradient = criterion_gradient(output_gradient, target_gradient) * 1.0
            loss_center = criterion_center(output_center, target_center) * 1.0
            loss_cov = criterion_cov(output_cov, target_cov) * 1.0

            # Total loss
            loss = loss_gradient + loss_center + loss_cov

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running loss for display
            running_loss += loss.item()
            progress_bar.set_postfix({
                "Total Loss": f"{loss.item():.6f}",
                "Gradient Loss": f"{loss_gradient.item():.6f}",
                "Covariance Loss": f"{loss_cov.item():.6f}"
            })
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}] Average Loss: {running_loss / len(train_loader):.6f}")
        print(f"Learning Rate: {scheduler.get_last_lr()}")
        # Save checkpoint every 100 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = chkpoint_save_path + f'/{network_type}_chkpoint_{epoch + 1:03d}.pth'
            save_checkpoint(epoch, model, optimizer, scheduler, checkpoint_path)

    torch.save(model.state_dict(), chkpoint_save_path + f'/EllipssianNet{network_type}.pth')
    print("Training Complete!")