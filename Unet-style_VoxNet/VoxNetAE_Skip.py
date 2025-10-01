import torch
from torch import nn
import torch.nn.functional as F

class VoxNetAE_Skip(nn.Module):
    def __init__(self, latent_dim=256):
        super(VoxNetAE_Skip, self).__init__()
        self.latent_dim = latent_dim

        self.enc1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=5, stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(32 * 6 * 6 * 6, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 32 * 6 * 6 * 6)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(64, 1, kernel_size=5, stride=2, output_padding=1),
        )

    def forward(self, x):
        x1 = self.enc1(x) 
        x2 = self.enc2(x1) 

        x3 = F.max_pool3d(x2, 2)
        x3 = F.dropout(x3, p=0.3)

        x_flat = x3.view(x3.size(0), -1)
        x_latent = self.fc1(x_flat)
        x_recon = self.fc2(x_latent)
        x_recon = x_recon.view(-1, 32, 6, 6, 6)

        x_recon = self.deconv1(x_recon)
        x_recon = self.deconv2(torch.cat([x2, x_recon], dim=1))
        x_recon = self.deconv3(torch.cat([x1, x_recon], dim=1))

        return x_recon  # Use BCEWithLogitsLoss on this
