import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

class VoxNetAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VoxNetAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential( 
            nn.Conv3d(1, 32, kernel_size=5, stride=2),   # (32, 14, 14, 14)
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv3d(32, 32, kernel_size=3, stride=1),  # (32, 12, 12, 12)
            nn.ReLU(),
            nn.MaxPool3d(2),                             # (32, 6, 6, 6)
            nn.Dropout(p=0.3)
        )
        self.fc1 = nn.Linear(32 * 6 * 6 * 6, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 32 * 6 * 6 * 6)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2),                 # 6 → 12
            nn.ReLU(),
            nn.ConvTranspose3d(32, 32, kernel_size=3, stride=1),                 # 12 → 14
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=5, stride=2, output_padding=1) # 14 → 32
        )


    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)            # (B, 6912), flatten
        x = self.fc1(x)                      # (B, latent_dim)
        x = self.fc2(x)                      # (B, 6912)
        x = x.view(-1, 32, 6, 6, 6)          # (B, 32, 6, 6, 6)
        x = self.decoder(x)
        return x
