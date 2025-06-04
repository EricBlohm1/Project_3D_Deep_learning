from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
# Dropout values from this repo: https://github.com/MonteYang/VoxNet.pytorch/blob/master/voxnet.py

class VoxNet(nn.Module):
    def __init__(self, num_classes):
        super(VoxNet,self).__init__()
        self.conv = nn.Sequential( 
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2),  # Output: (32, 14, 14, 14)
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1), # Output: (32, 12, 12, 12)
            nn.ReLU(),
            nn.MaxPool3d(2),                                                     # Output: (32, 6, 6, 6)
            nn.Dropout(p=0.3)
        )

        self.classification = nn.Sequential(
            nn.Linear(32*6*6*6, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )

        #self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0, std=0.01)  # Zero-mean Gaussian with σ=0.01
            elif isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_in',nonlinearity='relu') 


    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classification(x)
        return x
        

