import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Comments:
A_feat is passed through for training, used in the loss.
"""

class TNet(nn.Module):
    def __init__(self, dim, num_points=1024):
        super(TNet, self).__init__()

        self.dim = dim

        self.conv1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, dim*dim)

        ## Apply after each block
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.max_pool = nn.MaxPool1d(kernel_size=num_points)

    def forward(self, x):
        bs = x.shape[0]

        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))

        # Keep batches, max pool everything else!
        x = self.max_pool(x).view(bs, -1)

        x = self.bn4(F.relu(self.fc1(x)))
        x = self.bn5(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))

        # Identity for stability, to not converge toward 0 matrix.
        identity = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            identity = identity.cuda()
        
        # reshape to dim x dim matrix.
        x = x.view(-1, self.dim, self.dim) + identity

        return x
    


class PointNetBackbone(nn.Module):
    def __init__(self, num_points=1024):
        super(PointNetBackbone, self).__init__()
        self.num_points = num_points
        
        # Spatial transformer networks
        self.tnet1 = TNet(dim=3, num_points=num_points)
        self.tnet2 = TNet(dim=64, num_points=num_points)

        # Shared MLP 1
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)

        # Shared MLP 2
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, self.num_points, kernel_size=1)

        # Batch norms
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.num_points)

        # Maxpool for global feats
        self.maxpool = nn.MaxPool1d(self.num_points)

    def forward(self, x):
        bs = x.shape[0]

        # Input transformation
        A_input = self.tnet1(x)
        x = torch.bmm(x.transpose(2,1), A_input).transpose(2,1)

    	# Pass through first MLP
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))

        # Feature transformation
        A_feat = self.tnet2(x)

        x = torch.bmm(x.transpose(2,1), A_feat).transpose(2,1)

        # Pass through second MLP
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))

        global_features = self.maxpool(x)
        global_features = global_features.view(bs, -1)

        return global_features, A_feat

class PointNetCls(nn.Module):
    def __init__(self, num_points=1024, num_classes = 10): 
        super(PointNetCls,self).__init__()
        self.num_points = num_points
        self.num_classes = num_classes

        self.backbone = PointNetBackbone()

        self.fc1 = nn.Linear(num_points,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,num_classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        # The paper states that batch norm was only added to the layer 
        # before the classication layer, but another version adds dropout  
        # to the first 2 layers
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x, A_feat = self.backbone(x)

        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        # return logits
        return x, A_feat

        

## Classifier without transformations.
class PointNet(nn.Module):
    def __init__(self, classes=10):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, classes),
            nn.LogSoftmax(dim=1) # Used because of NLLLoss()
        )

    def forward(self, input):
        # input shape: (B, N, 3) → transpose to (B, 3, N) to work with pytorch
        xb = input.transpose(1, 2)

        xb = self.conv_block(xb)
        
        # Global max pooling
        xb = torch.max(xb, 2)[0]  # (B, 1024)
        output = self.classifier(xb)
        
        return output



def main():
    test_data = torch.rand(32, 3, 1024)

    ## test T-net
    tnet = TNet(dim=3)
    transform = tnet(test_data)
    print(f'T-net output shape: {transform.shape}')

    ## test backbone
    pointfeat = PointNetBackbone()
    out, _ = pointfeat(test_data)
    print(f'Global Features shape: {out.shape}')

    # test on single batch (should throw error if there is an issue)
    pointfeat = PointNetBackbone().eval()
    out, _ = pointfeat(test_data[0, :, :].unsqueeze(0))

    ## test classification head
    classifier = PointNetCls(num_classes=10)
    out, _ = classifier(test_data)
    print(f'Class output shape: {out.shape}')


if __name__ == "__main__":
    main()