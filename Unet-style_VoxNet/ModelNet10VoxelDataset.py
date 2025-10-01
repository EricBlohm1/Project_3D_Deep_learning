import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ModelNet10VoxelDataset(Dataset):
    """
        root_dir: Folder where the dataset is located
        split: either train or test. Train default
        res: resolution, 32 for voxnet, and 32 default. 
        mode: choose either 'classification' or 'partial' or slicing.
        p: percentage of voxels to mask
    """
    def __init__(self, root_dir, split='train', res=32, mode='classification', p=0.3):
        self.root_dir = root_dir
        self.split = split
        self.res = res
        self.mode = mode
        self.p = p

        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        for idx, class_name in enumerate(os.listdir(self.root_dir)):
            if class_name == 'README.txt':
                continue

            class_path = os.path.join(self.root_dir, class_name)
            split_path = os.path.join(class_path, self.split)

            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name

            for file in os.listdir(split_path):
                file_path = os.path.join(split_path, file)
                self.samples.append((file_path, idx))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        file_path, class_idx = sample


        voxel_grid = np.load(file_path)
        voxel_tensor = torch.from_numpy(voxel_grid).unsqueeze(0).float()

        if self.mode == 'classification':
            return voxel_tensor, class_idx

        elif self.mode == 'partial':
            # Create corrupted voxel grid
            corrupted = voxel_tensor.clone()
            flat = corrupted.view(-1)
            non_zero_indices = (flat > 0).nonzero(as_tuple=False).view(-1)

            if len(non_zero_indices) == 0:
                return voxel_tensor, voxel_tensor  # nothing to mask

            num_zero = int(self.p * len(non_zero_indices))
            indices_to_zero = non_zero_indices[torch.randperm(len(non_zero_indices))[:num_zero]]
            flat[indices_to_zero] = 0.0

            corrupted = flat.view_as(voxel_tensor)
            return corrupted, voxel_tensor
        
        elif self.mode == 'sliced_x':
            corrupted = voxel_tensor.clone()
            # Simulate seeing the object from above, thus the bottom of the object would disapear.
            n_remove = int(self.p * self.res)  
            corrupted[: ,:n_remove, :, :] = 0.0  
            return corrupted, voxel_tensor
        elif self.mode == 'sliced_y':
            corrupted = voxel_tensor.clone()
            # Simulate seeing the object from above, thus the bottom of the object would disapear.
            n_remove = int(self.p * self.res)  
            corrupted[: ,:, :n_remove, :] = 0.0  
            return corrupted, voxel_tensor
        elif self.mode == 'sliced_z':
            corrupted = voxel_tensor.clone()
            # Simulate seeing the object from above, thus the bottom of the object would disapear.
            n_remove = int(self.p * self.res)  
            corrupted[: ,:, :, :n_remove] = 0.0  
            return corrupted, voxel_tensor


