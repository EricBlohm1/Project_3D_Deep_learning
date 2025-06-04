import os
import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset

class ModelNet10PointNetDataset(Dataset):
    """
        root_dir: Folder where the dataset is located
        split: either train or test. Train default
    """
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split

        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Filter only valid class names, this removes the readme.
        class_names = sorted([d for d in os.listdir(self.root_dir)
                            if os.path.isdir(os.path.join(self.root_dir, d))])

        for idx, class_name in enumerate(class_names):
            class_path = os.path.join(self.root_dir, class_name)
            split_path = os.path.join(class_path, self.split)

            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name

            for file in os.listdir(split_path):
                file_path = os.path.join(split_path, file)
                self.samples.append((file_path, idx))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        file_path, class_idx = self.samples[idx]
        mesh = trimesh.load(file_path)

        sampled_points = mesh.sample(1024)

        # Normalize point cloud over unit sphere
        sampled_points = sampled_points - np.mean(sampled_points, axis=0)  # Center the cloud
        max_dist = np.max(np.linalg.norm(sampled_points, axis=1))  # Furthest point from origin
        sampled_points = sampled_points / max_dist  # Scale so that all points lie inside a radius of 1. (unit sphere)

        pointcloud_tensor = torch.from_numpy(sampled_points.astype(np.float32))  # Convert to tensor

        return pointcloud_tensor.T, class_idx
        