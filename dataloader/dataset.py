import os
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from collections import namedtuple

from SalsaNext.laserscan import SemLaserScan
from dataloader.dataset import WADS




class WADS(Dataset):
    def __init__(self, root, config):
        self.root = root
        self.scans = []
        self.labels = []
        self.projection_H = config['projection_H']
        self.projection_W = config['projection_W']
        self.sensor_fov_up = config['sensor_fov_up']
        self.sensor_fov_down = config['sensor_fov_down']
        self.sensor_img_means = torch.tensor(config['sensor_img_means'], dtype=torch.float)
        self.sensor_img_stds = torch.tensor(config['sensor_img_stds'], dtype=torch.float)
        self.max_points=config['max_points']
        base_dir = os.path.join(root, 'C:\Users\pudhin\Downloads\Lidar-WeatherNet-main\Lidar-WeatherNet-main\dataloader\Cleaned-WADS')  # Update with the actual structure

        # Iterate through the dataset directory
    

        for folder in os.walk(base_dir):
            # Add .label files to labels list
            if "labels" in folder[0]:
                for file in sorted(folder[-1]):
                    label_path = os.path.join('C:\Users\pudhin\Downloads\Lidar-WeatherNet-main\Lidar-WeatherNet-main\dataloader\Cleaned-WADS\sequences\11\labels',folder[0], file)  # Full path to label file
                    self.labels.append(label_path)
            # Add .bin files to scans list
            elif "velodyne" in folder[0]:
                for file in sorted(folder[-1]):
                    scan_path = os.path.join('C:\Users\pudhin\Downloads\Lidar-WeatherNet-main\Lidar-WeatherNet-main\dataloader\Cleaned-WADS\sequences\11\labels',folder[0], file)  # Full path to scan file
                    self.scans.append(scan_path)
        # Expand the dataset directory
        for folder in os.walk(self.root):
            # Add .label files to labels list
            if "labels" in folder[0]:
                for file in sorted(folder[-1]):
                    self.labels.append(os.path.join(folder[0], file))
            # Add .bin files to scans list
            elif "velodyne" in folder[0]:
                for file in sorted(folder[-1]):
                    self.scans.append(os.path.join(folder[0], file))
                    
    def __len__(self):
        return len(self.labels)
                
    def __getitem__(self, idx):
        scan_file = self.scans[idx]
        label_file = self.labels[idx]

        scan = SemLaserScan(project=True,
                            H=self.projection_H,
                            W=self.projection_W,
                            #fov_up=self.sensor_fov_up,
                            #fov_down=self.sensor_fov_down
                           )
        
        scan.open_scan(scan_file)
        scan.open_label(label_file)
        
        ################################### This part is inspired by SalsaNext ###################################
        # https://github.com/Halmstad-University/SalsaNext/blob/master/train/tasks/semantic/dataset/kitti/parser.py
        # Original unprojected points
        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
        unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
        unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)

        # Projected points
        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)
        proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
        proj_labels = proj_labels * proj_mask

        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
        proj = torch.cat([proj_range.unsqueeze(0).clone(),
                          proj_xyz.clone().permute(2, 0, 1),
                          proj_remission.unsqueeze(0).clone()])
        
        proj = (proj - self.sensor_img_means[:, None, None]
                ) / self.sensor_img_stds[:, None, None]
        
        proj = proj * proj_mask.float()

        path_norm = os.path.normpath(scan_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")
        ##########################################################################################################

        # If label is snow, change it to 1 else to 0
        proj_labels = torch.where(proj_labels==110, 1, 0)
        unproj_labels = torch.where(unproj_labels==110, 1, 0)

        # Add channel dimensions and reorder channel
        proj_xyz = proj_xyz.permute(2, 0, 1)
        proj_range = proj_range.unsqueeze(0)
        proj_remission = proj_remission.unsqueeze(0)
        proj_labels = proj_labels.unsqueeze(0)

        output = namedtuple('dataset_output',
                            ['proj',
                             'proj_mask',
                             'proj_labels',
                             'unproj_labels',
                             'path_seq',
                             'path_name',
                             'proj_x',
                             'proj_y',
                             'proj_range',
                             'unproj_range',
                             'proj_xyz',
                             'unproj_xyz',
                             'proj_remission',
                             'unproj_remissions',
                             'unproj_n_points',])

        return output(proj,
                      proj_mask,
                      proj_labels,
                      unproj_labels,
                      path_seq,
                      path_name, proj_x,
                      proj_y,
                      proj_range,
                      unproj_range,
                      proj_xyz,
                      unproj_xyz,
                      proj_remission,
                      unproj_remissions,
                      unproj_n_points)