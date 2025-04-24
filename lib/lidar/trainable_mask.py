import torch
from torch import nn

class LiDARMask(nn.Module):
    def __init__(self, lidar_shapes: dict, lr):
        super().__init__()
        self.params = {}
        self.params = {
            lidar_id: {
                "depth": nn.Parameter(torch.ones(lidar_shapes[lidar_id], dtype=torch.float32, device='cuda')),
                "intensity": nn.Parameter(torch.ones(lidar_shapes[lidar_id], dtype=torch.float32, device='cuda')),
                "raydrop": nn.Parameter(torch.ones(lidar_shapes[lidar_id], dtype=torch.float32, device='cuda'))
            }
            for lidar_id in lidar_shapes
        }
        optim_params = []
        for lidar_id, mask_dict in self.params.items():
            optim_params.append(mask_dict["depth"])
            optim_params.append(mask_dict["intensity"])
            optim_params.append(mask_dict["raydrop"])

        self.optimizer = torch.optim.Adam(optim_params, lr=lr)
            
    def forward(self, depth, intensity, raydrop, lidar_id):
        depth = depth * self.params[lidar_id]["depth"]
        intensity = intensity * self.params[lidar_id]["intensity"]
        raydrop = raydrop * self.params[lidar_id]["raydrop"]
        return depth, intensity, raydrop
    
    def update_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad()