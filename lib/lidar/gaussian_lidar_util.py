import torch
import os
import json
import numpy as np
from plyfile import PlyData, PlyElement



class LiDARCamera():
    def __init__(self, data_path, frame_id, camera_id, device='cuda'):
        self.data_path = data_path
        self.frame_id = frame_id
        self.camera_id = camera_id
        self.device = device

        interval = self.union_mask(camera_id != 0)
        slices = []
        if interval[0] < interval[1]:
            slices.append(slice(interval[0], interval[1]))
        else:
            slices.append(slice(None, interval[1]))
            slices.append(slice(interval[0], None))

        lidar_ray = np.load(os.path.join(data_path, f'lidar_ray/{frame_id:06d}_{camera_id}.npz'))
        self.lidar_coordinate = torch.from_numpy(lidar_ray['lidar_coordinate'])
        self.lidar_ray_directions = torch.from_numpy(np.concatenate([lidar_ray['lidar_ray_directions'][:,s] for s in slices], axis=1))
        self.h, self.w, _ = self.lidar_ray_directions.shape
        
        self.range_image_1 = torch.from_numpy(np.concatenate([lidar_ray['range_image_1'][:,s] for s in slices], axis=1))
        self.mask_1 = torch.from_numpy(np.concatenate([lidar_ray['mask_1'][:,s] for s in slices], axis=1))
        self.range_image_2 = torch.from_numpy(np.concatenate([lidar_ray['range_image_2'][:,s] for s in slices], axis=1))
        self.mask_2 = torch.from_numpy(np.concatenate([lidar_ray['mask_2'][:,s] for s in slices], axis=1))

        with open(os.path.join(data_path, 'timestamps.json'), 'r') as f:
            timestamps = json.load(f)

        self.timestamp = timestamps['FRAME'][f'{frame_id:06d}']

    def union_mask(self, masking = True):
        union_mask = np.load(os.path.join(self.data_path, f'lidar_union_mask/{self.camera_id}.npy'))
        if not masking:
            return 0, union_mask.shape[1]
        
        col_mask = np.any(union_mask, axis=0)
        intervals = []
        in_interval = False
        for i, col_bool in enumerate(col_mask):
            if not col_bool:
                if not in_interval:
                    in_interval = True
                    intervals.append([i, i])
                else:
                    intervals[-1][1] = i
            else:
                if in_interval:
                    in_interval = False
                else:
                    continue
        if intervals[0][0] == 0 and intervals[-1][1] == len(col_mask) - 1:
            intervals[0][0] = intervals.pop()[0]
        interval_id = np.argsort([(tup[1] - tup[0]) % len(col_mask) for tup in intervals])[-1]
        interval = intervals[interval_id]

        # return_mask = np.ones_like(union_mask)
        # if interval[0] < interval[1]:
        #     return_mask[interval[0]:interval[1] + 1] = False
        # else:
        #     return_mask[interval[0]:] = False
        #     return_mask[:interval[1] + 1] = False

        return interval


    def get_ray_o(self):
        return self.lidar_coordinate.repeat(self.h * self.w, 1).to(self.device).to(torch.float32)
    
    def get_ray_d(self):
        ray_d_world = self.lidar_ray_directions.reshape(-1, 3)
        ray_d_world /= torch.linalg.vector_norm(ray_d_world, ord=2, dim=-1, keepdim=True)
        return ray_d_world.to(self.device).to(torch.float32)
    
    def get_depth(self, second_response=False):
        if not second_response:
            return self.range_image_1[:, :, 0].reshape(-1).to(self.device).to(torch.float32)
        else:
            return self.range_image_2[:, :, 0].reshape(-1).to(self.device).to(torch.float32)
        
    def get_intensity(self, second_response=False):
        if not second_response:
            return self.range_image_1[:, :, 1].reshape(-1).to(self.device).to(torch.float32)
        else:
            return self.range_image_2[:, :, 1].reshape(-1).to(self.device).to(torch.float32)
        
    def get_mask(self, second_response=False):
        if not second_response:
            return self.mask_1.reshape(-1).to(self.device)
        else:
            return self.mask_2.reshape(-1).to(self.device)

    def get_raydrop(self, second_response=False):
        return self.get_mask(second_response).to(torch.float32)



def save_ply(points_np, weights_np = None, lidar_position = None, gt_np = None, ply_path = 'pc.ply'):
    num_points = points_np.shape[0]
    if lidar_position is not None:
        num_points += 1
    if gt_np is not None:
        num_points += gt_np.shape[0]
    vertices = np.zeros(num_points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')])
    vertices['x'][:points_np.shape[0]] = points_np[:, 0]
    vertices['y'][:points_np.shape[0]] = points_np[:, 1]
    vertices['z'][:points_np.shape[0]] = points_np[:, 2]
    if weights_np is not None and weights_np.shape[0] == points_np.shape[0]:
        vertices['red'][:points_np.shape[0]] = weights_np * 127 + 128
        vertices['green'][:points_np.shape[0]] = weights_np * 127 + 128
        vertices['blue'][:points_np.shape[0]] = weights_np * 127 + 128
    else:
        vertices['red'][:points_np.shape[0]] = 255
        vertices['green'][:points_np.shape[0]] = 255
        vertices['blue'][:points_np.shape[0]] = 255
    if gt_np is not None:
        vertices['x'][points_np.shape[0]:points_np.shape[0] + gt_np.shape[0]] = gt_np[:, 0]
        vertices['y'][points_np.shape[0]:points_np.shape[0] + gt_np.shape[0]] = gt_np[:, 1]
        vertices['z'][points_np.shape[0]:points_np.shape[0] + gt_np.shape[0]] = gt_np[:, 2]
        vertices['red'][points_np.shape[0]:points_np.shape[0] + gt_np.shape[0]] = 63
        vertices['green'][points_np.shape[0]:points_np.shape[0] + gt_np.shape[0]] = 63
        vertices['blue'][points_np.shape[0]:points_np.shape[0] + gt_np.shape[0]] = 255
    if lidar_position is not None:
        vertices['x'][-1] = lidar_position[0]
        vertices['y'][-1] = lidar_position[1]
        vertices['z'][-1] = lidar_position[2]
        vertices['red'][-1] = 255
        vertices['green'][-1] = 63
        vertices['blue'][-1] = 63

    vertex_element = PlyElement.describe(vertices, 'vertex')
    ply_data = PlyData([vertex_element], text=True)
    ply_data.write(ply_path)


