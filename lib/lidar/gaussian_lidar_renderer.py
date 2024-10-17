import torch
from tqdm import tqdm
import numpy as np
from plyfile import PlyData, PlyElement
from lib.models.gaussian_model import GaussianModel


def unstrip_symmetric(stripped: torch.Tensor):
    L = torch.zeros((stripped.shape[0], 3, 3), dtype=torch.float, device="cuda")
    L[:, 0, 0] = stripped[:, 0]
    L[:, 0, 1] = stripped[:, 1]
    L[:, 0, 2] = stripped[:, 2]
    L[:, 1, 0] = stripped[:, 1]
    L[:, 1, 1] = stripped[:, 3]
    L[:, 1, 2] = stripped[:, 4]
    L[:, 2, 0] = stripped[:, 2]
    L[:, 2, 1] = stripped[:, 4]
    L[:, 2, 2] = stripped[:, 5]
    return L
    

class GaussianLidarRenderer():
    def __init__(self) -> None:
        pass    

    def render(self, lidar_position: torch.Tensor, beams: torch.Tensor, gaussian: GaussianModel, lidar_range: float=74, angle_range: float=0.999, weight_treshold: float=-1000, step: float=100):
        progress_bar = tqdm(total=beams.shape[0])

        covariance = unstrip_symmetric(gaussian.get_covariance()[::step])
        covariance = torch.inverse(covariance)
        xyz = gaussian.get_xyz[::step]
        opacity = gaussian.get_opacity[::step]
        
        reflect_points = []
        weights = []
        for beam in beams:
            reflect_point, weight = self.render_beams(lidar_range, angle_range, weight_treshold, lidar_position, beam, covariance, xyz, opacity)
            if reflect_point is not None:
                reflect_points.append(reflect_point)
                weights.append(weight)
            
            progress_bar.set_postfix({"cuda mem": "{:.5f} G".format(torch.cuda.memory_allocated('cuda') / (1024 ** 3))})
            progress_bar.update()

        reflect_points = torch.stack(reflect_points)
        weights = torch.tensor(weights)

        return reflect_points, weights

    def render_beams(self, lidar_range: float, angle_range: float, weight_treshold: float, lidar_position: torch.Tensor, beam: torch.Tensor, covariance: torch.Tensor, xyz: torch.Tensor, opacity: torch.Tensor, w_gaussian: float=1, w_distance: float=0):
        distance = xyz - lidar_position
        line_distance = (distance - torch.einsum('i,j->ij', ((distance @ beam) / (beam @ beam)), beam)).norm(dim=1)
        intersect_position_parameters = torch.einsum('bi,bij,j->b', distance, covariance, beam) / torch.einsum('i,bij,j->b', beam, covariance, beam)
        intersect_positions = torch.einsum('b,i->bi', intersect_position_parameters, beam) + lidar_position
        front_mask = (distance @ beam / (beam.norm() * distance.norm(dim=1)) > angle_range) & (beam.norm() * intersect_position_parameters < lidar_range)
        intersect_mu = intersect_positions - xyz
        
        weights = torch.log(opacity.reshape(-1)) + torch.log(front_mask) + w_gaussian * (-torch.einsum('bi,bij,bj->b', intersect_mu, covariance, intersect_mu)) - w_distance * torch.log(line_distance)
        
        reflect_mask = weights > weight_treshold
        if torch.any(reflect_mask):
            reflect_points = intersect_positions[reflect_mask]
            reflect_point_parameters = intersect_position_parameters[reflect_mask]
            reflect_weights = weights[reflect_mask]
            # print(reflect_weights.shape[0], torch.max(reflect_weights), torch.mean(reflect_weights), torch.min(reflect_weights))
            # print(reflect_point_parameters.shape[0], torch.max(reflect_point_parameters), torch.mean(reflect_point_parameters), torch.min(reflect_point_parameters))
            response1_index = torch.argmin(reflect_point_parameters + 0.1 * reflect_weights)
            
            return reflect_points[response1_index].clone(), float(reflect_weights[response1_index])
        else:
            return None, None


            