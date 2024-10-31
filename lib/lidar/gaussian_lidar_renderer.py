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

    def render(self, lidar_position: torch.Tensor, beams: torch.Tensor, gaussian: GaussianModel, lidar_range: float=74, angle_range: float=0.999, weight_treshold: float=-100, step: float=100):
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

    def render_beams(self, lidar_range: float, angle_range: float, weight_treshold: float, lidar_position: torch.Tensor, beam: torch.Tensor, covariance: torch.Tensor, xyz: torch.Tensor, opacity: torch.Tensor, w_gaussian: float=0.01, w_distance: float=0):
        distance = xyz - lidar_position
        line_distance = (distance - torch.einsum('i,j->ij', ((distance @ beam) / (beam @ beam)), beam)).norm(dim=1)
        intersect_position_coefficient = torch.einsum('bi,bij,j->b', distance, covariance, beam) / torch.einsum('i,bij,j->b', beam, covariance, beam)
        front_mask = (intersect_position_coefficient > 0) & (distance @ beam / (beam.norm() * distance.norm(dim=1)) > angle_range) & (beam.norm() * intersect_position_coefficient < lidar_range)
        
        distance = distance[front_mask]
        line_distance = line_distance[front_mask]
        intersect_position_coefficient = intersect_position_coefficient[front_mask]
        covariance = covariance[front_mask]
        xyz = xyz[front_mask]
        opacity = opacity[front_mask]
        
        intersect_positions = torch.einsum('b,i->bi', intersect_position_coefficient, beam) + lidar_position
        intersect_mu = intersect_positions - xyz
        
        do_ablend = True
        if not do_ablend:
            weights = torch.log(opacity.reshape(-1)) + w_gaussian * (-torch.einsum('bi,bij,bj->b', intersect_mu, covariance, intersect_mu)) - w_distance * torch.log(line_distance)
            # print(f"weights = {torch.log(opacity.reshape(-1))} + {torch.log(front_mask)} + {w_gaussian * (-torch.einsum('bi,bij,bj->b', intersect_mu, covariance, intersect_mu))}")

            reflect_mask = weights > weight_treshold
            if torch.any(reflect_mask):
                reflect_points = intersect_positions[reflect_mask]
                reflect_point_coefficient = intersect_position_coefficient[reflect_mask]
                reflect_weights = weights[reflect_mask]
                # print(reflect_weights.shape[0], torch.max(reflect_weights), torch.mean(reflect_weights), torch.min(reflect_weights))
                # print(reflect_point_coefficient.shape[0], torch.max(reflect_point_coefficient), torch.mean(reflect_point_coefficient), torch.min(reflect_point_coefficient))
                response1_index = torch.argmin(0 * reflect_point_coefficient - 1 * reflect_weights)

                if (reflect_points[response1_index] - lidar_position).norm() > lidar_range:
                    print('[weight] ', reflect_weights[response1_index])
                    print('[distance] ', reflect_points[response1_index] - lidar_position)
                    print('[mask] ', front_mask[reflect_mask][response1_index])
                    print(f'[mask angle] ({distance[reflect_mask][response1_index]} @ {beam} / ({beam.norm()} * {distance[reflect_mask][response1_index].norm()} = ', (distance[reflect_mask][response1_index] @ beam / (beam.norm() * distance[reflect_mask][response1_index].norm())))
                    print('[mask distance] ', beam.norm() * reflect_point_coefficient[response1_index])
                    print('[coeff] ', reflect_point_coefficient[response1_index])
                    raise Exception
                
                return reflect_points[response1_index].clone(), float(reflect_weights[response1_index])
            else:
                return None, None
            
        else:
            remaining_opacity = 1
            intersect_position_coefficient_toreturn = 0
            weight_sum = 0
            sort_index = torch.argsort(intersect_position_coefficient)
            for w, power, coefficient in zip(opacity.reshape(-1)[sort_index], -torch.einsum('bi,bij,bj->b', intersect_mu, covariance, intersect_mu)[sort_index], intersect_position_coefficient[sort_index]):
                if power < -2.5 or power > 0:
                    continue
                alpha = min(0.99, w * torch.exp(power).item())
                if alpha < 0.004:
                    continue

                weight_sum
                intersect_position_coefficient_toreturn += remaining_opacity * alpha * coefficient
                remaining_opacity *= 1 - alpha
                
            if remaining_opacity < 0.5:
                return (intersect_position_coefficient_toreturn / (1 - remaining_opacity)) * beam + lidar_position, 1 - remaining_opacity
            return None, None


            