# srun --pty -p v100 --qos v100 --cpus-per-task=8 --gres=gpu:1 python lidar_test.py --config configs/example/waymo_train_002.yaml

import torch
import numpy as np
from plyfile import PlyData, PlyElement
from lib.models.gaussian_model import GaussianModel
from lib.lidar.gaussian_lidar_renderer import GaussianLidarRenderer as PyRenderer
from gaussian_lidar_renderer import GaussianLidarRenderer as CUDARenderer


path = 'output/waymo_full_exp/waymo_train_002/point_cloud/iteration_50000/point_cloud.ply'


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

def strip_symmetric(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def render(lidar_position, beams, gaussian, step, method):
    if method == 'py':
        beams = beams[::1]
        beams = beams / torch.norm(beams, dim=1, keepdim=True)

        covariance = unstrip_symmetric(gaussian.get_covariance()[::step])
        covariance = torch.inverse(covariance)
        covariance = strip_symmetric(covariance)
        xyz = gaussian.get_xyz[::step]
        opacity = gaussian.get_opacity[::step]
        renderer = PyRenderer()
        reflect_points, weights = renderer.render(lidar_positions=[lidar_position], beams=[beams], lidar_ranges=74, covariance=covariance, xyz=xyz, opacity=opacity)
        reflect_points = torch.cat(reflect_points)
        weights = torch.cat(weights)

        return reflect_points, weights
    
    elif method == 'cuda':
        beams = beams[::1]
        beams = beams / torch.norm(beams, dim=1, keepdim=True)
        lidar_position = lidar_position.repeat(beams.shape[0], 1)

        step = 1
        means3D = gaussian.get_xyz[::step]
        scales = gaussian.get_scaling[::step]
        rotations = gaussian.get_rotation[::step]
        opacity = gaussian.get_opacity[::step]

        # ray_tracer = RayTracer(means3D, scales, rotations, 10)
        # trace_visibility_return = ray_tracer.trace_visibility(lidar_position, beams, means3D, symm_inv, opacity, normals)
        # t_values = trace_visibility_return['tvalue']
        # weights = trace_visibility_return['visibility']
        n_contribute, weights, t_values = CUDARenderer.apply(means3D, scales, rotations, opacity, lidar_position, beams)
        mask = (t_values.reshape(-1) > 0.1) & (t_values.reshape(-1) < 74) & (weights.reshape(-1) > 0.5)

        reflect_points = lidar_position + t_values.repeat(1, 3) * beams
        reflect_points = reflect_points[mask]
        weights = weights[mask].reshape(-1)
        return reflect_points, weights

def render_ply(method):
    with torch.no_grad():
        gaussian = GaussianModel()
        gaussian.load_ply(path)

        lidar_position = torch.tensor([60,9,3], dtype=torch.float, device="cuda")
        beams = torch.tensor(np.load('lidar_beams.npy'), dtype=torch.float, device="cuda")
        reflect_points, weights = render(lidar_position, beams, gaussian, 1, 'cuda')

        print("weights", torch.min(weights).item(), torch.max(weights).item())
        weights = (weights - torch.min(weights)) / (torch.max(weights) - torch.min(weights)) * 128 + 128

        points_np = reflect_points.cpu().numpy()
        weights_np = weights.cpu().numpy()
        gaussian_np = gaussian.get_xyz.cpu().numpy()[::100]

        vertices = np.zeros(points_np.shape[0] + gaussian_np.shape[0] + 1, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')])
        vertices['x'][:points_np.shape[0]] = points_np[:, 0]
        vertices['y'][:points_np.shape[0]] = points_np[:, 1]
        vertices['z'][:points_np.shape[0]] = points_np[:, 2]
        vertices['red'][:points_np.shape[0]] = weights_np
        vertices['green'][:points_np.shape[0]] = weights_np
        vertices['blue'][:points_np.shape[0]] = weights_np
        vertices['x'][points_np.shape[0]:-1] = gaussian_np[:, 0]
        vertices['y'][points_np.shape[0]:-1] = gaussian_np[:, 1]
        vertices['z'][points_np.shape[0]:-1] = gaussian_np[:, 2]
        vertices['red'][points_np.shape[0]:-1] = 63
        vertices['green'][points_np.shape[0]:-1] = 63
        vertices['blue'][points_np.shape[0]:-1] = 255
        vertices['x'][-1] = lidar_position[0]
        vertices['y'][-1] = lidar_position[1]
        vertices['z'][-1] = lidar_position[2]
        vertices['red'][-1] = 255
        vertices['green'][-1] = 63
        vertices['blue'][-1] = 63

        vertex_element = PlyElement.describe(vertices, 'vertex')
        ply_data = PlyData([vertex_element], text=True)
        ply_data.write(f'output_lidar_{method}/pc_combined.ply')


        vertices = np.zeros(points_np.shape[0] + 1, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')])
        vertices['x'][:points_np.shape[0]] = points_np[:, 0]
        vertices['y'][:points_np.shape[0]] = points_np[:, 1]
        vertices['z'][:points_np.shape[0]] = points_np[:, 2]
        vertices['red'][:points_np.shape[0]] = weights_np
        vertices['green'][:points_np.shape[0]] = weights_np
        vertices['blue'][:points_np.shape[0]] = weights_np
        vertices['x'][-1] = lidar_position[0]
        vertices['y'][-1] = lidar_position[1]
        vertices['z'][-1] = lidar_position[2]
        vertices['red'][-1] = 255
        vertices['green'][-1] = 63
        vertices['blue'][-1] = 63

        vertex_element = PlyElement.describe(vertices, 'vertex')
        ply_data = PlyData([vertex_element], text=True)
        ply_data.write(f'output_lidar_{method}/pc_lidar.ply')

render_ply(method='cuda')
# render_ply(method='py')
