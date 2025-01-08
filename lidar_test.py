# srun --pty -p a100 --qos a100 --cpus-per-task=8 --gres=gpu:1 python lidar_test.py --config configs/example/waymo_train_002.yaml

import torch
import numpy as np
from plyfile import PlyData, PlyElement
# from lib.models.gaussian_model import GaussianModel
# from lib.lidar.gaussian_lidar_renderer import GaussianLidarRenderer as PyRenderer
from gaussian_lidar_renderer import GaussianLidarRenderer as CUDARenderer

import time


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
        start_time = time.time()
        reflect_points, weights = renderer.render(lidar_positions=[lidar_position], beams=[beams], lidar_ranges=74, covariance=covariance, xyz=xyz, opacity=opacity)
        render_time = time.time() - start_time
        reflect_points = torch.cat(reflect_points)
        weights = torch.cat(weights)

        return reflect_points, None, weights, render_time  # mask
    
    elif method == 'cuda':
        beams = beams[::1]
        beams = beams / torch.norm(beams, dim=1, keepdim=True)
        lidar_position = lidar_position.repeat(beams.shape[0], 1)

        step = 1
        means3D = gaussian.get_xyz[::step]
        scales = gaussian.get_scaling[::step]
        rotations = gaussian.get_rotation[::step]
        opacity = gaussian.get_opacity[::step]

        print(torch.min(means3D, dim=0).values, torch.max(means3D, dim=0).values)

        start_time = time.time()
        n_contribute, weights, t_values = CUDARenderer.apply(means3D, scales, rotations, opacity, lidar_position, beams, 50)
        render_time = time.time() - start_time
        mask = (t_values.reshape(-1) > 0.1) & (t_values.reshape(-1) < 74) & (weights.reshape(-1) > 0.5)

        reflect_points = lidar_position + t_values.repeat(1, 3) * beams
        weights = weights.reshape(-1)
        return reflect_points, mask, weights, render_time

def render_ply(method):
    print(method)
    with torch.no_grad():
        gaussian = GaussianModel()
        gaussian.load_ply(path)

        frame_arr = np.load(f"npz_file/0.npz", allow_pickle=True)

        lidar_position = torch.tensor([60,9,3], dtype=torch.float, device="cuda")
        beams = torch.tensor(np.load('lidar_beams.npy'), dtype=torch.float, device="cuda")

        reflect_points, mask, weights, render_time = render(lidar_position, beams, gaussian, 1, method)
        try:
            reflect_points = reflect_points[mask]
            weights = weights[mask]
        except:
            pass

        print(f"lidar render {render_time} seconds         weights {torch.min(weights).item()} {torch.max(weights).item()}")
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



def save_ply(points_np, weights_np, lidar_position, gaussian_np, ply_path):
    vertices = np.zeros(points_np.shape[0] + gaussian_np.shape[0] + 1, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')])
    vertices['x'][:points_np.shape[0]] = points_np[:, 0]
    vertices['y'][:points_np.shape[0]] = points_np[:, 1]
    vertices['z'][:points_np.shape[0]] = points_np[:, 2]
    vertices['red'][:points_np.shape[0]] = weights_np * 255
    vertices['green'][:points_np.shape[0]] = weights_np * 255
    vertices['blue'][:points_np.shape[0]] = weights_np * 255
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
    ply_data.write(ply_path)


def rand_data_gen(nx = 10, ny = 10, hor_res = torch.pi / 6, ver_res = torch.pi / 12, ver_min = 0.5 * torch.pi, ver_max = 0.8 * torch.pi):
    means3D = torch.hstack([torch.stack(torch.meshgrid(torch.arange(-(nx // 2), nx // 2 + nx % 2), torch.arange(-(ny // 2), ny // 2 + ny % 2)), dim=2).reshape(-1, 2), torch.zeros((nx * ny, 1))]).to(torch.float).to("cuda")
    
    scaling = torch.tensor([1, 1, 0.1]).repeat(nx * ny, 1).to(torch.float).to("cuda")
    
    rotations = torch.tensor([1, 0, 0, 0]).repeat(nx * ny, 1).to(torch.float).to("cuda")
    
    opacity = torch.ones(nx * ny).to(torch.float).to("cuda")
    
    lidar_sh = torch.rand((9, 2)).repeat(nx * ny, 1, 1).to(torch.float).to("cuda")
    max_sh_degree = 2# 0:1, 1:4, 2:9, 3:16
    
    lidar_position = torch.tensor([0,0,2], dtype=torch.float, device="cuda")

    alpha, sigma = torch.meshgrid(torch.arange(0, 2 * torch.pi, hor_res), torch.arange(ver_min, ver_max, ver_res))
    beams = torch.stack([torch.sin(sigma) * torch.sin(alpha), torch.sin(sigma) * torch.cos(alpha), torch.cos(sigma)], dim=2).reshape(-1, 3).to(torch.float).to("cuda")
    
    return means3D, scaling, rotations, opacity, lidar_sh, max_sh_degree, lidar_position, beams

def print_lidar_returns(beams, lidar_n_contribute, lidar_weights, lidar_t_values, lidar_intensity, lidar_raydrop, mask):
    for beam, n_contribute, weight, t_value, intensity, raydrop, m in zip(beams, lidar_n_contribute, lidar_weights, lidar_t_values, lidar_intensity, lidar_raydrop, mask):
        print(f"[{m}] beam: {beam} n_contribute: {n_contribute} weight: {weight} t_value: {t_value} intensity: {intensity} raydrop: {raydrop}")


def test_lidar_function():
    means3D, scaling, rotations, opacity, lidar_sh, max_sh_degree, lidar_position, beams = rand_data_gen()

    aabb_scale = 20
    lidar_n_contribute, lidar_weights, lidar_t_values, lidar_intensity, lidar_raydrop = CUDARenderer.apply(
        means3D,
        scaling,
        rotations,
        opacity,
        lidar_sh,
        max_sh_degree,
        lidar_position,
        beams,
        aabb_scale)
    lidar_mask_rule = (lidar_t_values.reshape(-1) > 0.1) & (lidar_t_values.reshape(-1) < 74) & (lidar_weights.reshape(-1) > 0.5)
    lidar_mask_raydrop = lidar_raydrop > 0.5
    mask = lidar_mask_raydrop

    print_lidar_returns(beams, lidar_n_contribute, lidar_weights, lidar_t_values, lidar_intensity, lidar_raydrop, mask)
    lidar_pointcloud = lidar_position + lidar_t_values[mask].repeat(1, 3) * beams[mask]
    lidar_weights = lidar_weights[mask]
    save_ply(lidar_pointcloud.cpu().numpy(), lidar_weights.cpu().numpy(), lidar_position.cpu().numpy(), means3D.cpu().numpy(), "pc.ply")


if __name__ == "__main__":
    test_lidar_function()