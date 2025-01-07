import torch
import numpy as np
from plyfile import PlyData, PlyElement

from lib.models.gaussian_model import GaussianModel
from gaussian_lidar_renderer import GaussianLidarRenderer as CUDARenderer
path = 'output/waymo_full_exp/waymo_train_002/point_cloud/iteration_50000/point_cloud.ply'


def chamfer_distance(pointcloud1, mask1, pointcloud2, mask2):
    """
    Calculate Chamfer Distance between two point clouds.
    
    Args:
        pointcloud1 (torch.Tensor): Tensor of shape (N, 3) representing the first point cloud.
        mask1 (bool tensor): size (N)
        pointcloud2 (torch.Tensor): Tensor of shape (N, 3) representing the second point cloud.
        mask2 (bool tensor): size (N)
    
    Returns:
        torch.Tensor: Scalar value of Chamfer Distance.
    """
    chunk_size = 8

    pointcloud1 = pointcloud1[mask1]
    pointcloud2 = pointcloud2[mask2]
    
    chamfer_dist1 = 0
    for point1 in pointcloud1:
        dist1 = torch.cdist(point1.reshape(1, 3), pointcloud2, p=2)
        chamfer_dist1 += torch.min(dist1, dim=1).values.item()
    chamfer_dist1 /= len(pointcloud1)

    chamfer_dist2 = 0
    for point2 in pointcloud2:
        dist2 = torch.cdist(point2.reshape(1, 3), pointcloud1, p=2)
        chamfer_dist2 += torch.min(dist2, dim=1).values.item()
    chamfer_dist2 /= len(pointcloud2)

    return chamfer_dist1 + chamfer_dist2 

def f1_score(pointcloud1, mask1, pointcloud2, mask2, threshold=0.05):
    distance_close = torch.norm(pointcloud1 - pointcloud2, dim=1) < threshold
    # tp = torch.sum(mask1 & mask2 & distance_close)
    tp = torch.sum(mask1 & mask2)
    fp = torch.sum(mask1) - tp
    tn = mask1.shape[0] - torch.sum(mask1 | mask2)
    fn = torch.sum(mask2) - torch.sum(mask1 & mask2)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)

def RMSE(pointcloud1, mask1, pointcloud2, mask2):
    pc1 = pointcloud1[mask1 & mask2]
    pc2 = pointcloud2[mask1 & mask2]
    assert pc1.shape == pc2.shape, "RMSE Point clouds must have the same shape."
    rmse = torch.sqrt(torch.mean(torch.sum((pc1 - pc2) ** 2, dim=1)))
    return rmse

def MAE(pointcloud1, mask1, pointcloud2, mask2):
    pc1 = pointcloud1[mask1 & mask2]
    pc2 = pointcloud2[mask1 & mask2]
    assert pc1.shape == pc2.shape, "MAE Point clouds must have the same shape."
    mae = torch.mean(torch.mean(torch.abs(pc1 - pc2), dim=1))
    return mae

def eval(pointcloud1, mask1, pointcloud2, mask2, verbose=True):
    cd = chamfer_distance(pointcloud1, mask1, pointcloud2, mask2)
    f_score = f1_score(pointcloud1, mask1, pointcloud2, mask2)
    rmse = RMSE(pointcloud1, mask1, pointcloud2, mask2)
    mae = MAE(pointcloud1, mask1, pointcloud2, mask2)
    if verbose:
        print(f'CD: {cd}, F-score: {f_score}, RMSE: {rmse}, MAE: {mae}')
    return cd, f_score, rmse, mae


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

if __name__ == "__main__":
    cd_sum, f_score_sum, rmse_sum, mae_sum = 0, 0, 0, 0
    cd_n, f_score_n, rmse_n, mae_n = 0, 0, 0, 0
    for frame_idx in range(146, 200, 2):
        frame_arr = np.load(f"npz_file/{frame_idx}.npz", allow_pickle=True)
        lidar_pos = frame_arr['lidar_pos'].reshape(3) + np.array([0.32755102778553224, 0.11152884235235824, 0])
        frame_beams = frame_arr['beams'] + np.array([[0, 0, -2.0537412522959393]])
        t_values = np.ones(frame_beams.shape[0])
        gt_pointcloud = np.zeros((len(t_values), 3))
        mask = ~np.isnan(t_values)
        t_values[~mask] = 0
        
        gt_pointcloud = lidar_pos + frame_beams * t_values.reshape(-1, 1)
        gt_pointcloud = torch.tensor(gt_pointcloud, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.bool)

        with torch.no_grad():
            gaussian = GaussianModel()
            gaussian.load_ply(path)

            beams = torch.tensor(frame_beams, dtype=torch.float, device="cuda")
            beams = beams / torch.norm(beams, dim=1, keepdim=True)
            lidar_position = torch.tensor(lidar_pos, dtype=torch.float, device="cuda")
            lidar_position = lidar_position.repeat(beams.shape[0], 1)

            means3D = gaussian.get_xyz
            scales = gaussian.get_scaling
            rotations = gaussian.get_rotation
            opacity = gaussian.get_opacity

            n_contribute, rendered_weights, rendered_t_values = CUDARenderer.apply(means3D, scales, rotations, opacity, lidar_position, beams, 20)
            render_mask = (rendered_t_values.reshape(-1) > 0.1) & (rendered_t_values.reshape(-1) < 74) & (rendered_weights.reshape(-1) > 0.5)

            render_pointcloud = lidar_position + rendered_t_values.repeat(1, 3) * beams
            rendered_weights = rendered_weights.reshape(-1)

            render_pointcloud = render_pointcloud.cpu()
            render_mask = render_mask.cpu()
            cd, f_score, rmse, mae = eval(render_pointcloud, render_mask, gt_pointcloud, mask, True)
            cd_sum += cd
            cd_n += 1
            f_score_sum += f_score
            f_score_n += 1
            rmse_sum += rmse
            rmse_n += 1
            mae_sum += mae
            mae_n += 1
        
            save_ply(gt_pointcloud[mask].numpy(), np.ones(len(gt_pointcloud[mask])), lidar_pos, gaussian.get_xyz.cpu().numpy()[::100], f'npz_file/{frame_idx}_gt.ply')
            save_ply(render_pointcloud[render_mask].numpy(), np.ones(len(render_pointcloud[render_mask])), lidar_pos, gaussian.get_xyz.cpu().numpy()[::100], f'npz_file/{frame_idx}_re.ply')

    print(f'Average CD: {cd_sum / cd_n}, F-score: {f_score_sum / f_score_n}, RMSE: {rmse_sum / rmse_n}, MAE: {mae_sum / mae_n}')
