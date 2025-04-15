import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim


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
    tp = torch.sum(mask1 & mask2 & distance_close)
    # tp = torch.sum(mask1 & mask2)
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
    mae = torch.mean(torch.abs(pc1 - pc2))
    return mae

def MedAE(pointcloud1, mask1, pointcloud2, mask2):
    pc1 = pointcloud1[mask1 & mask2]
    pc2 = pointcloud2[mask1 & mask2]
    assert pc1.shape == pc2.shape, "MedAE Point clouds must have the same shape."
    medae = torch.median(torch.abs(pc1 - pc2))
    return medae

def MAE_intensity(intensity1, mask1, intensity2, mask2):
    intensity1 = intensity1[mask1 & mask2]
    intensity2 = intensity2[mask1 & mask2]
    assert intensity1.shape == intensity2.shape, "MAE Intensities must have the same shape."
    mae = torch.mean(torch.abs(intensity1 - intensity2))
    return mae

def RMSE_intensity(intensity1, mask1, intensity2, mask2):
    intensity1 = intensity1[mask1 & mask2]
    intensity2 = intensity2[mask1 & mask2]
    assert intensity1.shape == intensity2.shape, "RMSE Intensities must have the same shape."
    print(intensity1.mean(), intensity2.mean(), torch.max((intensity1 - intensity2) ** 2))
    rmse = torch.sqrt(torch.mean((intensity1 - intensity2) ** 2))
    return rmse

def PSNR_intensity(intensity1, mask1, intensity2, mask2):
    max_value = max(torch.max(intensity1), torch.max(intensity2), 1.0)
    intensity1 = intensity1[mask1 & mask2]
    intensity2 = intensity2[mask1 & mask2]
    assert intensity1.shape == intensity2.shape, "PSNR Intensities must have the same shape."
    mse = torch.mean((intensity1 - intensity2) ** 2)
    psnr = 10 * torch.log10(max_value ** 2 / mse)
    return psnr

def SSIM_intensity(intensity1, mask1, intensity2, mask2):
    max_value = max(torch.max(intensity1), torch.max(intensity2), 1.0)
    min_value = min(torch.min(intensity1), torch.min(intensity2), 0.0)
    intensity1 = intensity1[mask1 & mask2].cpu().numpy()
    intensity2 = intensity2[mask1 & mask2].cpu().numpy()
    intensity1 = intensity1.copy()
    intensity2 = intensity2.copy()
    assert intensity1.shape == intensity2.shape, "SSIM Intensities must have the same shape."
    ssim_val = ssim(intensity1, intensity2, data_range=max_value - min_value)
    return ssim_val

def pointcloud_eval(pointcloud1, mask1, pointcloud2, mask2, verbose=True):
    cd = chamfer_distance(pointcloud1, mask1, pointcloud2, mask2)
    f_score = f1_score(pointcloud1, mask1, pointcloud2, mask2)
    rmse = RMSE(pointcloud1, mask1, pointcloud2, mask2)
    mae = MAE(pointcloud1, mask1, pointcloud2, mask2)
    if verbose:
        print(f'[pointcloud_eval] CD: {cd:.4f}, F-score: {f_score:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}')
    return cd, f_score, rmse, mae

def intensity_eval(intensity1, mask1, intensity2, mask2, verbose=True):
    mae = MAE_intensity(intensity1, mask1, intensity2, mask2)
    rmse = RMSE_intensity(intensity1, mask1, intensity2, mask2)
    psnr = PSNR_intensity(intensity1, mask1, intensity2, mask2)
    ssim = SSIM_intensity(intensity1, mask1, intensity2, mask2)
    if verbose:
        print(f'[intensity_eval] MAE: {mae:.4f}, RMSE: {rmse:.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}')
    return mae, rmse, psnr, ssim

def dynamic_eval(pointcloud1, intensity1, mask1, pointcloud2, intensity2, mask2, verbose=True):
    intensity1 = torch.clamp(intensity1, 0.0, 1.0)
    intensity2 = torch.clamp(intensity2, 0.0, 1.0)
    mae = MAE(pointcloud1, mask1, pointcloud2, mask2)
    medae = MedAE(pointcloud1, mask1, pointcloud2, mask2)
    cd = chamfer_distance(pointcloud1, mask1, pointcloud2, mask2)
    rmse = RMSE_intensity(intensity1, mask1, intensity2, mask2)
    if verbose:
        print(f'[dynamic_eval] MAE: {mae:.4f}, MedAE: {medae:.4f}, CD: {cd:.4f}, Intensity-RMSE: {rmse:.4f}')
    return mae, medae, cd, rmse


if __name__ == "__main__":
    print('start eval')
    
    from lib.lidar.gaussian_lidar_util import save_ply, LiDARCamera
    import open3d as o3d
    import os

    data_path = 'data/waymo/lidargs/004'
    output_path = 'output/waymo_lidar_test/waymo_train_004_0_60-copy1/trajectory/ours_10000/lidar'

    raydrop_stat = {'num': 0, 'MAE': 0, 'MedAE': 0, 'CD': 0, 'Intensity-RMSE': 0}
    weights_stat = {'num': 0, 'MAE': 0, 'MedAE': 0, 'CD': 0, 'Intensity-RMSE': 0}
    for frame_id, camera_id in [(22, 0), (25, 0), (28, 0), (31, 0), (34, 0), (37, 0)]:
        lidar_camera = LiDARCamera(data_path, frame_id, camera_id, 'cpu')

        pointcloud_weight = o3d.io.read_point_cloud(os.path.join(output_path, f'{frame_id:06d}_{camera_id}_weights.ply'))
        points_weight = np.asarray(pointcloud_weight.points)
        colors_weight = np.asarray(pointcloud_weight.colors)
        ri = np.load(os.path.join(output_path, f'{frame_id:06d}_{camera_id}.npz'))
        range_image = ri['range_image']
        weights = (colors_weight[:,0] - 0.5) / 0.5

        result = {}
        result['depth'] = torch.from_numpy(range_image.reshape(-1, 3)[:, 0])
        result['intensity'] = torch.from_numpy(range_image.reshape(-1, 3)[:, 1])
        result['raydrop'] = torch.from_numpy(range_image.reshape(-1, 3)[:, 2])
        result['weights'] = torch.from_numpy(weights)

        pointcloud_full = torch.from_numpy(points_weight).to(torch.float32)
        pointcloud_gt = (lidar_camera.get_ray_o() + lidar_camera.get_ray_d() * lidar_camera.get_depth().reshape(-1, 1))#[lidar_camera.get_mask()]

        raydrop_mask = result['raydrop'] > 0.02
        weights_mask = result['weights'] > 0.04

        print(lidar_camera.get_intensity()[lidar_camera.get_mask()].mean(), torch.clamp(lidar_camera.get_intensity()[lidar_camera.get_mask()], 0.0, 1.0).mean())

        print(f'[frame {frame_id}, camera {camera_id}]', torch.sum(lidar_camera.get_mask()), torch.sum(raydrop_mask), torch.sum(weights_mask))

        mae, medae, cd, rmse = dynamic_eval(pointcloud_gt, lidar_camera.get_intensity(), lidar_camera.get_mask(), pointcloud_full, result['intensity'], raydrop_mask)
        raydrop_stat['num'] += 1
        raydrop_stat['MAE'] += mae
        raydrop_stat['MedAE'] += medae
        raydrop_stat['CD'] += cd
        raydrop_stat['Intensity-RMSE'] += rmse
        mae, medae, cd, rmse = dynamic_eval(pointcloud_gt, lidar_camera.get_intensity(), lidar_camera.get_mask(), pointcloud_full, result['intensity'], weights_mask)
        weights_stat['num'] += 1
        weights_stat['MAE'] += mae
        weights_stat['MedAE'] += medae
        weights_stat['CD'] += cd
        weights_stat['Intensity-RMSE'] += rmse

        # save_ply(points_np=pointcloud_full[raydrop_mask].numpy(), weights_np=result['intensity'][raydrop_mask].numpy(), ply_path='pc_raydrop.ply')
        # save_ply(points_np=pointcloud_full[weights_mask].numpy(), weights_np=result['intensity'][weights_mask].numpy(), ply_path='pc_weights.ply')

    print('AVERAGE')
    print(f'[raydrop_stat] MAE: {raydrop_stat["MAE"]/raydrop_stat["num"]:.4f}, MedAE: {raydrop_stat["MedAE"]/raydrop_stat["num"]:.4f}, CD: {raydrop_stat["CD"]/raydrop_stat["num"]:.4f}, Intensity-RMSE: {raydrop_stat["Intensity-RMSE"]/raydrop_stat["num"]:.4f}')
    print(f'[weights_stat] MAE: {weights_stat["MAE"]/weights_stat["num"]:.4f}, MedAE: {weights_stat["MedAE"]/weights_stat["num"]:.4f}, CD: {weights_stat["CD"]/weights_stat["num"]:.4f}, Intensity-RMSE: {weights_stat["Intensity-RMSE"]/weights_stat["num"]:.4f}')


        

