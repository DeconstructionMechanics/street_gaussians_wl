from plyfile import PlyData, PlyElement
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.config import cfg
from lib.datasets.dataset import Dataset
from lib.utils.camera_utils import Camera, CameraInfo, loadmask, loadmetadata, WARNED
from lib.visualizers.street_gaussian_visualizer import StreetGaussianVisualizer
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.utils.general_utils import PILtoTorch, NumpytoTorch, matrix_to_quaternion
from lib.utils.img_utils import visualize_depth_numpy
from lib.models.scene import Scene
import copy
import time
from PIL import Image
import numpy as np
import tqdm
import math
from copy import deepcopy
import cv2
from lib.utils.sh_utils import RGB2SH
import torch
import torchvision
import imageio
import torch.nn as nn
import os
from scipy.ndimage import median_filter, generic_filter, zoom

from perlin_numpy import generate_fractal_noise_2d, generate_perlin_noise_2d


def loadCamData(cam_info: CameraInfo, resolution_scale):
    orig_w, orig_h = cam_info.image.size
    if cfg.resolution in [1, 2, 4, 8]:
        scale = resolution_scale * cfg.resolution
        resolution = round(orig_w / scale), round(orig_h / scale)
    else:  # should be a type that converts to float
        if cfg.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / cfg.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    K = copy.deepcopy(cam_info.K)
    K[:2] /= scale

    image = PILtoTorch(cam_info.image, resolution,
                       resize_mode=Image.BILINEAR)[:3, ...]
    masks = loadmask(cam_info, resolution, resize_mode=Image.NEAREST)
    metadata = loadmetadata(cam_info.metadata, resolution)
    return {
        "id": cam_info.uid,
        "FoVx": cam_info.FovX,
        "FoVy": cam_info.FovY,
        "K": K,
        "image": image,
        "masks": masks,
        "image_name": cam_info.image_name,
        "metadata": metadata,
    }


def loadCam(camdata, R, T, foVx=None, foVy=None):
    if foVx is None:
        foVx = deepcopy(camdata["FoVx"])
    if foVy is None:
        foVy = deepcopy(camdata["FoVy"])
    return Camera(
        id=deepcopy(camdata["id"]),
        R=R,
        T=T,
        FoVx=foVx,
        FoVy=foVy,
        K=deepcopy(camdata["K"]),
        image=deepcopy(camdata["image"]),
        masks=deepcopy(camdata["masks"]),
        image_name=deepcopy(camdata["image_name"]),
        metadata=deepcopy(camdata["metadata"]),
    )

def is_diff_depth(ori, dst, th = 0.1):
    if math.isnan(ori) or math.isnan(dst):
        return True
    if abs(ori - dst) > th:
        return True
    return False

dX = [-1, -1, -1,  0, 0,  1, 1, 1]
dY = [-1,  0,  1, -1, 1, -1, 0, 1]
ld = 8

def count_adj(i, j, depth, th=0.5):
    flg = 0
    for k in range(ld):
        if i+dX[k] < 0 or i+dX[k] >= depth.shape[0] or j+dY[k] < 0 or j+dY[k] >= depth.shape[1]:
            continue
        if not is_diff_depth(depth[i][j], depth[i+dX[k]][j+dY[k]], th):
            flg += 1
    return flg

def is_floating_depth(i, j, depth, th=0.1):
    return count_adj(i, j, depth, th) < 4

import queue

def _count_adj_points(x, y, depth, vis, q):
    cnt_q = queue.Queue()
    cnt_q.put((x, y))
    q.put((x, y))
    cnt = 1
    while not cnt_q.empty():
        i, j = cnt_q.get()
        for k in range(ld):
            idx = i + dX[k]
            jdy = j + dY[k]
            if idx < 0 or idx >= depth.shape[0] or jdy < 0 or jdy >= depth.shape[1]:
                continue
            if (not is_diff_depth(depth[idx][jdy], depth[i][j], th=0.5)) and vis[idx][jdy] == 0:
            # if (not np.isnan(depth[idx][jdy])) and vis[idx][jdy] == 0:
                vis[idx][jdy] = 1
                cnt_q.put((idx, jdy))
                q.put((idx, jdy))
                cnt += 1
    return cnt

def count_adj_points(i, j, depth, cnt_arr):
    vis = np.zeros(depth.shape)
    vis[i][j] = 1
    q = queue.Queue()
    cnt = _count_adj_points(i, j, depth, vis, q)
    while not q.empty():
        x, y = q.get()
        cnt_arr[x][y] = cnt

def adaptive_median_filter(image, n=3, A=0.5):
    """
    Apply an adaptive median filter to smooth an image while preserving edges.
    
    Parameters:
        image (np.array): Grayscale image as a NumPy array.
        n (int): Size of the filter window (must be odd).
        A (int): Intensity difference threshold.
    
    Returns:
        np.array: The filtered image.
    """
    pdw = n//2
    padded = np.pad(image, pad_width=pdw, mode='reflect')
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract local neighborhood
            window = padded[i-pdw:i+pdw, j-pdw:j+pdw].flatten()
            center_value = image[i, j]
            
            # Keep only values within threshold
            valid_pixels = window[np.abs(window - center_value) <= A] if not np.isnan(center_value) else window
            
            # Compute median of valid pixels
            if valid_pixels.size > 0:
                output[i, j] = np.median(valid_pixels)
            else:
                output[i, j] = np.median(window)  # If no valid pixels, floating

    return output

def adaptive_median_filter__(image, n = 3, A = 0.5):
    pdw = n//2
    padded = np.pad(image, pad_width=pdw, mode='reflect')
    
    def filter_func(window: np.ndarray):
        center_value = window[len(window) // 2]
        valid_pixels = window[np.abs(window - center_value) <= A] if not np.isnan(center_value) else window
        return np.median(valid_pixels) if valid_pixels.size > 0 else np.median(window)
    
    footprint = np.ones((2 * pdw + 1, 2 * pdw + 1))
    output = generic_filter(padded, filter_func, footprint=footprint, mode='nearest')
    
    return output[pdw:-pdw, pdw:-pdw]  # Crop back to original size

def depth_to_point_cloud_torch(depth: np.ndarray, fovX: float, fovY: float, world_view_transform: torch.Tensor):
    """
    Converts a depth map to a 3D point cloud in world coordinates.

    Parameters:
        depth (np.ndarray): Depth map of shape (W, H, 1)
        fovX (float): Horizontal field of view in degrees
        fovY (float): Vertical field of view in degrees
        world_view_transform (torch.Tensor): A (4, 4) transformation matrix to convert from world to camera coordinates

    Returns:
        torch.Tensor: A tensor of shape (N, 3) where N is the number of points, and each row is a (x, y, z) coordinate
    """
    # Convert depth to a torch tensor
    depth = depth.squeeze()

    # calc time used
    start = time.time()
    depth = median_filter(depth, size=5)
    print('median_filter time:', time.time()-start)
    start = time.time()

    # remove discontinuous points
    cnt_arr = np.zeros(depth.shape)
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            if depth[i][j] != np.nan and cnt_arr[i][j] == 0:
                count_adj_points(i, j, depth, cnt_arr)
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            if depth[i][j] != np.nan and cnt_arr[i][j] < 300:
                depth[i][j] = np.nan
    print('remove discontinuous points time:', time.time()-start)
    start = time.time()
    depth = adaptive_median_filter(depth, n=5, A=0.5)
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            cnt_arr[i][j] = count_adj(i, j, depth)
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            depth[i][j] = depth[i][j] if cnt_arr[i][j] > 2 else np.nan
    # depth = adaptive_median_filter(depth, n=5, A=0.5)
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            cnt_arr[i][j] = count_adj(i, j, depth)
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            depth[i][j] = depth[i][j] if cnt_arr[i][j] > 2 else np.nan
    depth = adaptive_median_filter(depth, A=0.5)
    print('adaptive_median_filter time:', time.time()-start)
    start = time.time()
    
    t_shape = depth.shape

    # Apply median filter with a window size (e.g., 3x3)
    depth = torch.from_numpy(depth)

    # depth = torch.from_numpy(depth)
    print(depth.shape)

    # Get image dimensions
    W, H = depth.shape
    pixel_x, pixel_y = torch.meshgrid(torch.arange(W), torch.arange(H))

    # Convert pixel coordinates to normalized camera coordinates
    # We assume that the image center is at (W/2, H/2) and scale by field of view.
    # fovX and fovY are in degrees, need to convert them to radians and compute focal lengths
    fx = (W / 2) / math.tan(fovY / 2)
    fy = (H / 2) / math.tan(fovX / 2)

    # Normalize pixel coordinates (centered)
    x_normalized = (pixel_x - W / 2) / fx
    y_normalized = (pixel_y - H / 2) / fy
    print(x_normalized.shape, y_normalized.shape)

    valid_mask = ((~torch.isnan(depth)) & (depth >= 470)) # TODO: move isnan after noise
    depth = depth[valid_mask]
    x_normalized = x_normalized[valid_mask]
    y_normalized = y_normalized[valid_mask]

    # Stack the normalized coordinates and depth to form the 3D point cloud in camera space
    z = depth  # (N)
    x = x_normalized * z  # (N)
    y = y_normalized * z  # (N)

    z = z.flatten()
    x = x.flatten()
    y = y.flatten()
    # Stack the (x, y, z) points together
    points_camera_space = torch.stack((y, x, z), dim=-1)  # Shape: (N, 3)

    # Add a column of ones for homogeneous coordinates
    points_camera_space_homogeneous = torch.cat([points_camera_space, torch.ones(
        points_camera_space.shape[0], 1)], dim=-1).cuda()  # (H * W, 4)

    # Transform the points to world coordinates
    points_world_space = points_camera_space_homogeneous @ torch.linalg.inv(
        world_view_transform)  # (H * W, 4)

    print(world_view_transform, torch.linalg.inv(world_view_transform))

    print(points_world_space[torch.randint(
        0, points_world_space.shape[0], (20,)), 3])

    # Keep only the (x, y, z) coordinates
    # (H * W, 3)
    points_world_space = points_world_space[:, :3] / points_world_space[:, 3].squeeze()[:, None]

    # points_world_space[:, [0, 1]] = points_world_space[:, [1, 0]]
    points_world_space[:, 0] = points_world_space[:, 0] + 3
    points_world_space[:, 1] = points_world_space[:, 1] - 1

    return points_world_space.cpu(), valid_mask.cpu(), t_shape


def save_ply_mesh(filename, points, edges, IOR):
    """
    Save a mesh to a PLY file format.

    Args:
        filename (str): Path to save the PLY file.
        points (np.ndarray): Array of shape (N, 3) representing the 3D points (x, y, z).
        edges (np.ndarray): Array of shape (M, 3) representing the triangle mesh.
        IOR (float): Object's IOR.
    """
    num_vertices = points.shape[0]
    num_faces = edges.shape[0]
    
    with open(filename, 'w') as ply_file:
        # Write PLY header
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element IOR 1\n")
        ply_file.write("property float val\n")
        ply_file.write(f"element vertex {num_vertices}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write(f"property float ior\n")  # Custom property for IOR
        ply_file.write(f"element face {num_faces}\n")
        ply_file.write("property list uchar int vertex_indices\n")
        ply_file.write("end_header\n")
        
        # Write IOR data
        ply_file.write(f"{IOR}\n")
        
        # Write vertex data
        for point in points:
            ply_file.write(f"{point[0]} {point[1]} {point[2]}\n")
        
        # Write face data
        for face in edges:
            ply_file.write(f"3 {face[0]} {face[1]} {face[2]}\n")
            
def create_water_mesh(min_xyz, max_xyz, Z, dx = 32, dy = 32, dz = 0.2, height=10):
    """
    create a water mesh.

    Args:
        min_xyz (np.ndarray): Min val of x, y, z.
        max_xyz (np.ndarray): Max val of x, y, z.
        Z (float): water_avg_height, sea level.
        dx (int): x axis resolution.
        dy (int): y axis resolution.
        dz (int): water wave height.
        height (int): water height.
    Returns:
        points: shape like (N, 3), position
        edges: shape like (M, 3), a triangle mesh, each row is a triangle, each element is the index of points.
    """
    noise = generate_perlin_noise_2d((dx, dy), (dx//4, dy//4))
    # noise = zoom(noise, ((max_xyz[0]-min_xyz[0])/noise.shape[0], (max_xyz[1]-min_xyz[1])/noise.shape[1]))
    noise = noise*dz+Z
    noise = noise.flatten()
    xx, yy = np.meshgrid(np.linspace(min_xyz[0], max_xyz[0], dx), np.linspace(min_xyz[1], max_xyz[1], dy), indexing='ij')
    xx = xx.flatten()
    yy = yy.flatten()
    points = np.stack((xx, yy, noise), axis=-1)
    # print(points)
    edges = []
    for i in range(dx-1):
        for j in range(dy-1):
            edges.append([i*dy+j, (i+1)*dy+j, i*dy+j+1])
            edges.append([i*dy+j+1, (i+1)*dy+j, (i+1)*dy+j+1])

    points = points.tolist()
    
    xx, yy, zz = np.meshgrid(np.linspace(min_xyz[0], max_xyz[0], dx), np.array(min_xyz[1]), np.array(Z-dz))
    points_left = np.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=-1)

    xx, yy, zz = np.meshgrid(np.array(min_xyz[0]), np.linspace(min_xyz[1], max_xyz[1], dy), np.array(Z-dz))
    points_upper = np.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=-1)

    xx, yy, zz = np.meshgrid(np.array(max_xyz[0]), np.linspace(min_xyz[1], max_xyz[1], dy), np.array(Z-dz))
    points_bottom = np.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=-1)

    xx, yy, zz = np.meshgrid(np.linspace(min_xyz[0], max_xyz[0], dx), np.array(max_xyz[1]), np.array(Z-dz))
    points_right = np.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=-1)
    # print(points_left)
    # print(points_upper)
    # print(points_bottom)
    # print(points_right)
    lu_pid = now_ptid = len(points)
    points.append(points_upper[0])
    now_ptid += 1
    i = j = 0
    
    for j in range(1, dy):
        points.append(points_upper[j])
        edges.append([i*dy+j, now_ptid, now_ptid-1])
        edges.append([i*dy+j, now_ptid-1, i*dy+j-1])
        now_ptid += 1
    
    j = dy-1
    ru_pid = now_ptid-1
    for i in range(1, dx):
        points.append(points_right[i])
        edges.append([i*dy+j, now_ptid, now_ptid-1])
        edges.append([i*dy+j, now_ptid-1, (i-1)*dy+j])
        now_ptid += 1
    
    i = dx-1
    rd_pid = now_ptid-1
    for j in range(dy-2, -1, -1):
        points.append(points_bottom[j])
        edges.append([i*dy+j, now_ptid, now_ptid-1])
        edges.append([i*dy+j, now_ptid-1, i*dy+j+1])
        now_ptid += 1
    
    j = 0
    ld_pid = now_ptid-1
    for i in range(dx-2, 0, -1):
        points.append(points_left[i])
        edges.append([i*dy+j, now_ptid, now_ptid-1])
        edges.append([i*dy+j, now_ptid-1, (i+1)*dy+j])
        now_ptid += 1
    i = 0
    edges.append([i*dy+j, lu_pid, now_ptid-1])
    edges.append([i*dy+j, now_ptid-1, (i+1)*dy+j])
    
    lud_pid = len(points)
    points.append([min_xyz[0], min_xyz[1], Z-height])
    ldd_pid = len(points)
    points.append([max_xyz[0], min_xyz[1], Z-height])
    rdd_pid = len(points)
    points.append([max_xyz[0], max_xyz[1], Z-height])
    rud_pid = len(points)
    points.append([min_xyz[0], max_xyz[1], Z-height])
    
    edges.append([lu_pid, lud_pid, ldd_pid])
    edges.append([lu_pid, ldd_pid, ld_pid])
    edges.append([ld_pid, ldd_pid, rdd_pid])
    edges.append([ld_pid, rdd_pid, rd_pid])
    edges.append([rd_pid, rdd_pid, rud_pid])
    edges.append([rd_pid, rud_pid, ru_pid])
    edges.append([ru_pid, rud_pid, lud_pid])
    edges.append([ru_pid, lud_pid, lu_pid])
    
    edges.append([lud_pid, rdd_pid, ldd_pid])
    edges.append([lud_pid, rud_pid, rdd_pid])
    
    return points, edges

import sys

with torch.no_grad():
    cfg.data.use_test_cam = False
    dataset = Dataset()
    sys.stdout.flush()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    print('done load model')
    sys.stdout.flush()
    scene = Scene(gaussians=gaussians, dataset=dataset)
    # gaussians.load_ply('./output/waymo_full_exp/waymo_train_002/point_cloud/iteration_50000/point_cloud.ply')

    print(gaussians.model_name_id)
    print(gaussians.metadata['scene_center'])
    
    sys.stdout.flush()

    # cam = Camera(1, R, T, 2000, 2000, None, None, "BEV_Depth")
    cam_old: Camera = dataset.train_cameras[cfg.resolution_scales[0]][0]
    print(cam_old.FoVx, math.tan((cam_old.FoVx / 2)))
    print(cam_old.FoVy, math.tan((cam_old.FoVy / 2)))
    print(cam_old.projection_matrix)
    sys.stdout.flush()
    camd = loadCamData(
        dataset.scene_info.train_cameras[0], cfg.resolution_scales[0])
    # cam = Camera(cam_old.id, R, T, cam_old.FoVx, cam_old.FoVy, cam_old.K.cpu().numpy(), cam_old.original_image, 'BEV', cam_old.trans, cam_old.scale, cam_old.meta, )
    # cam.ego_pose = cam_old.ego_pose
    # cam.extrinsic = cam_old.extrinsic
    sc_m, _  = torch.max(gaussians.background._scaling, dim=1)
    gaussians.background.background_mask = ((sc_m < 0) & (gaussians.background._opacity.flatten() > -5.7))
    # print('mask: ', torch.mean(gaussians.background.background_mask.to(torch.float16)))
    print('sc_shape', gaussians.background.get_scaling.shape, gaussians.background._scaling.shape)
    sys.stdout.flush()

    R = np.array(cam_old.R)
    theta_max = -1.57
    # T = gaussians.metadata['scene_center']
    T = cam_old.T
    print(T)

    renderer = StreetGaussianRenderer()
    vis = StreetGaussianVisualizer("./output/try/")

    theta = theta_max
    RRR = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta), np.cos(theta)]])
    RR = R @ RRR
    TT = np.array(T)
    TT[1] += 500
    TT[2] -= 150
    cam = loadCam(camd, RR, np.linalg.inv(RRR) @ TT)
    print('K:', camd['K'])

    ret = renderer.render_background(cam, gaussians)

    name = cam.image_name
    rgb = ret['rgb']
    torchvision.utils.save_image(rgb, os.path.join(
        "./output/try/", f'{name}_rgb.png'))
    depth = ret['depth']
    depth = depth.detach().permute(1, 2, 0).detach().cpu().numpy()  # [H, W, 1]
    imageio.imwrite(os.path.join("./output/try/", f'{name}_depth.png'), visualize_depth_numpy(
        depth, cmap=cv2.COLORMAP_TURBO)[0][..., [2, 1, 0]])

    try:
        print(np.min(depth), np.max(depth))
    except:
        pass

    pc, mask, t_shape = depth_to_point_cloud_torch(
        depth, cam.FoVx, cam.FoVy, cam.world_view_transform)
    
    min_xyz, _ = torch.min(pc, dim=0)
    print(min_xyz)
    max_xyz, _ = torch.max(pc, dim=0)
    print(max_xyz)
    
    Z = 0
    points, edges = create_water_mesh(min_xyz.cpu().numpy(), max_xyz.cpu().numpy(), Z)
    save_ply_mesh(os.path.join(f"./output/waymo_full_exp/waymo_train_002/point_cloud/iteration_{str(scene.loaded_iter)}", 'water_mesh.ply'), np.array(points), np.array(edges), 1.33)
    

    # save_ply(f"./output/waymo_full_exp/waymo_train_002/point_cloud/iteration_{str(scene.loaded_iter)}/snow_ply.ply", pcl.numpy(), scales=sc, rots=rt,
    #          opacity=oc,
    #          features_dc=features[:, :, 0:1].transpose(
    #              1, 2).contiguous().cpu().numpy(),
    #          features_extra=features[:, :, 1:].transpose(1, 2).contiguous().cpu().numpy())

    # # TODO: add snow here(or whatever changes)
    # gaussians.background.extend_ply(
    #     f"./output/waymo_full_exp/waymo_train_002/point_cloud/iteration_{str(scene.loaded_iter)}/snow_ply.ply")

    # gaussians.save_ply('./output/waymo_full_exp/waymo_train_002/point_cloud/iteration_50000/point_cloud_changed.ply')
    # checkpoint_path = os.path.join(
    #     cfg.trained_model_dir, f"iteration_{str(scene.loaded_iter)}.pth")
    # assert os.path.exists(checkpoint_path)
    # torch.save(gaussians.save_state_dict(True), checkpoint_path)

    print(f'ADD_OBJ: MAX MEMORY: {(torch.cuda.max_memory_allocated() / (1024 * 1024)): .2f} MB')