import torch 
import os
import json
from tqdm import tqdm
from lib.models.street_gaussian_model import StreetGaussianModel 
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.datasets.dataset import Dataset
from lib.models.scene import Scene
from lib.utils.general_utils import safe_state
from lib.config import cfg
from lib.visualizers.base_visualizer import BaseVisualizer as Visualizer
from lib.visualizers.street_gaussian_visualizer import StreetGaussianVisualizer
import time

from lib.lidar.gaussian_lidar_renderer import LiDARCamera, render_lidar
from lib.lidar.gaussian_lidar_util import save_ply
import random
import numpy as np

def render_sets():
    cfg.render.save_image = True
    cfg.render.save_video = False

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        times = []
        if not cfg.eval.skip_train:
            save_dir = os.path.join(cfg.model_path, 'train', "ours_{}".format(scene.loaded_iter))
            visualizer = Visualizer(save_dir)
            cameras = scene.getTrainCameras()
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Training View")):
                
                torch.cuda.synchronize()
                start_time = time.time()
                result = renderer.render(camera, gaussians)
                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                
                visualizer.visualize(result, camera)

        if not cfg.eval.skip_test:
            save_dir = os.path.join(cfg.model_path, 'test', "ours_{}".format(scene.loaded_iter))
            visualizer = Visualizer(save_dir)
            cameras =  scene.getTestCameras()
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Testing View")):
                
                torch.cuda.synchronize()
                start_time = time.time()
                
                result = renderer.render(camera, gaussians)
                                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                
                visualizer.visualize(result, camera)
        
        print(times)        
        print('average rendering time: ', sum(times[1:]) / len(times[1:]))

        print('\nLiDAR rendering start: ', len(range(*cfg.train_lidar.testset_frames)), '\n')
        lidar_save_dir = os.path.join(save_dir, 'lidar')
        os.makedirs(lidar_save_dir, exist_ok=True)
        times = []

        testset_frames = []
        for range_param in cfg.train_lidar.testset_frames:
            testset_frames += list(range(*range_param))
        testset_frames = sorted(testset_frames)
        for frame_id in testset_frames:
            for camera_id in cfg.train_lidar.testset_cameras:
                lidar_camera = LiDARCamera(cfg.source_path, frame_id, camera_id)
                start_time = time.time()
                result = render_lidar(lidar_camera, gaussians, cfg.train_lidar.aabb_scale)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)

                range_image = torch.cat((result['depth'].reshape(-1, 1), result['intensity'].reshape(-1, 1), result['raydrop'].reshape(-1, 1)), dim=1).reshape(lidar_camera.h, lidar_camera.w, 3)
                pointcloud = lidar_camera.get_ray_o() + lidar_camera.get_ray_d() * result['depth'].reshape(-1, 1)

                np.savez_compressed(os.path.join(lidar_save_dir, f'{frame_id:06d}_{camera_id}.npz'),
                                    range_image=range_image.cpu().numpy(),
                                    pointcloud=pointcloud.cpu().numpy()
                                    )
                save_ply(points_np=pointcloud.cpu().numpy(), weights_np=result['intensity'].cpu().numpy(), ply_path=os.path.join(lidar_save_dir, f'{frame_id:06d}_{camera_id}.ply'))

                # depth loss
                depth_error = torch.abs((lidar_camera.get_depth()[lidar_camera.get_mask()] - result['depth'][lidar_camera.get_mask()]))
                depth_error, _ = torch.topk(depth_error, int(0.95 * depth_error.size(0)), largest=False)
                depth_loss = cfg.train_lidar.depth_lr * depth_error.mean()

                # intensity loss
                intensity_error = torch.abs((lidar_camera.get_intensity()[lidar_camera.get_mask()] - result['intensity'][lidar_camera.get_mask()]))
                intensity_error, _ = torch.topk(intensity_error, int(0.95 * intensity_error.size(0)), largest=False)
                intensity_loss = cfg.train_lidar.intensity_lr * intensity_error.mean()

                # raydrop loss
                raydrop_error = torch.abs((lidar_camera.get_raydrop()[lidar_camera.get_mask()] - result['raydrop'][lidar_camera.get_mask()]))
                raydrop_error, _ = torch.topk(raydrop_error, int(0.95 * raydrop_error.size(0)), largest=False)
                raydrop_loss = cfg.train_lidar.raydrop_lr * raydrop_error.mean()

                print(f'[{frame_id:06d}_{camera_id}], time: {(end_time - start_time) * 1000}, depth_loss: {depth_loss}, intensity_loss: {intensity_loss}, raydrop_loss: {raydrop_loss}')

            
        print("\nLiDAR rendering complete")
        print("average rendering time: ", sum(times) / len(times))
        sorted_times = sorted(times, reverse=True)
        num_1low = round(len(sorted_times) * 0.01)
        if num_1low > 0:
            print("1% Low: ", sum(sorted_times[:num_1low]) / num_1low)


                
def render_trajectory():
    cfg.render.save_image = False
    cfg.render.save_video = True
    # Assuming A (fog color) and beta are given
    cfg.render.fog = False
    fog_color = torch.tensor([0.9, 0.9, 0.9]).to(cfg.data_device)
    fog_beta = 0.03
    cfg.render.fog_color = fog_color.tolist()
    
    print('begin render traj')

    with torch.no_grad():
        dataset = Dataset()        
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        gaussians.sky_cubemap.sky_color[:] = fog_color

        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        
        save_dir = os.path.join(cfg.model_path, 'trajectory', "ours_{}".format(scene.loaded_iter))
        visualizer = StreetGaussianVisualizer(save_dir)
        
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))

        print('prepare done')
        for idx, camera in enumerate(tqdm(cameras, desc="Rendering Trajectory")):
            result = renderer.render_all(camera, gaussians)
            if cfg.render.get('fog', False):
                rgb:torch.Tensor = result['rgb']
                depth:torch.Tensor = result['depth']

                normalized_depth = torch.nan_to_num(depth, nan=torch.inf).to(cfg.data_device)

                dx = torch.exp(-fog_beta * normalized_depth).to(cfg.data_device)
                dx = dx.squeeze()

                # fog_img = rgb * dx + A * (1 - dx)
                result['rgb'] = (rgb.to(cfg.data_device) * dx.unsqueeze(0) + fog_color.unsqueeze(1).unsqueeze(2) * (1 - dx.unsqueeze(0))).cpu()
                # print(result['rgb'].shape)
            
            
            visualizer.visualize(result, camera)

        visualizer.summarize()

        lidar_save_dir = os.path.join(save_dir, 'lidar')
        os.makedirs(lidar_save_dir, exist_ok=True)
        times = []

        traject_frames = []
        for range_param in cfg.train_lidar.traject_frames:
            traject_frames += list(range(*range_param))
        traject_frames = sorted(traject_frames)
        print('\nLiDAR rendering start: ', len(traject_frames), '\n')
        for frame_id in traject_frames:
            for camera_id in cfg.train_lidar.traject_cameras:
                lidar_camera = LiDARCamera(cfg.source_path, frame_id, camera_id)
                start_time = time.time()
                result = render_lidar(lidar_camera, gaussians, cfg.train_lidar.aabb_scale)
                end_time = time.time()
                times.append(end_time - start_time)

                range_image = torch.cat((result['depth'].reshape(-1, 1), result['intensity'].reshape(-1, 1), result['raydrop'].reshape(-1, 1)), dim=1).reshape(lidar_camera.h, lidar_camera.w, 3)
                pointcloud_raydrop = (lidar_camera.get_ray_o() + lidar_camera.get_ray_d() * result['depth'].reshape(-1, 1))[result['raydrop'] > 0.5]
                pointcloud_weights = (lidar_camera.get_ray_o() + lidar_camera.get_ray_d() * result['depth'].reshape(-1, 1))[result['weights'] > -0.5]
                pointcloud_gt = (lidar_camera.get_ray_o() + lidar_camera.get_ray_d() * lidar_camera.get_depth().reshape(-1, 1))[lidar_camera.get_mask()]

                np.savez_compressed(os.path.join(lidar_save_dir, f'{frame_id:06d}_{camera_id}.npz'),
                                    range_image=range_image.cpu().numpy(),
                                    pointcloud=pointcloud_raydrop.cpu().numpy()
                                    )
                save_ply(points_np=pointcloud_raydrop.cpu().numpy(), weights_np=result['intensity'][result['raydrop'] > 0.5].cpu().numpy(), ply_path=os.path.join(lidar_save_dir, f'{frame_id:06d}_{camera_id}_raydrop.ply'))
                save_ply(points_np=pointcloud_weights.cpu().numpy(), weights_np=result['weights'][result['weights'] > -0.5].cpu().numpy(), ply_path=os.path.join(lidar_save_dir, f'{frame_id:06d}_{camera_id}_weights.ply'))
                save_ply(points_np=pointcloud_gt.cpu().numpy(), weights_np=lidar_camera.get_intensity()[lidar_camera.get_mask()].cpu().numpy(), ply_path=os.path.join(lidar_save_dir, f'{frame_id:06d}_{camera_id}_gt.ply'))
    
                print(f'[{frame_id:06d}_{camera_id}], time: {(end_time - start_time) * 1000} ms')

        print("\nLiDAR rendering complete")
        print("average FPS: ", len(times) / sum(times))
        sorted_times = sorted(times, reverse=True)
        num_1low = round(len(sorted_times) * 0.1)
        if num_1low > 0:
            print("10% Low: ", num_1low / sum(sorted_times[:num_1low]))
            
if __name__ == "__main__":
    print("Rendering " + cfg.model_path)
    safe_state(cfg.eval.quiet)
    
    if cfg.mode == 'evaluate':
        render_sets()
    elif cfg.mode == 'trajectory':
        render_trajectory()
    else:
        raise NotImplementedError()
