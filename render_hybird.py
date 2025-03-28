import torch 
import os
import json
from tqdm import tqdm
from lib.models.street_gaussian_model import StreetGaussianModel 
# from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.models.street_gaussian_RTrenderer import StreetGaussianRTRenderer
from lib.datasets.dataset import Dataset
from lib.models.scene import Scene
from lib.utils.general_utils import safe_state
from lib.config import cfg
from lib.visualizers.base_visualizer import BaseVisualizer as Visualizer
from lib.visualizers.street_gaussian_visualizer import StreetGaussianVisualizer
import time
import numpy as np

def render_sets():
    cfg.render.save_image = True
    cfg.render.save_video = False

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRTRenderer()

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

def load_ply_mesh(filename):
    """
    Load a mesh from a PLY file.

    Args:
        filename (str): Path to the PLY file.

    Returns:
        tuple: (points, edges, IOR), where
            - points (np.ndarray): Array of shape (N, 3) representing the 3D points (x, y, z).
            - edges (np.ndarray): Array of shape (M, 3) representing the triangle mesh.
            - IOR (float): The object's IOR.
    """
    with open(filename, 'r') as ply_file:
        lines = ply_file.readlines()
    
    # Read header
    vertex_count = 0
    face_count = 0
    ior = 1.0
    header_ended = False
    vertex_start = 0
    face_start = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        elif line.startswith("element face"):
            face_count = int(line.split()[-1])
        # elif line.startswith("element IOR"):
        #     ior = float(lines[i + 2].strip())  # IOR value is stored after 'property float val'
        elif line == "end_header":
            header_ended = True
            vertex_start = i + 2  # Skip the IOR value line
            face_start = vertex_start + vertex_count
            break
    ior = float(lines[i + 1].strip())
    # Read vertex data
    points = np.array([list(map(float, lines[j].split()))[:3] for j in range(vertex_start, vertex_start + vertex_count)])
    
    # Read face data
    edges = np.array([list(map(int, lines[j].split()[1:])) for j in range(face_start, face_start + face_count)])
    
    return points, edges, ior

def render_trajectory():
    cfg.render.save_image = False
    cfg.render.save_video = True
    # Assuming A (fog color) and beta are given
    cfg.render.fog = False
    fog_color = torch.tensor([0.9, 0.9, 0.9]).to(cfg.data_device)
    fog_beta = 0.1
    cfg.render.fog_color = fog_color.tolist()
    
    points, edges, IOR = load_ply_mesh(os.path.join(f"./output/waymo_full_exp/waymo_train_002/point_cloud/iteration_100000", 'water_mesh.ply'))

    points = torch.tensor(points).to(cfg.data_device)
    edges = torch.tensor(edges).to(cfg.data_device)

    print('begin render traj')

    with torch.no_grad():
        dataset = Dataset()        
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        gaussians.sky_cubemap.sky_color[:] = fog_color

        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRTRenderer()
        # only enable first half points
        gaussians.background.background_mask = torch.zeros(gaussians.background._xyz.shape[0], dtype=torch.bool).to(cfg.data_device)
        items = gaussians.background.background_mask.shape[0]
        gaussians.background.background_mask[:items//4] = True
        
        print('background.shape=', gaussians.background.get_xyz.shape)

        save_dir = os.path.join(cfg.model_path, 'trajectory', "ours_{}".format(scene.loaded_iter))
        visualizer = StreetGaussianVisualizer(save_dir)

        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))
        
        gaussians.set_mesh_vertices(points, edges.flatten())

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

if __name__ == "__main__":
    print("Rendering " + cfg.model_path)
    safe_state(cfg.eval.quiet)
    
    if cfg.mode == 'evaluate':
        render_sets()
    elif cfg.mode == 'trajectory':
        render_trajectory()
    else:
        raise NotImplementedError()
    print(f'RENDER: MAX MEMORY: {(torch.cuda.max_memory_allocated() / (1024 * 1024)): .2f} MB')
