from lib.models.street_gaussian_model import StreetGaussianModel
from lib.config import cfg
from lib.datasets.dataset import Dataset
from lib.utils.camera_utils import Camera, CameraInfo, loadmask, loadmetadata, WARNED
from lib.visualizers.street_gaussian_visualizer import StreetGaussianVisualizer
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.utils.general_utils import PILtoTorch, NumpytoTorch, matrix_to_quaternion
from lib.models.scene import Scene
import copy
from PIL import Image
import numpy as np
import tqdm
import math
from copy import deepcopy
import torch
import torch.nn as nn

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

    image = PILtoTorch(cam_info.image, resolution, resize_mode=Image.BILINEAR)[:3, ...]
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

def loadCam(camdata, R, T, foVx = None, foVy = None):
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

with torch.no_grad():
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    scene = Scene(gaussians=gaussians, dataset=dataset)
    # gaussians.load_ply('./output/waymo_full_exp/waymo_train_002/point_cloud/iteration_100000/point_cloud.ply')

    print(gaussians.model_name_id)
    print(gaussians.metadata['scene_center'])
    sc = gaussians.background._scaling
    print(sc.min(), sc.max(), sc.mean())
    oc = gaussians.background._opacity
    print(oc.min(), oc.max(), oc.mean())
    print(torch.count_nonzero(sc>=1))

    # cam = Camera(1, R, T, 2000, 2000, None, None, "BEV_Depth")
    cam_old : Camera = dataset.train_cameras[cfg.resolution_scales[0]][0]
    print(cam_old.FoVx, math.tan((cam_old.FoVx / 2)))
    print(cam_old.FoVy, math.tan((cam_old.FoVy / 2)))
    print(cam_old.projection_matrix)
    camd = loadCamData(dataset.scene_info.train_cameras[0], cfg.resolution_scales[0])
    # cam = Camera(cam_old.id, R, T, cam_old.FoVx, cam_old.FoVy, cam_old.K.cpu().numpy(), cam_old.original_image, 'BEV', cam_old.trans, cam_old.scale, cam_old.meta, )
    # cam.ego_pose = cam_old.ego_pose
    # cam.extrinsic = cam_old.extrinsic

    R = np.array(cam_old.R)
    theta_max = -1.57
    # T = gaussians.metadata['scene_center']
    T = cam_old.T
    print(T)

    vis = StreetGaussianVisualizer("./output/try/")
    vis.save_image = False
    vis.save_video = True
    renderer = StreetGaussianRenderer()

    cam = loadCam(camd, R, T)
    ret = renderer.render_all(cam, gaussians)
    vis.visualize(ret, cam)
    I_MAX = 100
    tt = tqdm.tqdm(range(1, I_MAX+1), desc="rendering")
    for i in tt:
        theta = theta_max * I_MAX / I_MAX
        RRR = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
        RR = R @ RRR
        TT = np.array(T)

        TT[1] += i/I_MAX * 500
        TT[2] -= I_MAX/I_MAX * 150

        cam = loadCam(camd, RR, np.linalg.inv(RRR) @ TT)
        cam.image_name += f"_upper{i}"
        # try:
        ret = renderer.render_all(cam, gaussians)
        vis.visualize(ret, cam)
        # except:
        #    pass

    vis.summarize()
    exit(0)