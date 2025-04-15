from lib.lidar.gaussian_lidar_util import LiDARCamera
from lib.models.street_gaussian_model import StreetGaussianModel
from gaussian_lidar_renderer import GaussianLidarRenderer
import torch


def render_lidar(
        viewpoint_lidarcamera: LiDARCamera,
        pc: StreetGaussianModel,
        aabb_scale
    ):
        # set_visibility, parse_camera
        include_list = list(set(pc.model_name_id.keys()))
        pc.set_visibility(include_list)
        pc.parse_lidar_camera(viewpoint_lidarcamera)
        
        lidar_feature_map = [False, False, False, True, True]
        # if torch.isnan(pc.get_xyz)
        lidar_sh = pc.get_features[:, :, lidar_feature_map]

        # print('isnan(pc.get_xyz)', torch.isnan(pc.get_xyz).any())
        # print('isnan(pc.get_scaling)', torch.isnan(pc.get_scaling).any())
        # print('isnan(pc.get_rotation)', torch.isnan(pc.get_rotation).any())
        # print('isnan(pc.get_opacity)', torch.isnan(pc.get_opacity).any())
        # print('isnan(lidar_sh)', torch.isnan(lidar_sh).any())
        # print('isnan(viewpoint_lidarcamera.get_ray_o())', torch.isnan(viewpoint_lidarcamera.get_ray_o()).any())
        # print('isnan(viewpoint_lidarcamera.get_ray_d())', torch.isnan(viewpoint_lidarcamera.get_ray_d()).any())
        

        
        lidar_n_contribute, lidar_weights, lidar_t_values, lidar_intensity, lidar_raydrop = GaussianLidarRenderer.apply(
            pc.get_xyz,
            pc.get_scaling,
            pc.get_rotation,
            pc.get_opacity,
            lidar_sh,
            pc.max_sh_degree,
            viewpoint_lidarcamera.get_ray_o(),
            viewpoint_lidarcamera.get_ray_d(),
            aabb_scale)
        # lidar_mask_rule = (lidar_t_values.reshape(-1) > 0.1) & (lidar_t_values.reshape(-1) < 74) & (lidar_weights.reshape(-1) > 0.5)
        # lidar_mask_raydrop = lidar_raydrop > 0.5

        result = {
            "depth": lidar_t_values.reshape(-1),
            "intensity": lidar_intensity.reshape(-1),
            "raydrop": lidar_raydrop.reshape(-1),
            "n_contribute": lidar_n_contribute.reshape(-1),
            "weights": lidar_weights.reshape(-1),
        }

        if torch.isnan(result['depth']).any():
            print('[render_lidar] isnan(result[depth])', torch.isnan(result['depth']).any())
        if torch.isnan(result['intensity']).any():
            print('[render_lidar] isnan(result[intensity])', torch.isnan(result['intensity']).any())
        if torch.isnan(result['raydrop']).any():
            print('[render_lidar] isnan(result[raydrop])', torch.isnan(result['raydrop']).any())
        if torch.isnan(result['n_contribute']).any():
            print('[render_lidar] isnan(result[n_contribute])', torch.isnan(result['n_contribute']).any())
        if torch.isnan(result['weights']).any():
            print('[render_lidar] isnan(result[weights])', torch.isnan(result['weights']).any())


        return result
