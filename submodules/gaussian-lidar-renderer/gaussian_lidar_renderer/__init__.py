import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from .utils import build_rotation, covariance_activation, covariance_activation_unstrip, backward_covariance

try:
    from . import _C
except Exception as e:
    _src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _C = load(
        name='gaussian_lidar_renderer',
        extra_cuda_cflags=["-O3", "--expt-extended-lambda"],
        extra_cflags=["-O3"],
        sources=[os.path.join(_src_path, 'src', f) for f in [
            'bvh.cu',
            'trace.cu',
            'construct.cu',
            'bindings.cpp',
        ]],
        extra_include_paths=[
            os.path.join(_src_path, 'include'),
        ],
        verbose=True)


class GaussianLidarRenderer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3D, scales, rotations, opacity, shs, sh_degree, rays_o, rays_d, aabb_scale: float=10):
        P = means3D.shape[0]
        rot = build_rotation(rotations)
        nodes = torch.full((2 * P - 1, 5), -1, device="cuda").int()
        nodes[:P - 1, 4] = 0
        nodes[P - 1:, 4] = 1
        aabbs = torch.zeros(2 * P - 1, 6, device="cuda").float()
        aabbs[:, :3] = 100000
        aabbs[:, 3:] = -100000

        a, b, c = rot[:, :, 0], rot[:, :, 1], rot[:, :, 2]
        m = 3 / aabb_scale
        sa, sb, sc = m * scales[:, 0], m * scales[:, 1], m * scales[:, 2]

        x111 = means3D + a * sa[:, None] + b * sb[:, None] + c * sc[:, None]
        x110 = means3D + a * sa[:, None] + b * sb[:, None] - c * sc[:, None]
        x101 = means3D + a * sa[:, None] - b * sb[:, None] + c * sc[:, None]
        x100 = means3D + a * sa[:, None] - b * sb[:, None] - c * sc[:, None]
        x011 = means3D - a * sa[:, None] + b * sb[:, None] + c * sc[:, None]
        x010 = means3D - a * sa[:, None] + b * sb[:, None] - c * sc[:, None]
        x001 = means3D - a * sa[:, None] - b * sb[:, None] + c * sc[:, None]
        x000 = means3D - a * sa[:, None] - b * sb[:, None] - c * sc[:, None]
        aabb_min = torch.minimum(torch.minimum(
            torch.minimum(torch.minimum(torch.minimum(torch.minimum(torch.minimum(x111, x110), x101), x100), x011),
                          x010), x001), x000)
        aabb_max = torch.maximum(torch.maximum(torch.maximum(torch.maximum(
            torch.maximum(torch.maximum(torch.maximum(x111, x110), x101), x100), x011), x010), x001), x000)

        aabbs[P - 1:] = torch.cat([aabb_min, aabb_max], dim=-1)

        tree, aabb, morton = _C.create_bvh(means3D, scales, rotations, nodes, aabbs)

        rays_o = rays_o + rays_d * 0.05
        symm_inv = covariance_activation(scales, rotations)
        needs_grad = ctx.needs_input_grad[0]
        cotrib, opa, tvalue, intensity, raydrop = _C.trace_bvh_opacity(tree, aabb, rays_o, rays_d, means3D, symm_inv, opacity, shs, sh_degree)
        if needs_grad:
            ctx.save_for_backward() # TODO
        else:
            ctx.save_for_backward(None)

        n_contribute, weights, tvalues, intensities, raydrops = cotrib.unsqueeze(-1), opa.unsqueeze(-1), tvalue.unsqueeze(-1), intensity.unsqueeze(-1), raydrop.unsqueeze(-1)
        return n_contribute, weights, tvalues, intensities, raydrops
    
    @staticmethod
    def backward(ctx, grad_n_contribute, grad_weights, grad_tvalues, grad_intensity, grad_raydrop):
        contribute_gid, contribute_T, contribute_clamp, weights, means3D, scales, rotations, opacity, shs, sh_degree, rays_o, rays_d = ctx.saved_tensors
        '''
        contribute_gid: (n_ray, n_gaussian), contributed gaussian index, int, -1 means gaussian not exist
        contribute_T: (n_ray, n_gaussian), remaining opacity, float, 0 means gaussian not exist
        contribute_clamp: (n_ray, n_gaussian), clamp value, bool
        ''' 
        grad_means3D = torch.zeros_like(means3D)
        grad_scales = torch.zeros_like(scales)
        grad_rotations = torch.zeros_like(rotations)
        grad_opacity = torch.zeros_like(opacity)
        grad_shs = torch.zeros_like(shs)
        grad_sh_degree, grad_rays_o, grad_rays_d, grad_aabb_scale = None, None, None, None
        
        # grad outputs from alpha-blending
        diff_params = contribute_T * opacity[contribute_gid] # (T_i * alpha_i) opacity?
        diff_params[contribute_gid < 0] = 0
        diff_tprime = diff_params / torch.max(weights.reshape(-1, 1), torch.tensor(0.00001)) # (T_i * alpha_i) / (weight)
        grad_tprime = diff_tprime * grad_tvalues.reshape(-1, 1)
        grad_intensityprime = diff_params * grad_intensity.reshape(-1, 1)
        grad_raydropprime = diff_params * grad_raydrop.reshape(-1, 1)
        
        # grad sh features
        grad_shs_from_shs, grad_means_from_shs = _C.backward_shs(contribute_gid, contribute_clamp, grad_intensityprime, grad_raydropprime, rays_o, rays_d, means3D, shs, sh_degree)
        grad_shs += grad_shs_from_shs
        grad_means3D += grad_means_from_shs

        # grad means3D
        symm_inv_mat = covariance_activation_unstrip(scales, rotations)
        sigma_raysd = torch.einsum('rgab,rb->rga', symm_inv_mat[contribute_gid], rays_d)
        raysd_sigma_raysd = torch.einsum('ra,rga->rg', rays_d, sigma_raysd)
        grad_means_from_tprime = sigma_raysd / (raysd_sigma_raysd * grad_tprime).unsqueeze(-1) # ？
        grad_means3D.index_add_(0, torch.max(torch.tensor(0), contribute_gid.flatten()), grad_means_from_tprime.reshape(-1, 3))
        
        # grad cov
        ray_mu = means3D[contribute_gid] - rays_o.unsqueeze(1)
        grad_cov_from_tprime = torch.einsum('rga,rb,rg->rgab', (ray_mu / raysd_sigma_raysd.unsqueeze(-1) - (torch.einsum('rga,rga->rg', ray_mu, sigma_raysd) / (raysd_sigma_raysd ** 2)).unsqueeze(-1) * rays_d.unsqueeze(1)), rays_d, grad_tprime)
        grad_cov = torch.zeros(grad_means3D.shape[0], 3, 3, device='cuda').float()
        grad_cov.index_add_(0, torch.max(torch.tensor(0), contribute_gid.flatten()), grad_cov_from_tprime.reshape(-1, 3, 3))
        grad_scales_from_tprime, grad_rotations_from_tprime = backward_covariance(grad_cov, scales, rotations)
        grad_scales += grad_scales_from_tprime
        grad_rotations += grad_rotations_from_tprime

        # grad opacity

        return grad_means3D, grad_scales, grad_rotations, grad_opacity, grad_shs, grad_sh_degree, grad_rays_o, grad_rays_d, grad_aabb_scale

