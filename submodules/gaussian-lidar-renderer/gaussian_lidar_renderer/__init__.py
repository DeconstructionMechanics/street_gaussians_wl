import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from .utils import build_rotation, covariance_activation, backward_covariance

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
        n_gaussian_backward = 32

        cotrib, opa, tvalue, intensity, raydrop,\
        contribute_gid, contribute_depth, contribute_T, contribute_clamp,\
        contribute_tprime, contribute_intensityprime, contribute_raydropprime,\
        = _C.trace_bvh_opacity(tree, aabb, rays_o, rays_d, means3D, symm_inv, opacity, shs, sh_degree, needs_grad, n_gaussian_backward)
        
        # n_contribute, weights, tvalues, intensities, raydrops = cotrib.unsqueeze(-1), opa.unsqueeze(-1), tvalue.unsqueeze(-1), intensity.unsqueeze(-1), raydrop.unsqueeze(-1)
        n_contribute, weights, tvalues, intensities, raydrops = cotrib, opa, tvalue, intensity, raydrop
        
        if needs_grad:
            ctx.save_for_backward(contribute_gid, contribute_depth, contribute_T, contribute_clamp,\
                                  contribute_tprime, contribute_intensityprime, contribute_raydropprime,\
                                  weights, tvalues, intensities, raydrops,\
                                  means3D, scales, rotations, opacity, shs,\
                                  rays_o, rays_d)
            ctx.sh_degree = sh_degree
        else:
            ctx.save_for_backward(None)
        return n_contribute, weights, tvalues, intensities, raydrops
    
    @staticmethod
    def backward(ctx, grad_n_contribute, grad_weights, grad_tvalues, grad_intensity, grad_raydrop):

        contribute_gid, contribute_depth, contribute_T, contribute_clamp,\
        contribute_tprime, contribute_intensityprime, contribute_raydropprime,\
        weights, tvalues, intensity, raydrop,\
        means3D, scales, rotations, opacity, shs,\
        rays_o, rays_d = ctx.saved_tensors
        sh_degree = ctx.sh_degree
        '''
        contribute_gid: (n_ray, n_gaussian), contributed gaussian index, int, -1 means gaussian not exist
        contribute_depth: (n_ray, n_gaussian), contributed depth of gaussian, int, start with 0, -1 means gaussian not exist
        contribute_T: (n_ray, n_gaussian), remaining opacity, float, 0 means gaussian not exist
        contribute_clamp: (n_ray, n_gaussian), clamp value, bool
        ''' 
        grad_means3D = torch.zeros_like(means3D)
        grad_scales = torch.zeros_like(scales)
        grad_rotations = torch.zeros_like(rotations)
        grad_opacity = torch.zeros_like(opacity)
        grad_shs = torch.zeros_like(shs)
        grad_sh_degree, grad_rays_o, grad_rays_d, grad_aabb_scale = None, None, None, None

        # sort contribute_gid by depth
        contribute_depth[contribute_depth < 0] = torch.max(contribute_depth) + 1
        contribute_gid_sorted = torch.gather(contribute_gid, 1, torch.argsort(contribute_depth, dim=1))

        # backward _C.trace_bvh_opacity
        covs3D = covariance_activation(scales, rotations)
        grad_means3D_from_trace, grad_covs3D_from_trace, grad_opacity_from_trace, grad_shs_from_trace = _C.backward_trace(\
            contribute_gid_sorted, contribute_T, contribute_clamp,\
            contribute_tprime, contribute_intensityprime, contribute_raydropprime,\
            weights, tvalues, intensity, raydrop,\
            means3D, covs3D, opacity, shs, sh_degree,\
            rays_o, rays_d,\
            grad_weights, grad_tvalues, grad_intensity, grad_raydrop)
        
        grad_means3D += grad_means3D_from_trace
        grad_opacity += grad_opacity_from_trace
        grad_shs += grad_shs_from_trace
        
        # backward covariance_activation
        grad_scales_from_covs3D, grad_rotations_from_covs3D = backward_covariance(grad_covs3D_from_trace, scales, rotations)
        grad_scales += grad_scales_from_covs3D
        grad_rotations += grad_rotations_from_covs3D

        return grad_means3D, grad_scales, grad_rotations, grad_opacity, grad_shs, grad_sh_degree, grad_rays_o, grad_rays_d, grad_aabb_scale

