import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from .utils import build_rotation, covariance_activation

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
    def forward(ctx, means3D, scales, rotations, opacity, rays_o, rays_d, aabb_scale: float=10):
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
        cotrib, opa, tvalue = _C.trace_bvh_opacity(tree, aabb, rays_o, rays_d, means3D, symm_inv, opacity)

        n_contribute, weights, tvalues = cotrib.unsqueeze(-1), opa.unsqueeze(-1), tvalue.unsqueeze(-1)
        return n_contribute, weights, tvalues
    
    @staticmethod
    def backward(ctx, grad_n_contribute, grad_weights, grad_tvalues):
        grad_means3D, grad_scales, grad_rotations, grad_opacity, grad_rays_o, grad_rays_d, grad_aabb_scale = None, None, None, None, None, None, None
        return grad_means3D, grad_scales, grad_rotations, grad_opacity, grad_rays_o, grad_rays_d, grad_aabb_scale

