import torch


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L

def strip_symmetric(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def covariance_activation(scales, rotations):
    L = build_scaling_rotation(scales, rotations)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm

def covariance_activation_unstrip(scales, rotations):
    L = build_scaling_rotation(scales, rotations)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def backward_covariance(grad_cov, scales, rotations):
    '''backward func covariance_activation_unstrip'''

    # covariance_activation
    L = build_scaling_rotation(scales, rotations)
    grad_L = torch.einsum('gab,gbc->gac', grad_cov, L) + torch.einsum('gab,gbc->gac', grad_cov.transpose(1, 2), L)

    # build_scaling_rotation
    S = torch.zeros((scales.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(rotations)
    S[:, 0, 0] = scales[:, 0]
    S[:, 1, 1] = scales[:, 1]
    S[:, 2, 2] = scales[:, 2]
    grad_R = torch.einsum('gab,gbc->gac', grad_L, S)
    grad_S = torch.einsum('gab,gbc->gac', R.transpose(1, 2), grad_L)
    grad_scales = torch.zeros_like(scales)
    grad_scales[:, 0] = grad_S[:, 0, 0]
    grad_scales[:, 1] = grad_S[:, 1, 1]
    grad_scales[:, 2] = grad_S[:, 2, 2]

    # build_rotation
    grad_rotations_norm = torch.zeros_like(rotations)

    grad_rotations_norm[:, 0] = - 2 * rotations[:, 3] * grad_R[:, 0, 1]\
                           + 2 * rotations[:, 2] * grad_R[:, 0, 2]\
                           + 2 * rotations[:, 3] * grad_R[:, 1, 0]\
                           - 2 * rotations[:, 1] * grad_R[:, 1, 2]\
                           - 2 * rotations[:, 2] * grad_R[:, 2, 0]\
                           + 2 * rotations[:, 1] * grad_R[:, 2, 1]
    
    grad_rotations_norm[:, 1] = 2 * rotations[:, 2] * grad_R[:, 0, 1]\
                           + 2 * rotations[:, 3] * grad_R[:, 0, 2]\
                           + 2 * rotations[:, 2] * grad_R[:, 1, 0]\
                           - 4 * rotations[:, 1] * grad_R[:, 1, 1]\
                           - 2 * rotations[:, 0] * grad_R[:, 1, 2]\
                           + 2 * rotations[:, 3] * grad_R[:, 2, 0]\
                           + 2 * rotations[:, 0] * grad_R[:, 2, 1]\
                           - 4 * rotations[:, 1] * grad_R[:, 2, 2]
                           
    grad_rotations_norm[:, 2] = - 4 * rotations[:, 2] * grad_R[:, 0, 0]\
                           + 2 * rotations[:, 1] * grad_R[:, 0, 1]\
                           + 2 * rotations[:, 0] * grad_R[:, 0, 2]\
                           + 2 * rotations[:, 1] * grad_R[:, 1, 0]\
                           + 2 * rotations[:, 3] * grad_R[:, 1, 2]\
                           - 2 * rotations[:, 0] * grad_R[:, 2, 0]\
                           + 2 * rotations[:, 3] * grad_R[:, 2, 1]\
                           - 4 * rotations[:, 2] * grad_R[:, 2, 2]
                           
    grad_rotations_norm[:, 3] = - 4 * rotations[:, 3] * grad_R[:, 0, 0]\
                           - 2 * rotations[:, 0] * grad_R[:, 0, 1]\
                           + 2 * rotations[:, 1] * grad_R[:, 0, 2]\
                           + 2 * rotations[:, 0] * grad_R[:, 1, 0]\
                           - 4 * rotations[:, 3] * grad_R[:, 1, 1]\
                           + 2 * rotations[:, 2] * grad_R[:, 1, 2]\
                           + 2 * rotations[:, 1] * grad_R[:, 2, 0]\
                           + 2 * rotations[:, 2] * grad_R[:, 2, 1]

    norm = torch.sqrt(rotations[:, 0] * rotations[:, 0] + rotations[:, 1] * rotations[:, 1] + rotations[:, 2] * rotations[:, 2] + rotations[:, 3] * rotations[:, 3])
    grad_rotations = (-1 * rotations * torch.sum(rotations * grad_rotations_norm, dim=1).unsqueeze(-1)) / (norm.unsqueeze(-1) ** 3) + grad_rotations_norm / norm.unsqueeze(-1) # bit different with torch autograd
    
    return grad_scales, grad_rotations
