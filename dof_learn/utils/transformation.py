import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.quaternion import Quaternion


class GaussianTransformNet(nn.Module):
    def __init__(self):
        super(GaussianTransformNet, self).__init__()

        self.quaternion = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]))  

        self.scale = nn.Parameter(torch.tensor(1.0))  

        self.tvec_x = nn.Parameter(torch.tensor(0.0))  
        self.tvec_y = nn.Parameter(torch.tensor(0.0))  
        self.tvec_z = nn.Parameter(torch.tensor(0.0))  

    def forward(self):
        scale = F.softplus(self.scale) + 1e-6  

        quat = F.normalize(self.quaternion, p=2, dim=0)
        rotmat = kornia.geometry.quaternion_to_rotation_matrix(quat.unsqueeze(0)).squeeze(0)

        tvec = torch.stack([self.tvec_x, self.tvec_y, self.tvec_z])

        return scale, rotmat, tvec


def scale_gaussians(gaussian, scale, scale_factor=1.0):
    scale = scale.unsqueeze(-1)
    gaussian._xyz = gaussian._xyz * scale * scale_factor
    g_scale = gaussian.get_scaling * scale * scale_factor
    gaussian._scaling = torch.log(g_scale + 1e-7)


def rotate_gaussians(gaussian, rotmat):
    rot_q = Quaternion.from_matrix(rotmat[None, ...])
    g_qvec = Quaternion(gaussian.get_rotation)
    gaussian._rotation = (rot_q * g_qvec).data

    gaussian._xyz = torch.einsum("ij,bj->bi", rotmat, gaussian._xyz)

def translate_gaussians(gaussian, tvec):
    tvec = tvec.to(gaussian._xyz.device)
    tvec_expanded = tvec.repeat(gaussian._xyz.shape[0], 1)
    gaussian._xyz = gaussian._xyz + tvec_expanded


def apply_gaussian_transform(gaussian, scale, rotmat, tvec , scale_factor=1.0):
    scale_gaussians(gaussian, scale, scale_factor)
    rotate_gaussians(gaussian, rotmat)
    translate_gaussians(gaussian, tvec)
    return gaussian

def scale_gaussians_mask(gaussian, scale, scale_factor=1.0):
    scale = scale.unsqueeze(-1)
    affected_mask = gaussian.mask == 1
    scale_tensor = torch.ones_like(gaussian._xyz)
    scale_tensor[affected_mask] = scale * scale_factor
    gaussian._xyz = gaussian._xyz * scale_tensor
    g_scale = gaussian.get_scaling * scale_tensor.squeeze(-1)
    gaussian._scaling = torch.log(g_scale + 1e-7)

def rotate_gaussians_mask(gaussian, rotmat):
    rot_q = Quaternion.from_matrix(rotmat[None, ...])
    g_qvec = Quaternion(gaussian.get_rotation)

    affected_mask = gaussian.mask == 1
    new_rotation = (rot_q * g_qvec).data

    gaussian._rotation = gaussian._rotation.clone()
    gaussian._rotation[affected_mask] = new_rotation[affected_mask]
    gaussian._xyz[affected_mask] = torch.einsum("ij,bj->bi", rotmat, gaussian._xyz[affected_mask])

def translate_gaussians_mask(gaussian, tvec):
    tvec = tvec.to(gaussian._xyz.device)
    affected_mask = gaussian.mask == 1
    if tvec.dim() == 2 and tvec.size(0) == 1:
        tvec = tvec.squeeze(0)
    if tvec.dim() != 1 or tvec.size(0) != 3:
        raise ValueError("tvec must be a 1D tensor of size 3 or a 2D tensor of size (1, 3)")
    
    gaussian._xyz[affected_mask] += tvec

def apply_gaussian_transform_mask(gaussian, scale, rotmat, tvec, scale_factor=1.0):
    scale_gaussians_mask(gaussian, scale, scale_factor=scale_factor)
    rotate_gaussians_mask(gaussian, rotmat)
    translate_gaussians_mask(gaussian, tvec)
    return gaussian

def rotate_object_gaussians(gaussian, rotmat):
    affected_mask = gaussian.mask == 1
    affected_points = gaussian._xyz[affected_mask]

    center = affected_points.mean(dim=0, keepdim=True)

    new_xyz = gaussian._xyz.clone()

    affected_points = affected_points - center

    affected_points = torch.einsum("ij,bj->bi", rotmat, affected_points)

    affected_points = affected_points + center

    new_xyz[affected_mask] = affected_points

    gaussian._xyz = new_xyz

    rot_q = Quaternion.from_matrix(rotmat[None, ...])
    g_qvec = Quaternion(gaussian.get_rotation)
    new_rotation = (rot_q * g_qvec).data

    gaussian._rotation = gaussian._rotation.clone()
    gaussian._rotation[affected_mask] = new_rotation[affected_mask]
