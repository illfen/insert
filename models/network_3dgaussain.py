import math
import os
from typing import NamedTuple

import numpy as np
import pymeshlab
import scipy.spatial
import torch
import torch.nn as nn
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2

from models.utils import RGB2SH, SH2RGB, eval_sh


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def get_expon_lr_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    def helper(step):
        if lr_init == lr_final:
            # constant lr, ignore other params
            return lr_init
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def inside_test(points, vertices):
    deln = scipy.spatial.Delaunay(vertices)
    mask = deln.find_simplex(points) + 1
    mask[mask > 0] = 1
    return mask

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

    # eps must be small enough !!!
    inv_det = 1 / (a * d * f + 2 * e * c * b - e ** 2 * a - c ** 2 * d - b ** 2 * f + 1e-24)
    inv_a = (d * f - e ** 2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c ** 2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b ** 2) * inv_det

    power = -0.5 * (x ** 2 * inv_a + y ** 2 * inv_d + z ** 2 * inv_f) - x * y * inv_b - x * z * inv_c - y * z * inv_e

    power[power > 0] = -1e10  # abnormal values... make weights 0

    return torch.exp(power)


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


class GSNetwork:

    def __init__(self, opt, device=None, sh_degree=0):
        self.opt = opt
        self.active_sh_degree = 0
        self.max_sh_degree = opt.sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 10
        self.setup_functions()
        self.device = device

        if self.opt.bg_color:
            self.bg_color = torch.tensor(self.opt.bg_color,dtype=torch.float32,device="cuda", )
        else:
            self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda", )


        self.active_sh_degree = 0
        self._generation = torch.empty(0)
        self.anchor = {}
        self.localize = False


    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.spatial_lr_scale,
        )
    def load_state_dict(self, checkpoint_dict, strict=False):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            self.spatial_lr_scale,
        ) = checkpoint_dict
        self_features_rest = checkpoint_dict[3]
        print('self_features_rest',self_features_rest)

        self.training_setup(self.opt)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.set_mask(
            torch.ones(
                self._opacity.shape[0],
                dtype=torch.bool,
                device="cuda",
                requires_grad=False,
            )
        )
        self._generation = torch.zeros(
            self._opacity.shape[0],
            dtype=torch.int64,
            device="cuda",
            requires_grad=False,
        ) 
    def delete_gs_bbox(self, delete_min_xyz, delete_max_xyz):
        print(f"delete_min_xyz: {delete_min_xyz}, delete_max_xyz: {delete_max_xyz}")

        delete_mask = (
            (self._xyz[:, 0] >= delete_min_xyz[0]) & (self._xyz[:, 0] <= delete_max_xyz[0]) &
            (self._xyz[:, 1] >= delete_min_xyz[1]) & (self._xyz[:, 1] <= delete_max_xyz[1]) &
            (self._xyz[:, 2] >= delete_min_xyz[2]) & (self._xyz[:, 2] <= delete_max_xyz[2])
        )

        keep_mask = ~delete_mask

        self._xyz = self._xyz[keep_mask]
        self._features_dc = self._features_dc[keep_mask]
        self._features_rest = self._features_rest[keep_mask]
        self._scaling = self._scaling[keep_mask]
        self._rotation = self._rotation[keep_mask]
        self._opacity = self._opacity[keep_mask]
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self._xyz.device)
        self.set_mask(torch.ones(
            self._opacity.shape[0],
            dtype=torch.bool,
            device="cuda",
            requires_grad=False,
        ))
        self._generation = torch.zeros(
            self._opacity.shape[0],
            dtype=torch.int64,
            device="cuda",
            requires_grad=False,
        )
    def inside_check(self, points, bounding_box):
        import open3d as o3d
        points = np.array(points).reshape(-1, 3)
        query_point = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
        occupancy = bounding_box.compute_occupancy(query_point)
        mask = occupancy.numpy().astype(bool)
        print("Mask:", mask)

        return mask  

    def add_from_checkpoint(self, checkpoint_dict, delete_min_xyz=None, delete_max_xyz=None,reset_color=False,delete_bounding_box=None):
        self._xyz = self._xyz.to('cuda')
        self._features_dc = self._features_dc.to('cuda')
        self._features_rest = self._features_rest.to('cuda')
        self._scaling = self._scaling.to('cuda')
        self._rotation = self._rotation.to('cuda')
        self._opacity = self._opacity.to('cuda')
        
        self.set_mask(
            torch.ones(
                self._opacity.shape[0],
                dtype=torch.bool,
                device="cuda",
                requires_grad=False,
            )
        )
        (
            self.active_sh_degree,
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_scaling,
            new_rotation,
            new_opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            self.spatial_lr_scale,
        ) = checkpoint_dict
        new_features_rest = checkpoint_dict[3]
        new_xyz = new_xyz.to('cuda')
        new_features_dc = new_features_dc.to('cuda')
        new_features_rest = new_features_rest.to('cuda')
        new_scaling = new_scaling.to('cuda')
        new_rotation = new_rotation.to('cuda')
        new_opacity = new_opacity.to('cuda')


        if(delete_min_xyz is not None and delete_max_xyz is not None):

            new_xyz = new_xyz.to('cuda')
            delete_max_xyz = torch.tensor(delete_max_xyz, device='cuda')
            delete_min_xyz = torch.tensor(delete_min_xyz, device='cuda')
            delete_mask = (
                (new_xyz[:, 0] >= delete_min_xyz[0]) & (new_xyz[:, 0] <= delete_max_xyz[0]) &
                (new_xyz[:, 1] >= delete_min_xyz[1]) & (new_xyz[:, 1] <= delete_max_xyz[1]) &
                (new_xyz[:, 2] >= delete_min_xyz[2]) & (new_xyz[:, 2] <= delete_max_xyz[2])
            )

            keep_mask = (~delete_mask).to('cuda')


            new_xyz = new_xyz[keep_mask]

            new_features_dc = new_features_dc[keep_mask]
            new_features_rest = new_features_rest[keep_mask]
            new_scaling = new_scaling[keep_mask]
            new_rotation = new_rotation[keep_mask]
            new_opacity = new_opacity[keep_mask]
        
        if delete_bounding_box is not None:
            delete_mask = self.inside_check(new_xyz.data.detach().cpu().numpy(), delete_bounding_box)
            keep_mask = torch.from_numpy(delete_mask).bool().to(new_xyz.device)
            new_xyz = new_xyz[keep_mask]

        self._xyz = nn.Parameter(torch.cat([self._xyz, new_xyz], dim=0))
        if reset_color:
            self._features_dc = nn.Parameter(torch.cat([self._features_dc, new_features_dc * 0 - 10], dim=0))
            new_features_rest = new_features_rest * 0
        else:
            self._features_dc = nn.Parameter(torch.cat([self._features_dc, new_features_dc], dim=0))
        if new_features_rest.shape != torch.Size([0]) and self._features_rest.shape != torch.Size([0]):
            if new_features_rest.size(1) == 0 and self._features_rest.size(1) != 0:
                num_rest = (self.max_sh_degree + 1) ** 2 - 1
                filled_features_rest = torch.zeros((new_features_rest.size(0), num_rest, new_features_rest.size(2)), device=new_features_rest.device)
                self._features_rest = nn.Parameter(torch.cat([self._features_rest, filled_features_rest], dim=0))
            elif new_features_rest.size(1) != 0 and self._features_rest.size(1) == 0:
                num_rest = (self.max_sh_degree + 1) ** 2 - 1
                filled_features_rest = torch.zeros((self._features_rest.size(0), num_rest, self._features_rest.size(2)), device=self._features_rest.device)
                self._features_rest = nn.Parameter(torch.cat([filled_features_rest, new_features_rest], dim=0))
            else:
                self._features_rest = nn.Parameter(torch.cat([self._features_rest, new_features_rest], dim=0))
        else:
            self._features_rest = nn.Parameter(torch.cat([self._features_rest, new_features_rest], dim=0))

        self._opacity = nn.Parameter(torch.cat([self._opacity, new_opacity], dim=0))
        self._scaling = nn.Parameter(torch.cat([self._scaling, new_scaling], dim=0))
        self._rotation = nn.Parameter(torch.cat([self._rotation, new_rotation], dim=0))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self._xyz.device)
        


        new_mask = torch.zeros(
            new_opacity.shape[0],
            dtype=torch.bool,
            device="cuda",
            requires_grad=False,
        )
        self.set_mask(torch.cat([self.mask, new_mask], dim=0))

        self.training_setup(self.opt)

    def add_from_checkpoint_no_color(self, checkpoint_dict):
        self.set_mask(
            torch.ones(
                self._opacity.shape[0],
                dtype=torch.bool,
                device="cuda",
                requires_grad=False,
            )
        )
        (
            self.active_sh_degree,
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_scaling,
            new_rotation,
            new_opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            self.spatial_lr_scale,
        ) = checkpoint_dict

        new_xyz = new_xyz.to(self._xyz.device)
        new_features_dc = new_features_dc.to(self._features_dc.device)
        new_features_rest = new_features_rest.to(self._features_rest.device)
        new_scaling = new_scaling.to(self._scaling.device)
        new_rotation = new_rotation.to(self._rotation.device)
        new_opacity = new_opacity.to(self._opacity.device)

        self._xyz = nn.Parameter(torch.cat([self._xyz, new_xyz], dim=0))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, new_features_dc], dim=0))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest, new_features_rest], dim=0))
        self._opacity = nn.Parameter(torch.cat([self._opacity, new_opacity - 10.0], dim=0))
        self._scaling = nn.Parameter(torch.cat([self._scaling, new_scaling], dim=0))
        self._rotation = nn.Parameter(torch.cat([self._rotation, new_rotation], dim=0))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self._xyz.device)
        

        new_mask = torch.zeros(
            new_opacity.shape[0],
            dtype=torch.bool,
            device="cuda",
            requires_grad=False,
        )
        self.set_mask(torch.cat([self.mask, new_mask], dim=0))

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        if self._rotation.dim() == 1:
            self._rotation = self._rotation.view(-1, 1)
        elif self._rotation.dim() > 2:
            raise ValueError("Expected _rotation to be 1D or 2D")

        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def add_from_box(self, pcd: BasicPointCloud, spatial_lr_scale: float = 1):

        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color  # + 10
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(torch.cat([self._xyz.data, fused_point_cloud.requires_grad_(True)]))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc.data,  # * 0 - 10,
                                                    features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(
                                                        True)]))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest.data,  # * 0,
                                                      features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(
                                                          True)]))
        self._scaling = nn.Parameter(torch.cat([self._scaling.data, scales.requires_grad_(True)]))
        self._rotation = nn.Parameter(torch.cat([self._rotation.data, rots.requires_grad_(True)]))
        self._opacity = nn.Parameter(torch.cat([self._opacity.data, opacities.requires_grad_(True)]))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")



    def add_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float = 1):


        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())


        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color  + 10
        features[:, 3:, 1:] = 0.0

        # print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.15 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(torch.cat([self._xyz.data, fused_point_cloud.requires_grad_(True)]))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc.data* 0 - 10,
                            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)]))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest.data * 0,
                         features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)]))
        self._scaling = nn.Parameter(torch.cat([self._scaling.data, scales.requires_grad_(True)]))
        self._rotation = nn.Parameter(torch.cat([self._rotation.data, rots.requires_grad_(True)]))
        self._opacity = nn.Parameter(torch.cat([self._opacity.data, opacities.requires_grad_(True)]))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
    def add_from_pcd_no_grad(self, pcd: BasicPointCloud, set_mask=False ,spatial_lr_scale: float = 1):
        
        if set_mask:
            self.set_mask(torch.zeros(
                self._opacity.shape[0],
                dtype=torch.bool,
                device="cuda",
                requires_grad=False,
            ))
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())


        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color  + 10
        features[:, 3:, 1:] = 0.0

        # print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.15 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # print('self._xyz.data_device',self._xyz.data.device)
        # print('fused_point_cloud.requires_grad_(False)',fused_point_cloud.requires_grad_(False).device)
        self._xyz = nn.Parameter(torch.cat([self._xyz.data.to('cuda'), fused_point_cloud.requires_grad_(False)]))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc.data.to('cuda')* 0 - 10,
                            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(False)]))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest.data.to('cuda') * 0,
                         features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(False)]))
        self._scaling = nn.Parameter(torch.cat([self._scaling.data.to('cuda'), scales.requires_grad_(False)]))
        self._rotation = nn.Parameter(torch.cat([self._rotation.data.to('cuda'), rots.requires_grad_(False)]))
        self._opacity = nn.Parameter(torch.cat([self._opacity.data.to('cuda'), opacities.requires_grad_(False)]))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if set_mask:
            new_mask = torch.zeros(
                opacities.shape[0],
                dtype=torch.bool,
                device="cuda",
                requires_grad=False,
            )
            self.set_mask(torch.cat([self.mask, new_mask], dim=0))

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float = 1):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.9 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]
        self.params_list = l
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def set_mask(self, mask):
        self.mask = mask

    def initialize_from_mesh(self, path, R_path, radius):

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(path)
        # ms.set_matrix(transformmatrix= np.load(R_path))
        pymesh = ms.current_mesh()
        xyz = pymesh.vertex_matrix()
        # --- replace pymeshlab.set_matrix with safe numpy transform ---
        R = np.load(R_path)
        if R.shape == (4, 4):
            ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
            xyz_h = np.concatenate([xyz, ones], axis=1)
            xyz = (R @ xyz_h.T).T[:, :3]
        elif R.shape == (3, 3):
            xyz = xyz @ R.T
        num_pts = xyz.shape[0]

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )
        self.create_from_pcd(pcd, radius)

    def add_from_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        
        if "opacity" in plydata.elements[0].data.dtype.names:
            opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        else:
            opacities = np.ones((len(plydata.elements[0].data), 1))
        
        # print("Number of points at adding : ", xyz.shape[0])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        # print(len(extra_f_names), self.max_sh_degree)
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        new_xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
        new_features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        new_features_rest = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        new_opacity = torch.tensor(opacities, dtype=torch.float, device="cuda")
        new_scaling = torch.tensor(scales, dtype=torch.float, device="cuda")
        new_rotation = torch.tensor(rots, dtype=torch.float, device="cuda")
        
        new_features_rest = new_features_rest[:, :0, :]

        new_object_gaussian = {
            "_xyz": new_xyz,
            "_features_dc": new_features_dc,
            "_features_rest": new_features_rest,
            "_opacity": new_opacity,
            "_scaling": new_scaling,
            "_rotation": new_rotation
        }

        self.concat_gaussians(new_object_gaussian)

        new_mask = torch.ones(new_xyz.shape[0], dtype=torch.bool, device="cuda", requires_grad=False)

        self.mask = torch.cat([self.mask, new_mask], dim=0)
        return new_xyz.shape[0]


    def add_from_ply_no_grad(self, path):
        self.set_mask(torch.zeros(self._xyz.shape[0], dtype=torch.bool, device="cuda", requires_grad=False))
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)

        if "opacity" in plydata.elements[0].data.dtype.names:
            opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        else:
            opacities = np.ones((len(plydata.elements[0].data), 1))

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        if "f_dc_0" in plydata.elements[0].data.dtype.names:
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        # 可选：自动推断 sh_degree（如果你希望自适应）
        # self.max_sh_degree = int(((len(extra_f_names) + 3) / 3) ** 0.5 - 1)

        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3, \
            f"PLY f_rest_* 数量与当前 max_sh_degree 不匹配: {len(extra_f_names)} vs {3 * (self.max_sh_degree + 1) ** 2 - 3}"

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        new_xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
        new_features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        new_features_rest = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        new_opacity = torch.tensor(opacities, dtype=torch.float, device="cuda")
        new_scaling = torch.tensor(scales, dtype=torch.float, device="cuda")
        new_rotation = torch.tensor(rots, dtype=torch.float, device="cuda")

        # 如果你要把 rest 系数设为 0，请保持形状一致
        # new_features_rest = torch.zeros_like(new_features_rest)

        # 与已有 features_rest 对齐（关键修复）
        target_rest_dim = self._features_rest.shape[1] if self._features_rest.numel() > 0 \
                        else (self.max_sh_degree + 1) ** 2 - 1
        if new_features_rest.shape[1] != target_rest_dim:
            if new_features_rest.shape[1] == 0:
                new_features_rest = torch.zeros(
                    (new_features_rest.shape[0], target_rest_dim, new_features_rest.shape[2]),
                    device=new_features_rest.device, dtype=new_features_rest.dtype
                )
            elif self._features_rest.shape[1] == 0:
                self._features_rest = nn.Parameter(torch.zeros(
                    (self._features_rest.shape[0], new_features_rest.shape[1], self._features_rest.shape[2]),
                    device=self._features_rest.device, dtype=self._features_rest.dtype
                ))
            elif new_features_rest.shape[1] < target_rest_dim:
                pad = target_rest_dim - new_features_rest.shape[1]
                new_features_rest = torch.cat([
                    new_features_rest,
                    torch.zeros(new_features_rest.shape[0], pad, new_features_rest.shape[2],
                                device=new_features_rest.device, dtype=new_features_rest.dtype)
                ], dim=1)
            else:
                new_features_rest = new_features_rest[:, :target_rest_dim, :]

        self._xyz = nn.Parameter(torch.cat([self._xyz.to('cuda'), new_xyz], dim=0).detach().requires_grad_(False))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc.to('cuda'), new_features_dc], dim=0).detach().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest.to('cuda'), new_features_rest], dim=0).detach().requires_grad_(False))
        self._opacity = nn.Parameter(torch.cat([self._opacity.to('cuda'), new_opacity], dim=0).detach().requires_grad_(False))
        self._scaling = nn.Parameter(torch.cat([self._scaling.to('cuda'), new_scaling], dim=0).detach().requires_grad_(False))
        self._rotation = nn.Parameter(torch.cat([self._rotation.to('cuda'), new_rotation], dim=0).detach().requires_grad_(False))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        new_mask = torch.ones(new_xyz.shape[0], dtype=torch.bool, device="cuda", requires_grad=False)
        self.mask = torch.cat([self.mask, new_mask], dim=0)
        return new_xyz.shape[0]


    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        if "opacity" in plydata.elements[0].data.dtype.names:
            opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        else:
            opacities = np.ones((len(plydata.elements[0].data), 1)) 

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        # print(len(extra_f_names), self.max_sh_degree)
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        self._generation = torch.zeros(
            self._opacity.shape[0],
            dtype=torch.int64,
            device="cuda",
            requires_grad=False,
        )  # generation list, begin from zero
        self.set_mask(
            torch.ones(
                self._opacity.shape[0],
                dtype=torch.bool,
                device="cuda",
                requires_grad=False,
            )
        )
        self.apply_grad_mask(self.mask)

        self.update_anchor()



    def load_ply_from_gs_editor(self, path):
        self.set_mask(torch.zeros(self._xyz.shape[0], dtype=torch.bool, device="cuda", requires_grad=False))

        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        if "opacity" in plydata.elements[0].data.dtype.names:
            opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        else:
            opacities = np.ones((len(plydata.elements[0].data), 1)) 

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        self.max_sh_degree = int(((len(extra_f_names) + 3) / 3) ** 0.5 - 1)
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree
        self._generation = torch.zeros(
            self._opacity.shape[0],
            dtype=torch.int64,
            device="cuda",
            requires_grad=False,
        )  # generation list, begin from zero
        new_mask = torch.ones(xyz.shape[0], dtype=torch.bool, device="cuda", requires_grad=False)

        # 如果当前 mask 长度为 0（模型最初为空），直接赋值；否则替换为新 mask，避免重复拼接导致长度不匹配
        try:
            if getattr(self, 'mask', None) is None or self.mask.numel() == 0:
                self.mask = new_mask
            else:
                # 如果已有 mask 长度等于当前 _xyz 的旧长度，替换为 new_mask
                # 否则直接用 new_mask（防止重复拼接）
                if self.mask.shape[0] == new_mask.shape[0]:
                    self.mask = new_mask
                else:
                    self.mask = new_mask
        except Exception:
            self.mask = new_mask
        self.apply_grad_mask(self.mask)

        self.update_anchor()

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:

                if group["params"][0].shape[1] == 0:
                    # If the second dimension is 0, reshape extension_tensor to have second dimension 0 as well
                    extension_tensor = extension_tensor[:, :0]  # This essentially makes the second dimension of extension_tensor 0


                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds) * 0.01
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                   torch.max(self.get_scaling,  dim=1).values <= self.percent_dense * scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def init_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        dist2 = torch.clamp_min(distCUDA2(self._xyz[selected_pts_mask].data.float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask] * 0 + scales

        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > self.percent_dense * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def densify_and_prune_with_mask(self, max_grad, min_opacity, extent, max_screen_size, old_mask):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        grads[old_mask==0] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        new_mask = torch.cat([old_mask.clone(), torch.ones(self._xyz.shape[0] - old_mask.shape[0]).cuda()])


        prune_mask = (self.get_opacity < min_opacity).squeeze()


        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > self.percent_dense * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            prune_mask[new_mask==0]=False
        self.prune_points(prune_mask)

        new_mask = new_mask[~prune_mask]
        torch.cuda.empty_cache()

        return new_mask

    def prune(self, min_opacity, extent, max_screen_size):

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
    def add_points(self, new_xyz, new_opacities, new_features_dc, new_features_rest, new_scaling, new_rotation):
        new_xyz = new_xyz.to(self.device)
        new_opacities = new_opacities.to(self.device)
        new_features_dc = new_features_dc.to(self.device)
        new_features_rest = new_features_rest.to(self.device)
        new_scaling = new_scaling.to(self.device)
        new_rotation = new_rotation.to(self.device)
        new_opacities = new_opacities.unsqueeze(-1)

        new_features_dc = new_features_dc.unsqueeze(-1)
        new_features_dc = new_features_dc.repeat(1, 1, 3)
        new_features_rest = new_features_rest.unsqueeze(-1)
        new_features_rest = new_features_rest.repeat(1, 1, 3)

        if new_rotation.shape[1] == 3: 
            new_rotation = torch.cat([new_rotation, torch.zeros(new_rotation.shape[0], 1, device=new_rotation.device)], dim=1)

        device = new_xyz.device

        self._xyz = torch.cat([self._xyz, new_xyz], dim=0)
        self._opacity = torch.cat([self._opacity, new_opacities], dim=0)
        self._features_dc = torch.cat([self._features_dc, new_features_dc], dim=0)
        self._features_rest = torch.cat([self._features_rest, new_features_rest], dim=0)

        self._scaling = torch.cat([self._scaling, new_scaling], dim=0)

        self._rotation = torch.cat([self._rotation, new_rotation], dim=0)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def add_densification_stats(self, viewspace_point_tensor, update_filter, mask=None):
        if not viewspace_point_tensor.requires_grad:
            viewspace_point_tensor.requires_grad_(True)

        if mask is not None:
            # 只选择可见点索引
            visible_idx = torch.nonzero(update_filter, as_tuple=True)[0]
            mask_visible = mask[visible_idx]  # 截取 mask 对应可见点

            # 安全判断 grad 是否为 None
            if viewspace_point_tensor.grad is not None:
                grad_visible = viewspace_point_tensor.grad[visible_idx]
                self.xyz_gradient_accum[visible_idx][mask_visible > 0] += torch.norm(grad_visible[mask_visible > 0][:, :2], dim=-1, keepdim=True)

            self.denom[visible_idx][mask_visible > 0] += 1

        else:
            # 没有 mask 的情况
            if viewspace_point_tensor.grad is not None:
                self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
            self.denom[update_filter] += 1



    def get_params(self, lr):

        params = [
            {'params': [self._xyz], 'lr': self.opt.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': self.opt.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': self.opt.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': self.opt.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': self.opt.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': self.opt.rotation_lr, "name": "rotation"}
        ]

        return params


    def render(self, viewpoint_camera,
        scaling_modifier=1.0,
        invert_bg_color=False,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        bg_color=None,
        mask=None):



        screenspace_points = (
                torch.zeros_like(
                    self.get_xyz,
                    dtype=self.get_xyz.dtype,
                    requires_grad=True,
                    device="cuda",
                )
                + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        if bg_color is None:
            bg_color = self.bg_color

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color if not invert_bg_color else 1 - bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.get_xyz
        # print("means3D shape:", means3D.shape)  
        means2D = screenspace_points
        opacity = self.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.get_covariance(scaling_modifier)
        else:
            scales = self.get_scaling
            rotations = self.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.get_features.transpose(1, 2).view(
                    -1, 3, (self.max_sh_degree + 1) ** 2
                )
                dir_pp = self.get_xyz - viewpoint_camera.camera_center.repeat(
                    self.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                print("colors_precomp shape:", colors_precomp.shape)
            else:
                shs = self.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        # start_point =63144
        # if mask.shape[0] > means3D.shape[0]:
        #     mask = mask[:means3D.shape[0]]
        # else:
        #     means3D = means3D[:mask.shape[0]]
        if mask is not None:
            # print(start_point)
            if mask.shape[0] > means3D.shape[0]:
                mask = mask[:means3D.shape[0]]
            else:
                means3D = means3D[:mask.shape[0]]

            if mask.shape[0] > means2D.shape[0]:
                mask = mask[:means2D.shape[0]]
            else:
                means2D = means2D[:mask.shape[0]]

            if mask.shape[0] > shs.shape[0]:
                mask = mask[:shs.shape[0]]
            else:
                shs = shs[:mask.shape[0]]

            if mask.shape[0] > opacity.shape[0]:
                mask = mask[:opacity.shape[0]]
            else:
                opacity = opacity[:mask.shape[0]]

            if mask.shape[0] > scales.shape[0]:
                mask = mask[:scales.shape[0]]
            else:
                scales = scales[:mask.shape[0]]

            if mask.shape[0] > rotations.shape[0]:
                mask = mask[:rotations.shape[0]]
            else:
                rotations = rotations[:mask.shape[0]]


            means3D = means3D[mask==1]
            means2D = means2D[mask==1]
            shs = shs[mask==1]
            opacity = opacity[mask==1]
            scales = scales[mask==1]
            rotations = rotations[mask==1]

        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
    
    def apply_grad_mask(self, mask):
        print("self.mask",self.mask.shape,"self._xyz",self._xyz.shape)
        assert self.mask.shape[0] == self._xyz.shape[0]
        self.set_mask(mask)

        def hook(grad):
            final_grad = grad * (
                self.mask[:, None] if grad.ndim == 2 else self.mask[:, None, None]
            )
            return final_grad

        fields = ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling"]

        self.hooks = []

        for field in fields:
            this_field = getattr(self, field)
            assert this_field.is_leaf and this_field.requires_grad
            self.hooks.append(this_field.register_hook(hook))

    def update_anchor(self):
        self.anchor = dict(
            _xyz=self._xyz.detach().clone(),
            _features_dc=self._features_dc.detach().clone(),
            _features_rest=self._features_rest.detach().clone(),
            _scaling=self._scaling.detach().clone(),
            _rotation=self._rotation.detach().clone(),
            _opacity=self._opacity.detach().clone(),
        )

    def prune_with_mask(self, new_mask=None):
        self.prune_points(self.mask)  # all the mask with value 1 are pruned
        if new_mask is not None:
            self.mask = new_mask
        else:
            self.mask[:] = 1  # all updatable
        # self.remove_grad_mask()
        self.apply_grad_mask(self.mask)
        self.update_anchor()

    def concat_gaussians(self, another_gaussian):
        new_xyz = another_gaussian["_xyz"]
        new_features_dc = another_gaussian["_features_dc"]
        new_features_rest = another_gaussian["_features_rest"]
        new_opacities = another_gaussian["_opacity"]
        new_scaling = another_gaussian["_scaling"]
        new_rotation = another_gaussian["_rotation"]
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )
        self.mask = ~self.mask
        self.mask = torch.cat([self.mask, torch.ones_like(new_opacities[:, 0], dtype=torch.bool)], dim=0)