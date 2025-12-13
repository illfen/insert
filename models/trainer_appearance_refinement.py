import abc
import glob
import os
import sys
import time

import imageio
import numpy as np
import open3d as o3d
import tensorboardX
import torch
import torch.distributed as dist
import tqdm
from rich.console import Console

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json

from dof_learn.utils.transformation import (
    GaussianTransformNet,
    apply_gaussian_transform_mask,
    rotate_gaussians_mask,
)
from dof_learn.utils.util import create_example_gaussian
from models.network_3dgaussain import GSNetwork
from models.provider import MiniCam
from models.sd import StableDiffusion


def load_transform_checkpoint(filename, model, optimizer=None):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    if  optimizer is not None:
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("Checkpoint does not contain 'optimizer_state_dict'.")
    # else:
        # print("Optimizer state not loaded.")

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

def pointcloud_to_mesh(ply_file, output_file):
    pcd = o3d.io.read_point_cloud(ply_file)

    hull, _ = pcd.compute_convex_hull()

    o3d.io.write_triangle_mesh(output_file, hull)
    print(f"mesh saved to {output_file}")

class Trainer_SDS(object):
    def __init__(self,
                 name,  # name of this experiment
                 opt,  # extra conf
                 model: GSNetwork,  # network
                 guidance: StableDiffusion,
                 metrics=[],
                 # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0,  # which GPU am I
                 world_size=1,  # total num of GPUs
                 device=None,  # device to use, usually setting to None is OK. (auto choose device)
                 mute=False,  # whether to mute all print
                 fp16=False,  # amp optimize level
                 eval_interval=1,  # eval once every $ epoch
                 max_keep_ckpt=5,  # max num of saved ckpts in disk
                 workspace='workspace',  # workspace to save logs & ckpts
                 best_mode='min',  # the smaller/larger result, the better
                 use_loss_as_metric=True,  # use loss as the first metric
                 report_metric_at_train=False,  # also report metrics at training
                 use_checkpoint="latest",  # which ckpt to use at init time
                 use_tensorboardX=True,  # whether to use tensorboard for logging,
                 sd_guidance=None,
                 ):
        self.text_local = None
        self.text_global = None
        self.near = 0.001
        self.far = 10.0
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.device = device if device is not None else torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.sd_guidance = sd_guidance





        if isinstance(model, torch.nn.Module):
            model.to(self.device)

        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if self.opt.bg_color is not None:
            self.bg_color = torch.tensor(self.opt.bg_color).cuda()
        else:
            self.bg_color = None

        self.guidance = guidance
        if self.sd_guidance is not None:
            self.prepare_text_embeddings()
        else:
            self.text = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }
        self.bbox_min = None
        self.bbox_max = None

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.ply_path = os.path.join(self.workspace, 'ply')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        if isinstance(model, torch.nn.Module):
            self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        self.log(self.opt)

        self.editing_points_mask = None

        
        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)


        if self.opt.bbox_path is not None:

            ply_file = self.opt.bbox_path
            output_file = self.opt.bbox_path
            pointcloud_to_mesh(ply_file, output_file)

            print('generate bounding_box', self.opt.bbox_path)
            mesh = o3d.io.read_triangle_mesh(self.opt.bbox_path)
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

            center = mesh.get_center()

            # print('generate bounding_box', self.opt.bbox_path)
            mesh = o3d.io.read_triangle_mesh(self.opt.bbox_path)
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

            vertices = np.asarray(mesh.vertex['positions'].numpy(), dtype=np.float64)

            bbox_min = vertices.min(axis=0)
            bbox_max = vertices.max(axis=0)
            self.bbox_min = bbox_min
            self.bbox_max = bbox_max

            center = vertices.mean(axis=0)

            vertices -= center
            vertices *= np.array(self.opt.bbox_size_factor, dtype=np.float64)
            vertices += center

            mesh.vertex['positions'] = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)

            mesh.compute_vertex_normals()

            self.bounding_box = o3d.t.geometry.RaycastingScene()
        
            _ = self.bounding_box.add_triangles(mesh)
        
        if self.opt.transform_ckpt is not None:

            net = GaussianTransformNet().to(self.device)
            load_transform_checkpoint(self.opt.transform_ckpt, net)
            # this.model.add_from_checkpoint(checkpoint_dict['model'])
            print('load object transform ckpt')
            self.model.add_from_ply_no_grad(self.opt.new_object_ply_path)
            scale, rotmat, tvec = net()
            print('scale', scale)  
            print('rotmat', rotmat)
            print('tvec', tvec)
            apply_gaussian_transform_mask(self.model, scale * opt.transform_scale_factor, rotmat, tvec - torch.tensor(opt.tvec_bias, device=self.device))
           

        if not opt.test:
            print('loading init 3dgs')
            # load scene gaussian
            if self.opt.load_path is not None:
                if self.opt.editing_type == 3:
                    checkpoint_dict = torch.load(self.opt.load_path, map_location=self.device)
                    if(opt.origin_bbox_path is not None):
                        gaussian_box = GSNetwork(opt, 'cuda')
                        import pymeshlab
                        ms = pymeshlab.MeshSet()
                        ms.load_new_mesh(opt.origin_bbox_path)
                        ms.meshing_surface_subdivision_midpoint(iterations=4)
                        pymesh = ms.current_mesh()
                        xyz = pymesh.vertex_matrix()

                        mean = xyz.mean(axis=0)
                        xyz = xyz - mean
                        xyz *= 1
                        xyz = xyz + mean

                        from PIL import Image

                        from models.network_3dgaussain import BasicPointCloud
                        from models.sh_utils import SH2RGB

                        num_pts = xyz.shape[0]
                        shs = np.ones((num_pts, 3))
                        pcd = BasicPointCloud(
                            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
                        )
                        
                        gaussian_box.add_from_pcd_no_grad(pcd)
                        gaussian_box.set_mask(torch.ones_like(gaussian_box._xyz[:, 0], dtype=torch.int32))  
                        net = GaussianTransformNet().to(self.device)
                        load_transform_checkpoint(self.opt.transform_ckpt, net) 
                        scale, rotmat, tvec = net()        
                        transformed_gaussian = apply_gaussian_transform_mask(gaussian_box, scale *1.8, rotmat, tvec - torch.tensor(opt.tvec_bias, device=device), float(opt.transform_scale_factor))
                        opt.rotation_matrix = json.loads(opt.rotation_matrix) if isinstance(opt.rotation_matrix, str) else opt.rotation_matrix
                        init_rotation_bias = torch.tensor(opt.rotation_matrix, dtype=torch.float32).to('cuda')
                        rotate_gaussians_mask(transformed_gaussian, init_rotation_bias)
                        delete_min_xyz = torch.min(gaussian_box._xyz, dim=0).values  # [3]
                        delete_max_xyz = torch.max(gaussian_box._xyz, dim=0).values  # [3]
                        
                    else:
                        delete_min_xyz = None
                        delete_max_xyz = None
                    self.model.add_from_checkpoint(checkpoint_dict['model'], delete_min_xyz, delete_max_xyz)

                # replace object
                elif self.opt.editing_type == 4:
                    checkpoint_dict = torch.load(self.opt.load_path, map_location=self.device)
                    if(opt.origin_bbox_path is not None):
                        gaussian_box = GSNetwork(opt, 'cuda')
                        import pymeshlab
                        ms = pymeshlab.MeshSet()
                        ms.load_new_mesh(opt.origin_bbox_path)
                        ms.meshing_surface_subdivision_midpoint(iterations=4)
                        pymesh = ms.current_mesh()
                        xyz = pymesh.vertex_matrix()

                        mean = xyz.mean(axis=0)
                        xyz = xyz - mean
                        xyz *= 1
                        xyz = xyz + mean

                        from PIL import Image

                        from models.network_3dgaussain import BasicPointCloud
                        from models.sh_utils import SH2RGB

                        num_pts = xyz.shape[0]
                        shs = np.ones((num_pts, 3))
                        pcd = BasicPointCloud(
                            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
                        )
                        
                        gaussian_box.add_from_pcd_no_grad(pcd)
                        gaussian_box.set_mask(torch.ones_like(gaussian_box._xyz[:, 0], dtype=torch.int32))  

                        delete_min_xyz = torch.min(gaussian_box._xyz, dim=0).values  # [3]
                        delete_max_xyz = torch.max(gaussian_box._xyz, dim=0).values  # [3]
                        
                    else:
                        delete_min_xyz = None
                        delete_max_xyz = None
                    self.model.add_from_checkpoint(checkpoint_dict['model'], delete_min_xyz, delete_max_xyz)

                else:
                    self.load_checkpoint(checkpoint=self.opt.load_path)
            else:
                if self.opt.ply_path is not None:
                    self.model.load_ply_from_gs_editor(self.opt.ply_path)

            self.ori_points = self.model.get_xyz.data.detach().cpu().numpy().shape[0]
            if self.opt.bbox_path is not None:
                mask = torch.LongTensor(
                    self.inside_check(self.model.get_xyz.data.detach().cpu().numpy())).to(self.device)
            else:
                mask = torch.zeros(self.model.get_xyz.data.shape[0], device=self.device)


            # layout object add
            if self.opt.editing_type == 3:
                print('edit existing object')
                if self.editing_points_mask is None:
                    self.editing_points_mask = self.model.mask.int()
                print('mask points: {}, all points: {}'.format(self.editing_points_mask.sum().item(),
                                                               self.editing_points_mask.size()[0]))
                if self.opt.reset_points:
                    self.model._opacity.data[mask > 0] *= 0
                    self.model._opacity.data[mask > 0] -= 4.5

            # layout object replace
            elif self.opt.editing_type == 4:
                print('edit existing object')
                if self.editing_points_mask is None:
                    self.editing_points_mask = self.model.mask.int()
                print('mask points: {}, all points: {}'.format(self.editing_points_mask.sum().item(),
                                                               self.editing_points_mask.size()[0]))
                if self.opt.reset_points:
                    self.model._opacity.data[mask > 0] *= 0
                    self.model._opacity.data[mask > 0] -= 4.5
        
    


        print('update_ori_parameter')
        self.backup_ori_parameter()

        print('active_sh_degree: {}'.format(self.model.active_sh_degree))
        size = self.model._features_rest.data.size()

        if len(size) > 1:
            self.sh_size = size[1]
        elif len(size) == 1:
            self.sh_size = size[0]
        wandb.init(project='FreeInsert_appearance_refinement', name=self.opt.text_local, config=self.opt)

    def backup_ori_parameter(self):
        if self.opt.update_color_only:
            self.ori_features_dc = self.model._features_dc.clone().data
            self.ori_features_rest = self.model._features_rest.clone().data
        else:
            self.ori_xyz = self.model._xyz.clone().data
            self.ori_features_dc = self.model._features_dc.clone().data
            self.ori_features_rest = self.model._features_rest.clone().data
            self.ori_opacity = self.model._opacity.clone().data
            self.ori_scaling = self.model._scaling.clone().data
            self.ori_rotation = self.model._rotation.clone().data

    def inside_check(self, points):
        points = np.array(points).reshape(-1, 3)
        query_point = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
        occupancy = self.bounding_box.compute_occupancy(query_point)
        mask = occupancy.numpy()
        print("Mask:", mask)

        return mask

    def remove_oob_points(self):

        # remove points out of bounding box
        if self.opt.editing_type == 0 or (self.opt.editing_type == 4 and self.opt.bbox_path is not None):
            remove_mask = (self.editing_points_mask > 0) * (1 - torch.LongTensor(
                self.inside_check(self.model.get_xyz.data.detach().cpu().numpy())).to(self.device))
            remove_mask += (self.editing_points_mask > 0) * (
                    torch.max(self.model.get_scaling, dim=1).values > self.opt.max_scale_size)
        else:
            # remove points with larger scale size
            remove_mask = (self.editing_points_mask > 0) * (
                    torch.max(self.model.get_scaling, dim=1).values > self.opt.max_scale_size)
        
        # remove points with low opacity
        remove_mask += (self.editing_points_mask > 0) * (self.model.get_opacity < self.opt.min_opacity).squeeze()

        prune_mask = remove_mask > 0

        self.model.prune_points(prune_mask)
        self.editing_points_mask = self.editing_points_mask[~prune_mask]
        self.backup_ori_parameter()
        torch.cuda.empty_cache()
        print('mask points: {}, all points: {}'.format(self.editing_points_mask.sum().item(),
                                                       self.editing_points_mask.size()[0]))

    # calculate the text embs.
    def prepare_text_embeddings(self):

        if self.opt.text_global is None:
            self.log(f"[WARN] text_global prompt is not provided.")
            sys.exit()
        if self.opt.text_local is None:
            self.log(f"[WARN] text_local prompt is not provided.")
            sys.exit()

        self.text_global = self.sd_guidance.get_text_embeds([self.opt.text_global] * int(self.opt.batch_size),
                                                         [self.opt.negative] * int(self.opt.batch_size))

        self.text_local = self.sd_guidance.get_text_embeds([self.opt.text_local] * int(self.opt.batch_size),
                                                        [self.opt.negative] * int(self.opt.batch_size))

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file


    def train_step(self, data):

        rgbs, mask, _, h, w, R, T, fx, fy, pose, index = data

        pred_global_rgb = []
        pred_local_rgb = []
        output_list = []
        for i in range(self.opt.batch_size):
            h_i = h[i]
            w_i = w[i]
            R_i = R[i]
            T_i = T[i]

            fx_i = fx[i]
            fy_i = fy[i]
            pose_i = pose[i]

            cur_cam = MiniCam(
                R_i,
                T_i,
                w_i,
                h_i,
                fy_i,
                fx_i,
                pose_i,
                self.near,
                self.far
            )

            if self.global_step > 500 and self.global_step % 2 == 0:
                # random background color
                r_color = torch.rand(3).cuda()
                outputs_nobg = self.model.render(cur_cam, mask=self.editing_points_mask, bg_color=r_color)
            else:
                outputs_nobg = self.model.render(cur_cam, mask=self.editing_points_mask)
            pred_local_rgb.append(outputs_nobg['image'])
            output_list.append(outputs_nobg)

            outputs = self.model.render(cur_cam, bg_color=self.bg_color)
            pred_global_rgb.append(outputs['image'])
            output_list.append(outputs)
            gaussian = create_example_gaussian(self.opt)
            net = GaussianTransformNet().to(self.device)
            load_transform_checkpoint(self.opt.transform_ckpt, net)
            gaussian.add_from_ply_no_grad(self.opt.new_object_ply_path)

            scale, rotmat, tvec = net()
            apply_gaussian_transform_mask(gaussian, scale * self.opt.transform_scale_factor, rotmat, tvec - torch.tensor(self.opt.tvec_bias).to('cuda'))
            

        pose = torch.tensor(pose).cuda().float()
        pred_local_rgb = torch.stack(pred_local_rgb)

            
        loss_sds_local = self.sd_guidance.train_step(self.text_local, pred_local_rgb, pose,
                                                  ratio=(self.global_step / self.opt.iters),
                                                  guidance_scale=self.opt.guidance_scale)

        pred_global_rgb = torch.stack(pred_global_rgb)
        loss_sds_global = self.sd_guidance.train_step(self.text_global, pred_global_rgb, pose,
                                                   ratio=(self.global_step / self.opt.iters),
                                                   guidance_scale=self.opt.guidance_scale)


        gamma = self.opt.start_gamma - (self.opt.start_gamma - self.opt.end_gamma) * (
                self.global_step / self.opt.iters)


        loss = gamma * loss_sds_global + (1 - gamma) * loss_sds_local
        loss_dict = {
            'loss_sds_local': loss_sds_local.item(),
            'loss': loss.item(),
        }




        return pred_global_rgb, loss, loss_dict, output_list

    def eval_step(self, data, if_mask=False):
        # rgbs, mask, h, w, R, T, fx, fy, index = data
        rgbs, mask, _, h, w, R, T, fx, fy, pose, index = data

        h = h[0]
        w = w[0]
        R = R[0]
        T = T[0]

        fx = fx[0]
        fy = fy[0]
        pose = pose[0]

        cur_cam = MiniCam(
            R,
            T,
            w,
            h,
            fy,
            fx,
            pose,
            self.near,
            self.far
        )

        B = 1
        if if_mask:
            outputs = self.model.render(cur_cam, invert_bg_color=True, mask=self.editing_points_mask)
        else:
            outputs = self.model.render(cur_cam)

        all_pred_rgb = outputs['image'].permute(1, 2, 0).reshape(B, h, w, 3)
        all_pred_depth = outputs['depth'].reshape(B, h, w)
        all_pred_depth /= torch.max(all_pred_depth)
        return all_pred_rgb, all_pred_depth

    def test_step(self, data, p_mask=None):

        rgbs, mask, _, h, w, R, T, fx, fy, pose, index = data

        h = h[0]
        w = w[0]
        R = R[0]
        T = T[0]

        fx = fx[0]
        fy = fy[0]
        pose = pose[0]

        cur_cam = MiniCam(
            R,
            T,
            w,
            h,
            fy,
            fx,
            pose,
            self.near,
            self.far
        )

        B = 1
        if p_mask is not None:
            outputs = self.model.render(cur_cam, invert_bg_color=True, mask=p_mask)  # invert_bg_color=True,
        else:
            outputs = self.model.render(cur_cam)

        all_pred_rgb = outputs['image'].permute(1, 2, 0).reshape(B, h, w, 3)
        all_pred_depth = outputs['depth'].reshape(B, h, w)
        all_pred_depth /= torch.max(all_pred_depth)
        all_pred_alpha = outputs['alpha'].reshape(B, h, w)

        return all_pred_rgb, all_pred_depth, all_pred_alpha

    def train(self, train_loader, valid_loader, max_epochs):
        self.valid_loader = valid_loader
        self.model.xyz_gradient_accum = torch.zeros((self.model.get_xyz.shape[0], 1), device="cuda")
        self.model.denom = torch.zeros((self.model.get_xyz.shape[0], 1), device="cuda")
        
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch
            
            if self.epoch == max_epochs:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint()
                # self.model.save_ply(os.path.join(self.workspace, 'ply', '{}.ply'.format(str(epoch))))
            else:
                self.evaluate_one_epoch(valid_loader, f"watch_{self.epoch}")

            self.train_one_epoch(train_loader)
            if not self.opt.update_color_only :
                self.remove_oob_points()
            # self.remove_oob_points()

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t) / 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True, if_gui=False):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                         bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():

            for i, data in enumerate(loader):
                preds, preds_depth = self.test_step(data)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)

            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8,
                             macro_block_size=1)

        if len(self.opt.bbox_path) > 0:

            mask = torch.LongTensor(
                self.inside_check(self.model.get_xyz.data.detach().cpu().numpy())).to(self.device)
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            if write_video:
                all_preds = []
            with torch.no_grad():
                for i, data in enumerate(loader):
                    preds, preds_depth = self.test_step(data, p_mask=mask)
                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)
                    if write_video:
                        all_preds.append(pred)
                    pbar.update(loader.batch_size)

            if write_video:
                all_preds = np.stack(all_preds, axis=0)
                imageio.mimwrite(os.path.join(save_path, f'{name}_rgb_cube.mp4'), all_preds, fps=25, quality=8,
                                 macro_block_size=1)

        self.log(f"==> Finished Test.")

    def train_one_epoch(self, loader):
        self.log(
            f"==> Start Training {self.workspace} Epoch {self.epoch}, lr={self.model.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        if isinstance(self.model, torch.nn.Module):
            self.model.train()


        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:

            self.local_step += 1
            self.global_step += 1

            self.model.update_learning_rate(self.global_step)

            pred_rgbs, loss, loss_dict, output_list = self.train_step(data)

            loss.backward()

            with torch.no_grad():
                if self.editing_points_mask.shape[0] > self.model.max_radii2D.shape[0]:
                    self.editing_points_mask = self.editing_points_mask[:self.model.max_radii2D.shape[0]]
                else:
                    self.model.max_radii2D = self.model.max_radii2D[:self.editing_points_mask.shape[0]]

                for outputs in output_list:
                    visibility_filter = outputs['visibility_filter']
                    radii = outputs['radii']
                    viewspace_point_tensor = outputs['viewspace_points']

                    if visibility_filter.size()[0] != self.model.max_radii2D.size()[0]:

                        self.model.max_radii2D[self.editing_points_mask > 0][visibility_filter] = torch.max(
                            self.model.max_radii2D[self.editing_points_mask > 0][visibility_filter],
                            radii[visibility_filter]
                        )
                        self.model.add_densification_stats(viewspace_point_tensor, visibility_filter, mask=self.editing_points_mask)
                    else:
                        self.model.max_radii2D[visibility_filter] = torch.max(
                            self.model.max_radii2D[visibility_filter], radii[visibility_filter]
                        )
                        self.model.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if (not self.opt.update_color_only) and self.global_step > self.opt.desification_start_percent * self.opt.iters and self.global_step < self.opt.desification_end_percent * self.opt.iters \
                        and self.global_step % self.opt.densification_interval == 0:
                    self.editing_points_mask = self.model.densify_and_prune_with_mask(self.opt.densify_grad_threshold,
                                                                                      min_opacity=self.opt.min_opacity,
                                                                                      extent=self.opt.extent,
                                                                                      max_screen_size=self.opt.max_screen_size,
                                                                                      old_mask=self.editing_points_mask)
                    self.backup_ori_parameter()
                    print('update_ori_parameter')
                    print('mask points: {}, all points: {}'.format(self.editing_points_mask.sum().item(),
                                                                   self.editing_points_mask.size()[0]))

                if self.global_step % self.opt.opacity_reset_interval == 0:
                    self.model.reset_opacity()

                self.model.optimizer.step()
                self.model.optimizer.zero_grad(set_to_none=True)
                if self.opt.update_color_only:
                    self.model._features_dc.data = (1 - self.editing_points_mask.reshape(-1, 1, 1).repeat(1, 1,
                                                                                                      3)) * self.ori_features_dc + \
                                                   self.editing_points_mask.reshape(-1, 1, 1).repeat(1, 1,
                                                                                                     3) * self.model._features_dc.data

                    self.model._features_rest.data = (1 - self.editing_points_mask.reshape(-1, 1, 1).repeat(1, self.sh_size,
                                                                                                        3)) * self.ori_features_rest + \
                                                     self.editing_points_mask.reshape(-1, 1, 1).repeat(1, self.sh_size,
                                                                                                       3) * self.model._features_rest.data
                else:
                    self.model._xyz.data = (1 - self.editing_points_mask.unsqueeze(1).repeat(1, 3)) * self.ori_xyz + \
                                        self.editing_points_mask.unsqueeze(1).repeat(1, 3) * self.model._xyz.data

                    self.model._features_dc.data = (1 - self.editing_points_mask.reshape(-1, 1, 1).repeat(1, 1,
                                                                                                        3)) * self.ori_features_dc + \
                                                self.editing_points_mask.reshape(-1, 1, 1).repeat(1, 1,
                                                                                                    3) * self.model._features_dc.data

                    self.model._features_rest.data = (1 - self.editing_points_mask.reshape(-1, 1, 1).repeat(1, self.sh_size,
                                                                                                            3)) * self.ori_features_rest + \
                                                    self.editing_points_mask.reshape(-1, 1, 1).repeat(1, self.sh_size,
                                                                                                    3) * self.model._features_rest.data

                    self.model._opacity.data = (1 - self.editing_points_mask.reshape(-1, 1)) * self.ori_opacity \
                                            + self.editing_points_mask.reshape(-1, 1) * self.model._opacity.data
                    self.model._scaling.data = (1 - self.editing_points_mask.reshape(-1, 1).repeat(1,
                                                                                                3)) * self.ori_scaling + \
                                            self.editing_points_mask.reshape(-1, 1).repeat(1,
                                                                                            3) * self.model._scaling.data
                    self.model._rotation.data = (1 - self.editing_points_mask.reshape(-1, 1).repeat(1,
                                                                                                    4)) * self.ori_rotation + \
                                                self.editing_points_mask.reshape(-1, 1).repeat(1,
                                                                                            4) * self.model._rotation.data

            loss_total = loss_dict['loss']
            
            # collision_loss = loss_dict['collision_loss']
            total_loss += loss_total

            if self.local_rank == 0:

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_total, self.global_step)
                    self.writer.add_scalar("train/lr", self.model.optimizer.param_groups[0]['lr'], self.global_step)

                    wandb.log({
                        "train/loss": loss_total,
                        # "train/collision_loss": collision_loss,
                        "train/lr": self.model.optimizer.param_groups[0]['lr'],
                    }, step=self.global_step)
                
                pbar.set_description(f"loss={loss_total:.4f} ({total_loss / self.local_step:.4f})")
                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        self.log(f"==> Finished Epoch {self.epoch}. average_loss {average_loss}")

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        all_preds = []
        all_preds_depth = []
        all_preds_nobg = []
        all_preds_depth_nobg = []
        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1
                preds, preds_depth = self.eval_step(data)
                preds_nobg, preds_depth_nobg = self.eval_step(data, if_mask=True)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in
                                  range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in
                                        range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    preds_list_nobg = [torch.zeros_like(preds_nobg).to(self.device) for _ in
                                       range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list_nobg, preds_nobg)
                    preds_nobg = torch.cat(preds_list_nobg, dim=0)

                    preds_depth_list_nobg = [torch.zeros_like(preds_depth_nobg).to(self.device) for _ in
                                             range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth_nobg)
                    preds_depth_nobg = torch.cat(preds_depth_list_nobg, dim=0)

                loss_val = 0
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:
                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_normal = os.path.join(self.workspace, 'validation',
                                                    f'{name}_{self.local_step:04d}_normal.png')
                    save_path_depth = os.path.join(self.workspace, 'validation',
                                                   f'{name}_{self.local_step:04d}_depth.png')

                    # self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth * 255).astype(np.uint8)

                    pred_nobg = preds_nobg[0].detach().cpu().numpy()
                    pred_nobg = (pred_nobg * 255).astype(np.uint8)

                    pred_depth_nobg = preds_depth_nobg[0].detach().cpu().numpy()
                    pred_depth_nobg = (pred_depth_nobg * 255).astype(np.uint8)

                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                    all_preds_nobg.append(pred_nobg)
                    all_preds_depth_nobg.append(pred_depth_nobg)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                    pbar.update(loader.batch_size)

        all_preds = np.stack(all_preds, axis=0)
        all_preds_depth = np.stack(all_preds_depth, axis=0)
        all_preds_nobg = np.stack(all_preds_nobg, axis=0)
        all_preds_depth_nobg = np.stack(all_preds_depth_nobg, axis=0)
        rgb_global_path = os.path.join(self.workspace, 'validation', f'{name}_rgb_global.mp4')
        depth_global_path = os.path.join(self.workspace, 'validation', f'{name}_depth_global.mp4')
        rgb_local_path = os.path.join(self.workspace, 'validation', f'{name}_rgb_local.mp4')
        depth_local_path = os.path.join(self.workspace, 'validation', f'{name}_depth_local.mp4')

        imageio.mimwrite(rgb_global_path, all_preds, fps=24, quality=8, macro_block_size=1)
        imageio.mimwrite(depth_global_path, all_preds_depth, fps=24, quality=8, macro_block_size=1)
        imageio.mimwrite(rgb_local_path, all_preds_nobg, fps=24, quality=8, macro_block_size=1)
        imageio.mimwrite(depth_local_path, all_preds_depth_nobg, fps=24, quality=8, macro_block_size=1)

        wandb.log({
            "rgb_global_video": wandb.Video(rgb_global_path, fps=24, format="mp4"),
            "depth_global_video": wandb.Video(depth_global_path, fps=24, format="mp4"),
            "rgb_local_video": wandb.Video(rgb_local_path, fps=24, format="mp4"),
            "depth_local_video": wandb.Video(depth_local_path, fps=24, format="mp4")
        })

        if os.path.exists(rgb_global_path):
            os.remove(rgb_global_path)
        if os.path.exists(depth_global_path):
            os.remove(depth_global_path)
        if os.path.exists(rgb_local_path):
            os.remove(rgb_local_path)
        if os.path.exists(depth_local_path):
            os.remove(depth_local_path)


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(
                    result if self.best_mode == 'min' else - result)  # if max mode, use -result
            else:
                self.stats["results"].append(average_loss)  # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        self.log(f"++> Evaluate epoch {self.epoch} Finished. loss: ({total_loss / self.local_step:.4f})")

    def save_checkpoint(self, name=None):
        file_path = f"{self.epoch}.pth"

        self.stats["checkpoints"].append(file_path)

        if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
            old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)
        state = {'epoch': self.epoch, 'global_step': self.global_step,
                 'model': self.model.capture(), 'mask': self.editing_points_mask}
        torch.save(state, os.path.join(self.ckpt_path, file_path))

    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        self.model.load_state_dict(checkpoint_dict['model'])
        if 'mask' in checkpoint_dict.keys():
            self.editing_points_mask = checkpoint_dict['mask']
        self.log("[INFO] loaded model.")
        self.log("[INFO] update optimizer.")
        return     