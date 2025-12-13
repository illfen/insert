import abc
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from utils.transformation import apply_gaussian_transform_mask, rotate_gaussians_mask

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.provider import SampleViewsDataset


def compute_geometric_center(image_tensor):
    B, H, W = image_tensor.shape
    assert B == 1, "Batch size must be 1"

    x_coords = torch.arange(W, dtype=torch.float32, device=image_tensor.device).view(1, 1, W)  # (1,1,W)
    y_coords = torch.arange(H, dtype=torch.float32, device=image_tensor.device).view(1, H, 1)  # (1,H,1)

    image_sum = torch.sum(image_tensor) + 1e-6
    C_x = torch.sum(image_tensor * x_coords) / image_sum
    C_y = torch.sum(image_tensor * y_coords) / image_sum

    return C_x, C_y

def load_target_centers(csv_path):
    df = pd.read_csv(csv_path)
    centers_dict = defaultdict(list)
    
    for _, row in df.iterrows():
        centers_dict[row["filename"]].append((row["point_x"], row["point_y"]))
    
    return centers_dict


def compute_covariance_matrix(points):
    mean = points.mean(dim=0, keepdim=True)
    centered_points = points - mean
    covariance_matrix = (centered_points.T @ centered_points) / (points.shape[0] - 1)
    return covariance_matrix



def gaussian_radius(cov_matrices):
    if cov_matrices.dim() == 2:
        eigenvalues = torch.linalg.eigvalsh(cov_matrices)
        max_radius = torch.sqrt(eigenvalues[-1])
    elif cov_matrices.dim() == 3:
        eigenvalues = torch.linalg.eigvalsh(cov_matrices)
        max_radius = torch.sqrt(eigenvalues[:, -1])
    else:
        raise ValueError(f"input cov_matrices must be 2D or 3D tensor, but got {cov_matrices.dim()}D tensor.")
    
    return max_radius


def compute_separation_threshold(covs1, covs2):
    r1 = gaussian_radius(covs1).mean()
    r2 = gaussian_radius(covs2).mean()
    return r1 + r2

import numpy as np
import torch


def compute_separation_loss(data1, mean2, cov2):
    eigenvalues = torch.linalg.eigvalsh(cov2)
    R = torch.sqrt(torch.max(eigenvalues))

    distances = torch.norm(data1 - mean2, dim=1)
    loss = torch.sum(torch.clamp(R - distances, min=0) ** 2)  

    return loss

import torch


def bhattacharyya_distance(mean1, cov1, mean2, cov2):
    cov_mean = (cov1 + cov2) / 2
    
    diff = mean1 - mean2
    
    cov_mean_inv = torch.linalg.inv(cov_mean)
    det_cov_mean = torch.linalg.det(cov_mean)
    det_cov1 = torch.linalg.det(cov1)
    det_cov2 = torch.linalg.det(cov2)
    
    term1 = 0.125 * (diff @ cov_mean_inv @ diff.T)
    
    term2 = 0.5 * torch.log(det_cov_mean / torch.sqrt(det_cov1 * det_cov2))
    
    return term1 + term2

def kl_divergence(mean1, cov1, mean2, cov2):
    cov2_inv = torch.linalg.inv(cov2)
    det_cov1 = torch.linalg.det(cov1)
    det_cov2 = torch.linalg.det(cov2)
    
    diff = (mean2 - mean1).view(-1)

    d = mean1.shape[0]
    term1 = torch.trace(cov2_inv @ cov1)
    term2 = diff.T @ cov2_inv @ diff
    term3 = torch.log(det_cov2 / det_cov1)
    
    kl = 0.5 * (term1 + term2 - d + term3)
    return kl


def separation_loss(means1, covs1, means2, covs2):
    C1 = means1.mean(dim=0)
    C2 = means2.mean(dim=0)

    D = torch.norm(C1 - C2)
    min_dist = compute_separation_threshold(covs1, covs2)

    loss = torch.relu(min_dist - D)
    return loss


def load_checkpoint(filename, model, optimizer=None):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    if  optimizer is not None:
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("Checkpoint does not contain 'optimizer_state_dict'.")
    else:
        print("Optimizer state not loaded.")

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inside_check(points, bounding_box):
    points = np.array(points).reshape(-1, 3)
    query_point = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
    occupancy = bounding_box.compute_occupancy(query_point)
    mask = occupancy.numpy()
    return mask 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text_local', default=None, help="local text prompt")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--save_video', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--eval_interval', type=int, default=1, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--ply_path', type=str, default=None)
    parser.add_argument('--bbox_path', type=str, default=None)
    parser.add_argument('--sd_path', type=str, default='./res_gaussion/colmap_doll/content_personalization')
    parser.add_argument('--object_ply_path', type=str,default=None)

    parser.add_argument('--sd_img_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--sd_max_step_start', type=float, default=0.75, help="sd_max_step")
    parser.add_argument('--sd_max_step_end', type=float, default=0.25, help="sd_max_step")
    parser.add_argument('--sd_min_step', type=float, default=0.02, help="sd_min_step")
    parser.add_argument('--sd_min_step_end', type=float, default=0.02, help="sd_min_step")
    parser.add_argument('--sd_min_step_start', type=float, default=0.02, help="sd_min_step")
    ### training options
    parser.add_argument('--sh_degree', type=int, default=3)
    parser.add_argument('--bbox_size_factor', type=float,nargs='*',  default=[1.0, 1.0 ,1.0], help="size factor of 3d bounding box")
    parser.add_argument('--start_gamma', type=float, default=0.99, help="initial gamma value")
    parser.add_argument('--end_gamma', type=float, default=0.5, help="end gamma value")
    parser.add_argument('--points_times', type=int, default=1, help="repeat editing points x times")
    parser.add_argument('--position_lr_init', type=float, default=0.001, help="initial learning rate")
    parser.add_argument('--position_lr_final', type=float, default=0.00002, help="initial learning rate")
    parser.add_argument('--position_lr_delay_mult', type=float, default=0.01, help="initial learning rate")
    parser.add_argument('--position_lr_max_steps', type=float, default=30000, help="initial learning rate")
    parser.add_argument('--feature_lr', type=float, default=0.01, help="initial learning rate")
    parser.add_argument('--opacity_lr', type=float, default=0.1, help="initial learning rate")
    parser.add_argument('--scaling_lr', type=float, default=0.005, help="initial learning rate")
    parser.add_argument('--rotation_lr', type=float, default=0.005, help="initial learning rate")
    parser.add_argument('--percent_dense', type=float, default=0.1, help="initial learning rate")
    parser.add_argument('--densification_interval', type=float, default=250)
    parser.add_argument('--opacity_reset_interval', type=float, default=30000)
    parser.add_argument('--densify_from_iter', type=float, default=500)
    parser.add_argument('--densify_until_iter', type=float, default=15_000)
    parser.add_argument('--densify_grad_threshold', type=float, default=5)
    parser.add_argument('--min_opacity', type=float, default=0.001)
    parser.add_argument('--max_screen_size', type=float, default=1.0)
    parser.add_argument('--max_scale_size', type=float, default=0.05)
    parser.add_argument('--extent', type=float, default=0.5)
    parser.add_argument('--iters', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--guidance_scale', type=float, default=10.0)
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--max_steps', type=int, default=1024,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--bg_color', type=float, nargs='+', default=None)
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")

    parser.add_argument("--editing_type", type=int, default=0, help="0:add new object, 1:edit existing object")
    parser.add_argument("--reset_points", type=bool, default=False, help="If reset color and size of the editing points")

    ### dataset options
    parser.add_argument("--pose_sample_strategy", type=str, default='uniform',
                        help='input data directory')
    parser.add_argument("--R_path", type=str, default=None,
                        help='input data directory')
    parser.add_argument('--batch_size', type=int, default=1, help="GUI width")
    parser.add_argument('--num_work', type=int, default=4, help="GUI width")
    parser.add_argument('--radius_range', type=float, nargs='*', default=[1.4, 1.6],
                        help="training camera radius range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[65, 65], help="training camera fovy range")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[-90, 90], help="training camera fovy range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[60, 90], help="training camera fovy range")
    parser.add_argument('--mask_type',type=str, default='box', help="mask type")
    parser.add_argument('--project_name', type=str,  help="mask size")
    parser.add_argument('--reset_opacity_offset', type=float, default=4.5)
    parser.add_argument('--desification_start_percent', type=float, default=0.25)
    parser.add_argument('--desification_end_percent', type=float, default=0.75)

    parser.add_argument('--update_color_only',action='store_true', help="test mode")
    parser.add_argument('--num_new_points', type=int, default=1, help="number of new points")
    parser.add_argument('--transform_ckpt', type=str, default=None, help="transform ckpt path")
    parser.add_argument('--origin_bbox_path', type=str, default=None, help="box from object ply")
    parser.add_argument('--transform_scale_factor',type=float, default=1.0, help="transform scale factor")
    parser.add_argument('--tvec_bias', type=float ,nargs='*', default=[0.0, 0.0, 0.0], help="transform offset")
    parser.add_argument('--num_epochs', type=int, default=100)

    parser.add_argument('--size_factor',type=float, default=1.0)
    parser.add_argument('--object_name', type=str, default='hat2')
    parser.add_argument('--wandb_name', type=str, default='hat2')
    parser.add_argument('--scene_ckpt', type=str, default='')


    parser.add_argument('--sample', action='store_true', help="sample views mode")
    parser.add_argument('--radius_list', type=float, nargs='*', default=[0.2, 0.4])
    parser.add_argument('--fovy', type=float, default=50)
    parser.add_argument('--phi_list', type=float, nargs='*', default=[-180, 180])
    parser.add_argument('--theta_list', type=float, nargs='*', default=[0, 90])
    parser.add_argument('--text_global', type=str, default='A doll wearing a pointy hat')
    parser.add_argument('--init_ckpt', type=str, default='')
    parser.add_argument('--scheduler_type', type=str, default='cosine')
    parser.add_argument('--tvec_change_max', type=float, default=0.1)
    parser.add_argument('--key_word', type=str, default='hat')
    parser.add_argument('--scale_lr', type=float, default=0.01)
    parser.add_argument('--rotmat_lr', type=float, default=0.01)
    parser.add_argument('--tvec_lr', type=float, default=0.01)
    parser.add_argument('--use_local', action='store_true')
    parser.add_argument('--use_global', action='store_true')
    parser.add_argument('--local_weight_start', type=float, default=0.99)
    parser.add_argument('--local_weight_end', type=float, default=0.5)
    parser.add_argument('--init_opacity_offset', type=float, default=3.0)
    parser.add_argument('--inside_box',type=str, default=None)
    parser.add_argument('--csv_path',type=str, default='',required=True)
    parser.add_argument('--init_rotation_matrix', type=str, default='[[1, 0, 0], [0, 1, 0], [0, 0, 1]]')

    seed = 42
    set_random_seed(seed)

    from transformers import set_seed
    set_seed(seed)


    opt = parser.parse_args()
    wandb.init(project=f'{opt.object_name}_molmo_refine', name=f'{opt.wandb_name}')

    wandb.config.update(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




    transform = transforms.ToTensor()

    from florence_box_train import (
        GaussianTransformNet,
        create_example_gaussian,
        save_checkpoint,
    )

    net = GaussianTransformNet().to(device)


    epoch, loss = load_checkpoint(opt.init_ckpt, net)
    import torch.optim as optim

    optimizer = torch.optim.AdamW([
        {'params': net.scale, 'lr': 0},
        {'params': net.quaternion, 'lr': 0},
        {'params': net.tvec_x, 'lr': opt.tvec_lr},
        {'params': net.tvec_y, 'lr': opt.tvec_lr},
        {'params': net.tvec_z, 'lr': opt.tvec_lr},
    ])
    from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

    def lr_lambda(epoch):
        if epoch < 10: 
            return 1.0 
        else:
            return 0.1
        
    scheduler_type = opt.scheduler_type
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.num_epochs)
    elif scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=00, gamma=0.5)
    elif scheduler_type == 'lambda':
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    else:
        scheduler = None
    
    all_preds = []
    all_names = []
 
    if opt.csv_path is not None:
        target_centers = load_target_centers(opt.csv_path)


    num_epochs = opt.num_epochs
    train_loader = SampleViewsDataset(opt, device=device, R_path=opt.R_path, type='test', H=512, W=512).dataloader()
    prev_rotmat = None
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            loss = torch.tensor(0.0).cuda()
            gaussian = create_example_gaussian(opt)

            gaussian.add_from_ply_no_grad(opt.object_ply_path)

            checkpoint_dict = torch.load(opt.scene_ckpt, map_location=device)

            gaussian.add_from_checkpoint(checkpoint_dict['model'])

            scale, rotmat, tvec = net()
            transformed_gaussian = apply_gaussian_transform_mask(gaussian, scale * opt.transform_scale_factor, rotmat, tvec - torch.tensor(opt.tvec_bias).to('cuda'), float(opt.size_factor))

            if isinstance(opt.init_rotation_matrix, str):
            # 每行分割，每个数字转换为 float
                opt.init_rotation_matrix = np.array(
                    [[float(num) for num in line.split()] for line in opt.init_rotation_matrix.strip().split('\n')]
                )

            # opt.init_rotation_matrix = json.loads(opt.init_rotation_matrix) if isinstance(opt.init_rotation_matrix, str) else opt.init_rotation_matrix
            init_rotation_bias = torch.tensor(opt.init_rotation_matrix, dtype=torch.float32).to('cuda')
            rotate_gaussians_mask(transformed_gaussian, init_rotation_bias)

            if epoch == 0 and i == 0:
                tvec_init = tvec.detach().clone()
            gaussian1 = gaussian._xyz[gaussian.mask == 1]
            gaussian2 = gaussian._xyz[gaussian.mask == 0]

            rgbs, mask, _, h, w, R, T, fx, fy, pose, index = data

            

            if opt.bbox_path is not None:

                mesh = o3d.io.read_triangle_mesh(opt.bbox_path)
                mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

                center = mesh.get_center()

                mesh = o3d.io.read_triangle_mesh(opt.bbox_path)
                mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

                vertices = np.asarray(mesh.vertex['positions'].numpy(), dtype=np.float64)

                bbox_min = vertices.min(axis=0)
                bbox_max = vertices.max(axis=0)

                center = vertices.mean(axis=0)

                vertices -= center
                vertices *= np.array(opt.bbox_size_factor, dtype=np.float64)
                vertices += center

                mesh.vertex['positions'] = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)

                mesh.compute_vertex_normals()

                bounding_box = o3d.t.geometry.RaycastingScene()
            
                _ = bounding_box.add_triangles(mesh)
        
                scene_mask = inside_check(gaussian._xyz[gaussian.mask == 0].detach().cpu().numpy(), bounding_box)

            surface_gaussians = gaussian._xyz[gaussian.mask == 0][scene_mask == 1]
            gaussian._features_dc = gaussian._features_dc.to('cuda')
            gaussian.mask = gaussian.mask.to('cuda')

            scene_mask = torch.tensor(scene_mask, device='cuda') 
            scene_mask = scene_mask.to('cuda')

            mask_indices = torch.nonzero(gaussian.mask == 0, as_tuple=True)
            scene_indices = torch.nonzero(scene_mask == 1, as_tuple=True)

            all_box_mask = torch.tensor(inside_check(gaussian._xyz.detach().cpu().numpy(), bounding_box)).to('cuda')
            final_indices = torch.nonzero((gaussian.mask == 0) & (all_box_mask== 1), as_tuple=True)

            with torch.no_grad():
                gaussian._features_dc[final_indices[0], :3] = torch.tensor([5.0, 0.0, 0.0], device=gaussian._features_dc.device)


            object_gaussians = gaussian._xyz[gaussian.mask == 1]

            # separation_loss_value = compute_separation_loss(surface_gaussians, object_gaussians.mean(dim=0), compute_covariance_matrix(object_gaussians))


            surface_means = surface_gaussians.mean(dim=0, keepdim=True)
            object_means = object_gaussians.mean(dim=0, keepdim=True) 

            surface_covs = compute_covariance_matrix(surface_gaussians)
            object_covs = compute_covariance_matrix(object_gaussians) 


            separation_loss_value = kl_divergence(surface_means,surface_covs,object_means,object_covs).item()
            h = h[0]
            w = w[0]
            R = R[0]
            T = T[0]

            fx = fx[0]
            fy = fy[0]
            pose = pose[0]
            from models.provider import MiniCam

            cam = MiniCam(
                R,
                T,
                w,
                h,
                fy,
                fx,
                pose,
                0.001,
                10.0
            )
            B = 1
            render_results = gaussian.render(cam, bg_color=None,mask=gaussian.mask)
            preview_results = gaussian.render(cam, bg_color=None)
            loss_render_image = render_results['image'].unsqueeze(0).cuda()
            preview_image = preview_results['image']
            save_image = preview_image.permute(1, 2, 0).reshape(B, h, w, 3)
            poses = data[-2][0]
            phi = int(data[0])
            theta = int(data[1])
            radius = data[2]
            out_name = f'{radius}_{theta}_{phi}.png'
            pred = save_image[0].detach().cpu().numpy()
            pred = (pred * 255).astype(np.uint8)
            pose = torch.tensor(pose).cuda().float()


            loss_render_image = loss_render_image.squeeze(0)


            import torch.nn.functional as F

            pred_gray = 0.2989 * loss_render_image[0] + 0.5870 * loss_render_image[1] + 0.1140 * loss_render_image[2]  # RGB to Grayscale
            pred_gray = pred_gray.unsqueeze(0)
            C_x, C_y = compute_geometric_center(pred_gray)

            if out_name in target_centers:
                target_centers_list = target_centers[out_name]

                target_x = sum(x for x, y in target_centers_list) / len(target_centers_list)
                target_y = sum(y for x, y in target_centers_list) / len(target_centers_list)
            else:
                print(f"warning: {out_name} not in target_centers")
                target_x, target_y = C_x, C_y

            target_tensor = torch.tensor([target_x, target_y], dtype=torch.float32, device="cuda", requires_grad=True)

            loss += F.mse_loss(torch.stack([C_x, C_y]), target_tensor)
               
            loss += separation_loss_value * 100
            image_path = f'{opt.bbox_path.replace("molmo_box.ply", "")}/{opt.object_name}/object_rgb'
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            cv2.imwrite(os.path.join(image_path,out_name), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
            wandb_pred = Image.fromarray(pred)
            all_preds.append(wandb.Image(wandb_pred, caption=out_name))
            all_names.append(out_name)



            wandb.log({
                'separation_loss': separation_loss_value,
                'total_loss': loss
                })


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        wandb.log({'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr']})
        if epoch % 5 == 0:
            wandb.log({"preditions": all_preds, "names": all_names})

        all_preds.clear()
        all_names.clear()
        if scheduler is not None:
            scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    save_checkpoint_path = f'{opt.bbox_path.replace("molmo_box.ply", "")}/{opt.object_name}/molmo_final.pth'
    save_checkpoint( net, optimizer, epoch, loss,save_checkpoint_path)
