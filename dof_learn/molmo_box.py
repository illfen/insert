import argparse
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from utils.transformation import GaussianTransformNet, apply_gaussian_transform
from utils.util import create_example_gaussian, load_checkpoint, save_checkpoint

import wandb
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

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    parser.add_argument('--init_ckpt', type=str, default=None)
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
    parser.add_argument('--csv_path',type=str, default='')


    seed = 42
    set_random_seed(seed)

    from transformers import set_seed
    set_seed(seed)


    opt = parser.parse_args()
    wandb.init(project=f'{opt.object_name}_molmo_refine', name=f'{opt.wandb_name}')

    wandb.config.update(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import torchvision.transforms as transforms
    from PIL import Image

    transform = transforms.ToTensor()

    from florence_box_train import (
        GaussianTransformNet,
        create_example_gaussian,
        save_checkpoint,
    )

    net = GaussianTransformNet().to(device)

    if opt.init_ckpt is not None:
        epoch, loss = load_checkpoint(opt.init_ckpt, net)
    # print(epoch, loss)
    import torch.optim as optim

    # optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    optimizer = torch.optim.AdamW([
        {'params': net.scale, 'lr': 0},
        {'params': net.quaternion, 'lr': 0},
        {'params': net.tvec_x, 'lr': opt.tvec_lr},
        {'params': net.tvec_y, 'lr': opt.tvec_lr},
        {'params': net.tvec_z, 'lr': opt.tvec_lr},
    ])
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR

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
 

    target_centers = load_target_centers(opt.csv_path)


    num_epochs = opt.num_epochs
    train_loader = SampleViewsDataset(opt, device=device, R_path=opt.R_path, type='test', H=512, W=512).dataloader()
    prev_rotmat = None

    for epoch in range(num_epochs):
        gaussian = create_example_gaussian(opt)
        import os

        from models.provider import SampleViewsDataset

        test_loader = SampleViewsDataset(opt, device=device, R_path=opt.R_path, type='test', H=512, W=512).dataloader()
        
        for data in test_loader:
            loss = torch.tensor(0.0, device=device)
            import pymeshlab
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(opt.object_ply_path)
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
            checkpoint_dict = torch.load(opt.scene_ckpt, map_location=device)

            gaussian.add_from_pcd_no_grad(pcd,set_mask=True)

            input_data = gaussian._xyz

            scale, rotmat, tvec = net()
            transformed_gaussian = apply_gaussian_transform(gaussian, scale, rotmat, tvec, opt.transform_scale_factor)

            rgbs, mask, _, h, w, R, T, fx, fy, pose, index = data
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
            render_image = gaussian.render(cam, bg_color=None)['image']
            
            poses = data[-2][0]
            phi = int(data[0])
            theta = int(data[1])
            radius = data[2]
            out_name = f'{radius}_{theta}_{phi}.png'
            pred = render_image[0].detach().cpu().numpy()
            pred = (pred * 255).astype(np.uint8)
            pred = Image.fromarray(pred).convert('L')

            pred_gray = render_image[0]
            pred_gray = pred_gray.unsqueeze(0)
            C_x, C_y = compute_geometric_center(pred_gray)

            if out_name in target_centers:
                target_centers_list = target_centers[out_name]

                target_x = sum(x for x, y in target_centers_list) / len(target_centers_list)
                target_y = sum(y for x, y in target_centers_list) / len(target_centers_list)


            else:
                print(f"Warning: {out_name} not found in target centers.")
                target_x, target_y = C_x, C_y

            target_tensor = torch.tensor([target_x, target_y], dtype=torch.float32, device="cuda", requires_grad=True)

            import torch.nn.functional as F
            loss = F.mse_loss(torch.stack([C_x, C_y]), target_tensor)
        
            wandb.log({
                'total_loss': loss
                })
            box_projected_mask_path = os.path.join(f'{opt.bbox_path.replace("molmo_box.ply", "")}/{opt.object_name}/molmo_projected_mask')
            if not os.path.exists(box_projected_mask_path):
                os.makedirs(box_projected_mask_path)
            pred_path = os.path.join(box_projected_mask_path, out_name)
            pred.save(pred_path)
            print('pred_path', pred_path)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({
                'total_loss': loss
                })
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")
    save_checkpoint_path = f'{opt.bbox_path.replace("molmo_box.ply", "")}/{opt.object_name}/molmo_box.pth'
    print("---------------------- Saving checkpoint ------------------")
    save_checkpoint( net, optimizer, epoch, loss,save_checkpoint_path)
