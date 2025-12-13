import argparse
import os
import random
import sys

from utils.transformation import GaussianTransformNet, apply_gaussian_transform
from utils.util import create_example_gaussian, save_checkpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pymeshlab
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

import wandb
from models.network_3dgaussain import BasicPointCloud
from models.provider import SampleViewsDataset
from models.sh_utils import SH2RGB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def combined_loss(pred, target):
    # Ensure pred and target have same channel shape: convert RGB pred to single-channel mask
    if pred.dim() == 3 and pred.size(0) == 3:
        pred = pred.mean(dim=0, keepdim=True)
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    bce = nn.BCEWithLogitsLoss()(pred, target)
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = 1 - (2. * intersection + 1) / (pred.sum() + target.sum() + 1)
    wandb.log({"bce": bce, "dice": dice})
    return bce + dice






def find_image(base_path, image_name):
    for root, dirs, files in os.walk(base_path):
        if image_name in files:
            return os.path.join(root, image_name)
    return None

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 





def main():
    guidance = None # no need to load guidance model at test
    wandb.init(project='dof_init_box')

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--eval_interval', type=int, default=50, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### sampling options
    parser.add_argument('--sample', action='store_true', help="sample views mode")
    parser.add_argument('--radius_list', type=float, nargs='*', default=[0.2, 0.4])
    parser.add_argument('--fovy', type=float, default=50)
    parser.add_argument('--phi_list', type=float, nargs='*', default=[-180, 180])
    parser.add_argument('--theta_list', type=float, nargs='*', default=[0, 90])


    ### training options
    parser.add_argument('--iters', type=int, default=30_000, help="training iters")
    parser.add_argument('--lr', type=float, default=5e-4, help="initial learning rate")
    parser.add_argument('--sh_degree', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--ckpt', type=str, default='latest')


    ### dataset options
    parser.add_argument("--data_path", type=str, default='/mnt/d/dataset/data_DTU/dtu_scan105/',
                        help='input data directory')
    parser.add_argument("--if_data_cuda", action='store_false')
    parser.add_argument("--data_type", type=str, default='dtu', help='input data')
    parser.add_argument('--initial_points', type=str, default=None)
    parser.add_argument('--bg_color', type=float, nargs='+', default=None)
    parser.add_argument("--R_path", type=str, default=None, help='input data directory')
    parser.add_argument("--sample_R_path", type=str, default=None, help='input data directory')
    parser.add_argument('--batch_size', type=int, default=1, help="GUI width")
    parser.add_argument('--num_work', type=int, default=0, help="GUI width")
    parser.add_argument(
        "--task_name",
        required=False,
        default='',
    )
    parser.add_argument('--init_ckpt', type=str, help="init_ckpt")
    parser.add_argument('--target_image_path', type=str, 
    default='/home/mobs/code/res_gaussion/colmap_doll_sunglasses/sample_views/detection_florence2', help="florence2 mask path")

    parser.add_argument('--init_box_path', type=str)
    parser.add_argument('--object_name', type=str, default='hat2')
    parser.add_argument('--transform_scale_factor',type=float, default=1.0, help="transform scale factor")
    parser.add_argument('--object_ply_path', type=str, default=None)
    parser.add_argument('--delete_gs_bbox_path', type=str, default=None)
    parser.add_argument('--init_ply', type=str, default=None, help="transform ckpt path")
    parser.add_argument('--ply_to_ckpt',action='store_true', help="sample views mode")
    parser.add_argument('--transform_ckpt', type=str, default=None, help="transform ckpt path")

    opt = parser.parse_args()
    seed = 42
    set_random_seed(seed)

    from transformers import set_seed
    set_seed(seed)

    net = GaussianTransformNet().to(device)

    optimizer = torch.optim.AdamW([
        {'params': net.scale, 'lr': 2e-2},
        {'params': net.quaternion, 'lr': 5e-3},
        {'params': net.tvec_x, 'lr': 5e-3},
        {'params': net.tvec_y, 'lr': 5e-3},
        {'params': net.tvec_z, 'lr': 5e-3},

    ])

    wandb.watch(net)

    target_image_path = opt.target_image_path
    num_epochs = 500
    

    all_preds = []  
    all_names = []  
    for epoch in range(num_epochs):
        loss = torch.tensor(0.0, device=device)
        gaussian = create_example_gaussian(opt)


        test_loader = SampleViewsDataset(opt, device=device, R_path=opt.R_path, type='test', H=512, W=512).dataloader()
        
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(opt.init_box_path)
        ms.meshing_surface_subdivision_midpoint(iterations=4)
        pymesh = ms.current_mesh()
        xyz = pymesh.vertex_matrix()

        mean = xyz.mean(axis=0)
        xyz = xyz - mean
        xyz *= 1
        # xyz = xyz + mean
        xyz[:, 2] += mean[2]



        num_pts = xyz.shape[0]
        shs = np.ones((num_pts, 3))
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )

        gaussian.add_from_pcd_no_grad(pcd,set_mask=True)

        scale, rotmat, tvec = net()


        apply_gaussian_transform(gaussian, scale, rotmat, tvec, opt.transform_scale_factor)

        for data in test_loader:
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

            target_image = Image.open(find_image(target_image_path, out_name))
            pred = render_image[0].detach().cpu().numpy()
            pred = (pred * 255).astype(np.uint8)
            pred = Image.fromarray(pred).convert('L')
            transform = transforms.ToTensor()
            target_tensor = transform(target_image).to(render_image.device)
            #构造一个目录路径，作为后续保存预测掩码的目标文件夹 
            box_projected_mask_path = os.path.join(f'./res_gaussion/colmap_doll_sunglasses/initial/{opt.object_name}/projected_mask')
            if not os.path.exists(box_projected_mask_path):
                os.makedirs(box_projected_mask_path)
            
            pred_path = os.path.join(box_projected_mask_path, out_name)
            pred.save(pred_path)
            all_preds.append(wandb.Image(pred, caption=out_name))
            all_names.append(out_name)


            loss += combined_loss(render_image, target_tensor)
        if epoch % 10 == 0:
            wandb.log({"predictions": all_preds, "names": all_names})
            wandb.log({'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr']})

        all_preds.clear()
        all_names.clear()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")
    print("Object 3D position (scene coordinates):", tvec.cpu().detach().numpy())
    print("Object scale:", scale.cpu().detach().numpy())
    print("Object rotation matrix (rotmat):\n", rotmat.cpu().detach().numpy())  
    gaussian.save_ply(f'./res_gaussion/colmap_doll_sunglasses/initial/{opt.object_name}/florence2_train_{epoch}.ply')

    checkpoint_path = f'./res_gaussion/colmap_doll_sunglasses/initial/{opt.object_name}/florence2_checkpoint.pth'
    save_checkpoint(net, optimizer, epoch=500, loss=loss.item(), filename=checkpoint_path)

if __name__ == "__main__":
    main()