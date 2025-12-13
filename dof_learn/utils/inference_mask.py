import argparse
import os

import numpy as np
import pymeshlab
import torch
from PIL import Image
from transformation import GaussianTransformNet, apply_gaussian_transform
from util import create_example_gaussian, load_checkpoint

from models.network_3dgaussain import BasicPointCloud
from models.provider import MiniCam, SampleViewsDataset
from models.sh_utils import SH2RGB
from models.trainer_3dgs import Trainer_3DGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
parser.add_argument('--bounding_box_path', type=str, default=None)
parser.add_argument('--bbox_size_factor', type=float, default=1.0)


### training options
parser.add_argument('--iters', type=int, default=30_000, help="training iters")
parser.add_argument('--lr', type=float, default=5e-4, help="initial learning rate")
parser.add_argument('--sh_degree', type=int, default=3)
parser.add_argument('--position_lr_init', type=float, default= 0.00016)
parser.add_argument('--position_lr_final', type=float, default= 0.0000016)
parser.add_argument('--position_lr_delay_mult', type=float, default=0.01)
parser.add_argument('--position_lr_max_steps', type=float, default=30_000)
parser.add_argument('--feature_lr', type=float, default=0.0025)
parser.add_argument('--opacity_lr', type=float, default=0.05)
parser.add_argument('--scaling_lr', type=float, default=0.005)
parser.add_argument('--rotation_lr', type=float, default=0.001)
parser.add_argument('--percent_dense', type=float, default=0.01)
parser.add_argument('--lambda_dssim', type=float, default=0.2)
parser.add_argument('--densification_interval', type=float, default=100)
parser.add_argument('--opacity_reset_interval', type=float, default=3000)
parser.add_argument('--densify_from_iter', type=float, default=500)
parser.add_argument('--densify_until_iter', type=float, default=15_000)
parser.add_argument('--densify_grad_threshold', type=float, default=0.0002)
parser.add_argument('--min_opacity', type=float, default=0.005)
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
parser.add_argument('--batch_rays', type=int, default=512, help="GUI width")
parser.add_argument('--train_resolution_level', type=float, default=1, help="GUI width")
parser.add_argument('--eval_resolution_level', type=float, default=4, help="GUI width")
parser.add_argument('--num_work', type=int, default=0, help="GUI width")
parser.add_argument('--train_batch_type', type=str, default='image')
parser.add_argument('--val_batch_type', type=str, default='image')
parser.add_argument('--radius_range', type=float, nargs='*', default=[0.15, 0.15], help="test camera radius range")
parser.add_argument('--fovy_range', type=float, nargs='*', default=[50, 70], help="test camera fovy range")
parser.add_argument('--phi_range', type=float, nargs='*', default=[-180, 180], help="test camera fovy range")
parser.add_argument('--theta_range', type=float, nargs='*', default=[60, 90], help="test camera fovy range")

parser.add_argument('--ckpt_path', type=str, default='workspace/ckpt')
parser.add_argument('--box_path', type=str, default='box')
parser.add_argument('--object_name', type=str, default='hat1')
parser.add_argument('--init_ckpt', type=str, help="init_ckpt")
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--object_ply_path', type=str, default=None)
parser.add_argument('--init_ply', type=str, default=None)
parser.add_argument('--transform_ckpt', type=str, default=None)
parser.add_argument('--ply_to_ckpt', type=str, default=None)
parser.add_argument('--delete_gs_bbox_path', type=str, default=None)


opt = parser.parse_args()


net = GaussianTransformNet().to(device)

epoch, loss = load_checkpoint(opt.ckpt_path, net)

gaussian = create_example_gaussian(opt)


test_loader = SampleViewsDataset(opt, device=device, R_path=opt.R_path, type='test', H=512, W=512).dataloader()


ms = pymeshlab.MeshSet()
ms.load_new_mesh(opt.box_path)
ms.meshing_surface_subdivision_midpoint(iterations=4)
pymesh = ms.current_mesh()
xyz = pymesh.vertex_matrix()

mean = xyz.mean(axis=0)
xyz = xyz - mean
xyz *= 1
xyz = xyz + mean



num_pts = xyz.shape[0]
shs = np.ones((num_pts, 3))
pcd = BasicPointCloud(
    points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
)

gaussian.add_from_pcd_no_grad(pcd)
guidance = None
gs_trainer = Trainer_3DGS('df', opt, gaussian, guidance, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)
input_data = gaussian._xyz

scale, rotmat, tvec = net()
transformed_gaussian = apply_gaussian_transform(gaussian, scale * opt.scale_factor, rotmat, tvec)


for data in test_loader:
    rgbs, mask, _, h, w, R, T, fx, fy, pose, index = data
    h = h[0]
    w = w[0]
    R = R[0]
    T = T[0]

    fx = fx[0]
    fy = fy[0]
    pose = pose[0]

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
    # pred = render_image[0].detach().cpu().numpy()
    # pred = (pred * 255).astype(np.uint8)
    # pred = Image.fromarray(pred).convert('L')

    # box_projected_mask_path = os.path.join(f'./res_gaussion/colmap_doll_glasses/initial/{opt.object_name}/projected_mask')
    # if not os.path.exists(box_projected_mask_path):
    #     os.makedirs(box_projected_mask_path)
    # pred_path = os.path.join(box_projected_mask_path, out_name)
    # pred.save(pred_path)
    # pred = (pred * 255).astype(np.uint8)
    # pred = Image.fromarray(pred).convert('L')

    # box_projected_mask_path = os.path.join(f'./res_gaussion/colmap_doll_glasses/initial/{opt.object_name}/projected_mask')
    # if not os.path.exists(box_projected_mask_path):
    #     os.makedirs(box_projected_mask_path)
    # pred_path = os.path.join(box_projected_mask_path, out_name)
    # pred.save(pred_path)
    pred = render_image[0].detach().cpu().numpy()  # tensor → numpy
    pred = (pred * 255).astype(np.uint8)          # 归一化到 0~255
    pred_image = Image.fromarray(pred).convert('L')  # numpy → PIL

    # 保存
    box_projected_mask_path = os.path.join(f'./res_gaussion/colmap_doll_glasses/initial/{opt.object_name}/projected_mask')
    os.makedirs(box_projected_mask_path, exist_ok=True)
    pred_image.save(os.path.join(box_projected_mask_path, out_name))