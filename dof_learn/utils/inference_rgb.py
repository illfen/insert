import argparse
import os

import cv2
import numpy as np
import torch
from flask import json
from transformation import (
    GaussianTransformNet,
    apply_gaussian_transform,
    rotate_gaussians,
)
from util import create_example_gaussian, load_checkpoint

from models.provider import SampleViewsDataset

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
parser.add_argument('--radius_list', type=float, nargs='*', default=[1.3])
parser.add_argument('--fovy', type=float, default=50)
parser.add_argument('--phi_list', type=float, nargs='*', default=[-30, -15, 0, 15, 30])
parser.add_argument('--theta_list', type=float, nargs='*', default=[60, 75, 90])
parser.add_argument('--sh_degree', type=int, default=0)


### dataset options
parser.add_argument('--bg_color', type=float, nargs='+', default=None)
parser.add_argument("--R_path", type=str, default=None, help='input data directory')


parser.add_argument('--ckpt_path', type=str, default='')
parser.add_argument('--object_name', type=str, default='sunglasses1')
parser.add_argument('--scale_factor', type=float, default=1.8)
parser.add_argument('--object_ply_path', type=str, default='')
parser.add_argument('--tvec_bias',  type=float, nargs='*', default=[0, 0, 0])
parser.add_argument('--rotation_matrix',  type=str,  default='[[1, 0, 0], [0, 1, 0], [0, 0, 1]]')
opt = parser.parse_args()


net = GaussianTransformNet().to(device)

epoch, loss = load_checkpoint(opt.ckpt_path, net)

gaussian = create_example_gaussian(opt)

multi_views_save_path = opt.ckpt_path.replace('ssds_final.pth', 'multi_views/rgb')
if not os.path.exists(multi_views_save_path):
    os.makedirs(multi_views_save_path)

test_loader = SampleViewsDataset(opt, device=device, R_path=opt.R_path, type='test', H=512, W=512).dataloader()

gaussian.add_from_ply_no_grad(opt.object_ply_path)
scale, rotmat, tvec = net()
transformed_gaussian = apply_gaussian_transform(gaussian, scale * opt.scale_factor, rotmat, tvec)
rotation_matrix = json.loads(opt.rotation_matrix) if isinstance(opt.rotation_matrix, str) else opt.rotation_matrix
init_rotation_bias = torch.tensor(rotation_matrix, dtype=torch.float32).to('cuda')
rotate_gaussians(transformed_gaussian, init_rotation_bias)

pose_dict = {}
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
    B = 1
    render_results = gaussian.render(cam, bg_color=None)
    loss_render_image = render_results['image'].unsqueeze(0).cuda()
    render_image = render_results['image']
    save_image = render_image.permute(1, 2, 0).reshape(B, h, w, 3)
    poses = data[-2][0]
    phi = int(data[0])
    theta = int(data[1])
    radius = data[2]
    out_name = f'{radius}_{theta}_{phi}.png'
    pred = save_image[0].detach().cpu().numpy()
    pred = (pred * 255).astype(np.uint8)
    pose = torch.tensor(pose).cuda().float()

    image_path = f'{multi_views_save_path}'
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    cv2.imwrite(os.path.join(image_path,out_name), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
    pose_dict[out_name] = poses

np.save(os.path.join(f'{multi_views_save_path.replace("rgb", "")}', 'pose_dict.npy'), pose_dict)
