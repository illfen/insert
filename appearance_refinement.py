import argparse
import sys

import numpy as np
import torch

from models.provider import SphericalSamplingDataset
from models.trainer_appearance_refinement import Trainer_SDS
from models.utils import seed_everything

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text_global', default=None, help="global text prompt")
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
    parser.add_argument('--new_object_ply_path', type=str,default=None)

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
    parser.add_argument('--reset_opacity_offset', type=float, default=4.5)
    parser.add_argument('--desification_start_percent', type=float, default=0.25)
    parser.add_argument('--desification_end_percent', type=float, default=0.75)

    parser.add_argument('--update_color_only',action='store_true', help="test mode")
    parser.add_argument('--num_new_points', type=int, default=1, help="number of new points")
    parser.add_argument('--transform_ckpt', type=str, default=None, help="transform ckpt path")
    parser.add_argument('--origin_bbox_path', type=str, default=None, help="box from object ply")
    parser.add_argument('--transform_scale_factor',type=float, default=1.0, help="transform scale factor")
    parser.add_argument('--rotation_matrix', type=str, default=None, help="initial rotation matrix")
    parser.add_argument('--tvec_bias', type=float ,nargs='*', default=[0.0, 0.0, 0.0], help="transform offset")
    parser.add_argument('--is_use_ssds',action='store_true')
    parser.add_argument('--reweight_param',type=float, default=1.0)
    parser.add_argument('--text_global_part', default=None, help="global text prompt")
    parser.add_argument('--radius_list', type=float, nargs='*', default=[1.0], help="training camera fovy range")
    parser.add_argument('--phi_list', type=float, nargs='*', default=[-180, 180], help="training camera fovy range")
    parser.add_argument('--theta_list', type=float, nargs='*', default=[0, 90], help="training camera fovy range")
    parser.add_argument('--fovy', type=float, default=50)
   
    opt = parser.parse_args()

    if opt.seed is not None:
        seed_everything(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from models.network_3dgaussain import GSNetwork

    model = GSNetwork(opt, device)

    if opt.test:
        guidance = None  # no need to load guidance model at test

        trainer = Trainer_SDS('df', opt, model, guidance, opt, device=device, workspace=opt.workspace,
                              fp16=opt.fp16,
                              use_checkpoint=opt.ckpt)

        if opt.save_video:
            test_loader = SphericalSamplingDataset(opt, device=device, R_path=opt.sample_R_path, type='test', H=512,
                                                   W=512, size=250).dataloader()
            trainer.test(test_loader)
            sys.exit()

    else:

        from models.sd import StableDiffusion

        sd_guidance = StableDiffusion(opt, device, sd_path=opt.sd_path)
        train_loader = SphericalSamplingDataset(opt, device=device, R_path=opt.R_path, size=100 * opt.batch_size).dataloader()

        trainer = Trainer_SDS('df', opt, model, None, device=device, workspace=opt.workspace,
                              fp16=opt.fp16, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval,sd_guidance=sd_guidance)

        valid_loader = SphericalSamplingDataset(opt, device=device, R_path=opt.R_path, type='test', H=512, W=512,
                                                size=120).dataloader()

        max_epoch = np.ceil(opt.iters / 100).astype(np.int32)
        trainer.train(train_loader, valid_loader, max_epoch)




