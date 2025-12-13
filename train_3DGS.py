import argparse
import torch
import os
import sys
import numpy as np
from models.utils import seed_everything
from models.provider import SceneDataset, SphericalSamplingDataset, SampleViewsDataset
from models.trainer_3dgs import Trainer_3DGS


if __name__ == '__main__':

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
    parser.add_argument(
        "--task_name",
        required=False,
        default='',
    )
    parser.add_argument('--init_ckpt', type=str, help="init_ckpt")
    parser.add_argument('--object_ply_path', type=str, help="object_ply_path", default=None)
    parser.add_argument('--delete_gs_bbox_path',type=str, help="delete_gs_bbox_path", default=None)
    parser.add_argument('--transform_ckpt', type=str, default=None, help="transform ckpt path")
    parser.add_argument('--init_ply', type=str, default=None, help="transform ckpt path")
    parser.add_argument('--ply_to_ckpt',action='store_true', help="sample views mode")
    parser.add_argument('--bbox_path',type=str, default=None, help="sample inside box")

    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from models.network_3dgaussain import GSNetwork


    seed_everything(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GSNetwork(opt, device)
    if isinstance(model, torch.nn.Module):
        print(model)

    # --- PLY 支持: 如果传入了 --init_ply，则把它作为初始点云（等同于 --initial_points）
    if getattr(opt, "init_ply", None):
        # 如果用户想把 ply 转为 checkpoint，执行转换并退出
        if getattr(opt, "ply_to_ckpt", False):
            try:
                print(f"Loading PLY from: {opt.init_ply}")
                model.load_ply(opt.init_ply)
                ckpt_path = opt.init_ckpt if opt.init_ckpt else os.path.join(opt.workspace, "init_from_ply.pth")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(model.capture(), ckpt_path)
                print(f"Converted PLY -> checkpoint: {ckpt_path}")
            except Exception as e:
                print("Failed to convert PLY to checkpoint:", e)
            sys.exit(0)

        # 否则把 init_ply 作为 initial_points 路径传入后续初始化
        if not opt.initial_points:
            opt.initial_points = opt.init_ply

        # 另外，在 test/sample 模式下我们需要把 ply 直接加载到 model 中，保证 render 时有 means3D
        try:
            from plyfile import PlyData
            from models.network_3dgaussain import BasicPointCloud

            print(f"Loading init PLY into model: {opt.init_ply}")
            plydata = PlyData.read(opt.init_ply)

            # 如果 PLY 是 3DGS 编辑器导出的（包含 f_dc_* 字段），使用专门的加载函数
            names = plydata.elements[0].data.dtype.names
            if names is not None and any(n.startswith('f_dc_') for n in names):
                print("Detected 3DGS-formatted PLY (SH features). Using load_ply_from_gs_editor().")
                model.load_ply_from_gs_editor(opt.init_ply)
            else:
                v = plydata['vertex']
                xyz = np.stack((np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])), axis=1)

                # 颜色字段可能是 red/green/blue 或 r/g/b 或 absent
                if 'red' in names and 'green' in names and 'blue' in names:
                    colors = np.stack((np.asarray(v['red']), np.asarray(v['green']), np.asarray(v['blue'])), axis=1)
                elif 'r' in names and 'g' in names and 'b' in names:
                    colors = np.stack((np.asarray(v['r']), np.asarray(v['g']), np.asarray(v['b'])), axis=1)
                else:
                    colors = np.ones_like(xyz) * 0.8

                # 规范颜色到 0..1
                colors = colors.astype(np.float32)
                if colors.max() > 1.1:
                    colors = colors / 255.0
                colors = np.clip(colors, 0.0, 1.0)

                normals = np.zeros_like(xyz)
                pcd = BasicPointCloud(points=xyz, colors=colors, normals=normals)
                model.create_from_pcd(pcd, spatial_lr_scale=1)
                print(f"Loaded PLY into model: {xyz.shape[0]} points")
        except Exception as e:
            # 如果读取失败，继续但会在渲染时报错
            print("init_ply load skipped or failed:", e)

    if opt.test:
        guidance = None # no need to load guidance model at test
        trainer = Trainer_3DGS('df', opt, model, guidance, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)

        if opt.save_video:
            test_loader = SphericalSamplingDataset(opt, device=device, R_path=opt.sample_R_path, type='test', H=512, W=512, size=250).dataloader()
            trainer.test(test_loader)
            sys.exit()
        elif opt.sample:
            test_loader = SampleViewsDataset(opt, device=device, R_path=opt.R_path, type='test', H=512, W=512).dataloader()
            trainer.sample_views(test_loader, os.path.join(opt.workspace, f'sample_views'))
            sys.exit()
        else:
            test_loader = SceneDataset(opt, device=device, R_path=opt.R_path, type='val').dataloader()
            trainer.test(test_loader, if_gui=False)
    else:

        train_loader = SceneDataset(opt, device=device, R_path=opt.R_path, type='train').dataloader()
        valid_loader = SceneDataset(opt, device=device, R_path=opt.R_path, type='val').dataloader()

        model.initialize_from_mesh(opt.initial_points, opt.R_path, 1)
        model.training_setup(opt)

        trainer = Trainer_3DGS('df', opt, model, device=device, workspace=opt.workspace, ema_decay=None, fp16=opt.fp16,
               use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True)

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)

        print('max_epoch : {}'.format(max_epoch))

        trainer.train(train_loader, valid_loader, max_epoch)

