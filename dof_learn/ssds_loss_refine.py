
import argparse
import json
import os
import random

import cv2
import numpy as np
import open3d as o3d
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR
from utils.attention_ctrl import (
    AttentionReweight,
    AttentionStore,
    get_equalizer,
    register_attention_control,
    tokenize_prompt,
)
from utils.transformation import (
    GaussianTransformNet,
    apply_gaussian_transform_mask,
    rotate_gaussians_mask,
)
from utils.util import create_example_gaussian, load_checkpoint, save_checkpoint

import wandb
from dof_learn.utils.ptp_utils import (
    AttentionStore,
    aggregate_attention,
    register_unt_attention_control,
)
from models.provider import SampleViewsDataset


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False


def inside_check( points, bounding_box):
    points = np.array(points).reshape(-1, 3)
    query_point = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
    occupancy = bounding_box.compute_occupancy(query_point)
    mask = occupancy.numpy()
    print("Mask:", mask)
    return mask 

LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 1000
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77


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

    parser.add_argument('--update_color_only',action='store_true', help="test mode")
    parser.add_argument('--num_new_points', type=int, default=1, help="number of new points")
    parser.add_argument('--transform_ckpt', type=str, default=None, help="transform ckpt path")
    parser.add_argument('--origin_bbox_path', type=str, default=None, help="box from object ply")
    parser.add_argument('--init_scale_factor',type=float, default=1.0, help="transform scale factor")
    parser.add_argument('--init_rotation_matrix', type=str, default='[[1, 0, 0], [0, 1, 0], [0, 0, 1]]', help="initial rotation matrix")
    parser.add_argument('--tvec_bias', type=float ,nargs='*', default=[0.0, 0.0, 0.0], help="transform offset")
    parser.add_argument('--num_epochs', type=int, default=100)

    parser.add_argument('--size_factor',type=float, default=1.0)
    parser.add_argument('--object_name', type=str, default='hat2')
    parser.add_argument('--wandb_nafme', type=str, default='hat2')
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
    parser.add_argument('--global_key_words', type=str, nargs='+',  default=[])
    parser.add_argument('--local_key_words', type=str, nargs='+',  default=[])
    parser.add_argument('--scale_lr', type=float, default=0.01)
    parser.add_argument('--rotmat_lr', type=float, default=0.01)
    parser.add_argument('--tvec_lr', type=float, default=0.01)
    parser.add_argument('--use_local', action='store_true')
    parser.add_argument('--use_global', action='store_true')
    parser.add_argument('--local_weight_start', type=float, default=0.99)
    parser.add_argument('--local_weight_end', type=float, default=0.5)
    parser.add_argument('--init_opacity_offset', type=float, default=0.0)
    parser.add_argument('--inside_box',type=str, default=None)
    parser.add_argument('--object_position',type=int, default=-1)
    parser.add_argument('--object_scale',type=int, default=-1)
    parser.add_argument('--use_attn', action='store_true')
    parser.add_argument('--wandb_name', type=str, default='test')

    seed = 42
    set_random_seed(seed)

    from transformers import set_seed
    set_seed(seed)


    opt = parser.parse_args()
    wandb.init(project=f'{opt.object_name}_ssds_refine', name=f'{opt.wandb_name}')

    wandb.config.update(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import torchvision.transforms as transforms
    from PIL import Image

    transform = transforms.ToTensor()

    net = GaussianTransformNet().to(device)


    epoch, loss = load_checkpoint(opt.init_ckpt, net)
    dof_refine_path = os.path.dirname(opt.init_ckpt).replace("initial", "dof_refine")
    if not os.path.exists(dof_refine_path):
        os.makedirs(dof_refine_path)

        
    optimizer = torch.optim.AdamW([
        {'params': net.scale, 'lr': opt.scale_lr},
        {'params': net.quaternion, 'lr': opt.rotmat_lr},
        {'params': net.tvec_x, 'lr': opt.tvec_lr},
        {'params': net.tvec_y, 'lr': opt.tvec_lr},
        {'params': net.tvec_z, 'lr': opt.tvec_lr},
    ])

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
 
    checkpoint_dict = torch.load(opt.scene_ckpt, map_location=device)
    from models.sd import StableDiffusion

    if opt.use_local:
        local_guidance = StableDiffusion(opt, device, sd_path=opt.sd_path)
    if opt.use_global:
        global_guidance = StableDiffusion(opt, device, sd_path=opt.sd_path)

    if opt.use_local:
        tokenizer = local_guidance.tokenizer
        local_controller = AttentionStore()
    if opt.use_global:
        tokenizer = global_guidance.tokenizer
        global_controller = AttentionStore()
    
    if opt.use_local is False and opt.use_global is False:
        guidance = StableDiffusion(opt, device, sd_path=opt.sd_path)
        tokenizer = guidance.tokenizer
    
        controller = AttentionStore()
        register_unt_attention_control(guidance.unet, controller)

    global_relationship_words = tuple(opt.global_key_words)
    local_relationship_words = tuple(opt.local_key_words)


    num_epochs = opt.num_epochs
    train_loader = SampleViewsDataset(opt, device=device, R_path=opt.R_path, type='test', H=512, W=512).dataloader()
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            loss = torch.tensor(0.0).cuda()

            prompts_global = [opt.text_global] * 2
            prompts_local = [opt.text_local] * 2

            if opt.use_local:
                equalizer_local = get_equalizer(prompts_local[1], local_relationship_words, (200,),tokenizer)

                controller_local = AttentionReweight(prompts_local, NUM_DIFFUSION_STEPS, cross_replace_steps=(.8),
                                            self_replace_steps=.4, equalizer=equalizer_local,tokenizer=tokenizer
                                            )
                register_attention_control(local_guidance, controller_local)
                text_local = local_guidance.get_text_embeds([opt.text_local] * 1,
                                                                        [''] * 1)
                                                
            if opt.use_global:

                equalizer_global = get_equalizer(prompts_global[1], global_relationship_words, (200,),tokenizer)
                controller_global = AttentionReweight(prompts_global, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
                                            self_replace_steps=.4, equalizer=equalizer_global,tokenizer=tokenizer
                                            )
                register_attention_control(global_guidance, controller_global)
                register_unt_attention_control(global_guidance.unet, global_controller)

                text_global = global_guidance.get_text_embeds([opt.text_global] * 1,
                                                                [''] * 1)


            if opt.use_local is False and opt.use_global is False:
                text = guidance.get_text_embeds([opt.text_global] * 1,
                                                [''] * 1)


            gaussian = create_example_gaussian(opt)

            gaussian.add_from_ply_no_grad(opt.object_ply_path)
            checkpoint_dict = torch.load(opt.scene_ckpt, map_location=device)

            obj_mask = torch.ones(gaussian.get_xyz.data.shape[0], device=device)


            scale, rotmat, tvec = net()





            if(opt.origin_bbox_path is not None):
                gaussian_box = create_example_gaussian(opt)
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
                apply_gaussian_transform_mask(gaussian_box, scale * 1.5, rotmat, tvec, float(opt.size_factor))
                delete_min_xyz = torch.min(gaussian_box._xyz, dim=0).values
                delete_max_xyz = torch.max(gaussian_box._xyz, dim=0).values

            else:
                delete_min_xyz = None
                delete_max_xyz = None

            ori_num = gaussian.get_xyz.data.shape[0]
            gaussian.add_from_checkpoint(checkpoint_dict['model'],delete_min_xyz,delete_max_xyz)
            cur_num = gaussian.get_xyz.data.shape[0]
            scene_mask = torch.zeros(cur_num - ori_num,device="cuda")
            mask = torch.cat((obj_mask,scene_mask ))
              
           
            transformed_gaussian = apply_gaussian_transform_mask(gaussian, scale * opt.init_scale_factor, rotmat, tvec - torch.tensor(opt.tvec_bias).to('cuda'), float(opt.size_factor))
            rotation_matrix = json.loads(opt.init_rotation_matrix) if isinstance(opt.init_rotation_matrix, str) else opt.rotation_matrix
            init_rotation_bias = torch.tensor(rotation_matrix, dtype=torch.float32).to('cuda')
            rotate_gaussians_mask(transformed_gaussian, init_rotation_bias)


            if epoch == 0 and i == 0:
                tvec_init = tvec.detach().clone()
            gaussian1 = gaussian._xyz[gaussian.mask == 1]
            gaussian2 = gaussian._xyz[gaussian.mask == 0]
            
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

            render_results = gaussian.render(cam, bg_color=None)
            loss_render_image = render_results['image'].unsqueeze(0).cuda()

            image_path = f'./personalization/box_learn/results/{opt.object_name}/rgb'
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            cv2.imwrite(os.path.join(image_path,out_name), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
            wandb_pred = Image.fromarray(pred)
            all_preds.append(wandb.Image(wandb_pred, caption=out_name))
            all_names.append(out_name)

            if opt.use_global:
                global_ssds_loss, global_t = global_guidance.train_dof_step_ssds(text_global, loss_render_image, pose,
                                                                ratio=epoch / num_epochs,
                                                                guidance_scale=7.5)

                if opt.use_attn:
                    attention_map_dict = aggregate_attention(
                                            attention_store=controller_global,
                                            from_where=("up", "down", "mid"),
                                            is_cross=True,
                                            select=0)

                    text_inputs = tokenize_prompt(tokenizer, opt.text_global)
                    attention_mask = text_inputs['attention_mask']

                    last_idx = torch.sum(attention_mask[0]).item() - 1
                    for key, value in attention_map_dict.items():
                        attention_map_dict[key] = value[:, :, 1:last_idx]
                        attention_map_dict[key] *= 100
                        attention_map_dict[key] = torch.nn.functional.softmax(attention_map_dict[key], dim=-1)

                    bg_mask_image = Image.open(f'./res_gaussion/colmap_doll_glasses/initial/sunglasses1/projected_mask/{out_name}')
                    import torch.nn.functional as F

                    bg_mask_16 = F.interpolate(torch.tensor(np.array(bg_mask_image)/255, dtype=torch.float32).unsqueeze(0).unsqueeze(0), size=(16, 16),
                                                mode='bilinear', antialias=True).squeeze()

                    ob_pos = opt.object_position

                    att_mask_loss_ob_neg_16 = attention_map_dict[16][bg_mask_16 == 0][:, ob_pos].mean()

                    att_mask_loss_ob_neg = att_mask_loss_ob_neg_16

                    att_mask_loss_ob_pos_16 = max(0, 1. - attention_map_dict[16][bg_mask_16 > 0][:, ob_pos].max())

                    att_mask_loss_ob_pos = att_mask_loss_ob_pos_16

                    att_loss = 1.0 * att_mask_loss_ob_pos + 1.0 * att_mask_loss_ob_neg
                
            if opt.use_local:
                local_ssds_loss, local_t = local_guidance.train_dof_step_ssds(text_local, loss_render_image, pose,
                                                                ratio=epoch / num_epochs,
                                                                guidance_scale=7.5)          
            if opt.use_local is False and opt.use_global is False:
                ssds_loss, t = guidance.train_dof_step_ssds(text, loss_render_image, pose,
                                                                ratio=epoch / num_epochs,
                                                                guidance_scale=7.5)


            alpha = opt.local_weight_start - (opt.local_weight_start - opt.local_weight_end) * (epoch / num_epochs)
            beta = 1 - alpha
            if opt.use_local:
                loss += local_ssds_loss * alpha
            if opt.use_global:
                if opt.use_local:
                    loss += global_ssds_loss * beta
                else:
                    loss += global_ssds_loss
                # loss += global_ssds_loss
            if opt.use_attn:
                loss += att_loss
            if opt.use_local is False and opt.use_global is False:
                loss += ssds_loss + att_loss
            # loss += tvec_loss * 100

            wandb.log({
                'global_ssds_loss': global_ssds_loss if opt.use_global else 0,            
                'local_ssds_loss': local_ssds_loss if opt.use_local else 0,
                'global_t': global_t if opt.use_global else 0,
                'local_t': local_t if opt.use_local else 0,
                'total_loss': loss,
                "attention_loss": att_loss if opt.use_attn else 0,
                "t": t if opt.use_local is False and opt.use_global is False else 0,
                # "tvec_change_loss": tvec_loss,
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
    
    save_checkpoint_path = f'{dof_refine_path}/ssds_final.pth'
    save_checkpoint( net, optimizer, epoch, loss,save_checkpoint_path)