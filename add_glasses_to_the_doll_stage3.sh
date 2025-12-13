# Stage3: DoF Refinement, init_ckpt can be set as florence2_checkpoint.pth when the translation initialization is good enough

export CUDA_VISIBLE_DEVICES=1
init_scale_factor=1.8
init_rot_matrix="[[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]]"

object_text="A pair of glasses"
attachment_region_text="eyes"
text_global="A doll wearing a glasses"
text_local="A glasses aligns with the eyes"
global_relationship_words="wearing"
local_relationship_words="aligns with"

# mask to solve the rare case
python dof_learn/utils/inference_mask.py \
    --radius_list 1.3 \
    --fovy 50 \
    --phi_list -45 -30 -15 0 15 30 45\
    --theta_list 60 75 90\
    --object_name "sunglasses1" \
    --box_path "./res_gaussion/colmap_doll_glasses/initial/aabb_mesh.ply" \
    --ckpt_path "./res_gaussion/colmap_doll_glasses/initial/sunglasses1/molmo_final.pth"

python dof_learn/ssds_loss_refine.py \
  --radius_list 1.3  \
  --fovy 50 \
  --phi_list   -15 -30 0 15\
  --theta_list 75\
  --sd_path /mnt/A/jiangzy/models/stable-diffusion-2-1-base \
  --object_name "sunglasses1" \
  --scene_ckpt ./res_gaussion/colmap_doll_glasses/checkpoints/df_ep0625.pth \
  --num_epochs 70 \
  --text_global "$text_global" \
  --text_local "$text_local" \
  --init_ckpt ./res_gaussion/colmap_doll_glasses/initial/sunglasses1/molmo_final.pth \
  --object_ply_path ./reconstruct_data/sunglasses1/gradio_output.ply \
  --global_key_words "$global_relationship_words" \
  --local_key_words "$local_relationship_words" \
  --init_scale_factor $init_scale_factor \
  --init_rotation_matrix "$init_rot_matrix" \
  --scheduler_type 'constant' \
  --wandb_name 'add_doll_glassea' \
  --scale_lr 5e-4 \
  --rotmat_lr 5e-4 \
  --tvec_lr 5e-4 \
  --sd_max_step_end 0.2 \
  --sd_max_step_start 0.2 \
  --sd_min_step_end 0.02 \
  --sd_min_step_start 0.02 \
  --local_weight_start 0.99 \
  --local_weight_end 0.1 \
  --object_position -1 \
  --use_global \
  --use_attn \
  --init_opacity_offset 0.0 \
  --sh_degree 0 


python ./dof_learn/utils/inference_rgb.py \
  --ckpt_path ./res_gaussion/colmap_doll_glasses/dof_refine/sunglasses1/ssds_final.pth \
  --object_name sunglasses1 \
  --scale_factor $init_scale_factor \
  --rotation_matrix "$init_rot_matrix" \
  --object_ply_path ./reconstruct_data/sunglasses1/gradio_output.ply 