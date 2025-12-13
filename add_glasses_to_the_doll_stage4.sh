# 4. Appearance Refinement
# lora-based diffusion finetune

init_scale_factor=1.8
init_rot_matrix="[[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]]"

object_text="A pair of glasses"
attachment_region_text="eyes"
text_global="A doll wearing a glasses"
text_local="A glasses aligns with the eyes"
global_relationship_words="wearing"
local_relationship_words="aligns with"



export MODEL_NAME="/mnt/A/jiangzy/models/stable-diffusion-2-1-base"
export OUTPUT_DIR="./res_gaussion/colmap_doll_glasses/lora_train"
export image_root="./res_gaussion/colmap_doll_glasses/dof_refine/sunglasses1/multi_views/rgb"

# replace selected angle of multi views with original object image
# DINO similarity-based selection
# python dof_learn/utils/dino_similarity_selection.py \
#   --target_image ./data/object/sunglasses1/sunglass1_clean.png \
#   --candidates_dir $image_root


# python ./models/lora_trainer.py \
#   --pretrained_model_name_or_path $MODEL_NAME  \
#   --enable_xformers_memory_efficient_attention \
#   --instance_data_dir "./res_gaussion/colmap_doll_glasses/dof_refine/sunglasses1/multi_views/rgb" \
#   --instance_prompt 'a photo of a <pth> glasses' \
#   --validation_prompt "a photo of a <pth> glasses"  \
#   --output_dir $OUTPUT_DIR \
#   --validation_images $image_root/1.3_75_-30.png \
#     $image_root/1.3_75_0.png  \
#     $image_root/1.3_75_30.png  \
#   --max_train_steps=1000 \
#   --report_to=wandb \
#   --specific_name_image '60_15' # selected by DINO similarity-based selection

python ./appearance_refinement.py \
  --editing_type 3 \
  --batch_size 2 \
  --seed 1 \
  --eval_interval 4 \
  --iters 1000 \
  --new_object_ply_path ./reconstruct_data/sunglasses1/gradio_output.ply \
  --transform_ckpt ./res_gaussion/colmap_doll_glasses/dof_refine/sunglasses1/ssds_final.pth \
  --load_path ./res_gaussion/colmap_doll_glasses/checkpoints/df_ep0625.pth \
  --text_global "a toy wearing a <pth> glasses" \
  --text_local "a photo of a <pth> glasses" \
  --sd_path ./res_gaussion/colmap_doll_glasses/lora_train/checkpoint-1000 \
  --radius_range 1.3 1.3 \
  --fovy_range 50 50 \
  --pose_sample_strategy 360 \
  --phi_range -30 30 \
  --theta_range 60 75 \
  --workspace ./res_gaussion/colmap_doll_glasses/appearance_refinement \
  --position_lr_init 5e-5 \
  --reweight_param 25 \
  --guidance_scale 7.5 \
  --sd_max_step_end 0.25 \
  --sd_max_step_start 0.5 \
  --sd_min_step_start 0.02 \
  --sd_min_step_end 0.02 \
  --end_gamma 0.0 \
  --start_gamma 0.0 \
  --transform_scale_factor $init_scale_factor \
  --rotation_matrix "$init_rot_matrix" \
  --sh_degree 0



