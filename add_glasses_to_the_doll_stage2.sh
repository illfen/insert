#2. 3D Object Generation & DoF initialization

#3D Object Generation
#(1). Using Stable Diffusion to generate a object image with object text or user provide image, refer to https://huggingface.co/spaces/stabilityai/stable-diffusion
#(2). Using LGM to generate a 3D object with the generated image, refer to https://github.com/3DTopia/LGM?tab=readme-ov-file#inference。We provide several reconstructed objects in reconstruct_data folder.

object_text="A pair of glasses"
attachment_region_text="vase"
text_global="A doll wearing a glasses"
text_local="A glasses aligns with the eyes"
global_relationship_words="wearing"
local_relationship_words="aligns with"


# python train_3DGS.py --fp16 --workspace ./res_gaussion/colmap_desk --test --sample \
#   --radius_list 3.0 --fovy 50 --phi_list -45 -30 -15 0 15 30 45 --theta_list 60 75 90 \
#   --init_ply /mnt/A/jiangzy/datasets/point_cloud.ply


# Attachment Region Detection
# 场景的不同视角视图生成2d框
# python florence-sam/open_voca_detection.py \
#   --text $attachment_region_text \
#   --image_path ./res_gaussion/colmap_desk/sample_views 
#物体的包围框
python dof_learn/ply_to_bbox.py \
  --input_ply_path ./reconstruct_data/sunglasses1/gradio_output_centered.ply \
  --output_path ./res_gaussion/colmap_desk
#训练得到一个3D变换，使AABB框与真实目标在多视角上的mask对齐
python dof_learn/florence_box_train.py  \
    --radius_list 3  \
    --fovy 50 \
    --phi_list -30 -15 0 15 30\
    --theta_list 60 75 90\
    --object_name "sunglasses1" \
    --target_image_path ./res_gaussion/colmap_desk/sample_views/detection_florence2 \
    --init_box_path ./res_gaussion/colmap_desk/initial/aabb_mesh.ply


# #2. init DOF, scale & rotation is initialized by MLLM reasoning, prompts are provided in prompt.json 
# init_scale_factor=1.8

# #rotation matrix is got by MLLM selection the most close one from the sampled rotations and then convert to rotation matrix

# # object_camera is the camera angle of the object-only view selected by MLLM, scene_camera is the camera angle of the scene view which is input to MLLM
# # 把物体的朝向转到场景的方向
# python dof_learn/utils/rotation_transfer.py \
#   --object_camera 0 90 \
#   --scene_camera 0 90 \
#   --output_path ./res_gaussion/colmap_doll_glasses/initial/sunglasses1/rotation_matrix.txt

# init_rot_matrix="$(cat ./res_gaussion/colmap_doll_glasses/initial/sunglasses1/rotation_matrix.txt)"
# #init_rot_matrix="[[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]]"

# # init Translation & molmo Annotation & dependencies transformers==4.45.1
# #基于 Molmo 模型的图像关键点/掩码生成脚本
# python dof_learn/molmo/index.py \
#   --folder ./res_gaussion/colmap_doll_glasses/sample_views/rgb \
#   --prompt 'Point the position to add a glasses to the doll'

# # Correct the attachment box position using molmo points and get attachment region gaussian
# #使用 Molmo 提供的 2D 点修正物体初始 3D 边界框，然后把修正后的区域转为高斯表示，用于后续插入/优化
# python dof_learn/molmo_box.py \
#     --radius_list 1.3  \
#     --fovy 50 \
#     --phi_list  -45 -30 -15 0 15 30 45\
#     --theta_list 60 75 90\
#     --object_name "sunglasses1" \
#     --object_ply_path "./res_gaussion/colmap_doll_glasses/initial/aabb_mesh.ply" \
#     --init_ckpt "./res_gaussion/colmap_doll_glasses/initial/sunglasses1/florence2_checkpoint.pth" \
#     --transform_scale_factor $init_scale_factor \
#     --num_epochs 30 \
#     --csv_path ./res_gaussion/colmap_doll_glasses/sample_views/rgb/points.csv \
#     --scene_ckpt ./res_gaussion/colmap_doll_glasses/checkpoints/df_ep0625.pth \
#     --sh_degree 0 \
#     --bbox_path ./res_gaussion/colmap_doll_glasses/initial/molmo_box.ply
# #3D 高斯（Gaussian）物体的姿态初始化和保存
# python dof_learn/utils/ckpt_to_box.py \
#   --object_ply_path ./res_gaussion/colmap_doll_glasses/initial/aabb_mesh.ply \
#   --save_ply_path ./res_gaussion/colmap_doll_glasses/initial/molmo_box.ply \
#   --transform_ckpt ./res_gaussion/colmap_doll_glasses/initial/sunglasses1/molmo_box.pth\
#   --transform_scale_factor $init_scale_factor 

# # translation initialization
# python dof_learn/florence_to_molmo.py \
#     --radius_list 1.3  \
#     --fovy 50 \
#     --phi_list  -30 -15 0 15 30\
#     --theta_list 75 90 60\
#     --object_name "sunglasses1" \
#     --object_ply_path ./reconstruct_data/sunglasses1/gradio_output.ply \
#     --init_ckpt ./res_gaussion/colmap_doll_glasses/initial/sunglasses1/florence2_checkpoint.pth \
#     --transform_scale_factor $init_scale_factor \
#     --init_rotation_matrix "$init_rot_matrix" \
#     --num_epochs 100 \
#     --csv_path ./res_gaussion/colmap_doll_glasses/sample_views/rgb/points.csv \
#     --scene_ckpt "./res_gaussion/colmap_doll_glasses/checkpoints/df_ep0625.pth" \
#     --bbox_path "./res_gaussion/colmap_doll_glasses/initial/molmo_box.ply" \
#     --sh_degree 0
