# 3DGS reconstruction and rendering, R_path can be got refer to https://github.com/zjy526223908/TIP-Editor?tab=readme-ov-file#data-process
# 训练模型 --test/--sample 未设置
# python train_3DGS.py --fp16 --workspace ./res_gaussion/colmap_doll_glasses  \
#   --min_opacity 0.001 \
#   --percent_dense 0.1 \
#   --iters 40000 \
#   --data_path ./data/colmap_doll  \
#   --R_path ./data/colmap_doll/Orient_R.npy  \
#   --initial_points  ./data/colmap_doll/sparse_points.ply \
#   --train_resolution_level 2 --eval_resolution_level 2 \
#   --data_type 'colmap'  --eval_interval 50

# 渲染/采样多视角
# 可选 init ckpt (示例): /mnt/A/jiangzy/FreeInsert/res_gaussion/colmap_doll_glasses/checkpoints/df_ep0625.pth
python train_3DGS.py --fp16 --workspace ./res_gaussion/colmap_desk --test --sample \
  --radius_list 3.0 --fovy 50 --phi_list -45 -30 -15 0 15 30 45 --theta_list 60 75 90 \
  --init_ply /mnt/A/jiangzy/datasets/point_cloud.ply

User_prompt="Add a pair of glasses to the doll"

#1. MLLM-based Parser, we provide the parser_prompt in prompt.json, you can also use GPT4 with <user_input> and one scene sample view which is default sampled on phi=0, theta=90
#and below is GPT4 Parser Output
object_text="A pair of glasses"
attachment_region_text="eyes"
text_global="A doll wearing a glasses"
text_local="A glasses aligns with the eyes"
global_relationship_words="wearing"
local_relationship_words="aligns with"