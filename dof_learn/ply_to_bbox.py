import trimesh
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_ply_path', type=str, default='/home/mobs/code/reconstruct_data/hat2/coarse_gs.ply')
parser.add_argument('--output_path', type=str, default=None,
                    help='Path to save the output AABB mesh in PLY format. If not specified, it will be created in the current directory.')
opt = parser.parse_args()


if opt.output_path is None:
    print("Please specify the output path for the AABB mesh")
    exit()
else:
    if not os.path.exists(opt.output_path+'/initial'):
        os.makedirs(os.path.join(opt.output_path+'/initial'))

mesh = trimesh.load(opt.input_ply_path)

bounds = mesh.bounds  # [[x_min, y_min, z_min], [x_max, y_max, z_max]]
aabb_min, aabb_max = bounds[0], bounds[1]

aabb_center = (aabb_min + aabb_max) / 2
aabb_size = aabb_max - aabb_min
print("----------AABB信息----------")
print("AABB最小坐标:", aabb_min)
print("AABB最大坐标:", aabb_max)
print("AABB中心坐标:", aabb_center)
print("AABB尺寸 (长宽高):", aabb_size)

aabb_vertices = np.array([
    [aabb_min[0], aabb_min[1], aabb_min[2]],  # 0
    [aabb_max[0], aabb_min[1], aabb_min[2]],  # 1
    [aabb_max[0], aabb_max[1], aabb_min[2]],  # 2
    [aabb_min[0], aabb_max[1], aabb_min[2]],  # 3
    [aabb_min[0], aabb_min[1], aabb_max[2]],  # 4
    [aabb_max[0], aabb_min[1], aabb_max[2]],  # 5
    [aabb_max[0], aabb_max[1], aabb_max[2]],  # 6
    [aabb_min[0], aabb_max[1], aabb_max[2]],  # 7
])

aabb_faces = [
    [0, 1, 2], [0, 2, 3],  # Bottom face
    [4, 5, 6], [4, 6, 7],  # Top face
    [0, 1, 5], [0, 5, 4],  # Front face
    [2, 3, 7], [2, 7, 6],  # Back face
    [0, 3, 7], [0, 7, 4],  # Left face
    [1, 2, 6], [1, 6, 5],  # Right face
]

aabb_mesh = trimesh.Trimesh(vertices=aabb_vertices, faces=aabb_faces)

ply_file_name = os.path.join(opt.output_path+'/initial','aabb_mesh.ply')
res = aabb_mesh.export(ply_file_name)
if res:
    print("AABB mesh saved successfully")
else:
    print("Failed to save AABB mesh")
