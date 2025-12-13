
import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def get_camera_direction(h_angle, v_angle):
    theta = np.deg2rad(h_angle)
    phi = np.deg2rad(v_angle)
    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)
    return np.array([x, y, z])

def compute_rotation_between_views(object_h, object_v, scene_h=0, scene_v=90):
    src_dir = get_camera_direction(object_h, object_v)
    dst_dir = get_camera_direction(scene_h, scene_v)

    if np.allclose(src_dir, dst_dir):
        return np.eye(3)

    axis = np.cross(src_dir, dst_dir)
    angle = np.arccos(np.dot(src_dir, dst_dir) / (np.linalg.norm(src_dir) * np.linalg.norm(dst_dir)))
    rot = R.from_rotvec(axis / np.linalg.norm(axis) * angle)
    rot_matrix = rot.as_matrix()

    return rot_matrix

# rotation initialization, src_h, src_v are the camera angles of the selected view by MLLM
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_camera', type=float, nargs=2, required=True, help='source view angles: h_angle v_angle')
    parser.add_argument('--scene_camera', type=float, nargs=2, default=[0, 90], help='target view angles: h_angle v_angle')
    parser.add_argument('--output_path', type=str, default='./rotation_matrix.txt', help='output path for rotation matrix')
    args = parser.parse_args()
    src_h, src_v = args.object_camera
    tgt_h, tgt_v = args.scene_camera
    rot_matrix = compute_rotation_between_views(src_h, src_v, tgt_h, tgt_v)
    res = np.savetxt(args.output_path, rot_matrix)
    print(f"Rotation matrix saved to {args.output_path}")
    print(f"Rotation matrix saved to {args.output_path}")
    print(f"Rotation matrix saved to {args.output_path}")
