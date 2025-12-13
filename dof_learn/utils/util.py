import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.network_3dgaussain import GSNetwork


def load_checkpoint(filename, model, optimizer=None):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    if  optimizer is not None:
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("Checkpoint does not contain 'optimizer_state_dict'.")
    else:
        print("Optimizer state not loaded.")

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)


def create_example_gaussian(opt):
    gs_model = GSNetwork(opt, 'cuda')

    return gs_model

def rotation_matrix_z(angle_degrees):
    theta = np.radians(angle_degrees)
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
def rotation_matrix_y(theta):
    theta = np.radians(theta)
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
def rotation_matrix_x(theta):
    theta = np.radians(theta)
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])



def get_camera_direction(h_angle, v_angle):
    theta = np.deg2rad(h_angle)
    phi = np.deg2rad(v_angle)
    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)
    return np.array([x, y, z])

def get_rotated_degree( src_h, src_v):
    src_dir = get_camera_direction(src_h, src_v)
    dst_dir = get_camera_direction(0, 90)

    axis = np.cross(src_dir, dst_dir)
    angle = np.arccos(np.dot(src_dir, dst_dir) / (np.linalg.norm(src_dir) * np.linalg.norm(dst_dir)))
    rot = R.from_rotvec(axis / np.linalg.norm(axis) * angle)
    return rot.as_matrix()
