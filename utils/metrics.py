import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
import os
from os import path
import pickle

from utils import rotations


def comp_pos_error(curr_pos, targ_pos):
    return np.linalg.norm(curr_pos - targ_pos)


def comp_quat_error(curr_quat, targ_quat):
    return 1 - np.dot(curr_quat, targ_quat) ** 2


def comp_pos_error_traj(pos_traj, targ_pos):
    return np.linalg.norm(pos_traj - targ_pos, axis=1)


def comp_quat_error_traj(quat_traj, targ_quat):
    quat_error_traj = []
    for t in range(quat_traj.shape[0]):
        quat_error_traj.append(comp_quat_error(quat_traj[t], targ_quat))
    return np.array(quat_error_traj)


def comp_angle_error_traj(quat_traj, targ_quat):
    quat_error_traj = []
    for t in range(quat_traj.shape[0]):
        quat_error_traj.append(np.abs(rotations.quat_to_rpy(quat_traj[t])[2] - rotations.quat_to_rpy(targ_quat)[2]))
    return np.array(quat_error_traj)


def save_data(data, data_name, save_dir=None):
    if save_dir is None:
        save_dir = os.getcwd()
    else:
        save_dir = path.join(os.getcwd(), save_dir)

    if not path.exists(save_dir):
        os.makedirs(save_dir)

    saved_path = path.join(save_dir, data_name)

    with open(saved_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_data(data_name, save_dir=None):
    if save_dir is None:
        save_dir = os.getcwd()

    saved_path = path.join(save_dir, data_name)

    try:
        with open(saved_path, 'rb') as f:
            data = pickle.load(f)
    except:
        assert False

    return data
