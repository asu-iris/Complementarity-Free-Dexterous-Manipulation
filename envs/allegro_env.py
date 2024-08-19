import mujoco
import mujoco.viewer
import pathlib
import numpy as np
import time
import casadi as cs

import utils.rotations as rot


class MjSimulator():
    def __init__(self, param):

        self.param_ = param

        # init model data
        self.model_ = mujoco.MjModel.from_xml_path(self.param_.model_path_)
        self.data_ = mujoco.MjData(self.model_)

        self.fingertip_names_ = ['ftp_0', 'ftp_1', 'ftp_2', 'ftp_3']

        self.test_ft1_cmd = np.zeros(3)
        self.keyboard_sensitivity = 0.1
        self.break_out_signal_ = False
        self.dyn_paused_ = False

        self.set_goal(self.param_.target_p_, self.param_.target_q_)
        self.reset_env()

        self.viewer_ = mujoco.viewer.launch_passive(self.model_, self.data_, key_callback=self.keyboardCallback)

        self.allegro_fd_fn()

    def keyboardCallback(self, keycode):
        if chr(keycode) == ' ':
            self.dyn_paused_ = not self.dyn_paused_
            if self.dyn_paused_:
                print('simulation paused!')
            elif chr(keycode) == 'ĉ':
                self.test_ft1_cmd[1] += 0.001 * self.keyboard_sensitivity
            elif chr(keycode) == 'Ĉ':
                self.test_ft1_cmd[1] -= 0.001 * self.keyboard_sensitivity
            elif chr(keycode) == 'ć':
                self.test_ft1_cmd[0] -= 0.001 * self.keyboard_sensitivity
            elif chr(keycode) == 'Ć':
                self.test_ft1_cmd[0] += 0.001 * self.keyboard_sensitivity
            elif chr(keycode) == 'O':
                self.test_ft1_cmd[2] += 0.0005 * self.keyboard_sensitivity
            elif chr(keycode) == 'P':
                self.test_ft1_cmd[2] -= 0.0005 * self.keyboard_sensitivity
            elif chr(keycode) == 'R':
                self.test_ft1_cmd = np.array([0.0, 0.0, 0.0])
            else:
                print('simulation resumed!')
        elif chr(keycode) == 'Ā':
            self.break_out_signal_ = True

    def reset_env(self):
        self.data_.qpos[:] = np.hstack((self.param_.init_robot_qpos_, self.param_.init_obj_qpos_))
        self.data_.qvel[:] = np.zeros(22)

        mujoco.mj_forward(self.model_, self.data_)

    def step(self, jpos_cmd):
        curr_jpos = self.get_jpos()
        target_jpos = (curr_jpos + jpos_cmd)
        for i in range(self.param_.frame_skip_):
            self.data_.ctrl = target_jpos
            mujoco.mj_step(self.model_, self.data_)
            self.viewer_.sync()
            # print('error = ', np.linalg.norm(target_jpos - self.get_jpos()))

    def reset_fingers_qpos(self):
        for iter in range(self.param_.frame_skip_):
            self.data_.ctrl = self.param_.init_robot_qpos_
            mujoco.mj_step(self.model_, self.data_)
            time.sleep(0.001)
            self.viewer_.sync()

    def get_state(self):
        obj_pos = self.data_.qpos.flatten().copy()[-7:]
        robot_pos = self.data_.qpos.flatten().copy()[0:16]
        return np.concatenate((obj_pos, robot_pos))

    def get_jpos(self):
        return self.data_.qpos.flatten().copy()[0:16]

    def get_fingertips_position(self):
        fts_pos = []
        for ft_name in self.fingertip_names_:
            fts_pos.append(self.data_.site(ft_name).xpos)
        return np.concatenate(fts_pos).flatten().copy()

    def set_goal(self, goal_pos=None, goal_quat=None):
        # goal_id = mujoco.mj_name2id(self.model_, mujoco.mjtObj.mjOBJ_GEOM, 'goal')
        if goal_pos is not None:
            self.model_.body('goal').pos = goal_pos
            # self.model_.geom_pos[goal_id]=goal_pos
        if goal_quat is not None:
            self.model_.body('goal').quat = goal_quat
            # self.model_.geom_quat[goal_id] = goal_quat
        mujoco.mj_forward(self.model_, self.data_)
        pass

    # forward kinematics of allegro hand
    def allegro_fd_fn(self):

        t_palm = rot.quattmat_fn(np.array([0, 1, 0, 1]) / np.linalg.norm([0, 1, 0, 1]))

        # first finger
        ff_qpos = cs.SX.sym('ff_qpos', 4)
        ff_t_base = t_palm @ rot.ttmat_fn([0, 0.0435, -0.001542]) @ rot.quattmat_fn([0.999048, -0.0436194, 0, 0])
        ff_t_proximal = ff_t_base @ rot.rztmat_fn(ff_qpos[0]) @ rot.ttmat_fn([0, 0, 0.0164])
        ff_t_medial = ff_t_proximal @ rot.rytmat_fn(ff_qpos[1]) @ rot.ttmat_fn([0, 0, 0.054])
        ff_t_distal = ff_t_medial @ rot.rytmat_fn(ff_qpos[2]) @ rot.ttmat_fn([0, 0, 0.0384])
        ff_t_ftp = ff_t_distal @ rot.rytmat_fn(ff_qpos[3]) @ rot.ttmat_fn([0, 0, 0.0384])
        self.fftp_pos_fd_fn = cs.Function('ff_t_ftp_fn', [ff_qpos], [ff_t_ftp[0:3, -1]])

        # middle finger
        mf_qpos = cs.SX.sym('mf_qpos', 4)
        mf_t_base = t_palm @ rot.ttmat_fn([0, 0, 0.0007])
        mf_t_proximal = mf_t_base @ rot.rztmat_fn(mf_qpos[0]) @ rot.ttmat_fn([0, 0, 0.0164])
        mf_t_medial = mf_t_proximal @ rot.rytmat_fn(mf_qpos[1]) @ rot.ttmat_fn([0, 0, 0.054])
        mf_t_distal = mf_t_medial @ rot.rytmat_fn(mf_qpos[2]) @ rot.ttmat_fn([0, 0, 0.0384])
        mf_t_ftp = mf_t_distal @ rot.rytmat_fn(mf_qpos[3]) @ rot.ttmat_fn([0, 0, 0.0384])
        self.mftp_pos_fd_fn = cs.Function('mftp_pos_fd_fn', [mf_qpos], [mf_t_ftp[0:3, -1]])

        # ring finger
        rf_qpos = cs.SX.sym('rf_qpos', 4)
        rf_t_base = t_palm @ rot.ttmat_fn([0, -0.0435, -0.001542]) @ rot.quattmat_fn([0.999048, 0.0436194, 0, 0])
        rf_t_proximal = rf_t_base @ rot.rztmat_fn(rf_qpos[0]) @ rot.ttmat_fn([0, 0, 0.0164])
        rf_t_medial = rf_t_proximal @ rot.rytmat_fn(rf_qpos[1]) @ rot.ttmat_fn([0, 0, 0.054])
        rf_t_distal = rf_t_medial @ rot.rytmat_fn(rf_qpos[2]) @ rot.ttmat_fn([0, 0, 0.0384])
        rf_t_ftp = rf_t_distal @ rot.rytmat_fn(rf_qpos[3]) @ rot.ttmat_fn([0, 0, 0.0384])
        self.rftp_pos_fd_fn = cs.Function('rftp_pos_fd_fn', [rf_qpos], [rf_t_ftp[0:3, -1]])

        # Thumb
        th_qpos = cs.SX.sym('th_qpos', 4)
        th_t_base = t_palm @ rot.ttmat_fn([-0.0182, 0.019333, -0.045987]) @ rot.quattmat_fn(
            [0.477714, -0.521334, -0.521334, -0.477714])
        th_t_proximal = th_t_base @ rot.rxtmat_fn(-th_qpos[0]) @ rot.ttmat_fn([-0.027, 0.005, 0.0399])
        th_t_medial = th_t_proximal @ rot.rztmat_fn(th_qpos[1]) @ rot.ttmat_fn([0, 0, 0.0177])
        th_t_distal = th_t_medial @ rot.rytmat_fn(th_qpos[2]) @ rot.ttmat_fn([0, 0, 0.0514])
        th_t_ftp = th_t_distal @ rot.rytmat_fn(th_qpos[3]) @ rot.ttmat_fn([0, 0, 0.054])
        self.thtp_pos_fd_fn = cs.Function('thtp_pos_fd_fn', [th_qpos], [th_t_ftp[0:3, -1]])

        return 0
