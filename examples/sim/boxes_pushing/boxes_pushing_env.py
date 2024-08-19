import mujoco
import mujoco.viewer

import numpy as np


class MjSimulator():
    def __init__(self, param):

        self.param_ = param

        # init model data
        self.model_ = mujoco.MjModel.from_xml_path('boxes_pushing_10.xml')
        self.data_ = mujoco.MjData(self.model_)

        self.break_out_signal_ = False
        self.dyn_paused_ = False

        self.reset_mj_env()

        self.viewer_ = mujoco.viewer.launch_passive(self.model_, self.data_, key_callback=self.keyboardCallback)

    def keyboardCallback(self, keycode):
        if chr(keycode) == ' ':
            self.dyn_paused_ = not self.dyn_paused_
            if self.dyn_paused_:
                print('simulation paused!')
            else:
                print('simulation resumed!')
        elif chr(keycode) == 'Ā':
            self.break_out_signal_ = True

    def reset_mj_env(self):
        np.random.seed(200)

        curr_q = self.get_qpos()
        for i in range(10):
            rand_quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(rand_quat, [0, 0, 1], 2 * np.pi * np.random.randn(1))
            curr_q[7 * i + 3: 7 * i + 7] = rand_quat

        self.data_.qpos[:] = curr_q
        mujoco.mj_forward(self.model_, self.data_)

    def step(self, cmd=0.0):
        mujoco.mj_step(self.model_, self.data_, nstep=1)

        pass

    def get_qpos(self):
        return self.data_.qpos.flatten().copy()
