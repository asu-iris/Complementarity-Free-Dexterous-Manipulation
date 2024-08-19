import mujoco
import mujoco.viewer

import numpy as np

# ---------------------------------------------------------------------------------------------
#       ! This class implement a visualization module in Mujoco, which only used for
# rendering the object and robot states calculated by our models or other models, it
# doesn't call any models in Mujoco
# ---------------------------------------------------------------------------------------------

# -------------------------------
#   keyboard control parameters
# -------------------------------
keyboard_sensitivity = 0.1
cmd = np.array([0.0])
dyn_paused = False
break_out_signal = False


# -------------------------------
#  keyboard control callback fn
# -------------------------------
def keyboardCallback(keycode):
    global keyboard_sensitivity, cmd, dyn_paused, break_out_signal
    if chr(keycode) == ' ':
        dyn_paused = not dyn_paused
        if dyn_paused:
            print('simulation paused!')
        else:
            print('simulation resumed!')

    elif chr(keycode) == 'ć':
        cmd -= 0.001 * keyboard_sensitivity
    elif chr(keycode) == 'Ć':
        cmd += 0.001 * keyboard_sensitivity
    elif chr(keycode) == 'R':
        cmd = np.array([0.0])

    elif chr(keycode) == 'Ā':
        break_out_signal = True


class MjDynViewer():
    def __init__(self, param):
        self.param_ = param

        self.model_ = mujoco.MjModel.from_xml_path(self.param_.model_path_)
        self.data_ = mujoco.MjData(self.model_)

        self.viewer_ = mujoco.viewer.launch_passive(self.model_, self.data_, key_callback=keyboardCallback)

    def init_qpos(self):
        np.random.seed(200)

        curr_q = self.get_qpos()
        for i in range(self.param_.n_obj_):
            rand_quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(rand_quat, [0, 0, 1], 2 * np.pi * np.random.randn(1))
            curr_q[7 * i + 2] += 0.00
            curr_q[7 * i + 3: 7 * i + 7] = rand_quat

        self.data_.qpos[:] = curr_q
        mujoco.mj_forward(self.model_, self.data_)

    def get_qpos(self):
        return self.data_.qpos.flatten().copy()

    def generate_cmd(self):
        return cmd

    def rendering(self, q_pos):
        self.data_.qpos[:] = q_pos
        mujoco.mj_forward(self.model_, self.data_)
        self.viewer_.sync()

    def step(self, cmd):
        self.data_.ctrl = self.get_qpos()[-1] + cmd
        mujoco.mj_step(self.model_, self.data_)

        return self.get_qpos()
