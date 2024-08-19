import mujoco
import mujoco.viewer

import numpy as np
import casadi as cs

np.set_printoptions(suppress=True)


class ExplicitModel:
    def __init__(self, param):
        self.param_ = param

        self.init_utils()
        self.init_model()

    def init_utils(self):
        # -------------------------------
        #    quaternion integration fn
        # -------------------------------
        quat = cs.SX.sym('quat', 4)
        H_q_body = cs.vertcat(cs.horzcat(-quat[1], quat[0], quat[3], -quat[2]),
                              cs.horzcat(-quat[2], -quat[3], quat[0], quat[1]),
                              cs.horzcat(-quat[3], quat[2], -quat[1], quat[0]))
        self.cs_qmat_body_fn_ = cs.Function('cs_qmat_body_fn', [quat], [H_q_body.T])

        # -------------------------------
        #    state integration fn
        # -------------------------------
        qvel = cs.SX.sym('qvel', self.param_.n_qvel_)
        qpos = cs.SX.sym('qpos', self.param_.n_qpos_)
        next_obj_pos = qpos[0:3] + self.param_.h_ * qvel[0:3]
        next_robot_qpos = qpos[-(self.param_.n_robot_qpos_):] + self.param_.h_ * qvel[-(self.param_.n_robot_qpos_):]
        next_obj_quat = (qpos[3:7] + 0.5 * self.param_.h_ * self.cs_qmat_body_fn_(qpos[3:7]) @ qvel[3:6])
        # next_obj_quat = next_obj_quat / cs.norm_2(next_obj_quat)
        next_qpos = cs.vertcat(next_obj_pos, next_obj_quat, next_robot_qpos)
        self.cs_qposInteg_ = cs.Function('cs_qposInte', [qpos, qvel], [next_qpos])

    def init_model(self):
        curr_q = cs.SX.sym('qvel', self.param_.n_qpos_)
        cmd = cs.SX.sym('cmd', self.param_.n_cmd_)
        phi_vec = cs.SX.sym('phi_vec', self.param_.max_ncon_ * 4)
        jac_mat = cs.SX.sym('jac_mat', self.param_.max_ncon_ * 4, self.param_.n_qvel_)

        # b vector in the QP formulation
        b_o = cs.DM(self.param_.obj_mass_ * self.param_.gravity_)
        b_r = self.param_.robot_stiff_ @ cmd
        b = cs.vertcat(b_o, b_r)

        # Q matrix in the QP formulation
        Q = self.param_.Q
        Q_inv = np.linalg.inv(Q)

        # K matrix in the explicit model
        model_params = cs.SX.sym('sigma', 1)
        # K = sigma * cs.DM.eye(self.param_.max_ncon_ * 4)

        # time step h
        h = self.param_.h_

        # calculate the non-contact term
        v_non_contact = Q_inv @ b / h

        # calculate the contact term
        # contact_force = cs.fmax(-K @ (jac_mat @ Q_inv @ b + phi_vec), 0)
        contact_force = -model_params @ (jac_mat @ Q_inv @ b + phi_vec) - 0.1 * model_params @ jac_mat @ Q_inv @ b / h
        beta = 100.0
        contact_force = cs.log(1 + cs.exp(beta * contact_force)) / beta
        v_contact = Q_inv @ jac_mat.T @ contact_force / h

        # combine the velocity
        v = v_non_contact + v_contact

        # time integration
        next_qpos = self.cs_qposInteg_(curr_q, v)

        # assemble the casadi function
        self.step_once_fn = cs.Function('step_once', [curr_q, cmd, phi_vec, jac_mat, model_params], [next_qpos])

    def step(self, curr_q, cmd, phi_vec, jac_mat, sigma):
        return self.step_once_fn(curr_q, cmd, phi_vec, jac_mat, sigma)