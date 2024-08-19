import mujoco
import mujoco.viewer

import numpy as np

np.set_printoptions(suppress=True)
import casadi as cs


class QPModel:
    def __init__(self, param):
        self.param_ = param

        self.init_qp_model()

        self.init_utils()

    def init_qp_model(self):
        Q = self.param_.Q
        h = self.param_.h_

        v_o = cs.SX.sym('v_o', self.param_.n_v_obj_)
        v_r = cs.SX.sym('v_r', self.param_.n_cmd_)

        phi_vec = cs.SX.sym('phi_vec', self.param_.max_ncon_ * 4)
        jac_mat = cs.SX.sym('jac_mat', self.param_.max_ncon_ * 4, self.param_.n_qvel_)

        v = cs.vertcat(v_o, v_r)
        u = cs.SX.sym('u', self.param_.n_cmd_)

        # construct b vector
        b_o = np.zeros(6 * self.param_.n_obj_)
        for i in range(self.param_.n_obj_):
            b_o[6 * i:6 * i + 6] = self.param_.obj_mass_ * self.param_.gravity_
        b_r = self.param_.robot_stiff_ @ u
        b = cs.vertcat(b_o, b_r)

        # objective function
        qp_cost = 0.5 * cs.dot(v, Q @ v) - cs.dot(b, v) / h

        # constraint
        qp_g = phi_vec / self.param_.h_ + jac_mat @ v

        self.qp_model = dict(qp_cost_fn=cs.Function('qp_cost_fn', [v, u], [qp_cost]),
                             qp_g_fn=cs.Function('qp_g_fn', [v, phi_vec, jac_mat], [qp_g]))

        # construct qp solver
        qp_param = cs.vertcat(u, cs.vec(phi_vec), cs.vec(jac_mat))
        quadprog = {'x': v, 'f': qp_cost, 'g': qp_g, 'p': qp_param}

        # opts = {'error_on_fail': False, 'printLevel': 'none'}
        # self.qp_solver_fn_ = cs.qpsol('qp_solver', 'qpoases', quadprog, opts)

        opts = {'print_time': 0, 'osqp': {'verbose': False}}
        self.qp_solver_fn_ = cs.qpsol('qp_solver', 'osqp', quadprog, opts)

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
        obj_qvel = cs.SX.sym('qvel', 6)
        obj_qpos = cs.SX.sym('qpos', 7)
        next_obj_pos = obj_qpos[0:3] + self.param_.h_ * obj_qvel[0:3]
        next_obj_quat = (obj_qpos[3:7] + 0.5 * self.param_.h_ * self.cs_qmat_body_fn_(obj_qpos[3:7]) @ obj_qvel[3:6])
        next_obj_qpos = cs.vertcat(next_obj_pos, next_obj_quat)
        self.obj_poseInteg_fn = cs.Function('obj_poseInteg', [obj_qpos, obj_qvel], [next_obj_qpos])

    def step(self, curr_q, cmd, phi_vec, jac_mat):
        qp_param = cs.vertcat(cmd, cs.vec(phi_vec), cs.vec(jac_mat))
        sol = self.qp_solver_fn_(p=qp_param, lbg=0.0)
        v = sol['x'].full().flatten()
        next_qpos = []
        for i in range(self.param_.n_obj_):
            next_qpos.append(self.obj_poseInteg_fn(curr_q[7 * i:7 * i + 7], v[6 * i:6 * i + 6]))
        next_qpos.append(curr_q[-1] + self.param_.h_ * v[-1])
        next_qpos = cs.vcat(next_qpos).full().flatten()
        return next_qpos
