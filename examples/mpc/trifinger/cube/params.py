import numpy as np
import casadi as cs
import envs.trifinger_fkin as trifinger_fkin

from utils import rotations


class ExplicitMPCParams:
    def __init__(self, rand_seed=1, target_type='rotation'):
        # ---------------------------------------------------------------------------------------------
        #      simulation parameters 
        # ---------------------------------------------------------------------------------------------
        self.model_path_ = 'envs/xmls/env_trifinger_cube.xml'
        self.object_names_ = ['obj']

        self.h_ = 0.1
        self.frame_skip_ = int(20)

        # system dimensions:
        self.n_robot_qpos_ = 9
        self.n_qpos_ = 16
        self.n_qvel_ = 15
        self.n_cmd_ = 9

        # internal joint controller for each finger
        self.jc_kp_ = 10
        self.jc_damping_ = 0.05

        # ---------------------------------------------------------------------------------------------
        #      initial state and target state
        # ---------------------------------------------------------------------------------------------
        np.random.seed(100 + rand_seed)

        # random initial pose for object
        init_height = 0.030
        init_xy_rand = 0.1 * np.random.rand(2) - 0.05
        yaw_angle = - np.pi * np.random.rand(1) + np.pi / 2
        init_obj_quat_rand = rotations.rpy_to_quaternion(np.hstack([yaw_angle, 0, 0]))

        self.init_obj_qpos_ = np.hstack((init_xy_rand, init_height, init_obj_quat_rand))
        self.init_robot_qpos_ = np.array([
            0.0, -0.7, -1.1,
            0.0, -0.7, -1.1,
            0.0, -0.7, -1.1
        ])

        # random target pose for object
        if target_type == 'rotation':
            target_xy_rand = 0.1 * np.random.rand(2) - 0.05
            self.target_p_ = np.hstack([target_xy_rand, init_height])
            yaw_angle = np.pi * np.random.rand(1) - np.pi / 2
            self.target_q_ = rotations.rpy_to_quaternion(np.hstack([yaw_angle, 0, 0]))
        else:
            raise ValueError(f'Target type {target_type} not supported')

        # ---------------------------------------------------------------------------------------------
        #      contact parameters
        # ---------------------------------------------------------------------------------------------
        self.mu_object_ = 0.5
        self.n_mj_q_ = self.n_qpos_
        self.n_mj_v_ = self.n_qvel_
        self.max_ncon_ = 15

        # ---------------------------------------------------------------------------------------------
        #      models parameters
        # ---------------------------------------------------------------------------------------------
        self.obj_inertia_ = np.identity(6)
        self.obj_inertia_[0:3, 0:3] = 50 * np.eye(3)
        self.obj_inertia_[3:, 3:] = 0.1 * np.eye(3)
        self.robot_stiff_ = np.diag(self.n_cmd_ * [10])

        Q = np.zeros((self.n_qvel_, self.n_qvel_))
        Q[:6, :6] = self.obj_inertia_
        Q[6:, 6:] = self.robot_stiff_
        self.Q = Q

        self.obj_mass_ = 0.01
        self.gravity_ = np.array([0.00, 0.00, -9.8, 0.0, 0.0, 0.0])

        self.model_params = 0.5

        # ---------------------------------------------------------------------------------------------
        #      planner parameters
        # ---------------------------------------------------------------------------------------------
        self.mpc_horizon_ = 4
        self.ipopt_max_iter_ = 50
        self.mpc_model = 'explicit'

        self.mpc_u_lb_ = -0.01
        self.mpc_u_ub_ = 0.01
        obj_pos_lb = np.array([-0.99, -0.99, 0])
        obj_pos_ub = np.array([0.99, 0.99, 0.99])
        self.mpc_q_lb_ = np.hstack((obj_pos_lb, -1e7 * np.ones(4), -1e7 * np.ones(9)))
        self.mpc_q_ub_ = np.hstack((obj_pos_ub, 1e7 * np.ones(4), 1e7 * np.ones(9)))
        self.sol_guess_ = None

    # ---------------------------------------------------------------------------------------------
    #      cost function for MPC
    # ---------------------------------------------------------------------------------------------

    def init_cost_fns(self):
        x = cs.SX.sym('x', self.n_qpos_)
        u = cs.SX.sym('u', self.n_cmd_)

        obj_pose = x[0:7]
        f0_qpos = x[7:10]
        f120_qpos = x[10:13]
        f240_qpos = x[13:16]

        # forward kinematics to compute the position of fingertip
        ftp_1_position = trifinger_fkin.f0tip_pos_fd_fn(f0_qpos)
        ftp_2_position = trifinger_fkin.f120tip_pos_fd_fn(f120_qpos)
        ftp_3_position = trifinger_fkin.f240tip_pos_fd_fn(f240_qpos)

        # target cost
        target_position = cs.SX.sym('target_position', 3)
        target_quaternion = cs.SX.sym('target_quaternion', 4)
        position_cost = cs.sumsqr(obj_pose[0:3] - target_position)
        quaternion_cost = 1 - cs.dot(obj_pose[3:7], target_quaternion) ** 2
        contact_cost = (
                cs.sumsqr(obj_pose[0:3] - ftp_1_position) +
                cs.sumsqr(obj_pose[0:3] - ftp_2_position) +
                cs.sumsqr(obj_pose[0:3] - ftp_3_position)
        )

        # grasp cost
        obj_dirmat = rotations.quat2dcm_fn(x[3:7])
        obj_v0 = obj_dirmat.T @ (ftp_1_position - x[0:3])
        obj_v1 = obj_dirmat.T @ (ftp_2_position - x[0:3])
        obj_v2 = obj_dirmat.T @ (ftp_3_position - x[0:3])
        grasp_closure = cs.sumsqr(obj_v0 / cs.norm_2(obj_v0) + obj_v1 / cs.norm_2(obj_v1) + obj_v2 / cs.norm_2(obj_v2))

        # control cost
        control_cost = cs.sumsqr(u)

        # cost params
        phi_vec = cs.SX.sym('phi_vec', self.max_ncon_ * 4)
        jac_mat = cs.SX.sym('jac_mat', self.max_ncon_ * 4, self.n_qvel_)
        cost_param = cs.vvcat([target_position, target_quaternion, phi_vec, jac_mat])

        # base cost
        base_cost = 0.5 * contact_cost + 0.05 * grasp_closure
        final_cost = 500 * position_cost + 5.0 * quaternion_cost

        path_cost_fn = cs.Function('path_cost_fn', [x, u, cost_param], [base_cost + 10 * control_cost])
        final_cost_fn = cs.Function('final_cost_fn', [x, cost_param], [10 * final_cost])

        return path_cost_fn, final_cost_fn
