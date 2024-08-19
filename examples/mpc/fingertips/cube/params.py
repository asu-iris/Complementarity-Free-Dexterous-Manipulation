import casadi as cs
import numpy as np

from utils import rotations


class ExplicitMPCParams:
    def __init__(self, rand_seed=1, target_type=True):
        # ---------------------------------------------------------------------------------------------
        #      simulation parameters 
        # ---------------------------------------------------------------------------------------------
        self.model_path_ = 'envs/xmls/env_fingertips_cube.xml'
        self.object_names_ = ['obj']

        self.h_ = 0.1
        self.frame_skip_ = int(10)

        # system dimensions:
        self.n_robot_qpos_ = 9
        self.n_qpos_ = 16
        self.n_qvel_ = 15
        self.n_cmd_ = 9

        # ---------------------------------------------------------------------------------------------
        #      initial state and target state
        # ---------------------------------------------------------------------------------------------
        np.random.seed(100 + rand_seed)

        # random initial pose for object
        init_height = 0.03
        init_xy_rand = 0.05 * np.random.rand(2) - 0.025
        init_angle_rand = 2 * np.pi * np.random.rand(1) - np.pi
        init_obj_quat_rand = rotations.axisangle2quat(np.hstack(([0, 0, 1.0], init_angle_rand)))
        self.init_obj_qpos_ = np.hstack((init_xy_rand, init_height, init_obj_quat_rand))
        self.init_robot_qpos_ = np.array([0.2, 0.0, 0.0, 0.0, 0.2, 0.0, -0.2, 0.0, 0.0])

        # random target pose for object
        if target_type == 'ground-rotation':
            target_xy_rand = 0.2 * np.random.rand(2) - 0.1
            self.target_p_ = np.hstack([target_xy_rand, init_height])
            yaw_angle = 2 * np.pi * np.random.rand(1) - np.pi
            self.target_q_ = rotations.rpy_to_quaternion(np.hstack([yaw_angle, 0, 0]))

        elif target_type == 'ground-flip':
            init_height = 0.03+0.02
            target_xy_rand = 0.2 * np.random.rand(2) - 0.1
            self.target_p_ = np.hstack([target_xy_rand, init_height])
            yaw_angle = 2 * np.pi * np.random.rand(1) - np.pi
            pitch_angle = np.pi * np.random.rand(1) - np.pi/2
            roll_angle = np.pi * np.random.rand(1) - np.pi/2
            self.target_q_ = rotations.rpy_to_quaternion(np.hstack([yaw_angle, pitch_angle, roll_angle]))

        elif target_type == 'in-air':
            target_height = 0.05 + 0.05 * np.random.rand(1)
            target_xy_rand = 0.2 * np.random.rand(2) - 0.1
            self.target_p_ = np.hstack([target_xy_rand, target_height])

            angle = 2 * np.pi * np.random.rand(1) - np.pi
            axis = np.array([0, 1, 1]) + np.random.randn(3) * 0.1
            self.target_q_ = rotations.axisangle2quat(np.hstack([axis, angle]))

        else:
            raise ValueError('Invalid target type')

        # ---------------------------------------------------------------------------------------------
        #      contact parameters 
        # ---------------------------------------------------------------------------------------------
        self.mu_object_ = 0.5
        self.n_mj_q_ = self.n_qpos_
        self.n_mj_v_ = self.n_qvel_
        self.max_ncon_ = 8

        # ---------------------------------------------------------------------------------------------
        #      models parameters
        # ---------------------------------------------------------------------------------------------
        self.obj_inertia_ = np.identity(6)
        self.obj_inertia_[0:3, 0:3] = 50 * np.eye(3)
        self.obj_inertia_[3:, 3:] = 0.05 * np.eye(3)
        self.robot_stiff_ = np.diag(self.n_cmd_ * [100])

        Q = np.zeros((self.n_qvel_, self.n_qvel_))
        Q[:6, :6] = self.obj_inertia_
        Q[6:, 6:] = self.robot_stiff_
        self.Q = Q

        self.obj_mass_ = 0.01
        self.gravity_ = np.array([0.00, 0.00, -9.8, 0.0, 0.0, 0.0])

        self.model_params = 1

        # ---------------------------------------------------------------------------------------------
        #      planner parameters
        # ---------------------------------------------------------------------------------------------
        self.mpc_horizon_ = 4
        self.ipopt_max_iter_ = 500
        self.mpc_model = 'explicit'

        self.mpc_u_lb_ = -0.005
        self.mpc_u_ub_ = 0.005
        fts_q_lb = np.array([-100, -100, 0.0, -100, -100, 0.0, -100, -100, 0.0])
        fts_q_ub = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100])
        self.mpc_q_lb_ = np.hstack((-1e7 * np.ones(7), fts_q_lb))
        self.mpc_q_ub_ = np.hstack((1e7 * np.ones(7), fts_q_ub))

        self.sol_guess_ = None

    def init_cost_fns(self):
        x = cs.SX.sym('x', self.n_qpos_)
        u = cs.SX.sym('u', self.n_cmd_)

        # target cost
        target_position = cs.SX.sym('target_position', 3)
        target_quaternion = cs.SX.sym('target_quaternion', 4)
        position_cost = cs.sumsqr(x[0:3] - target_position)
        quaternion_cost = 1 - cs.dot(x[3:7], target_quaternion) ** 2
        contact_cost = cs.sumsqr(x[0:3] - x[7:10]) + cs.sumsqr(x[0:3] - x[10:13]) + cs.sumsqr(x[0:3] - x[13:16])
        control_cost = cs.sumsqr(u)

        obj_dirmat = rotations.quat2dcm_fn(x[3:7])
        obj_v0 = obj_dirmat.T @ (x[7:10] - x[0:3])
        obj_v1 = obj_dirmat.T @ (x[10:13] - x[0:3])
        obj_v2 = obj_dirmat.T @ (x[13:16] - x[0:3])
        grasp_closure = cs.sumsqr(
            obj_v0 / cs.norm_2(obj_v0) + obj_v1 / cs.norm_2(obj_v1) + obj_v2 / cs.norm_2(obj_v2))

        # cost params
        phi_vec = cs.SX.sym('phi_vec', self.max_ncon_ * 4)
        jac_mat = cs.SX.sym('jac_mat', self.max_ncon_ * 4, self.n_qvel_)
        cost_param = cs.vvcat([target_position, target_quaternion, phi_vec, jac_mat])

        # base cost
        base_cost = 1 * contact_cost + 0.05 * grasp_closure
        final_cost = 500 * position_cost + 5.0 * quaternion_cost

        path_cost_fn = cs.Function('path_cost_fn', [x, u, cost_param], [base_cost + 50 * control_cost])
        final_cost_fn = cs.Function('final_cost_fn', [x, cost_param], [10 * final_cost])

        return path_cost_fn, final_cost_fn
