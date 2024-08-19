import casadi as cs
import numpy as np
import time

from models.qp_model import QPModel


class MPCImplicit:
    def __init__(self, param):
        self.param_ = param

        # parse mpc model
        self.mpc_model = self.param_.mpc_model
        if self.mpc_model == 'qp_model':
            self.model = QPModel(param)

        # cost function
        self.path_cost_fn, self.final_cost_fn = self.param_.init_cost_fns()

        # initialize mpc solver
        self.init_MPC()

    def plan_once(self, target_p, target_q, curr_x, phi_vec, jac_mat, sol_guess=None):
        if sol_guess is None:
            sol_guess = dict(x0=self.nlp_w0_, lam_x0=self.nlp_lam_x0_, lam_g0=self.nlp_lam_g0_)

        cost_params = cs.vvcat([target_p, target_q, phi_vec, jac_mat])
        nlp_param = self.nlp_params_fn_(curr_x, phi_vec, jac_mat, cost_params)

        nlp_lbw, nlp_ubw = self.nlp_bounds_fn_(self.param_.mpc_u_lb_, self.param_.mpc_u_ub_, self.param_.mpc_q_lb_,
                                               self.param_.mpc_q_ub_)

        st = time.time()
        raw_sol = self.ipopt_solver(x0=self.nlp_w0_,
                                    lam_x0=self.nlp_lam_x0_,
                                    lam_g0=self.nlp_lam_g0_,
                                    lbx=nlp_lbw, ubx=nlp_ubw,
                                    lbg=0.0, ubg=0.0,
                                    p=nlp_param)
        print("mpc solve time:", time.time() - st)
        print('return_status = ', self.ipopt_solver.stats()['return_status'])

        w_opt = raw_sol['x'].full().flatten()
        cost_opt = raw_sol['f'].full().flatten()

        # extract the solution from the raw solution
        sol_traj = np.reshape(w_opt, (self.param_.mpc_horizon_, -1))
        opt_u_traj = sol_traj[:, 0:self.param_.n_cmd_]

        return dict(action=opt_u_traj[0, :],
                    sol_guess=dict(x0=w_opt,
                                   lam_x0=raw_sol['lam_x'],
                                   lam_g0=raw_sol['lam_g'],
                                   opt_cost=raw_sol['f'].full().item()),
                    cost_opt=cost_opt,
                    solve_status=self.ipopt_solver.stats()['return_status'])

    def init_MPC(self):

        phi_vec = cs.SX.sym('contact_distances', self.param_.max_ncon_ * 4)
        jac_mat = cs.SX.sym('contact_jacobians', self.param_.max_ncon_ * 4, self.param_.n_qvel_)

        dyn_cost_fn = self.model.qp_model['qp_cost_fn']
        dyn_g_fn = self.model.qp_model['qp_g_fn']

        # set variable
        u = cs.SX.sym('u', self.param_.n_cmd_)
        v = cs.SX.sym('v', self.param_.n_qvel_)
        lam = cs.SX.sym('lams', dyn_g_fn.size_out(0))
        dim_lam = dyn_g_fn.size_out(0)[0]

        # add kkt condition(LCP) of models(QP) into MPC problem
        dyn_lag = dyn_cost_fn(v, u) - cs.dot(lam, dyn_g_fn(v, phi_vec, jac_mat))
        dyn_equ_fn = cs.Function('dyn_equ_fn', [v, u, lam, phi_vec, jac_mat],
                                 [cs.gradient(dyn_lag, v)])
        dyn_comple_fn = cs.Function('dyn_comple_fn', [v, u, lam, phi_vec, jac_mat],
                                    [cs.diag(lam) @ dyn_g_fn(v, phi_vec, jac_mat)])

        # cost function params
        cost_params = cs.SX.sym('cost_params', self.path_cost_fn.size_in(2))
        lbu = cs.SX.sym('lbu', self.param_.n_cmd_)
        ubu = cs.SX.sym('ubu', self.param_.n_cmd_)

        lbq = cs.SX.sym('lbq', self.param_.n_qpos_)
        ubq = cs.SX.sym('ubq', self.param_.n_qpos_)

        # start with empty NLP
        w, w0, lbw, ubw, g = [], [], [], [], []
        J = 0.0
        q0 = cs.SX.sym('q', self.param_.n_qpos_)
        qk = q0
        for k in range(self.param_.mpc_horizon_):
            # control at time k
            uk = cs.SX.sym('u' + str(k), u.shape)
            w += [uk]
            lbw += [lbu]
            ubw += [ubu]
            w0 += [cs.DM.zeros(self.param_.n_cmd_)]

            # v at time k
            vk = cs.SX.sym('v' + str(k), self.param_.n_qvel_)
            w += [vk]
            lbw += [-cs.inf * cs.DM.ones(self.param_.n_qvel_)]
            ubw += [cs.inf * cs.DM.ones(self.param_.n_qvel_)]
            w0 += [cs.DM.zeros(self.param_.n_qvel_)]

            # lam at time k
            lamk = cs.SX.sym('lam' + str(k), lam.shape)
            w += [lamk]
            lbw += [cs.DM.zeros(lam.shape)]
            ubw += [cs.inf * cs.DM.ones(lam.shape)]
            w0 += [self.param_.comple_relax * cs.DM.ones(dim_lam)]

            # add models equality
            g += [dyn_equ_fn(vk, uk, lamk, phi_vec, jac_mat)]
            g += [dyn_comple_fn(vk, uk, lamk, phi_vec, jac_mat) - self.param_.comple_relax]

            # compute the predicted q_new
            pred_q = self.model.cs_qposInteg_(qk, vk)

            # compute the cost function
            J += self.path_cost_fn(qk, uk, cost_params)

            # q at time k+1 .... q_new
            qk = cs.SX.sym('q' + str(k + 1), self.param_.n_qpos_)
            w += [qk]
            w0 += [cs.DM.zeros(self.param_.n_qpos_)]
            lbw += [-cs.inf * cs.DM.ones(self.param_.n_qpos_)]
            ubw += [cs.inf * cs.DM.ones(self.param_.n_qpos_)]

            # add the concatenation constraint
            g += [pred_q - qk]

        # compute the final cost
        J += self.final_cost_fn(qk, cost_params)

        # create an NLP solver
        nlp_params = cs.vvcat([q0, phi_vec, jac_mat, cost_params])
        nlp_prog = {'f': J, 'x': cs.vcat(w), 'g': cs.vcat(g), 'p': nlp_params}
        nlp_opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0,
                    'ipopt.max_iter': self.param_.ipopt_max_iter_}
        self.ipopt_solver = cs.nlpsol('solver', 'ipopt', nlp_prog, nlp_opts)

        # useful mappings
        self.nlp_w0_ = cs.vcat(w0)
        self.nlp_lam_x0_ = cs.DM.zeros(self.nlp_w0_.shape)
        self.nlp_lam_g0_ = cs.DM.zeros(cs.vcat(g).shape)
        self.nlp_bounds_fn_ = cs.Function('nlp_bounds_fn', [lbu, ubu, lbq, ubq], [cs.vcat(lbw), cs.vvcat(ubw)])
        self.nlp_params_fn_ = cs.Function('nlp_params_fn',
                                          [q0, phi_vec, jac_mat, cost_params], [nlp_params])
