import mujoco

import numpy as np

np.set_printoptions(suppress=True)

from examples.sim.boxes_pushing.boxes_pushing_sim import MjDynViewer

class Contact:
    def __init__(self, param):
        self.param_ = param
        self.n_active_contact = 0.0

    def detect_once(self, curr_q, simulator: MjDynViewer):

        simulator.data_.qpos[:] = curr_q

        mujoco.mj_forward(simulator.model_, simulator.data_)
        mujoco.mj_collision(simulator.model_, simulator.data_)

        # extract the contacts
        n_con = simulator.data_.ncon
        contacts = simulator.data_.contact
        self.n_active_contact = n_con

        # solve the contact Jacobian
        con_phi_list = []
        con_frame_list = []
        con_pos_list = []
        con_jac_list = []

        for i in range(n_con):
            contact_i = contacts[i]

            geom1_name = mujoco.mj_id2name(simulator.model_, mujoco.mjtObj.mjOBJ_GEOM, contact_i.geom1)
            body1_id = simulator.model_.geom_bodyid[contact_i.geom1]
            geom2_name = mujoco.mj_id2name(simulator.model_, mujoco.mjtObj.mjOBJ_GEOM, contact_i.geom2)
            body2_id = simulator.model_.geom_bodyid[contact_i.geom2]

            # print(geom2_name, geom1_name)

            # contact point
            con_pos = contact_i.pos
            con_dist = contact_i.dist * 0.5
            con_mu = self.param_.mu_object_

            con_frame = contact_i.frame.reshape((-1, 3)).T
            con_frame_pmd = np.hstack((con_frame, -con_frame[:, -2:]))

            jacp1 = np.zeros((3, self.param_.n_mj_v_))
            mujoco.mj_jac(simulator.model_, simulator.data_, jacp=jacp1, jacr=None, point=con_pos, body=body1_id)
            con_jacp1 = con_frame_pmd.T @ jacp1

            jacp2 = np.zeros((3, self.param_.n_mj_v_))
            mujoco.mj_jac(simulator.model_, simulator.data_, jacp=jacp2, jacr=None, point=con_pos, body=body2_id)
            con_jacp2 = con_frame_pmd.T @ jacp2

            # hint:
            # jacobian direction: from contact pair to obj
            con_jacp = (con_jacp2 - con_jacp1)
            con_jacp_n = con_jacp[0]
            con_jacp_f = con_jacp[1:]
            con_jac = con_jacp_n + con_mu * con_jacp_f

            con_pos_list.append(con_pos)
            con_phi_list.append(con_dist)
            con_frame_list.append(con_frame)
            con_jac_list.append(con_jac)

        phi_vec, jac_mat = self.reformat(
            dict(
                con_pos_list=con_pos_list,
                con_phi_list=con_phi_list,
                con_frame_list=con_frame_list,
                con_jac_list=con_jac_list)
        )

        return phi_vec, jac_mat

    def reformat(self, contacts=None):
        # parse the input
        con_jac_list = contacts['con_jac_list']
        con_phi_list = contacts['con_phi_list']

        # fill the phi_vec
        phi_vec = np.ones((self.param_.max_ncon_ * 4,))  # this is very,very important for soft sensitivity analysis
        jac_mat = np.zeros((self.param_.max_ncon_ * 4, self.param_.n_qvel_))
        for i in range(len(con_phi_list)):
            phi_vec[4 * i: 4 * i + 4] = con_phi_list[i]
            jac_mat[4 * i: 4 * i + 4] = con_jac_list[i]

        return phi_vec, jac_mat
