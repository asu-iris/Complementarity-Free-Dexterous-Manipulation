import numpy as np

from examples.mpc.fingertips.bunny.params import ExplicitMPCParams
from planning.mpc_explicit import MPCExplicit

from envs.fingertips_env import MjSimulator
from contact.fingertips_collision_detection import Contact

from utils import metrics

# -------------------------------
#       loop trials
# -------------------------------
save_flag = False

if save_flag:
    save_dir = './examples/mpc/fingertips/bunny/save-air/'
    prefix_data_name = 'ours_'
    save_data = dict()

trial_num = 20
success_pos_threshold = 0.02
success_quat_threshold = 0.015
consecutive_success_time_threshold = 20
max_rollout_length = 2000

trial_count = 0
while trial_count < trial_num:

    # -------------------------------
    #        init param and MPC solver
    # -------------------------------
    param = ExplicitMPCParams(rand_seed=trial_count, target_type='in-air')
    mpc = MPCExplicit(param)

    # -------------------------------
    #        init contact
    # -------------------------------
    contact = Contact(param)

    # -------------------------------
    #        init envs
    # -------------------------------
    env = MjSimulator(param)

    # -------------------------------
    #        MPC rollout
    # -------------------------------
    rollout_step = 0
    consecutive_success_time = 0

    rollout_q_traj = []
    while rollout_step < max_rollout_length:
        # get state
        curr_q = env.get_state()
        rollout_q_traj.append(curr_q)

        # -----------------------
        #     contact detect
        # -----------------------
        phi_vec, jac_mat = contact.detect_once(env)

        # -----------------------
        #        planning
        # -----------------------
        sol = mpc.plan_once(
            param.target_p_,
            param.target_q_,
            curr_q,
            phi_vec,
            jac_mat,
            sol_guess=param.sol_guess_)
        param.sol_guess_ = sol['sol_guess']
        action = sol['action']

        # -----------------------
        #        simulate
        # -----------------------
        env.step(action)
        rollout_step = rollout_step + 1

        # -----------------------
        #        success check
        # -----------------------
        curr_q = env.get_state()
        if (metrics.comp_pos_error(curr_q[0:3], param.target_p_) < success_pos_threshold) \
                and (metrics.comp_quat_error(curr_q[3:7], param.target_q_) < success_quat_threshold):
            consecutive_success_time = consecutive_success_time + 1
        else:
            consecutive_success_time = 0

        # -----------------------
        #       early termination
        # -----------------------
        if consecutive_success_time > consecutive_success_time_threshold:
            break

    # -------------------------------
    #        close viewer
    # -------------------------------
    env.viewer_.close()

    # -------------------------------
    #        save data
    # -------------------------------
    if save_flag:
        # object target pos
        save_data.update(target_obj_pos=param.target_p_)
        # object target quat
        save_data.update(target_obj_quat=param.target_q_)
        # env rollout trajectory
        save_data.update(rollout_traj=np.array(rollout_q_traj))
        # success index
        if rollout_step < max_rollout_length:
            save_data.update(success=True)
        else:
            save_data.update(success=False)
        # save
        metrics.save_data(save_data, data_name=prefix_data_name + 'trial_' + str(trial_count) + '_rollout',
                          save_dir=save_dir)

    trial_count = trial_count + 1
