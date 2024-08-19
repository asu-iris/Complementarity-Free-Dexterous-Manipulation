import numpy as np

from examples.mpc.allegro.rubber_duck.params import ExplicitMPCParams
from planning.mpc_explicit import MPCExplicit
from envs.allegro_env import MjSimulator
from contact.allegro_collision_detection import Contact

from utils import metrics

# -------------------------------
#       loop trials
# -------------------------------
save_flag=False
if save_flag:
    save_dir = './examples/mpc/allegro/rubber_duck/save/'
    prefix_data_name = 'ours_'
    save_data = dict()

trial_num = 20
success_pos_threshold = 0.02
success_quat_threshold = 0.04
consecutive_success_time_threshold = 20
max_rollout_length = 500

trial_count = 0
while trial_count < trial_num:

    # -------------------------------
    #        init parameters
    # -------------------------------
    param = ExplicitMPCParams(rand_seed=trial_count, target_type='rotation')

    # -------------------------------
    #        init contact
    # -------------------------------
    contact = Contact(param)

    # -------------------------------
    #        init envs
    # -------------------------------
    env = MjSimulator(param)

    # -------------------------------
    #        init planner
    # -------------------------------
    mpc = MPCExplicit(param)

    # -------------------------------
    #        MPC rollout
    # -------------------------------
    rollout_step = 0
    consecutive_success_time = 0

    rollout_q_traj = []
    while rollout_step < max_rollout_length:
        if not env.dyn_paused_:
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
            print('trial no:', trial_count)
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
            if (metrics.comp_quat_error(curr_q[3:7], param.target_q_) < success_quat_threshold):
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
        # save
        save_data.update(target_obj_pos=param.target_p_)
        save_data.update(target_obj_quat=param.target_q_)
        save_data.update(rollout_traj=np.array(rollout_q_traj))
        # success index
        if rollout_step < max_rollout_length:
            save_data.update(success=True)
        else:
            save_data.update(success=False)
        # save to file
        metrics.save_data(save_data, data_name=prefix_data_name + 'trial_' + str(trial_count) + '_rollout',
                          save_dir=save_dir)

    trial_count = trial_count + 1
