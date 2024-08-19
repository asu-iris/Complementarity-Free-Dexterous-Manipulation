import casadi as cs
import numpy as np

import utils.rotations as rot

t_palm = rot.quattmat_fn(np.array([0, 1, 0, 1]) / np.linalg.norm([0, 1, 0, 1]))

# first finger
ff_qpos = cs.SX.sym('ff_qpos', 4)
ff_t_base = t_palm @ rot.ttmat_fn([0, 0.0435, -0.001542]) @ rot.quattmat_fn([0.999048, -0.0436194, 0, 0])
ff_t_proximal = ff_t_base @ rot.rztmat_fn(ff_qpos[0]) @ rot.ttmat_fn([0, 0, 0.0164])
ff_t_medial = ff_t_proximal @ rot.rytmat_fn(ff_qpos[1]) @ rot.ttmat_fn([0, 0, 0.054])
ff_t_distal = ff_t_medial @ rot.rytmat_fn(ff_qpos[2]) @ rot.ttmat_fn([0, 0, 0.0384])
ff_t_ftp = ff_t_distal @ rot.rytmat_fn(ff_qpos[3]) @ rot.ttmat_fn([0, 0, 0.0384])
fftp_pos_fd_fn = cs.Function('ff_t_ftp_fn', [ff_qpos], [ff_t_ftp[0:3, -1]])

# middle finger
mf_qpos = cs.SX.sym('mf_qpos', 4)
mf_t_base = t_palm @ rot.ttmat_fn([0, 0, 0.0007])
mf_t_proximal = mf_t_base @ rot.rztmat_fn(mf_qpos[0]) @ rot.ttmat_fn([0, 0, 0.0164])
mf_t_medial = mf_t_proximal @ rot.rytmat_fn(mf_qpos[1]) @ rot.ttmat_fn([0, 0, 0.054])
mf_t_distal = mf_t_medial @ rot.rytmat_fn(mf_qpos[2]) @ rot.ttmat_fn([0, 0, 0.0384])
mf_t_ftp = mf_t_distal @ rot.rytmat_fn(mf_qpos[3]) @ rot.ttmat_fn([0, 0, 0.0384])
mftp_pos_fd_fn = cs.Function('mftp_pos_fd_fn', [mf_qpos], [mf_t_ftp[0:3, -1]])

# ring finger
rf_qpos = cs.SX.sym('rf_qpos', 4)
rf_t_base = t_palm @ rot.ttmat_fn([0, -0.0435, -0.001542]) @ rot.quattmat_fn([0.999048, 0.0436194, 0, 0])
rf_t_proximal = rf_t_base @ rot.rztmat_fn(rf_qpos[0]) @ rot.ttmat_fn([0, 0, 0.0164])
rf_t_medial = rf_t_proximal @ rot.rytmat_fn(rf_qpos[1]) @ rot.ttmat_fn([0, 0, 0.054])
rf_t_distal = rf_t_medial @ rot.rytmat_fn(rf_qpos[2]) @ rot.ttmat_fn([0, 0, 0.0384])
rf_t_ftp = rf_t_distal @ rot.rytmat_fn(rf_qpos[3]) @ rot.ttmat_fn([0, 0, 0.0384])
rftp_pos_fd_fn = cs.Function('rftp_pos_fd_fn', [rf_qpos], [rf_t_ftp[0:3, -1]])

# Thumb
th_qpos = cs.SX.sym('th_qpos', 4)
th_t_base = t_palm @ rot.ttmat_fn([-0.0182, 0.019333, -0.045987]) @ rot.quattmat_fn(
    [0.477714, -0.521334, -0.521334, -0.477714])
th_t_proximal = th_t_base @ rot.rxtmat_fn(-th_qpos[0]) @ rot.ttmat_fn([-0.027, 0.005, 0.0399])
th_t_medial = th_t_proximal @ rot.rztmat_fn(th_qpos[1]) @ rot.ttmat_fn([0, 0, 0.0177])
th_t_distal = th_t_medial @ rot.rytmat_fn(th_qpos[2]) @ rot.ttmat_fn([0, 0, 0.0514])
th_t_ftp = th_t_distal @ rot.rytmat_fn(th_qpos[3]) @ rot.ttmat_fn([0, 0, 0.054])
thtp_pos_fd_fn = cs.Function('thtp_pos_fd_fn', [th_qpos], [th_t_ftp[0:3, -1]])
