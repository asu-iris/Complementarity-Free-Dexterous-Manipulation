import casadi as cs
import numpy as np

import utils.rotations as rot

t_base = rot.ttmat_fn([0, 0, 0])

#  finger0
f0_qpos = cs.SX.sym('f0_qpos', 3)
f0_t_upper = t_base @ rot.ttmat_fn([0, 0.04, 0.29])
f0_t_middle = f0_t_upper @ rot.rytmat_fn(f0_qpos[0]) @ rot.ttmat_fn([0, 0, 0.0])
f0_t_lower = f0_t_middle @ rot.rxtmat_fn(-f0_qpos[1]) @ rot.ttmat_fn([0, 0, -0.16])
f0_t_tip = (f0_t_lower @ rot.rxtmat_fn(f0_qpos[2]) @ rot.ttmat_fn([0.0, 0, -0.16]))
f0tip_pos_fd_fn = cs.Function('f0tp_pos_fd_fn', [f0_qpos], [f0_t_tip[0:3, -1]])

#  finger120
f120_qpos = cs.SX.sym('f120_qpos', 3)
f120_t_upper = t_base @ rot.ttmat_fn([0.034641, -0.02, 0.29]) @ rot.quattmat_fn([0.5, 0, 0, -0.866025])
f120_t_middle = f120_t_upper @ rot.rytmat_fn(f120_qpos[0]) @ rot.ttmat_fn([0, 0, 0.0])
f120_t_lower = f120_t_middle @ rot.rxtmat_fn(-f120_qpos[1]) @ rot.ttmat_fn([0, 0, -0.16])
f120_t_tip = (f120_t_lower @ rot.rxtmat_fn(f120_qpos[2]) @ rot.ttmat_fn([0.0, 0, -0.16]))
f120tip_pos_fd_fn = cs.Function('f120tip_pos_fd_fn', [f120_qpos], [f120_t_tip[0:3, -1]])

# finger 240
f240_qpos = cs.SX.sym('f240_qpos', 3)
f240_t_upper = t_base @ rot.ttmat_fn([-0.034641, -0.02, 0.29]) @ rot.quattmat_fn([-0.5, 0, 0, -0.866025])
f240_t_middle = f240_t_upper @ rot.rytmat_fn(f240_qpos[0]) @ rot.ttmat_fn([0, 0, 0.0])
f240_t_lower = f240_t_middle @ rot.rxtmat_fn(-f240_qpos[1]) @ rot.ttmat_fn([0, 0, -0.16])
f240_t_tip = (f240_t_lower @ rot.rxtmat_fn(f240_qpos[2]) @ rot.ttmat_fn([0.0, 0, -0.16]))
f240tip_pos_fd_fn = cs.Function('f240tip_pos_fd_fn', [f240_qpos], [f240_t_tip[0:3, -1]])
