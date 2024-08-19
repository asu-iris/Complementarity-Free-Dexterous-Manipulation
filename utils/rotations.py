import numpy as np
import numpy as numpy
import math
import casadi as cs


#########################################
#########################################

# converter to quaternion from (radian angle, direction)
def angle_dir_to_quat(angle, dir):
    if type(dir) == list:
        dir = np.array(dir)
    dir = dir / np.linalg.norm(dir)
    quat = np.zeros(4)
    quat[0] = math.cos(angle / 2)
    quat[1:] = math.sin(angle / 2) * dir
    return quat


def rpy_to_quaternion(angles):
    yaw, pitch, roll = angles[0], angles[1], angles[2]

    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return np.array([qw, qx, qy, qz])


def quat_to_rpy(q):
    x, y, z, w = q[1], q[2], q[3], q[0]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(np.min([2 * (w * y - x * z),1]))
    # pitch = - math.pi / 2 + 2 * math.atan2(math.sqrt(1 + 2 * (w * y - x * z)),
    #                                        math.sqrt(1 - 2 * (w * y - x * z)+ 1e-10) )
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return np.array([roll, pitch, yaw])


# alternative form
def axisangle2quat(axisangle):
    dir = axisangle[0:3]
    angle = axisangle[3]
    dir = dir / np.linalg.norm(dir)
    quat = np.zeros(4)
    quat[0] = math.cos(angle / 2)
    quat[1:] = math.sin(angle / 2) * dir
    return quat


#########################################
#########################################

# https://github.com/Khrylx/Mujoco-modeler/blob/master/transformation.py
# quaternion multiplication
def quaternion_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions.
    """
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return numpy.array([
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=numpy.float64)


def quaternion_mat(q):
    return np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1], q[0], -q[3], q[2]],
        [q[2], q[3], q[0], -q[1]],
        [q[3], -q[2], q[1], q[0]],
    ], dtype=numpy.float64)


def quaternionToAxisAngle(p: np.ndarray) -> list:
    # """Compute rotation parameters (axis and angle in radians) from a quaternion p defining a given orientation.

    # Parameters
    # ----------
    # p : [4x1] np.ndarray
    #     quaternion defining a given orientation

    # Returns
    # -------
    # axis : [3x1] np.ndarray, when undefined=[0. 0. 0.]
    # angle : float      
    # """
    # if isinstance(p, list) and len(p)==4:
    #     e0 = np.array(p[0])
    #     e = np.array(p[1:])  
    # elif isinstance(p, np.ndarray) and p.size==4:
    #     e0 = p[0]
    #     e = p[1:]
    # else:
    #     raise TypeError("The quaternion \"p\" must be given as [4x1] np.ndarray quaternion or a python list of 4 elements")    

    # if np.linalg.norm(e) == 0:
    #     axis = np.array([1,0,0]) #To be checked again
    #     angle = 0
    # elif np.linalg.norm(e) != 0:
    #     axis = e/np.linalg.norm(e) 
    #     if e0 == 0:
    #         angle = np.pi
    #     else:
    #         angle = 2*np.arctan(np.linalg.norm(e)/e0) 

    # return axis, angle

    p = p / np.linalg.norm(p)

    angle = 2 * np.arccos(p[0])

    axis = p[1:] / np.sin(angle / 2)

    return axis, angle


def quaternion_mul(q1, q2):
    return quaternion_mat(q1) @ q2


def quaternion_conjugate(quaternion):
    """Return conjugate of quaternion.
    """
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    numpy.negative(q[1:], q[1:])
    return q


def quaternion_inverse(quaternion):
    """Return inverse of quaternion.
    """
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    numpy.negative(q[1:], q[1:])
    return q / numpy.dot(q, q)


def quaternion_real(quaternion):
    """Return real part of quaternion.
    """
    return float(quaternion[0])


def quaternion_imag(quaternion):
    """Return imaginary part of quaternion.
    """
    return numpy.array(quaternion[1:4], dtype=numpy.float64, copy=True)


def quaternion_slerp(quat0, quat1, N):
    """Return spherical linear interpolation between two quaternions.
    """
    _EPS = 1e-6

    q0 = quat0 / np.linalg.norm(quat0)
    q1 = quat1 / np.linalg.norm(quat1)

    d = numpy.dot(q0, q1)

    if abs(abs(d) - 1.0) < _EPS:
        return np.tile(q0, (N, 1))

    angle = math.acos(d)
    isin = 1.0 / math.sin(angle)

    fractions = np.linspace(0, 1, N)

    q = []
    for frac in fractions:
        q.append(math.sin((1.0 - frac) * angle) * isin * q0 +
                 math.sin(frac * angle) * isin * q1)

    return np.array(q)


def random_quaternion(rand=None):
    """Return uniform random unit quaternion.
    rand: array like or None
        Three independent random variables that are uniformly distributed
        between 0 and 1.
    """
    if rand is None:
        rand = numpy.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = numpy.sqrt(1.0 - rand[0])
    r2 = numpy.sqrt(rand[0])
    pi2 = math.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return numpy.array([numpy.cos(t2) * r2, numpy.sin(t1) * r1,
                        numpy.cos(t1) * r1, numpy.sin(t2) * r2])


#########################################
#########################################

def quat2rotmat(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix


#########################################
# below is problematic
#########################################

def quat2angle(q):
    return 2.0 * math.acos(q[0]) * np.sign(q[-1])


#########################################
#########################################

def angle2mat(angle):
    mat = np.array(
        [[math.cos(angle), -math.sin(angle)],
         [math.sin(angle), math.cos(angle)]
         ]
    )
    return mat


#########################################
#########################################

# ----------------
# conjugate quaternion matrix
# https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/
# https://opensource.docs.anymal.com/doxygen/kindr/master/cheatsheet_latest.pdf

q = cs.SX.sym('q', 4)
cqm = cs.vertcat(
    cs.horzcat(q[0], -q[1], -q[2], -q[3]),
    cs.horzcat(q[1], q[0], q[3], -q[2]),
    cs.horzcat(q[2], -q[3], q[0], q[1]),
    cs.horzcat(q[3], q[2], -q[1], q[0]),
)
conjquat_mat_fn = cs.Function('conjquat_mat_fn', [q], [cqm])

wb = cs.SX.sym('wb', 3)
wb_cqm = cs.vertcat(
    cs.horzcat(0, -wb[0], -wb[1], -wb[2]),
    cs.horzcat(wb[0], 0, wb[2], -wb[1]),
    cs.horzcat(wb[1], -wb[2], 0, wb[0]),
    cs.horzcat(wb[2], wb[1], -wb[0], 0),
)
conjquatmat_wb_fn = cs.Function('conjquatmat_wb_fn', [wb], [wb_cqm])

# ----------------
# quaternion to dcm (casadi function)
q = cs.SX.sym('q', 4)
dcm = cs.vertcat(
    cs.horzcat(
        1 - 2 * (q[2] ** 2 + q[3] ** 2),
        2 * (q[1] * q[2] - q[0] * q[3]),
        2 * (q[1] * q[3] + q[0] * q[2]),
    ),
    cs.horzcat(
        2 * (q[1] * q[2] + q[0] * q[3]),
        1 - 2 * (q[1] ** 2 + q[3] ** 2),
        2 * (q[2] * q[3] - q[0] * q[1]),
    ),
    cs.horzcat(
        2 * (q[1] * q[3] - q[0] * q[2]),
        2 * (q[2] * q[3] + q[0] * q[1]),
        1 - 2 * (q[1] ** 2 + q[2] ** 2),
    ),
)
quat2dcm_fn = cs.Function('quat2dcm_fn', [q], [dcm])

# ----------------
# position to homogeneous transformation mat (casadi_fn)
pos = cs.SX.sym('pos', 3)
ttmat = cs.vertcat(
    cs.horzcat(1, 0, 0, pos[0]),
    cs.horzcat(0, 1, 0, pos[1]),
    cs.horzcat(0, 0, 1, pos[2]),
    cs.horzcat(0, 0, 0, 1)
)
ttmat_fn = cs.Function('ttmat_fn', [pos], [ttmat])

# ----------------
# rotations to homogeneous transformation mat (casadi_fn)
alpha = cs.SX.sym('alpha', 1)
rxtmat = cs.vertcat(
    cs.horzcat(1, 0, 0, 0),
    cs.horzcat(0, cs.cos(alpha), -cs.sin(alpha), 0),
    cs.horzcat(0, cs.sin(alpha), cs.cos(alpha), 0),
    cs.horzcat(0, 0, 0, 1)
)
rxtmat_fn = cs.Function('rxtmat_fn', [alpha], [rxtmat])

beta = cs.SX.sym('beta', 1)
rytmat = cs.vertcat(
    cs.horzcat(cs.cos(beta), 0, cs.sin(beta), 0),
    cs.horzcat(0, 1, 0, 0),
    cs.horzcat(-cs.sin(beta), 0, cs.cos(beta), 0),
    cs.horzcat(0, 0, 0, 1)
)
rytmat_fn = cs.Function('rytmat_fn', [beta], [rytmat])

theta = cs.SX.sym('theta', 1)
rztmat = cs.vertcat(
    cs.horzcat(cs.cos(theta), -cs.sin(theta), 0, 0),
    cs.horzcat(cs.sin(theta), cs.cos(theta), 0, 0),
    cs.horzcat(0, 0, 1, 0),
    cs.horzcat(0, 0, 0, 1),
)
rztmat_fn = cs.Function('rztmat_fn', [theta], [rztmat])

# ----------------
# quaterion to homogeneous transformation mat (casadi_fn)
q = cs.SX.sym('q', 4)
dcm = cs.vertcat(
    cs.horzcat(
        1 - 2 * (q[2] ** 2 + q[3] ** 2),
        2 * (q[1] * q[2] - q[0] * q[3]),
        2 * (q[1] * q[3] + q[0] * q[2]),
    ),
    cs.horzcat(
        2 * (q[1] * q[2] + q[0] * q[3]),
        1 - 2 * (q[1] ** 2 + q[3] ** 2),
        2 * (q[2] * q[3] - q[0] * q[1]),
    ),
    cs.horzcat(
        2 * (q[1] * q[3] - q[0] * q[2]),
        2 * (q[2] * q[3] + q[0] * q[1]),
        1 - 2 * (q[1] ** 2 + q[2] ** 2),
    ),
)
quattmat = cs.vertcat(
    cs.horzcat(dcm, cs.vertcat(0, 0, 0)),
    cs.horzcat(0, 0, 0, 1)
)
quattmat_fn = cs.Function('quattmat_fn', [q], [quattmat])
