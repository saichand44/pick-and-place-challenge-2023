import numpy as np
from lib.IK_velocity import IK_velocity
from lib.calcJacobian import calcJacobian


def IK_velocity_null(q_in, v_in, omega_in, b):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :param b: 7 x 1 Secondary task joint velocity vector
    :return:
    dq + null - 1 x 7 vector corresponding to the joint velocities + secondary task null velocities
    """

    dq = np.zeros((1, 7))
    null = np.zeros((1, 7))
    b = b.reshape((7, 1))
    v_in = np.array(v_in)
    v_in = v_in.reshape((3,1))
    omega_in = np.array(omega_in)
    omega_in = omega_in.reshape((3,1))

    # check if any coordinate of linear or angular velocity is unconstrained/np.nan
    # unconstrained => can take any values for that particular coordinate
    nan_indices_v = np.argwhere(np.isnan(v_in.flatten()))

    nan_indices_w = np.argwhere(np.isnan(omega_in).flatten())
    nan_indices_w += 3                # since the omega values start from index 3
        
    v_in = v_in.reshape((3,1))
    omega_in = omega_in.reshape((3,1))

    # compute the Jacobian
    J = calcJacobian(q_in)
    
    # initialize a list to store the indices where the entires of end effector
    # velocity is np.nan
    indices2remove = []
    indices2remove.extend(nan_indices_v)    # add indices related to v_in
    indices2remove.extend(nan_indices_w)    # add indices related to omega_in

    # modify the Jacobian by removing the rows that correspond to np.nan in the
    # end effector velocity
    J_updated = np.delete(J, indices2remove, axis=0)

    # initialize the desired velocity of the end effector
    velocity = np.vstack((v_in, omega_in))

    # modify the velocity by removing the rows that correspond to np.nan in the
    # end effector velocity
    velocity_updated = np.delete(velocity, indices2remove, axis=0)

    # calculate the joint velocities
    # joint_velocities = pseudo inverse of J * end effector velocity 
    dq = np.matmul(np.linalg.pinv(J_updated), velocity_updated)
    dq = dq.flatten()

    # add the null space velocity to the existing velocity
    null = np.matmul((np.eye(7)-np.matmul(np.linalg.pinv(J_updated),J_updated)),b)
    null = null.flatten()

    return dq + null

