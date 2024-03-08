import numpy as np 
from lib.calcJacobian import calcJacobian



def IK_velocity(q_in, v_in, omega_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
         are infeasible, then dq should minimize the least squares error. If v_in
         and omega_in have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """

    dq = np.zeros((1, 7))

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
    # using, np.linalg.lstsq, we solve for cases where rank of A not a full rank
    # matrix i.e. in this case task space != DOF of robot
    # For NO solutions, we find the solution by minimizing the ||b-a@x||^2
    # For INFINITE solutions, we find the solution by minimizing ||x||^2
    dq = np.linalg.lstsq(J_updated, velocity_updated)[0]

    # return dq after changing the shape to 1 X 7 vector

    return dq.flatten()
