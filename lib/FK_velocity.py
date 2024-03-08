import numpy as np 
from lib.calcJacobian import calcJacobian

def FK_velocity(q_in, dq):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param dq: 1 x 7 vector corresponding to the joint velocities.
    :return:
    velocity - 6 x 1 vector corresponding to the end effector velocities.    
    """


    velocity = np.zeros((6, 1))

    # compute the Jacobian
    J = calcJacobian(q_in)

    # compute the velocity
    velocity = np.matmul(J, np.transpose(dq))

    return velocity
