import numpy as np
from lib.calcJacobian import calcJacobian

def calcManipulability(q_in):
    """
    Helper function for calculating manipulability ellipsoid and index

    INPUTS:
    q_in - 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]

    OUTPUTS:
    mu - a float scalar, manipulability index
    M  - 3 x 3 manipulability matrix for the linear portion
    """
    J = calcJacobian(q_in)

    J_pos = J[:3,:]
    M = J_pos @ J_pos.T

    # using the equation manipulability index = sqrt(det(J * J.T))
    # we can use svd to get the singular values of J_pos 
    # (reference: Paper by Kevin Dufour, Wael Suleiman)
    U, S, Vt = np.linalg.svd(J_pos)
    mu = np.prod(S)

    return mu, M