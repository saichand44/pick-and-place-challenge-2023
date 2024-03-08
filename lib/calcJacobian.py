import numpy as np
from lib.calculateFK import FK

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))

    # initialize the class FK
    fk = FK()

    # get the required parameters from forward kinematics
    Ai_list = fk.compute_Ai(q_in)            # list containing Ai matrices
    z_i = fk.get_axis_of_rotation(q_in)      # axis of rotation of each joint
    joint_positions, T0e = fk.forward(q_in)  

    #===========================================================================
    # get linear velocity jacobian
    #===========================================================================
    # linear velocity jacobian matrix:
    # Jv_{i} = Z^0_{i-1} x (O_{n} - O_{i})

    # coordinates of the end effector in world frame
    On = T0e[:3, -1]

    for i in range(len(fk.joint_info)):

        # get the position of the (i+1)th joint
        O_joint = joint_positions[i]
        
        z_axis = z_i[:, i]

        # calculate the columns in the linear velocity jacobian matrix
        cross_prod = np.cross(z_axis,On-O_joint)

        J[:3, i] = cross_prod

    #===========================================================================
    # get angular velocity jacobian
    #===========================================================================
    # linear velocity jacobian matrix:
    # Jw_{i} = Z^0_{i-1}

    for i in range(len(fk.joint_info)):
        J[3:, i] = z_i[:, i]

    return J

if __name__ == '__main__':
    # q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    # q= np.array([0, 0, 0, 0, 0, 0, 0])
    q= np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    print(np.round(calcJacobian(q),3))
    print(np.linalg.matrix_rank(np.round(calcJacobian(q),3)))
