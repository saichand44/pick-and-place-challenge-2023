import numpy as np


def calcAngDiff(R_des, R_curr):
    """
    Helper function for the End Effector Orientation Task. Computes the axis of rotation 
    from the current orientation to the target orientation

    This data can also be interpreted as an end effector velocity which will
    bring the end effector closer to the target orientation.

    INPUTS:
    R_des - 3x3 numpy array representing the desired orientation from
    end effector to world
    R_curr - 3x3 numpy array representing the "current" end effector orientation

    OUTPUTS:
    omega - 0x3 a 3-element numpy array containing the axis of the rotation from
    the current frame to the end effector frame. The magnitude of this vector
    must be sin(angle), where angle is the angle of rotation around this axis
    """
    omega = np.zeros(3)

    def _get_omega(R):
        '''
        This function computes the angular velocity vector from the given matrix R
        '''
        # initialize the skew-symmeric matrix
        S  = np.zeros((3, 3))

        # calculate the skew symmejointstric matrix from the given rotation matrix
        S = (1/2) * (R - np.transpose(R))

        # extract the angular velocity components from S
        omegaX = S[2, 1]
        omegaY = S[0, 2]
        omegaZ = S[1, 0]
        omega = np.array([omegaX, omegaY, omegaZ])

        return omega

    # compute the relative rotation matrix between the current and target frame
    #  {R_2}^0 =  {R_1}^0 *  {R_2}^1
    # 1 --> current frame, 2 --> target / desired frame
    R_rel = np.matmul(np.transpose(R_curr), R_des)

    #===========================================================================
    # compute the vector
    #===========================================================================
    # compute the relative angular velocity w.r.t world frame
    omega_rel = _get_omega(R_rel)
    omega_rel = np.matmul(R_curr, omega_rel)

    #===========================================================================
    # compute the magnitude
    #===========================================================================
    # compute the angle
    cos_angle = (np.trace(R_rel) - 1) / 2

    # check the bounds of cos inverse and limit the value such that
    # cos_angle = 1 for all values > 1
    # cos_angle = -1 for all values < -1
    cos_angle = max(-1, min(cos_angle, 1))
    
    angle = np.arccos(cos_angle)

    # check for edge case when omega is a zero vector
    if np.linalg.norm(omega_rel) != 0:
        omega = (omega_rel / np.linalg.norm(omega_rel)) * np.abs(np.sin(angle))
    else:
        omega = np.array([0, 0, 0])
        
    omega = omega.flatten()

    return omega