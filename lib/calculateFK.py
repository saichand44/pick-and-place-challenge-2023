import numpy as np
from math import pi

class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout

        pass
    
    def _joint_dict(self, q):
        '''
        Create a dictionary containing joint information.

        Inputs
        ------
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]
        '''
        # create a dictionary conataining the joint information
        self.joint_info = {#"joint_name": ["frame_no.", --> joint defined in this frame
                           #                 x_off,     --> x offset of joint wrt frame
                           #                 y_off,     --> y offset of joint wrt frame
                           #                 z_off,     --> z offset of joint wrt frame
                           #                 theta_off, --> offset of joint angle in DH
                           #                 a,         --> a in DH convention
                           #                 alpha,     --> alpha in DH convention
                           #                 d,         --> d in DH convention
                           #                 theta      --> theta in DH convention
                           #              ]

            "joint_1" : ["frame_0", 0, 0, 0.141, 0,    0      , -pi/2, 0.333, q[0]],
            "joint_2" : ["frame_1", 0, 0, 0.000, 0,    0      ,  pi/2, 0    , q[1]],
            "joint_3" : ["frame_2", 0, 0, 0.195, 0,    0.0825 ,  pi/2, 0.316, q[2]],
            "joint_4" : ["frame_3", 0, 0, 0.000, 0,    -0.0825, -pi/2, 0    , q[3]],
            "joint_5" : ["frame_4", 0, 0, 0.125, 0,    0      ,  pi/2, 0.384, q[4]],
            "joint_6" : ["frame_5", 0, 0, -.015, 0,    0.088  ,  pi/2, 0    , q[5]],
            "joint_7" : ["frame_6", 0, 0, 0.051, -pi/4,    0  , 0    , 0.21 , q[6]],
                          }
# Link to the DH parameters convention:
# https://drive.google.com/file/d/1x7LrsrqWvBUMpzqUhth4MfYx7GKLYELz/view?usp=drive_link
    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the 
              FK of the robot. Transformations are not necessarily located at 
              the joint locations
        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        # update the joint information
        self._joint_dict(q)

        # numpy array containing the list of all Ai
        self.Ai_list = np.zeros((7,4,4))

        #=======================================================================
        # calculate Ai
        #=======================================================================
        for i in range(len(self.joint_info)):
            joint_name = "joint_%d"%(i+1)

            # initialize Ai with a 4X4 identity matrix
            Ai = np.identity(4)

            # get the DH paramaters
            a, alpha, d, theta = self.joint_info[joint_name][5:]
            theta += self.joint_info[joint_name][4]

            # populate the rows of Ai
            Ai[0] = [np.cos(theta), -1*np.sin(theta)*np.cos(alpha), 
                    np.sin(theta)*np.sin(alpha), a*np.cos(theta)]
            Ai[1] = [np.sin(theta), np.cos(theta)*np.cos(alpha),
                    -1*np.cos(theta)*np.sin(alpha), a*np.sin(theta)]
            Ai[2] = [0, np.sin(alpha), np.cos(alpha), d]

            self.Ai_list[i] = Ai

        return self.Ai_list

    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational 
                        joint of the robot or end effector
                        Each row contains the [x,y,z] coordinates in the world 
                        frame of the respective joint's center in meters.
                        The base of the robot is located at [0,0,0].

        T0e             -a 4 x 4 homogeneous transformation matrix,
                        representing the end effector frame expressed in the
                        world frame
        """

        # Your Lab 1 code starts here

        jointPositions = np.zeros((8,3))
        T0e = np.identity(4)

        # compute Ai
        _ = self.compute_Ai(q)

        #=======================================================================
        # calculate T0e
        #=======================================================================
        for i in range(len(self.joint_info)):
            T0e = np.matmul(T0e, self.Ai_list[i])

        #=======================================================================
        # get joint position
        #=======================================================================
        for i in range(len(self.joint_info)):
            # auxillary matrix to be multiplied to Ai for joint position correction
            A_aux = np.identity(4)

            # matrix containing the final positional information of the joint
            A_pos = np.identity(4)

            joint_name = "joint_%d"%(i+1)
            A_aux[:3, -1] = [offset for offset in self.joint_info[joint_name][1:4]]

            # special case for the position of joint 1
            if i == 0:
                A_pos = np.matmul(A_pos, A_aux)
            # position for rest of the joints
            else:
                for j in range(i):
                    A_pos = np.matmul(A_pos, self.Ai_list[j])
                A_pos = np.matmul(A_pos, A_aux)

            jointPositions[i] = A_pos[:3,-1]
        
        jointPositions[7] = T0e[:3, -1]

        # Your code ends here

        return jointPositions, T0e
    
    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis 
                                 of rotation for each joint in the world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        # numpy array containing the list of z axis of joints in the world frame
        z_i = np.zeros((3, len(q)))

        # coompute Ai
        Ai_list = self.compute_Ai(q) 

        for i in range(len(self.joint_info)):
            # initialize the matrix A_zaxis describing the rotation matrix of 
            # joint under consideration in the world frame
            A_zaxis = np.eye(4)

            for j in range(i):
                A_zaxis = np.matmul(A_zaxis, Ai_list[j])
            
            # extract the orientation of the z axis w.r.t base frame
            z_i[:, i] = A_zaxis[:3, -2]

        return z_i
    
if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    # q = np.array([ 0,    0,     0, 0,     0, 0, 0 ])
    q = np.array([0, -1, 0, -2, 0, 1.57, 0])

    joint_positions, T0e = fk.forward(q)
    
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)

