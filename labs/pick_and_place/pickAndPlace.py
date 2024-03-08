import numpy as np
from math import pi
from copy import deepcopy
import time  # Add this import statement

# for importing codes from lib folder
from lib.calculateFK import FK
from lib.IK_position_null import IK

ik = IK()
fk = FK()

class pickBlocks():
    '''
    '''
    def __init__(self, arm, detector, team):
        '''
        '''
        self.arm = arm
        self.detector = detector

        self.gripperPosTol   = 0.01
        self.gripperAlignTol = 0.01
        self.gripperClosePos = 0.04
        self.gripperForce    = 100
        self.blockSize       = 0.05
        self.blocksTransfered= 0

        if team == "red":
            # seeds for gradeint descent
            # for static blocks pickup
            self.qScanPose =  np.array([-0.05264077, 0.21943476, -0.26324824 ,
                                        -1.0763559 , 0.05900521, 1.28892532  ,
                                        2.0627702])
            self.qPrePickupPose1 = np.array([-0.14247908,  0.09731655, -0.16026504,
                                              -1.50608518,  0.01551351,  1.60215856,
                                               0.48291301])
            self.qPrePickupPose2 = np.array([-0.12878095,  0.14354131, -0.17021194,
                                             -1.96255347,  0.02813877,  2.10387491,
                                             .47382013])

            # for dynamic blocks pickup
            self.intermediatePose1 = np.array([0, -pi/4,  0.01978261, -3*pi/4, 0, pi/2, pi/4])
            self.intermediatePose2 = np.array([pi/2, -pi/4,  0.01978261, -3*pi/4, 0, pi/2, pi/4])
            self.prepPose = np.array([pi/4, pi/2, pi/2.1, -pi/8, -pi/8, pi/1.85, 3*pi/4-pi])
            self.fishPose = np.array([pi/4.25, pi/2, pi/2.1, -pi/3, -pi/8, pi/1.85, 3*pi/4-pi])
            self.DropPose = np.array([-0.12878095,  0.14354131, -0.17021194, 
                                      -1.96255347,  0.02813877,  2.10387491, .47382013])
            
            self.qPickUpSeed = []
            self.qPlaceSeed = [
            np.array([0.21581864, 0.14197394, 0.07948138, -1.96261353, -0.01304672, 2.10410797, 1.08653487]),
            np.array([0.1951742, 0.10210168, 0.10074635, -1.88333672, -0.01119784, 1.98490115, 1.08530339]),
            np.array([0.18421238,0.08006109,0.11207241,-1.78297707,-0.00933903,1.86252601, 1.08401301]),
            np.array([0.18297452,0.07735149,0.11415856,-1.65901614,-0.00892379,1.73585968, 1.08365919]),
            np.array([0.19258167,0.09666224,0.1065684,-1.5060993,-0.0102709,1.6022149, 1.08437714]),
            np.array([0.21573686,0.14409544,0.08650826,-1.31204593,-0.01248994,1.45561882, 1.08531567]),
            np.array([0.25668879,  0.23861266,  0.04606021, -1.04073551, -0.01136286, 1.27912238,  1.08357631]), # z=0.6
            np.array([-0.60441668, -0.42691443, 0.97327589, -2.22400281, 0.83523183, 3.55740389, 0.36158587]), # z=0.65
            np.array([-0.71868538, -0.54945537, 0.95110348, -2.16269818, 0.6592782, 3.37242764, 0.58287085]), # z=0.7
            ]

            # important poses
            self.placePose = np.array([[ 1.,     0.,    0.,    0.562],
                                       [ 0.,    -1.,    0.,    0.169],
                                       [ 0.,     0.,   -1.,    0.225],
                                       [ 0.,     0.,    0.,       1.]])
        else:
            # for static blocks pickup
            self.qScanPose = np.array([ 0.11175626, -0.10132531,  0.20100749,
                                        -1.74262799,  0.02025028,  1.64331442,
                                        1.09569086])

            self.qPrePickupPose1 = np.array(
                [0.19244415, 0.09656078, 0.10653278, -1.50622922, -0.01025674,
                    1.60224429, 1.08420506])
            self.qPrePickupPose2 = np.array(
                [0.21581864, 0.14197394, 0.07948138, -1.96261353,
                    -0.01304672, 2.10410797, 1.08653487])

            # for dynamic blocks pickup# for dynamic blocks pickup
            self.intermediatePose1 = np.array([0, -pi/4,  0.01978261, -3*pi/4, 0, pi/2, pi/4])
            self.intermediatePose2 = np.array([-pi/2, -pi/4,  0.01978261, -3*pi/4, 0, pi/2, pi/4])
            self.prepPose = np.array([-pi+pi/4, pi/2, pi/2.1, -pi/8, -pi/8, pi/1.85, 3*pi/4-pi])
            self.fishPose = np.array([-pi+pi/4.25, pi/2, pi/2.1, -pi/3, -pi/8, pi/1.85, 3*pi/4-pi])
            self.DropPose = np.array([0.21581864, 0.14197394, 0.07948138, -1.96261353, -0.01304672, 2.10410797, 1.08653487])

            self.qPickUpSeed = []
            self.qPlaceSeed = [
                np.array([-0.12474943, 0.17040306, -0.17450034, -1.99462759, 0.03547264,  2.16220022, 0.46884318]),  # z = 0.3
                np.array([-0.1349553, 0.10292467, -0.16336702, -1.8833162, 0.01825342, 1.98481856, 0.48058127]),  # z = 0.35
                np.array([-0.13872009, 0.0805614, -0.15935046, -1.78296793, 0.013333, 1.86248851, 0.48400135]),  # z = 0.40
                np.array([-0.14075522, 0.07780173, -0.15834905, -1.65900845, 0.01242517, 1.73582814, 0.48472361]),  # z = 0.45
                np.array([-0.14254567, 0.0973155, -0.16019364, -1.50608521, 0.01550649, 1.60215865, 0.48291771]),  # z = 0.50
                np.array([-0.14786998, 0.14533548, -0.16350073, -1.3120031, 0.02373343, 1.45545916,  0.47845288]),  # z = 0.55
                np.array([-0.16809721, 0.24084678, -0.16154061, -1.04057937,   0.04007258, 1.27862972, 0.47189338]), # z=0.6
                np.array([0.77914121, -0.65515323, -0.88583219, -2.35660407,  0.02295238, 3.47908044, 0.27233402]), # z=0.65
                np.array([0.72154166, -0.62754314, -0.77613096, -2.24484412, 0.20911296, 3.36328108, 0.15688971]), # z=0.7
            ]

            # important poses
            self.placePose = np.array([[ 1.,     0.,    0.,    0.562],
                                       [ 0.,    -1.,    0.,   -0.169],
                                       [ 0.,     0.,   -1.,    0.225],
                                       [ 0.,     0.,    0.,       1.]])
            
        self.qLiftBase = []
        self.pickPose = []

    def getJointConfig(self, qSeed, targetPose, stepSize=0.5):
        '''
        Compute the joint configuration for a given pose
        '''        
        _, pose = ik.fk.forward(qSeed)
        d, ang = IK.distance_and_angle(targetPose, pose)
        
        if np.linalg.norm(d) > self.gripperPosTol or ang > self.gripperAlignTol:
            q, _, success, _ = ik.inverse(targetPose, qSeed, method='J_pseudo', alpha=stepSize)
            return q
        elif np.linalg.norm(d) < self.gripperPosTol and ang < self.gripperAlignTol:
            return qSeed

    def go2PosePositionControl(self, q):
        '''
        Moves the robot to a pose with position control
        '''
        self.arm.safe_move_to_position(q)

    def go2PoseVeclocityControl(self):
        '''
        Moves the robot to a pose with position control
        '''
        pass
    
    def getDetectionsInfo(self, H_ee_camera, T0e=None, info="NO"):
        '''
        Get the camera detection, convert to wrt robot base
        '''
        # initialize a list to capture the detections
        self.detections = []
        
        # get the end effector pose wrt robot base frame
        if T0e is None:
            _, T0e = self.getState("joints", type="positions")

        # multiply with T0e, camera transformations to get blocks' pose wrt robot
        for (name, pose) in self.detector.get_detections():
            pose = np.matmul(H_ee_camera, pose)
            pose = np.matmul(T0e, pose)
            self.detections.append(pose)

            # if YES, print the transformations of the blocks
            if info == "YES":
                print(name+"_wrt_robot_base",'\n',pose)
    
    def pickUpBlock(self, blockID=0):
        '''
        Go to pick up pose, lower robot, close the gripper and move in z direction
        to a safe pose before moving to destination platform
        '''

        # set the desired Z orientation of the end effector (just parallel not exact)
        endEffZAxisRef = np.array([0., 0., 1.])

        # get the pose of the block
        blockPose = self.detections[blockID]

        # find out which column in the block pose is the axis parallel to world Z axis
        for i in range(4):
            if np.array_equal(np.abs(np.round(blockPose[:3, i])), endEffZAxisRef):
                break

        dummy = deepcopy(blockPose)
        dummy4Axis = np.delete(dummy, i, axis=1)
        
        newEndEffAxis1 = [dummy4Axis[0, 0], dummy4Axis[1, 0], 
                          dummy4Axis[2, 0], dummy4Axis[3, 0]]

        newEndEffAxis2 = [dummy4Axis[0, 1], dummy4Axis[1, 1], 
                          dummy4Axis[2, 1], dummy4Axis[3, 1]]

        robotPosePick = np.array([[ 1.,     0.,    0.,    0.],
                                  [ 0.,    -1.,    0.,    0.],
                                  [ 0.,     0.,   -1.,    0.],
                                  [ 0.,     0.,    0.,    0.]])

        # update the rotation part of the robotPose
        axis1dot = np.dot(robotPosePick[:3, 0], newEndEffAxis1[:3])
        axis2dot = np.dot(robotPosePick[:3, 0], newEndEffAxis2[:3])

        if axis1dot < axis2dot:
            if axis1dot>0:
                robotPosePick[:, 0] = newEndEffAxis1
            else:
                robotPosePick[:, 0] = [-value for value in newEndEffAxis1]
        else:
            if axis2dot>0:
                robotPosePick[:, 0] = newEndEffAxis2
            else:
                robotPosePick[:, 0] = [-value for value in newEndEffAxis2]

        robotPosePick[:3, 1] = np.cross(robotPosePick[:3, 2], robotPosePick[:3, 0])  

        # update the translation part of the robotPose
        robotPosePick[:3, -1] = [blockPose[0, -1], 
                                 blockPose[1, -1], 
                                 blockPose[2, -1]-0.01]
        
        robotPosePrePick = deepcopy(robotPosePick)
        robotPosePrePick[2, -1] += self.blockSize

        # move to a pre pick-up pose
        qPick1 = self.getJointConfig(self.qPrePickupPose2, robotPosePrePick, stepSize=0.5)

        # print(blockID)
        # print(qPick1)
        # _, pose = fk.forward(qPick1)
        # print(pose)
        # d, ang = IK.distance_and_angle(robotPosePrePick, pose)

        # if d > self.gripperPosTol or ang > self.gripperAlignTol:
        #     qPick1 = self.getJointConfig(qPick1, robotPosePrePick, stepSize=0.5)
        
        self.go2PosePositionControl(qPick1)
        qPick2 = self.getJointConfig(qPick1, robotPosePick, stepSize=0.5)
        
        # move and catch the block
        self.go2PosePositionControl(qPick2)
        self.gripperAction("catch", self.gripperClosePos)
        self.qPickUpSeed.append(qPick2)
        
        self.go2PosePositionControl(self.qPrePickupPose2)
        self.go2PosePositionControl(self.qPlaceSeed[self.blocksTransfered])
        # print(blockID)
        # print(self.getState(item="joints", type="angles"))

    def fishBlock(self):
        '''
        Repeatedly close and open grippers until a block is grabbed
        '''
        # move to a pre-fish pose
        # self.go2PosePositionControl(self.intermediatePose1)
        self.go2PosePositionControl(self.intermediatePose2)
        self.go2PosePositionControl(self.prepPose)

        # move into turntable area
        self.go2PosePositionControl(self.fishPose)
        # print(fk.forward(self.fishPose)[1])

        # wait until a block is caught
        count = 0
        block_grabbed = False
        while block_grabbed == False:
            newGripperClosePos = self.gripperClosePos - 0.01
            # time.sleep(5)       # TODO need to be fine tuned
            self.gripperAction("catch", newGripperClosePos)
            time.sleep(5)       # TODO need to be fine tuned
            print(self.getState("gripper")['position'])
            if self.getState("gripper")['position'][0]+ self.getState("gripper")['position'][1] > 1.2*newGripperClosePos:
            # if self.getState("gripper")['position'][0] >  0.02 and self.getState("gripper")['position'][1] > 0.02:
                block_grabbed = True
            else:
                self.gripperAction("open")
                print(self.getState("gripper")['position'])
                self.go2PosePositionControl(self.prepPose)
                self.go2PosePositionControl(self.fishPose)                         

        self.go2PosePositionControl(self.prepPose)
        self.go2PosePositionControl(self.intermediatePose2)
        self.go2PosePositionControl(self.intermediatePose1)

    def dropBlock(self):
        self.go2PosePositionControl(self.DropPose)
        self.gripperAction("open")

    def putDownBlock(self):
        '''
        Reach the destination platform and stack the blocks
        '''
        robotPosePlace = deepcopy(self.placePose)
        robotPosePlace[2, -1] += self.blockSize*(self.blocksTransfered)+0.01

        # calculate the joint configuration to place the block
        qPlace =  self.getJointConfig(self.qPlaceSeed[self.blocksTransfered], 
                                        robotPosePlace, stepSize=0.5)
        
        # go to the computed joint configuration to place the block
        self.go2PosePositionControl(qPlace)
        self.gripperAction("open")
        self.qPlaceSeed.append(qPlace)

        # go to a pre-defined position above latest placed block
        self.go2PosePositionControl(self.qPlaceSeed[self.blocksTransfered])

        # update the number of blocks stacked
        self.blocksTransfered+=1

    def gripperAction(self, action, distance=None):
        '''
        Open or Close the gripper --> open, close, catch
        '''
        if action == "open":
            self.arm.open_gripper()
        elif action == "close":
            self.arm.close_gripper()
        elif action == "catch":
            self.arm.exec_gripper_cmd(distance)

    def getState(self, item, type=None):
        '''
        Get the state of the gripper or robot
        '''
        if item == "gripper":
            return self.arm.get_gripper_state()
        elif item == "joints" and type=="angles":
            return self.arm.get_positions()
        elif item == "joints" and type=="positions":
            return fk.forward(self.arm.get_positions())
