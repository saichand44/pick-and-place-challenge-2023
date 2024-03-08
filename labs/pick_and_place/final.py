import sys
import numpy as np
from copy import deepcopy
from math import pi

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

# for importing codes from lib folder
from lib.calculateFK import FK
from lib.IK_position_null import IK
from labs.final.pickAndPlace import pickBlocks

ik = IK()
fk = FK()

if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()

    arm.open_gripper()
    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_position) # on your mark!

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    # STUDENT CODE HERE

    # get the transform from camera to panda_end_effector
    H_ee_camera = detector.get_H_ee_camera()

    pick = pickBlocks(arm, detector, team)

    #################
    # STATIC BLOCKS #
    #################
    pick.go2PosePositionControl(pick.qScanPose)
    pick.getDetectionsInfo(H_ee_camera, info="YES")

    for i in range(len(pick.detections)):
        print("loop", i)
        print("Picking...")
        pick.pickUpBlock(blockID=pick.blocksTransfered)
        pick.putDownBlock()

    ##################
    # DYNAMIC BLOCKS #
    ##################
    print("Switching to Dynamic Blocks")
    while True:
        pick.fishBlock()
        pick.dropBlock()
        pick.go2PosePositionControl(pick.qScanPose)
        pick.getDetectionsInfo(H_ee_camera, info="YES")
        if (len(pick.detections) == 1):
            for i in range(len(pick.detections)):
                print("Picking...")
                # print(pick.blocksTransfered)
                if pick.blocksTransfered < len(pick.qPlaceSeed):
                    pick.pickUpBlock(blockID=i)
                    # pick.go2PosePositionControl(pick.qPlaceSeed[pick.blocksTransfered]) #FOR TESTING
                    pick.putDownBlock()
                else:
                    break
    # # END STUDENT CODE