import numpy as np
import random
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from copy import deepcopy
from lib.calculateFK import FK
from lib.tree import Tree
from lib.tree import Node

def safeMarginObstacles(map, padding_dim = 0.1):
    '''
    Function to add safe margin to the obstacles' boundary dimensions
    :param map           :       the map struct
    :param padding_dim   :       padding dimension for boundaries
    :return              :       obstacles with padded dimensions
    '''
    obstacles = map.obstacles
    buffer = np.array([-1, -1, -1, 1, 1, 1])
    paddedObstacles = obstacles + buffer*padding_dim

    return paddedObstacles

def isRobotCollided(map, q):
    '''
    Function to check if a given robot configuration collides with the obstacles
    :param q             :       robot configuration
    :return              :       True, if robot collides. False, otherwise
    '''
    # instantiate the class for forward kinematics
    fk = FK()

    # get the joint positions, end effector position of the robot
    jointPositions, _ = fk.forward(q)

    # get the information about the obstacles
    if len(map.obstacles)!=0:
        obstacles = safeMarginObstacles(map)   # add safe margin to obstacles

        # loop through the obstacles to check if the robot collides with any of them
        for obstacle in obstacles:
            linePt1 = np.array([jointPositions[i] for i in range(0, len(jointPositions)-1)])
            linePt2 = np.array([jointPositions[i] for i in range(1, len(jointPositions))])
            checkCollision = detectCollision(linePt1, linePt2, obstacle)

            if True in checkCollision:
                return True
    
    # if the robot doesn't collide
    return False

def isPathCollided(map, qCurr, qNext):
    '''
    Fucntion to check if the robot collides with obstacles when moving from current
    configuration to the next configuration
    :param qCurr         :       current robot configuration
    :param qNext         :       next robot configuration
    :return              :       True, if robot collides. False, otherwise
    '''
    # set the step size
    steps = 50

    # get the direction of the progression from qCurr to qNext
    direction = (qCurr - qNext) / steps
        
    for i in range(steps):
        # calculate the intermediate configuration
        q = qNext + (i+1) * direction

        # check if the imtermediate configuration collides with the obstacles
        if isRobotCollided(map, q):
            return True

    return False

def rrt(map, start, goal):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (0x7).
    :param goal:        goal pose of the robot (0x7).
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """

    # initialize path
    path = []

    # get joint limits
    lowerLim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upperLim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    # check if start and goal are safe i.e no collision with obstacles, within joint limits
    if np.any(start < lowerLim) or np.any(start > upperLim) or isRobotCollided(map, start):
        return np.array(path)
    if np.any(goal < lowerLim) or np.any(goal > upperLim) or isRobotCollided(map, goal):
        return np.array(path)
    
    # set the maximum numner of iterations
    maxIterations = 1000

    # intiiate two trees, one from start config and another from goal config
    TStart = Tree(Node(start, 0))
    TGoal = Tree(Node(goal, 0))

    # set the random seed for reproducibility
    np.random.seed(42)

    currIter = 0
    while currIter <= maxIterations:
        qRand = np.random.uniform(low=lowerLim, high=upperLim)
        # if qRand collides with obstacles, look for a new qRand
        if isRobotCollided(map, qRand):
            continue
        
        # find the nearest node in both the trees to qRand
        nearestNeighbor_TStart, nnStartIdx = TStart.nearestNeighbour(qRand)
        nearestNeighbor_TGoal, nnGoalIdx  = TGoal.nearestNeighbour(qRand)

        # add qRand to start tree 
        if not isPathCollided(map, nearestNeighbor_TStart, qRand):
            newNode = Node(qRand, nnStartIdx)
            TStart.addNode(newNode)

        # add qRand to goal tree
        if not isPathCollided(map, nearestNeighbor_TGoal, qRand):
            newNode = Node(qRand, nnGoalIdx)
            TGoal.addNode(newNode)
        
        # check if qRand is present in both trees => both trees are connected
        if not isPathCollided(map, nearestNeighbor_TStart, qRand) and not isPathCollided(map, nearestNeighbor_TGoal, qRand):
            break
            
        currIter += 1
        if currIter == maxIterations:
            print("[INFO]: No start/goal node connection found")
            return np.array(path)

    # generate the path by tracing the start and goal trees
    latestNode = TStart.nodes[-1]
    while not np.any(latestNode.q == start):
        path.append(latestNode.q)
        latestNode = TStart.nodes[latestNode.parentID]
    path.append(start)

    # reverse the elements order so that "start" is the first element
    path = path[::-1]

    latestNode = TGoal.nodes[-1]
    while not np.any(latestNode.q == goal):
        latestNode = TGoal.nodes[latestNode.parentID]
        path.append(latestNode.q)

    return np.array(path)

if __name__ == '__main__':
    map_struct = loadmap("../maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))