import numpy as np

class Node():
    '''
    Class to store the joint angles of a configuration "q" and also the ID of the
    parent from which that configuration has emerged
    '''
    def __init__(self, q, parentID):
        self.q = q
        self.parentID = parentID

class Tree():
    '''
    Class to generate a tree for RRT
    '''
    def __init__(self, qStartNode):
        '''
        This initiates the tree with the root node as "qStartNode"
        '''
        self.qStartNode = qStartNode

        # initialize an empty list to store the permissible nodes of RRT
        self.nodes = []
        if len(self.nodes) == 0:
            self.nodes.append(self.qStartNode)

    def distance(self, qCurrent, qTarget):
        '''
        Find the distance norm between two robot configurations
        '''
        return np.linalg.norm(qTarget - qCurrent)

    def nearestNeighbour(self, q):
        '''
        Find the nearest neighbor node in the tree to the given q (random config.).
        '''
        # initialize nearest neighbor and its distance
        nearestNode = None
        minDistance = None

        for i, node in enumerate(self.nodes):
            currDistance = self.distance(node.q, q)

            # assign the nearest node and the min distance for the first time
            if minDistance is None and nearestNode is None:
                nearestNode = node
                minDistance = currDistance
                nodeIndex = i
            
            # update the nearest node and the min distance if a further nearest
            # node is found
            if currDistance < minDistance:
                nearestNode = node
                minDistance = currDistance
                nodeIndex = i

        return nearestNode.q, nodeIndex

    def addNode(self, newNode):
        '''
        Add a permissible / allowable node to the tree
        '''
        self.nodes.append(newNode)
