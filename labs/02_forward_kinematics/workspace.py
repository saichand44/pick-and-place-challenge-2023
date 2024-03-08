from lib.calculateFK import FK
from core.interfaces import ArmController

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

fk = FK()

# the dictionary below contains the data returned by calling arm.joint_limits()
limits = [
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -1.7628, 'upper': 1.7628},
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -3.0718, 'upper': -0.0698},
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -0.0175, 'upper': 3.7525},
    {'lower': -2.8973, 'upper': 2.8973}
        ]

# initialize an empty list to store the configuration space of panda robot
config_space = []

# initialize the paramater to evenly divide the values between the joint limits
size = 4

config_space = [np.array([q0, q1, q2, q3, q4, q5, q6])
                for q0 in np.linspace(limits[0]['lower'],limits[0]['upper'],size)
                for q1 in np.linspace(limits[1]['lower'],limits[1]['upper'],size)
                for q2 in np.linspace(limits[2]['lower'],limits[2]['upper'],size)
                for q3 in np.linspace(limits[3]['lower'],limits[3]['upper'],size)
                for q4 in np.linspace(limits[4]['lower'],limits[4]['upper'],size)
                for q5 in np.linspace(limits[5]['lower'],limits[5]['upper'],size)
                for q6 in np.linspace(limits[6]['lower'],limits[6]['upper'],size)
                ]

# initialize an array to store the end effector coordinates
end_effector_coor = np.zeros((len(config_space),3))

for i in range(len(config_space)):
    _, T0e = fk.forward(config_space[i])
    end_effector_coor[i] = T0e[:3, -1]

# calculate the range for x and y axes
x_min, x_max = np.min(end_effector_coor[:,0]), np.max(end_effector_coor[:,0])
y_min, y_max = np.min(end_effector_coor[:,1]), np.max(end_effector_coor[:,1])

# create a 3D plot
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

# set the x and y axis limits
ax1.set_xlim(1.5*x_min, 1.5*x_max)
ax1.set_ylim(1.5*y_min, 1.5*y_max)

ax1.scatter(end_effector_coor[:,0], 
            end_effector_coor[:,1], 
            end_effector_coor[:,2], 
            c='seagreen', marker='o', alpha=0.2)

# label the axes
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

plt.show()
