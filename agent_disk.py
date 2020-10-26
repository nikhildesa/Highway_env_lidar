# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 18:06:40 2020

@author: nikhi
"""
import numpy as np
import math
disc_collection = []


l = 5   # length of vehicle
w = 2   # width of vehicle

X = 2.5  #center X
Y = 1  #center Y

yaw = 0 # angle with horizontal lane

num = int(np.ceil(l/w)-1)    # the total number of discs will be 2*num + 1 = 5
s1 = np.array(range(-num,num+1)) 

x_circles = X*np.ones(num*2+1) + s1*w/2*math.cos(math.radians(yaw))    # the x coordinates of the discs
y_circles = Y*np.ones(num*2+1) + s1*w/2*math.sin(math.radians(yaw))    # the y-coordinates of the discs

disc = list(zip(x_circles,y_circles))
disc_collection.append(disc)

import gym
import highway_env
env = gym.make('highway-v0')
env.config
config = {
"observation": {
    "offscreen_rendering": True,
    "type": "Kinematics",
    'action': {'type': 'DiscreteMetaAction'},
     'simulation_frequency': 15,
     'policy_frequency': 1,
     'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
     'screen_width': 600,
     'screen_height': 150,
     'centering_position': [0.3, 0.5],
     'scaling': 5.5,
     'show_trajectories': False,
     'render_agent': True,
     "features": ["presence", "x", "y", "vx", "vy","cos_h","sin_h"],  # changed added
     'manual_control': False,
     'lanes_count': 4,
     'vehicles_count': 50,
     'duration': 40,
     'initial_spacing': 2,
     'collision_reward': -1,
     'offroad_terminal': False,
     'normalize': False,   # changed added
     "absolute": True      # changed added
                }

        }

env.configure(config)
env.reset()
for i in range(100):
    done = False
    while not done:
        observation_, reward, done, info = env.step(env.action_space.sample())
        env.render()