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

