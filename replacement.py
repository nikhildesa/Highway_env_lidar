# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 15:54:21 2020

@author: nikhi
"""
import math
rad = 25
center = (0,0)
x_line_t = []
y_line_t = []
mydegrees = 0
count = 0
while (count < 180):
    x_line_t.append(rad * math.cos(math.radians(mydegrees)) + center[0])
    y_line_t.append(rad * math.sin(math.radians(mydegrees)) + center[1])
    mydegrees+=2
    count+=1
    
"""
        
        for i in range(mydegrees,mydegrees+360,2):
            x_line_t.append(rad * math.cos(math.radians(i)) + center[0])
            y_line_t.append(rad * math.sin(math.radians(i)) + center[1])
        
        endpoints = list(zip(x_line_t,y_line_t))
"""