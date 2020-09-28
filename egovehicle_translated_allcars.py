# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 01:03:03 2020

@author: nikhi
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
import scipy
import math

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, 
            n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):   # The input here is a batch
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

class Agent():
    def __init__(self,max_mem_size,gamma, epsilon, lr, input_dims, batch_size, n_actions,lidar_input_dims,eps_end=0.05, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 20

        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                    fc1_dims=256, fc2_dims=256)
        #print(self.Q_eval)
        self.Q_next = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                    fc1_dims=256, fc2_dims=256)
        
        self.Q_next.load_state_dict(self.Q_eval.state_dict())
        self.Q_next.eval()


        self.state_memory = np.zeros((self.mem_size, 105), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, 105), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        
        self.lidar_memory = np.zeros((self.mem_size, *lidar_input_dims), dtype=np.float32)
        self.current_loss = []
    def line(self,p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C
    
    def intersection(self,L1, L2):
        D  = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x,y
        
    def display_lidar(self,pos):
        rectangle = []          # list stores point of each rectangle
        all_cars = []           # list of list stores all points of rectangle
        x_line_t = []             # list of x points of circle
        y_line_t = []             # list of y points of circle
        rectangle_line = []     # list store sides of each rectangle
        sides_of_rectangle = [] # list of list stores all sides of rectangle
        distance = [] # To store the length of all rays
        
        kinematics = pos
        kinematics[:,2] = 12 - kinematics[:,2]
        kinematics = kinematics[kinematics[:,0] == 1]  # consider only those vehicles whose presence is 1
    
        # My vehicle x,y
        center = (kinematics[0,1],kinematics[0,2])  # My vehicle x,y
    
       #Other vehicle x,y
        #kinematics = kinematics[1:,1:3]
        for i in range(len(kinematics)):
            x = kinematics[i][1]           # center x
            y = kinematics[i][2]           # center y
        
        
            # Need to find env.vehicle.position and env.vehicle.heading
        
            l = np.array([env.vehicle.LENGTH/2, 0])
            w = np.array([0, env.vehicle.WIDTH/2])
            points = np.array([- l - w, - l + w, + l - w, + l + w])
            c, s = np.cos(env.vehicle.heading), np.sin(env.vehicle.heading)
            R = np.array([[c, -s], [s, c]])
            rotated_points = R.dot(points.transpose()).transpose()
            translated_rotated_points = [env.vehicle.position + np.squeeze(p) for p in rotated_points]
        
            
            car_x_bottom_left = translated_rotated_points[0][0]  
            car_y_bottom_left = translated_rotated_points[0][1]
            
            car_x_bottom_right = translated_rotated_points[2][0]
            car_y_bottom_right = translated_rotated_points[2][1]
            
            car_x_top_right = translated_rotated_points[3][0]
            car_y_top_right = translated_rotated_points[3][1]
            
            car_x_top_left = translated_rotated_points[1][0]
            car_y_top_left = translated_rotated_points[1][1]
            
            rectangle = [[car_x_bottom_left,car_y_bottom_left],[car_x_bottom_right,car_y_bottom_right],[car_x_top_right,car_y_top_right],[car_x_top_left,car_y_top_left]]
            
            all_cars.append(rectangle)  # all_cars has points of all cars
            
        
        #------------------------------------------------------------------------------
        # using heading for angle
        car_angle = math.degrees(math.acos(env.vehicle.heading))  # if the car is going straight the angle is 90 degree
        mydegrees = car_angle - 90   # while taking left it is a 92 degrees, start the lidar from 2 degrees
        #------------------------------------------------------------------------------
        
        rad = 25
        count = 0
        while (count < 180):
            x_line_t.append(rad * math.cos(math.radians(mydegrees)) + center[0])
            y_line_t.append(rad * math.sin(math.radians(mydegrees)) + center[1])
            mydegrees+=2
            count+=1
        endpoints = list(zip(x_line_t,y_line_t))
        #<----------------------------------Making lines of cars---------------------------------------------------->
   
    
        for car in all_cars:
    
            L1 = self.line(car[0],car[1])
            L2 = self.line(car[1],car[2])
            L3 = self.line(car[2],car[3])
            L4 = self.line(car[3],car[0])
            rectangle_line = [L1,L2,L3,L4]  
            sides_of_rectangle.append(rectangle_line)
        
        
        for endpoint in endpoints:
            smallest_point = 0  
            intersect = []  # stores intersection point
            L = self.line(list(center),list(endpoint))         # L is (x,y,z)
        
            # rectangle sides loop
            for i in range(len(sides_of_rectangle)):
                for side in sides_of_rectangle[i]:
                    R = self.intersection(L,side)
                    if R:
                        if(R[0] >= all_cars[i][0][0] and R[0] <= all_cars[i][2][0] and R[1] >= all_cars[i][0][1] and R[1] <= all_cars[i][2][1] and np.linalg.norm(R) < np.linalg.norm(endpoint)):
                            if(np.linalg.norm(np.array(R)-np.array(endpoint)) < rad):  # checks if they lie in same quadrant
                                intersect.append(R)
    
            if len(intersect) > 1:
                dist = np.empty([len(intersect),3])
                dist[:,0:2] = np.array([intersect])
                dist[:,2] = euclidean_distances(np.array([center]),dist[:,0:2])
                row = dist[:,2].argmin()
                smallest_point = tuple(dist[row,0:2])
    
            if(len(intersect) == 1):
                smallest_point = intersect[0]
            
            if(len(intersect) == 0):
                smallest_point = endpoint
            
            distance.append(smallest_point)
            
        
        lidar = scipy.spatial.distance.cdist(np.array([center]), distance, 'euclidean')
        lidar = np.reshape(lidar,(-1,1))
        #degrees = np.arange(start,end_angle,deviation_angle)
        #degrees = np.reshape(degrees,(-1,1))
        #lidar = np.append(degrees,lidar,1)
        
        import matplotlib.pyplot as plt
        for i in range(len(distance)):
            plt.plot([center[0], distance[i][0]], [center[1], distance[i][1]], color = 'Red', linewidth = 1)
        plt.scatter(center[0],center[1],color = 'Blue',s=500)
        plt.show()
        
        return lidar

    def store_transition(self, state,lidar,action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        state = state.flatten().astype(np.float32)
        state_ = state_.flatten().astype(np.float32)
        lidar = lidar.flatten().astype(np.float32)
       
        
        self.state_memory[index] = state
        self.lidar_memory[index] = lidar
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal
        #print(self.mem_cntr)
        self.mem_cntr += 1

    def choose_action(self, observation,lidar):
        if np.random.random() > self.epsilon:
            observation = observation.flatten()         
            observation = np.float32(observation)
            lidar = np.float32(lidar)
            
            state = T.tensor([observation]).to(self.Q_eval.device)
            lidar = T.tensor([lidar]).to(self.Q_eval.device)
            
            state_lidar = T.cat([state,lidar],dim = 1)
            
            actions = self.Q_eval.forward(state_lidar)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        
        lidar_batch = T.tensor(self.lidar_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
    
        c = T.cat([state_batch, lidar_batch], dim=1)
        d = T.cat([new_state_batch,lidar_batch],dim = 1)
        
        q_eval = self.Q_eval.forward(c)[batch_index, action_batch]   # need to send lidar to the NN for learning
        q_next = self.Q_next.forward(d)
        
        q_next[terminal_batch] = 0.0
     
        q_target = reward_batch + self.gamma*T.max(q_next,dim=1)[0]

        loss = F.mse_loss(q_target,q_eval)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                       else self.eps_min
                       
        if self.iter_cntr % self.replace_target == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())
        return loss.item()

    
import gym
import highway_env
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
from IPython import display
import math
import time
 
if __name__ == '__main__':
    env = gym.make('highway-v0')
    
    config = {
    "observation": {
        "offscreen_rendering": True,
        "type": "Kinematics",
        "vehicles_count": 15,
        "initial_spacing":10,
        "features": ["presence", "x", "y", "vx", "vy","cos_h","sin_h"],
        "features_range": {
            "x": [0, 100],
            "y": [0, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "lanes_count": 2,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "absolute": True,
        "order": "sorted",
        'render_agent': True,
        "normalize":False,
        "duration":100,
        'simulation_frequency': 20,

                    }
            }

    #env = gym.make('highway-v0')
    env.configure(config)                       # Update our configuration in the environment
    env.reset()
    
    agent = Agent(max_mem_size=15000,gamma=0.99, epsilon=1.0, lr=0.003,input_dims=[105+540], batch_size=64, n_actions=5,lidar_input_dims=[540],eps_end=0.01)
    scores, eps_history,speed_at_collision,avg_speed,total_loss = [], [], [], [], []
    n_games = 1000
    crashed = 0
    lidar_first = np.zeros(180)
    lidar_second = np.zeros(180)

    for i in range(n_games):
        score = 0
        speed_in_episode = []
        done = False
        observation = env.reset()
        current_loss = []
        while not done:
            lidar = agent.display_lidar(observation)
            
            current_lidar = lidar.flatten()
            lidar = np.concatenate((lidar_first,lidar_second,current_lidar))
            
            action = agent.choose_action(observation,lidar)           # observation goes into the NN and then predicts action
            observation_, reward, done, info = env.step(action)
            """
            plt.imshow(env.render(mode='rgb_array'))
            display.display(plt.gcf())
            """
            env.render()
            score += reward
         
            agent.store_transition(observation,lidar, action, reward, 
                                    observation_, done)                # lidar is for the old observation
            loss = agent.learn()
            current_loss.append(loss)
            observation = observation_
            
            lidar_first = lidar_second
            lidar_second =  current_lidar
            speed_in_episode.append(math.sqrt(math.pow(observation[0][3],2)+math.pow(observation[0][4],2)))
            
            if(info['crashed'] == True):
                crashed+=1
        scores.append(score)
        eps_history.append(agent.epsilon)
        speed_at_collision.append(math.sqrt(math.pow(observation[0][3],2)+math.pow(observation[0][4],2)))
        avg_score = np.mean(scores[-100:])
        avg_speed.append(np.mean(speed_in_episode))
        total_loss.append(current_loss)
        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon,
                'speed at collsion %.2f'% speed_at_collision[-1],
                'average speed %.2f'% avg_speed[-1],
                'crashed',crashed)

