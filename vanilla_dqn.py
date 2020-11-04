# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:36:25 2020

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

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f'runs/vanilla1')


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
    def __init__(self,max_mem_size,gamma, epsilon, lr, input_dims, batch_size, n_actions,eps_end=0.05, eps_dec=5e-5):
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
        self.replace_target = 50

        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                    fc1_dims=512, fc2_dims=512)
    
        self.Q_next = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                    fc1_dims=512, fc2_dims=512)
        
        self.Q_next.load_state_dict(self.Q_eval.state_dict())
        self.Q_next.eval()


        self.state_memory = np.zeros((self.mem_size, 105), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, 105), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        
        self.current_loss = []    
    
    
    
    def store_transition(self, state,action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        state = state.flatten().astype(np.float32)
        state_ = state_.flatten().astype(np.float32)
       
        
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal
        #print(self.mem_cntr)
        self.mem_cntr += 1
        
    def choose_action_testing(self,observation):
        observation = observation.flatten()         
        observation = np.float32(observation)
        
        state = T.tensor([observation]).to(self.Q_eval.device)       
        
        actions = self.Q_eval.forward(state)
        action = T.argmax(actions).item()
        
        return action

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            observation = observation.flatten()         
            observation = np.float32(observation)
            
            state = T.tensor([observation]).to(self.Q_eval.device)
        
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self,i):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
    
        
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]   # need to send lidar to the NN for learning
        q_next = self.Q_next.forward(new_state_batch)
        
        q_next[terminal_batch] = 0.0
     
        q_target = reward_batch + self.gamma*T.max(q_next,dim=1)[0]


        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        writer.add_scalar('training loss',loss,global_step=i)
        loss.backward()
        self.Q_eval.optimizer.step()
        #print(loss)
        
        

        self.iter_cntr += 1
        
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                       else self.eps_min
                       
        #self.epsilon = self.eps_min + (1.0 - self.eps_min) * np.exp(-self.eps_dec * i)
                       
        if self.iter_cntr % self.replace_target == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())
        return loss.item()

    
      
    
    
"""

     *****************      Training         ***********************

"""    
import gym
from gym import wrappers
import highway_env
import numpy as np
from matplotlib import pyplot as plt
import math



if __name__ == '__main__':
    env = gym.make('highway-v0')
    
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
     'vehicles_count': 15,
     'duration': 40,
     'initial_spacing': 2,
     'collision_reward': -1,
     'offroad_terminal': False,
     'normalize': False,   # changed added
     "absolute": True      # changed added
                }

        }

    #env = gym.make('highway-v0')
    env.configure(config)                       # Update our configuration in the environment
    env.reset()
    env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True,force=True)
    agent = Agent(max_mem_size=50000,gamma=0.99, epsilon=1.0, lr=0.003,input_dims=[105], batch_size=32, n_actions=5,eps_end=0.05)
    n_games = 4000
    scores,eps_history,avg_score,speed_at_collision = [],[],[],[]
    #env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True,force=True)
    with open('Data.csv','w') as out_file:
        for i in range(n_games):
            score = 0
            speed_in_episode = []
            crashed = False
            observation = env.reset()
            done = False
            while not done:
                action = agent.choose_action(observation)
                
                observation_, reward, done, info = env.step(action)
                env.render()
                score+=reward
                agent.store_transition(observation,action, reward, 
                                        observation_, done)                # lidar is for the old observation
                agent.learn(i)
                observation = observation_
                
                speed_in_episode.append(math.sqrt(math.pow(observation[0][3],2)+math.pow(observation[0][4], 2)))
                
                if(info['crashed'] == True):
                    crashed = True
                
                
            
            scores.append(score)
            speed_at_collision = math.sqrt(math.pow(observation[0][3],2)+math.pow(observation[0][4], 2))
            eps_history.append(agent.epsilon)
            avg_score = np.mean(scores[-100:])
            avg_speed = np.mean(speed_in_episode)
            print('episode ', i, 'score %.2f' % score,
                    'average score %.2f' % avg_score,
                    'epsilon %.2f' % agent.epsilon)
            
            # Adding to tensorboard
            
            writer.add_scalar('reward',score,global_step=i)
            writer.add_scalar('epsilon',agent.epsilon,global_step=i)
            writer.add_scalar('avg_score',avg_score,global_step=i)
            writer.add_scalar('speed_at_collision',speed_at_collision,global_step=i)
            writer.add_scalar('avg_speed_epsiode',avg_speed,global_step=i)
            
            # Adding to local file
            out_string = ""
            out_string+=str(i)
            out_string+=","+str(score)
            out_string+=","+str(agent.epsilon)
            out_string+=","+str(avg_score)
            out_string+=","+str(speed_at_collision)
            out_string+=","+str(avg_speed)
            out_string+=","+str(crashed)
            out_string+="\n"
            out_file.write(out_string)
        
    # Testing    
    
    with open('Test.csv','w') as test_file:
        for i in range(500):
            score = 0
            speed_in_episode = []
            crashed = False
            observation = env.reset()
            done = False
            while not done:
                action = agent.choose_action_testing(observation)
                
                observation_, reward, done, info = env.step(action)
                env.render()
                score+=reward
                
                speed_in_episode.append(math.sqrt(math.pow(observation[0][3],2)+math.pow(observation[0][4], 2)))
                
                if(info['crashed'] == True):
                    crashed = True
                
                
            scores.append(score)
            speed_at_collision = math.sqrt(math.pow(observation[0][3],2)+math.pow(observation[0][4], 2))
            eps_history.append(agent.epsilon)
            avg_score = np.mean(scores[-100:])
            avg_speed = np.mean(speed_in_episode)
            print('episode ', i, 'score %.2f' % score,
                    'average score %.2f' % avg_score)
            
            # Adding to tensorboard
            
            writer.add_scalar('reward_testing',score,global_step=i)
            writer.add_scalar('avg_score_testing',avg_score,global_step=i)
            writer.add_scalar('speed_at_collision_testing',speed_at_collision,global_step=i)
            writer.add_scalar('avg_speed_epsiode_testing',avg_speed,global_step=i)
            
            # Adding to local file
            test_string = ""
            test_string+=str(i)
            test_string+=","+str(score)
            test_string+=","+str(avg_score)
            test_string+=","+str(speed_at_collision)
            test_string+=","+str(avg_speed)
            test_string+=","+str(crashed)
            test_string+="\n"
            test_file.write(test_string)




