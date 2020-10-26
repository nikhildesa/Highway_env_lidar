# -*- coding: utf-8 -*-

"""

      *******************      Deep Reinforcement learning  **********************
      
      vehicles = 15
      feature = presence,x,y,vx,vy,cosh,sinh
      gamma = 0.8
      batch_size=32
      max_mem_size=15000
      target update = 50
      

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
#writer = SummaryWriter(f'runs/try')


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
        
        self.lidar_memory = np.zeros((self.mem_size, *lidar_input_dims), dtype=np.float32)
        self.current_loss = []    
    
    
    
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
        
    def choose_action_testing(self,observation,lidar):
        observation = observation.flatten()         
        observation = np.float32(observation)
        lidar = np.float32(lidar)
        
        state = T.tensor([observation]).to(self.Q_eval.device)
        lidar = T.tensor([lidar]).to(self.Q_eval.device)
        
        state_lidar = T.cat([state,lidar],dim = 1)
        
        actions = self.Q_eval.forward(state_lidar)
        action = T.argmax(actions).item()
        
        return action

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

    def learn(self,i):
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


        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        #writer.add_scalar('training loss',loss,global_step=i)
        loss.backward()
        self.Q_eval.optimizer.step()
        #print(loss)
        
        

        self.iter_cntr += 1
        
        #self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                       #else self.eps_min
                       
        self.epsilon = self.eps_min + (1.0 - self.eps_min) * np.exp(-self.eps_dec * i)
                       
        if self.iter_cntr % self.replace_target == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())
        return loss.item()

    
      
    
    
"""

     *****************       Lidar + Testing           ***********************

"""    
import gym
from gym import wrappers
import highway_env
import numpy as np
from matplotlib import pyplot as plt
import math

def lidar_visualization(lidar,sensing_radius,agent_angle_copy,agent):
    for i in range(len(lidar)):
        x = lidar[i] * math.cos(math.radians(agent_angle_copy)) + agent[0] # get x coordinate
        y = lidar[i] * math.sin(math.radians(agent_angle_copy)) + agent[1] # get y coordinate
        plt.plot([agent[0], x], [agent[1], y], color = 'Red', linewidth = 1)  # plot the line
        agent_angle_copy+=2
    plt.scatter(agent[0],agent[1],color = 'Black',s=500)
    plt.show()


def find_endpoints(agent,agent_angle,sensing_radius):
    """<--------------------------- gives endpoint for the ray segment---------------------->"""
    count = 0
    x_point,y_point = [],[]
    while (count < 180):
        x_point.append(sensing_radius * math.cos(math.radians(agent_angle)) + agent[0])
        y_point.append(sensing_radius * math.sin(math.radians(agent_angle)) + agent[1])
        agent_angle+=2
        count+=1
    endpoints = list(zip(x_point,y_point))
    return endpoints


def get_intersection(Q,r,P1,P2):
    """<---------------------Gives closest point of intersection to the circle---------------------->"""
    Q = np.array(Q)    # Centre of circle
    P1 = np.array(P1)    # Radius of circle
    P2 = np.array(P2)    # Start of line segment
    V = P2 - P1    # Vector along line segment
   
    a = V.dot(V)
    b = 2 * V.dot(P1 - Q)
    c = P1.dot(P1) + Q.dot(Q) - 2 * P1.dot(Q) - r**2
    
    disc = b**2 - 4 * a * c
    if disc < 0:
        return False, None
    
    sqrt_disc = math.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    
    if not (0 <= t1 <= 1 or 0 <= t2 <= 1):
        return False, None
    
    t = min(t1,t2)
    return True,P1 + t * V


def display_lidar(position):
    """ <-------------------Agent-----------------------------> """
    position = position[position[:,0] == 1]  # considers only those vehicles which are in picture
    position[:,2] = 12 - position[:,2]     # change y axis from 4th duadrant to 1st quadrant
    
    agent_X = position[0][1]    # Agent centre x
    agent_Y = position[0][2]    # Agent centre y
    
    agent = (agent_X,agent_Y)   # Agent position
    agent_inclination = math.degrees(math.acos(env.vehicle.heading))
    agent_angle = agent_inclination - 90    # angle wrt lane
    agent_angle_copy = agent_angle    # copy of angle needed ahead
    
    """ <------------------------- Neighbor vehicles------------------> """
    disc_collection = []
    for i in range(1,len(position)):   # starts from 1 as 0 is our agent

        l = 5   # length of vehicle
        w = 2   # width of vehicle
        X = position[i][1]  # neighbor X
        Y = position[i][2]  # neighbor Y
        cos_h = position[i][5]
        sin_h = position[i][6]
        heading = np.arctan2(sin_h, cos_h)
        car_angle = math.degrees(math.acos(heading))
        yaw = car_angle - 90    # angle wrt lane

        num = int(np.ceil(l/w)-1)    # the total number of discs will be 2*num + 1 = 5
        s1 = np.array(range(-num,num+1)) 
        x_circles = X*np.ones(num*2+1) + s1*w/2*math.cos(math.radians(yaw))    # the x coordinates of the discs
        y_circles = Y*np.ones(num*2+1) + s1*w/2*math.sin(math.radians(yaw))    # the y-coordinates of the discs

        disc = list(zip(x_circles,y_circles))
        disc_collection.append(disc)
     
    """ <---------------------- check if the neighbor is within our sensing radius------------------> """ 
    neighbor = []
    sensing_radius = 25
    for n in disc_collection:
        disc_first = np.array(n[0])    # first disk
        disc_second = np.array(n[1])
        disc_third = np.array(n[2])
        disc_forth = np.array(n[3])
        disc_last = np.array(n[4])     # last disk

        # Checking if first or last disk is withing sensing radius (may be we dont need to check all disk)
        if np.linalg.norm(agent - disc_first) < sensing_radius or np.linalg.norm(agent - disc_last) < sensing_radius or np.linalg.norm(agent - disc_second) < sensing_radius or np.linalg.norm(agent - disc_third) < sensing_radius or np.linalg.norm(agent - disc_forth) < sensing_radius:
            neighbor.append(n)
    
    """<------------------------- ray end points------------------------------->"""
    endpoints = find_endpoints(agent,agent_angle,sensing_radius)    # end point of the ray
    lidar = []
    for endpoint in endpoints:
        intersection = [sensing_radius]    # if no intersection then sensing radius will be the length of ray
        for disk_set in neighbor:
            for disk in disk_set:
                flag,point = get_intersection(disk,1,agent,endpoint)
                if flag!=False:
                    distance = np.linalg.norm(agent - point)    #Euclidean distance from agent centre
                    intersection.append(distance)
        lidar.append(min(intersection))    # The smallest intersection is stored
    
    """<--------------------------- visualize the lidar--------------------------->"""    
    #lidar_visualization(lidar,sensing_radius,agent_angle_copy,agent)        
    lidar = np.array(lidar)
    return lidar


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
    agent = Agent(max_mem_size=15000,gamma=0.8, epsilon=1.0, lr=0.003,input_dims=[105+540], batch_size=32, n_actions=5,lidar_input_dims=[540],eps_end=0.05)
    n_games = 4000
    scores,eps_history,avg_score,speed_at_collision = [],[],[],[]
    env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True,force=True)
    with open('Data.csv','w') as out_file:
        for i in range(n_games):
            lidar_frame1 = np.zeros(180)
            lidar_frame2 = np.zeros(180)
            score = 0
            speed_in_episode = []
            crashed = False
            observation = env.reset()
            done = False
            while not done:
                lidar = display_lidar(observation)
                
                current_lidar = lidar.flatten()
                lidar = np.concatenate((current_lidar,lidar_frame1,lidar_frame2))   # current, (prev frame), (prev prev frame)
                action = agent.choose_action(observation,lidar)
                
                observation_, reward, done, info = env.step(action)
                env.render()
                score+=reward
                agent.store_transition(observation,lidar, action, reward, 
                                        observation_, done)                # lidar is for the old observation
                agent.learn(i)
                observation = observation_
                
                lidar_frame2 = lidar_frame1
                lidar_frame1 = current_lidar
                
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
            """
            writer.add_scalar('reward',score,global_step=i)
            writer.add_scalar('epsilon',agent.epsilon,global_step=i)
            writer.add_scalar('avg_score',avg_score,global_step=i)
            writer.add_scalar('speed_at_collision',speed_at_collision,global_step=i)
            writer.add_scalar('avg_speed_epsiode',avg_speed,global_step=i)
            """
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
            lidar_frame1 = np.zeros(180)
            lidar_frame2 = np.zeros(180)
            score = 0
            speed_in_episode = []
            crashed = False
            observation = env.reset()
            done = False
            while not done:
                lidar = display_lidar(observation)
                
                current_lidar = lidar.flatten()
                lidar = np.concatenate((current_lidar,lidar_frame1,lidar_frame2))   # current, (prev frame), (prev prev frame)
                action = agent.choose_action_testing(observation,lidar)
                
                observation_, reward, done, info = env.step(action)
                env.render()
                score+=reward
                
                lidar_frame2 = lidar_frame1
                lidar_frame1 = current_lidar
                
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
            """
            writer.add_scalar('reward_testing',score,global_step=i)
            writer.add_scalar('avg_score_testing',avg_score,global_step=i)
            writer.add_scalar('speed_at_collision_testing',speed_at_collision,global_step=i)
            writer.add_scalar('avg_speed_epsiode_testing',avg_speed,global_step=i)
            """
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




