
"""

      *******************  three (lidar) -> fc1-> fc2-> output+ three (vx,vy) ->fc3 ->fc4 ->output  **********************
      


"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f'runs')


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = 512
        self.fc3_dims = 512
        self.fc4_dims = 512

        
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(518,self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        self.fc5 = nn.Linear(self.fc4_dims, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, lidar,velocity): 
        x = F.relu(self.fc1(lidar))
        x = F.relu(self.fc2(x))
        x = T.cat([x,velocity],dim = 1)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        actions = self.fc5(x)

        return actions
    
    
    
class Agent():
    def __init__(self,max_mem_size,gamma, epsilon, lr, input_dims, batch_size, n_actions,eps_end=0.01, eps_dec=3e-5):
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

        self.Q_eval = DeepQNetwork(lr,input_dims=input_dims,
                                    fc1_dims=512, fc2_dims=512, n_actions=n_actions)
    
        self.Q_next = DeepQNetwork(lr, input_dims=input_dims,
                                    fc1_dims=512, fc2_dims=512, n_actions=n_actions)
        
        self.Q_next.load_state_dict(self.Q_eval.state_dict())
        self.Q_next.eval()


        self.lidar_memory = np.zeros((self.mem_size, 540), dtype=np.float32)
        self.new_lidar_memory = np.zeros((self.mem_size, 540), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.current_loss = [] 
        
        self.velocity_memory = np.zeros((self.mem_size,6), dtype=np.float32)
        self.new_velocity_memory = np.zeros((self.mem_size,6), dtype=np.float32)
           
    
    
    
    def store_transition(self,lidar,action, reward, lidar_, terminal,velocity,velocity_):
        index = self.mem_cntr % self.mem_size
        lidar = lidar.flatten().astype(np.float32)
        lidar_ = lidar_.flatten().astype(np.float32)
        
        velocity = velocity.flatten().astype(np.float32)
        velocity_ = velocity_.flatten().astype(np.float32)
       
        self.lidar_memory[index] = lidar
        self.new_lidar_memory[index] = lidar_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal
        self.velocity_memory[index] = velocity
        self.new_velocity_memory[index] = velocity_
        self.mem_cntr += 1
        
    def choose_action_testing(self,lidar,velocity):
        lidar = np.float32(lidar)
        lidar = T.tensor([lidar]).to(self.Q_eval.device)
        
        velocity = np.float32(velocity)
        velocity = T.tensor([velocity]).to(self.Q_eval.device)
        
        actions = self.Q_eval.forward(lidar,velocity)
        action = T.argmax(actions).item()
        
        return action

    def choose_action(self,lidar,velocity):
        if np.random.random() > self.epsilon:
            lidar = np.float32(lidar)
            lidar = T.tensor([lidar]).to(self.Q_eval.device)
            
            velocity = np.float32(velocity)
            velocity = T.tensor([velocity]).to(self.Q_eval.device)
            
            actions = self.Q_eval.forward(lidar,velocity)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self,global_step):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        lidar_batch = T.tensor(self.lidar_memory[batch]).to(self.Q_eval.device)
        new_lidar_batch = T.tensor(self.new_lidar_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        velocity_batch = T.tensor(self.velocity_memory[batch]).to(self.Q_eval.device)
        new_velocity_batch = T.tensor(self.new_velocity_memory[batch]).to(self.Q_eval.device)
        
        q_eval = self.Q_eval.forward(lidar_batch,velocity_batch)[batch_index, action_batch] 
        q_next = self.Q_next.forward(new_lidar_batch,new_velocity_batch)
        
        q_next[terminal_batch] = 0.0
     
        q_target = reward_batch + self.gamma*T.max(q_next,dim=1)[0]


        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        writer.add_scalar('training loss',loss,global_step = global_step)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.iter_cntr += 1
        
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                       else self.eps_min
                       
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

"""
def rollouts(current_lidar,global_step):
    reward_bag = []
    
    for i in range(10):
        action = agent.choose_action_testing(current_lidar)
        observation_, reward, done, info = env.step(action)
        reward_bag.append(reward)
    avg_reward_rollouts = sum(reward_bag)/len(reward_bag)
    writer.add_scalar('avg_rollout_score',avg_reward_rollouts,global_step)
"""
  
def rollouts(current_lidar,velocity,global_step):
    reward_bag = []
    for i in range(10):
        score = 0
        done = False
        while not done:
            action = agent.choose_action_testing(current_lidar,velocity)
            observation_, reward, done, info = env.step(action)
            score+=reward
        reward_bag.append(score)
    avg_reward_rollouts = sum(reward_bag)/len(reward_bag)
    writer.add_scalar('avg_rollout_score',avg_reward_rollouts,global_step)

def save_checkpoints(state,global_step):
    filename = 'my_checkpoints/checkpoint_'+str(global_step)+'.pth.tar'
    print('saving checkpoint',global_step)
    T.save(state,filename)
    
def load_checkpoint(checkpoint):
    print('loading checkpoint')
    agent.Q_eval.load_state_dict(checkpoint['state_dict'])
    agent.Q_eval.optimizer.load_state_dict(checkpoint['optimizer'])

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

    env.configure(config)                       # Update our configuration in the environment
    env.reset()
    #env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True,force=True)
    agent = Agent(max_mem_size=50000,gamma=0.99, epsilon=1.0, lr=0.003,input_dims=[540], batch_size=32, n_actions=5,eps_end=0.01)
    n_games = 4000
    scores,eps_history,avg_score,speed_at_collision = [],[],[],[]
    global_step = 0

    with open('Data.csv','w') as out_file:
        for i in range(n_games):
            score = 0
            lidar_frame1 = np.zeros(180)
            lidar_frame2 = np.zeros(180)
            velocity_frame1 = np.zeros(2)
            velocity_frame2 = np.zeros(2)
            speed_in_episode = []
            crashed = False
            observation = env.reset()
            done = False

            while not done:
                global_step+=1
                
                checkpoints = {'state_dict':agent.Q_eval.state_dict(),'optimizer':agent.Q_eval.optimizer.state_dict()}
                if global_step%500 == 0:
                    save_checkpoints(checkpoints,global_step)
                
                         
                """  Get Lidar   """
                lidar = display_lidar(observation)
                total_lidar = np.concatenate((lidar,lidar_frame1,lidar_frame2))
                vx = np.array([observation[0][3]])
                vy = np.array([observation[0][4]])
                velocity = np.concatenate((vx,vy))
                total_velocity = np.concatenate((velocity,velocity_frame1,velocity_frame2))
                
                """ Check if rollouts 
                if global_step%500 == 0:
                    rollouts(total_lidar,total_velocity,global_step)
                """
                
                """ Take action  """                
                action = agent.choose_action(total_lidar,total_velocity)
                observation_, reward, done, info = env.step(action)   
                
                """ Get Lidar for next observation  """
                lidar_ = display_lidar(observation_)
                total_lidar_ = np.concatenate((lidar_,lidar,lidar_frame1))
                vx_ = np.array([observation_[0][3]])
                vy_ = np.array([observation_[0][4]])
                velocity_ = np.concatenate((vx_,vy_))
                total_velocity_ = np.concatenate((velocity_,velocity,velocity_frame1))
                
                """ Render Store Learn """
                env.render()                        
                score+=reward
                agent.store_transition(total_lidar,action, reward,total_lidar_, done,total_velocity,total_velocity_) 
                agent.learn(global_step)

                speed_in_episode.append(math.sqrt(math.pow(observation[0][3],2)+math.pow(observation[0][4], 2)))
                if(info['crashed'] == True):
                    crashed = True
            
                observation = observation_
                lidar_frame2 = lidar_frame1
                lidar_frame1 = lidar
                
                velocity_frame2 = velocity_frame1
                velocity_frame1 = velocity
                writer.add_scalar('epsilon',agent.epsilon,global_step=global_step)
                
            scores.append(score)
            speed_at_collision = math.sqrt(math.pow(observation[0][3],2)+math.pow(observation[0][4], 2))
            eps_history.append(agent.epsilon)
            avg_score = np.mean(scores[-100:])
            avg_speed = np.mean(speed_in_episode)
            
            print('episode ', i, 'score %.2f' % score,
                    'average score %.2f' % avg_score,
                    'epsilon %.2f' % agent.epsilon,
                    'crashed',crashed,'step',global_step)
            
            # Adding to tensorboard
            
            writer.add_scalar('reward',score,global_step=global_step)
            writer.add_scalar('avg_score',avg_score,global_step=global_step)
            writer.add_scalar('speed_at_collision',speed_at_collision,global_step=global_step)
            writer.add_scalar('avg_speed_epsiode',avg_speed,global_step=global_step)
            
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
      
    #<--------------------------------------Testing ------------------------------->    
    
    
    """ 

    with open('Test.csv','w') as test_file:
        for i in range(500):
            score = 0
            speed_in_episode = []
            crashed = False
            observation = env.reset()
            done = False
            while not done:
                lidar = display_lidar(observation)  
                vx = np.array([observation[0][3]])
                vy = np.array([observation[0][4]])
                velocity = np.concatenate((vx,vy))
                
                action = agent.choose_action_testing(lidar,velocity)
                observation_, reward, done, info = env.step(action)   

                env.render()                        
                score+=reward
                observation = observation_

                speed_in_episode.append(math.sqrt(math.pow(observation[0][3],2)+math.pow(observation[0][4], 2)))
                if(info['crashed'] == True):
                    crashed = True
                
            scores.append(score)
            speed_at_collision = math.sqrt(math.pow(observation[0][3],2)+math.pow(observation[0][4], 2))
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
        """
    


