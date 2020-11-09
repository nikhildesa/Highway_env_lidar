
"""
    Neural networks

"""
class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, 
            n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = 512
        self.fc2_dims = 512
        self.fc3_dims = 512
        self.fc4_dims = 512

        
        self.n_actions = n_actions
        self.fc1 = nn.Linear(540, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(518,self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        self.fc5 = nn.Linear(self.fc4_dims, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state,velocity): 
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = T.cat([x,velocity],dim = 1)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        actions = self.fc5(x)

        return actions


"""
    Policy network and Target network

"""

self.Q_eval = DeepQNetwork(lr,input_dims=input_dims,
                                    fc1_dims=512, fc2_dims=512, n_actions=n_actions)
    
self.Q_next = DeepQNetwork(lr, input_dims=input_dims,
                                    fc1_dims=512, fc2_dims=512, n_actions=n_actions)
        
self.Q_next.load_state_dict(self.Q_eval.state_dict())
self.Q_next.eval()


"""
Learning

"""

q_eval = self.Q_eval.forward(lidar_batch,velocity_batch)[batch_index, action_batch] 
q_next = self.Q_next.forward(new_lidar_batch,new_velocity_batch)


"""
choose action

"""

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



"""
main loop

"""

for i in range(episodes):
    while not done:
        lidar = display_lidar(observation)
        total_lidar = np.concatenate((lidar,lidar_frame1,lidar_frame2))
        
        vx = np.array([observation[0][3]])
        vy = np.array([observation[0][4]])
        velocity = np.concatenate((vx,vy))
        total_velocity = np.concatenate((velocity,velocity_frame1,velocity_frame2))
        
        action = agent.choose_action(total_lidar,total_velocity)
        observation_, reward, done, info = env.step(action)   
        observation = observation_