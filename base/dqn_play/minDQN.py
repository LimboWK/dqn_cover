import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

class SimDQN(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super().__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))  
        actions = self.fc3(x) # no activated here !
        
        return actions 

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, 
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4, hidden_state=256,
        ) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        # self.n_actions = n_actions
        self.mem_size = max_mem_size
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.hidden_state = hidden_state
        self.actions_space = [i for i in range(n_actions)]
        self.mem_cntr = 0

        # use target network to improve stability
        self.Q_eval = SimDQN(self.lr, n_actions=n_actions, input_dims=self.input_dims, 
                             fc1_dims=self.hidden_state, fc2_dims=self.hidden_state)
        self.Q_target = SimDQN(self.lr, n_actions=n_actions, input_dims=self.input_dims, 
                               fc1_dims=self.hidden_state, fc2_dims=self.hidden_state)
        self.update_target_step = 200
        # replay
        self.state_memory = np.zeros((self.mem_size, *self.input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)


    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size # round back to zero when it's greater than mem size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = done

        self.mem_cntr += 1


    def choose_action(self, obs):
        if np.random.random() > self.epsilon:
            state = T.tensor([obs]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.actions_space)
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return None # warm-up
        self.Q_eval.optimizer.zero_grad()
        
        # create batch by randomly sampling from replay buffer
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = T.LongTensor(self.action_memory[batch])

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch] # q_eval return (q0 ~ qn)

        # update target network
        if self.mem_cntr % self.update_target_step == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())
        
        q_next = self.Q_target.forward(new_state_batch) # no target net here
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        
        loss = self.Q_eval.loss(q_eval, q_target).to(self.Q_eval.device)

        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min





        
