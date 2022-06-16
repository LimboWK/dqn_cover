import gym
import sys
# sys.path.append('../replay')
from .memory import ReplayBuffer, Experience
import numpy as np

import torch
import torch.nn as nn
class Agent:
    """Base Agent class handeling the interaction with the environment."""

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state = self.env.reset()
        
    def reset(self) -> None:
        """Resents the environment and updates the state."""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
            action = torch.tensor(action)
            _, action = torch.max(action, dim=1)
            action = int(action.item())
        else:
            # print(self.state)
            state = torch.tensor([self.state])
            
            if device not in ["cpu"]:
                state = state.cuda(device)

            """
            print('')
            print('Current Env step:', self.env.current_step)
            print('State:' , self.state, 'Shape:', self.state.shape)    
            """
            q_values = net(state)
            # print('Q-values:', q_values, 'Shape:', q_values.shape)

            _, action = torch.max(q_values.flatten(), dim=0)
            # print(_, action)
            action = int(action.item())
        # print(action)

        return action

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        device: str = "cpu",
    ) -> tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        action = self.get_action(net, epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)

        exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            print('env done, reset !')
            self.reset()
        return reward, done