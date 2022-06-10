import gym
from .models import DQN
import os, sys
from typing import List, Tuple 
from collections import OrderedDict, deque, namedtuple
import numpy as np
import pandas as pd

sys.path.append('..')
from replay.memory import ReplayBuffer, RLDataset
from replay.agent import Agent
from env.FXTradingTrainEnv import  FXTradingEnv

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import DistributedType
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torchsummary import summary

class DQNLightning(LightningModule):
    """Basic lightning Model of DQN"""

    def __init__(
        self,
        env,
        batch_size: int = 16,
        lr: float = 1e-2,
        gamma: float = 0.99,
        sync_rate: int = 10,
        replay_size: int = 1000,
        warm_start_size: int = 1000,
        eps_last_frame: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        episode_length: int = 200,
        warm_start_steps: int = 1000,
    ) -> None:
        """
        Args:
            batch_size: size of the batches")
            lr: learning rate
            env: gym environment tag
            gamma: discount factor
            sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            warm_start_size: how many samples do we use to fill our buffer at the start of training
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
            warm_start_steps: max episode reward in the environment
        """
        super().__init__()
        self.save_hyperparameters()

        # self.env = gym.make(self.hparams.env)
        self.env = env
        #print('Obs shape:',self.env.observation_space.shape)
        #print('Act shape:', self.env.action_space.shape)
        obs_size = self.env.observation_space.shape[1] # (1,3)
        n_actions = self.env.action_space.shape[1] # (1,10)
        print('Input dim:', obs_size, 'Output dim:', n_actions)

        self.net = DQN(obs_size=obs_size, n_actions=n_actions)
        print('Summary of DQN Net:')
        print(self.net)
        self.target_net = DQN(obs_size, n_actions)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)
        print('Finish init populate !')

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def get_epsilon(self, start: int, end: int, frames: int) -> float:
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch
        # print('batch[0]', states[0], actions[0], rewards[0])
        # print('actions:', actions)
        # print('states:', states)
        # actions = actions.reshape(len(actions), 1, 1)


        q_values_batch = self.net(states).squeeze(1)
        _index_actions = actions.long().unsqueeze(-1)

        """
        print('actions', actions, actions.shape)
        print('q_values', q_values_batch, q_values_batch.shape)
        print('rewards:', rewards, rewards.shape)
        print('indexed actions:', _index_actions)
        """
        # state_action_values = torch.gather(q_values_batch, 2, actions ).reshape(self.hparams.batch_size, 1)
        state_action_values = q_values_batch.gather(1, _index_actions).squeeze(-1)
        # _, state_action_values = torch.max(q_values_batch
        # , dim=2)
        state_action_values = state_action_values.reshape(-1,1)
        # print('state_action_values:', state_action_values, state_action_values.shape) 

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(2)[0] # target q value
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()
        # print('next_state_values:', next_state_values)
        expected_state_action_values = next_state_values * self.hparams.gamma + rewards.reshape(-1,1)
        _loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        if np.isnan(_loss.detach().numpy()):
            print()
            print('input state:', states)
            print('q_values_batch:', q_values_batch)
            print('state_action_values:', state_action_values, state_action_values.shape) 
            #print(rewards, rewards.shape)
            print('expected_state_action_values:', expected_state_action_values.shape)
            raise ValueError('Loss is NaN !')
            
        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = max(
            self.hparams.eps_end,
            self.hparams.eps_start - self.global_step + 1 / self.hparams.eps_last_frame,
        )

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        """ if self.trainer._distrib_type in {DistributedType.DP, DistributedType.DDP2}:
            loss = loss.unsqueeze(0)
        """
        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        
        self.log_dict(
            {
                "reward": reward,
                "train_loss": loss,
            }
        )

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())
            # print(f'Epoch:{self.current_epoch}, Global Step:{self.global_step}, Train_Loss:{loss}')
            self.env.render()

        """log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "train_loss": loss,
        }
        status = {
            "steps": torch.tensor(self.global_step).to(device),
            "total_reward": torch.tensor(self.total_reward).to(device),
        }"""
        # self.log("total_reward", self.total_reward, prog_bar=True)
        # self.log("steps", self.global_step, logger=False, prog_bar=True)

        # return OrderedDict({"loss": loss, "log": log, "progress_bar": status})
        return loss

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        return optimizer

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"