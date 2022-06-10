from distutils.command.config import config
import gym
from gym import spaces
from gym.utils import seeding

import os, sys
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from typing import List, Tuple 
from pickle import load

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from config.setup import setup
configuration = setup()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_NUM_UNIT = 1_000_000
MAX_SHARE_PRICE = 150
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

MAX_EXEC_AMOUNT = 2_000_000 # USD
INITIAL_ACCOUNT_BALANCE = 10_000_000 # JPY
MAX_ACCOUNT_BALANCE = 20_000_000

# load the scaler
scaler = load(open(os.path.join(configuration['SCALER_DIR'], 'minmaxscaler.pkl'), 'rb'))

class FXTradingEnv(gym.Env):
    "A prop trading env for FX "
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df):
        super().__init__()
        self.df = df 
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        
        
        # todo: get to understand the meaning of space.Box() classs
        # actions: buy or sell unit
        self.action_space = spaces.Box(low=0, high=3, shape=(1, 3), dtype=np.float16)
        
        # (balance, unit hold, open, close, high, low)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 8+2 ), dtype=np.float16)
        """
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.position = 0.
        self.cost_basis = 0.
        self.total_unit_sold = 0.
        self.total_sales_value = 0.
        """

    def reset(self, seed=42):
        # reset to init state
        print('Env reset is called ! ')
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.position = 0.
        self.cost_basis = 0.
        self.total_unit_sold = 0.
        self.total_sales_value = 0.
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        # set the current step to a random point within the data frame to start the new round.
        self.current_step = random.randint(0, len(self.df)-1)
        return self._next_observation()
    
    def _next_observation(self):
        # get last five steps and make the next obs
        # try minmax scaler later
        # loc[] inclusive so +5 means there are 6 elements
        frame = np.array([
            self.df.loc[self.current_step: self.current_step , 'BidOpen'].values,
            self.df.loc[self.current_step: self.current_step , 'BidClose'].values,
            self.df.loc[self.current_step: self.current_step , 'BidHigh'].values, 
            self.df.loc[self.current_step: self.current_step , 'BidLow'].values, 
            self.df.loc[self.current_step: self.current_step , 'AskOpen'].values, 
            self.df.loc[self.current_step: self.current_step , 'AskClose'].values,
            self.df.loc[self.current_step: self.current_step , 'AskHigh'].values, 
            self.df.loc[self.current_step: self.current_step , 'AskLow'].values,
            # self.df.loc[self.current_step: self.current_step , 'Volume'].values / MAX_SHARE_PRICE,
        ])
        if frame.shape != (8,1):
            print()
            print('step', self.current_step)
            print('frame', self.df.iloc[self.current_step])
            raise ValueError(f'input frame at step {self.current_step} is empty, check the source data !')
        
        frame = frame.reshape(1, 8)
        account = np.array([[
            self.balance / MAX_ACCOUNT_BALANCE,
            #self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.position / MAX_NUM_UNIT,
            #self.cost_basis / MAX_SHARE_PRICE,
            #self.total_unit_sold / MAX_NUM_SHARES,
            #self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
            ]])
        # states
        # print(f'balance: {self.balance} position: {self.position} net_worth: {self.net_worth}')
        if np.isnan(self.balance / MAX_ACCOUNT_BALANCE) or np.isnan( self.position / MAX_NUM_UNIT):
            raise ValueError('account info invalid !!')
        obs = np.append(frame, account).reshape(1, 8+2)
        # print(obs)
        return obs
    
    def step(self, action):
        # print('single step taken !')
        # every step gaps 5 frames
        self._take_action(action)
        self.current_step += 1
        
        if self.current_step > len(self.df)-1:
            self.current_step = 0
            
        delay_modifier = (self.current_step / MAX_STEPS)
        # balance will be changed due to self._take_action, so as net_worth
        reward = self.balance * delay_modifier 
        done = self.net_worth <= 0
        # print('reward: ', reward)
        # print('balance:', self.balance)
        obs = self._next_observation() # after updating the step
        # obs is new_state when call step ]
        return obs, reward, done, {}
    
    
    
    def _take_action(self, action, min_buy_amount=1000):
        # either buy, sell, hold
        # why not high or low
        # needs to seperate ask and bid prices
        # converted back to prices using scalers.
        #print('current_step', self.current_step)
        current_bid = scaler.inverse_transform(self.df.loc[self.current_step:self.current_step, 'BidClose'].values.reshape(-1,1))[0, 0]
        #print('bid', current_bid)
        current_ask = scaler.inverse_transform(self.df.loc[self.current_step:self.current_step, 'AskClose'].values.reshape(-1,1))[0, 0]
        
        action_type = action
        # use a const buy sell fraction.
        action_amount = 0.2
        
        # >0 buy, <0 sell
        if action_amount < 1:
            # print('buy action')
            # buy amount % of balance in shares
            total_possible = self.balance / current_ask
            # print(total_possible, current_ask)
            if total_possible >= min_buy_amount:
                unit_bought = total_possible * action_amount
                # prev_cost = self.cost_basis * self.position
                additional_cost = unit_bought * current_ask
                self.balance -= additional_cost
                # self.cost_basis = (prev_cost + additional_cost) / (self.position + share_bought)
                self.position += unit_bought
        
        elif action_amount < 2:
            # sell
            # print('sell action')
            shares_sold = self.position #* action_amount * (-1)
            self.balance += shares_sold * current_bid
            self.position -= shares_sold
            self.total_unit_sold += shares_sold
            self.total_sales_value += shares_sold * current_bid
            
        self.net_worth = self.balance + self.position * current_bid

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        
        if self.position == 0:
            self.cost_basis = 0
    
    
    def render(self, mode='human', close=False):
        # render the environment to the screen 
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.position} (Total sold: {self.total_unit_sold})')
        print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
        
        


