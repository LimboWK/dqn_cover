from tabnanny import verbose
import gym
import json
import datetime as dt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3 import PPO2

from env.stock import StockTradingEnvironment

import pandas as pd

df = pd.read_csv('./data/AAPL.csv')
df = df.sort_values('Date')

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnvironment(df)])
# env = StockTradingEnvironment(df)
model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=20000)

obs = env.reset()
for i in range(3000):
    action, _states = model.predict(obs)
    # the output obs is the new state for the next step
    obs, rewards, done, info = env.step(action)
    env.render()
