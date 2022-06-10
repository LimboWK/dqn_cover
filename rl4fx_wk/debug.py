from env.FXTradingTrainEnv import FXTradingTrainEnv
import pandas as pd
from config.setup import setup
import os, sys

configuration = setup()
df = pd.read_csv(os.path.join(configuration['DATA_DIR'], 'rate_example.csv'))
env = FXTradingTrainEnv(df)

print(env.state)