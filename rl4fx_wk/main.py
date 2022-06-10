from model.lightning import DQNLightning
import os
import torch
from pytorch_lightning import LightningModule, Trainer
import pandas as pd
from env.FXTradingTrainEnv import FXTradingEnv
from config.setup import setup

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())

configuration = setup()

df = pd.read_csv(os.path.join(configuration['DATA_DIR'], 'rate_example.csv'))
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print(True in df.isna())
env = FXTradingEnv(df)

model = DQNLightning(env)

trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=1000,
    val_check_interval=100,
)

trainer.fit(model)