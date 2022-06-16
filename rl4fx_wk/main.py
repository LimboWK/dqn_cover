from model.lightning import DQNLightning
import os
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
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

tb_logger = TensorBoardLogger(name='dqn_tb_logs', save_dir='./')

trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=1000,
    val_check_interval=100,
    logger = tb_logger,
    # logger=CSVLogger(save_dir="csv_logs/")
)

if os.path.exists(f'{trainer.logger.log_dir}'):
    print(f'{trainer.logger.log_dir} exists, overwritting ...')
    # os.remove(f'{trainer.logger.log_dir}')

trainer.fit(model)

if False:
    from IPython.display import display
    import seaborn as sn

    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    del metrics["step"]
    metrics.set_index("epoch", inplace=True)
    display(metrics.dropna(axis=1, how="all").head())
    sn.relplot(data=metrics, kind="line")