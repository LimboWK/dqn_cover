import os, sys

def setup():
    PROJECT_DIR = '/Users/kun.wan/workdir/gdsp/dqn_cover/rl4fx_wk'
    DATA_DIR = os.path.join(PROJECT_DIR, 'data')
    ENV_DIR = os.path.join(PROJECT_DIR, 'env')
    SCALER_DIR = os.path.join(PROJECT_DIR, 'scalers')
    return {'PROJECT_DIR':PROJECT_DIR, 'DATA_DIR': DATA_DIR, 'ENV_DIR':ENV_DIR, 'SCALER_DIR':SCALER_DIR}