import pandas as pd
import numpy as np
import os, sys
from utils.reader import create_dataframe_from_csv
from config.setup import setup
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from pickle import dump

configuration = setup()

FX_DATA = os.environ.get('FX_DATA_ROOT', '/Users/kun.wan/workdir/gdsp/FXData')
PERIOD = '1T'

df = create_dataframe_from_csv('2022-04-04', '2022-04-08', FX_DATA=FX_DATA, bestrates=True, executions=False)
df.dropna(inplace=True)
print(df.head())
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.data_max_ = 140
scaler.data_min_ = 100

out = pd.DataFrame()
out['BidOpen'] = scaler.fit_transform(df.resample(PERIOD).BestBid.first().values.reshape(-1, 1)).squeeze()
out['BidClose'] = scaler.fit_transform((df.resample(PERIOD).BestBid.last().values.reshape(-1,1))).squeeze()
out['BidHigh'] = scaler.fit_transform(df.resample(PERIOD).BestBid.max().values.reshape(-1,1)).squeeze()

out['BidLow'] = scaler.fit_transform(df.resample(PERIOD).BestBid.min().values.reshape(-1,1)).squeeze()
out['AskOpen'] = scaler.fit_transform(df.resample(PERIOD).BestAsk.first().values.reshape(-1,1)).squeeze()
out['AskClose'] = scaler.fit_transform(df.resample(PERIOD).BestAsk.last().values.reshape(-1,1)).squeeze()
out['AskHigh'] = scaler.fit_transform(df.resample(PERIOD).BestAsk.max().values.reshape(-1,1)).squeeze()
out['AskLow'] = scaler.fit_transform(df.resample(PERIOD).BestAsk.min().values.reshape(-1,1)).squeeze()
out.to_csv(os.path.join(configuration['DATA_DIR'], 'rate_example.csv'))

# save the scaler
dump(scaler, open(os.path.join(configuration['SCALER_DIR'], 'minmaxscaler.pkl'), 'wb'))