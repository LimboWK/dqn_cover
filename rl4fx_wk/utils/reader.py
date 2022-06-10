from datetime import datetime, timedelta
from lz4.block import decompress
import numpy as np
import pandas as pd
import os, sys

def create_dataframe_from_csv(start, end, FX_DATA='/mnt/data/FXData/', currency_pair = 'USDJPY', bestrates=True, executions=True):
    # for cpp sim the day starts from 0:00 but we use 6:00am as start
    formatter = '%Y-%m-%d'
    dt_start = datetime.strptime(start, formatter)
    dt_end = datetime.strptime(end, formatter)
    if dt_start > dt_end:
        raise ValueError('The starting time should be earlier than the ending time !!!')
    print(dt_start, dt_end)
    check_date = dt_start
    rate_files = []
    exec_files = []
    while check_date <= dt_end:
        check_date_str = check_date.date().strftime('%Y%m%d')
        check_date_year = check_date_str[0:4]

        rate_file = os.path.join(FX_DATA,'BestRate', check_date_year, f'bestrate.{check_date_str}_{currency_pair}.csv.gz')
        if os.path.exists(rate_file):
            rate_files.append(rate_file)

        exec_file = os.path.join(FX_DATA,'CustomerExecution', check_date_year,f'Customer_Execution_{currency_pair}_{check_date_str}.csv.gz')
        if os.path.exists(exec_file):
            exec_files.append(exec_file)
            if check_date == dt_end:
                extra_day = (check_date+timedelta(days=1)).date().strftime('%Y%m%d')
                print(check_date_year, extra_day)
                check_date_year = extra_day[0:4]
                rate_file = os.path.join(FX_DATA,'BestRate', check_date_year, f'bestrate.{extra_day}_{currency_pair}.csv.gz')
                if os.path.exists(rate_file):
                    rate_files.append(rate_file)
        check_date += timedelta(days=1)
        
    def buy_sell_fn(x):
        return -1 if x==2 else 1
    

    if executions:
        lst_prod = []
        for file in exec_files:
            _df = pd.read_csv(file, header=None)
            #print(_df)
            lst_prod.append(_df)
        df_prod = pd.concat(lst_prod)
        col_names_old = ['Date', 'OfficeID', 'ClientID', 'PairCode', 'BuySell', 'Time', 'Price', 'Amount', 'WLCode']
        col_names_new = ['Date', 'OfficeID', 'ClientID', 'PairCode', 'BuySell', 'Time', 'Price', 'Amount', 'WLCode']
        col_names = col_names_old if dt_start < datetime.strptime('2021-12-01', formatter) else col_names_new
        # print(f'Using {col_names}')
        df_prod.rename(columns={i:col_names[i] for i in range(len(col_names))}, inplace=True)
        df_prod.Time = pd.to_datetime(df_prod.Time)
        # df_prod = df_prod[(df_prod.Time < dt_end+timedelta(days=1)) & (df_prod.Time > dt_start)]
        # df_prod.Time = pd.to_datetime(df_prod.Time)
        df_prod.reset_index(inplace=True)
        df_prod['BuySell'] = df_prod['BuySell'].apply(lambda x: buy_sell_fn(x))

        # two accounts: JPY deposit and USD position
        df_prod['DepositVar'] = df_prod['BuySell']*df_prod['Amount']*df_prod['Price']
        df_prod['PositionVar'] = -1*df_prod['BuySell']*df_prod['Amount']

        df_prod['Deposit'] = df_prod['DepositVar'].cumsum()
        df_prod['Position'] = df_prod['PositionVar'].cumsum()
        # df_prod['AvgPosPrice'] = -1*df_prod['Deposit'] / df_prod['Position']
        # df_prod['UserProfit'] = df_prod['Deposit'] + df_prod['Position']*df_prod['Price']
        #deposit_history = df_prod.DepositVar.to_numpy()
        #position_history = df_prod.PositionVar.to_numpy()
        #ts_history = df_prod.Time.to_numpy()
        df_prod['Time'] = pd.to_datetime(df_prod['Time'])
        df_prod['TS'] = df_prod.Time
        df_prod.set_index('TS', inplace=True)
    # df_prod.tail()
    
    if bestrates:
        lst_rate = []
        for file in rate_files:
            _df = pd.read_csv(file, header=None)
            lst_rate.append(_df)
        df_rate = pd.concat(lst_rate)
        col_names = ['Time', 'CurrencyPairCode', 'CoreRate','BestBid', 'BestAsk', 'SecBestBid', 'SecBestAsk', 'BestBidCP', 'BestAskCP', 'SecBestBidCP', 'SecBestAskCP', 'Logic']
        df_rate.rename(columns={i:col_names[i] for i in range(len(col_names))}, inplace=True)
        df_rate.Time = pd.to_datetime(df_rate.Time)

        #bestbid_history = df_rate.BestBid.to_numpy()
        #bestask_history = df_rate.BestAsk.to_numpy()
        #ts_cp_history = df_rate.Time.to_numpy()

        df_rate = df_rate[df_rate.Time <= df_prod.Time.iloc[-1]] if executions else df_rate
        df_rate['TS'] = df_rate['Time']
        df_rate.set_index('TS', inplace=True)
    # df_rate.head() # be sure not to use more than prod data ts
    
    if bestrates and executions:
        return df_prod, df_rate
    elif bestrates and not executions:
        return df_rate
    elif not bestrates and executions:
        return df_prod
    else:
        print('Return Nothing !')
        return None