import pandas as pd
import os
import time
import yahoo_fin.stock_info as si
from datetime import datetime
from Config.Paths import DATA_DIR
import shutil
from Scripts.utils import load_df


# format the date
today_date = datetime.now().strftime('%d-%m-%Y')



def update_daily_price_data_yahoo(ticker, save=True):
    '''
    Get historical daily price data from yahoo_fin
    '''
    # Get historical price data from yahoo_fin
    df = si.get_data(ticker)
    df.drop('ticker', axis=1, inplace=True)
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'date'})

    # saving to data store
    if save:
        df.to_csv(os.path.join(DATA_DIR, 'Daily_price_data', 'yahoo_finance', f'{ticker}.csv'))

    return df