import pandas as pd
import os
import time
import yahoo_fin.stock_info as si
from datetime import datetime
from config.paths import DATA_DIR
import shutil
from scripts.utils import load_df


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
    df.rename(
        {
            'index': 'datetime',
            'adjclose': 'adj_close',
        },
        axis=1,
        inplace=True,
    )
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    # saving to data store
    if save:
        df.to_pickle(os.path.join(DATA_DIR, 'daily_price', 'yahoo_finance', f'{ticker}.pkl'))

    return df