import os
import pandas as pd
import numpy as np


def load_df(df_path):
    '''
    Load dataframe from csv or excel files
    '''

    file_type = df_path.split('.')[-1]
    if file_type == 'csv':
        df = pd.read_csv(df_path)
    elif file_type == 'xlsx':
        df = pd.read_excel(df_path)
    else:
        assert False, 'WRONG FILE TYPE!'

    try:
        df.drop('Unnamed: 0', axis='columns', inplace=True)
    except KeyError:
        pass
    return df



def create_folder(folder_path):
    '''
    Creates folder at folder_path.
    Does not create anything if folder exists
    '''
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        pass
    return



def calculate_performance_metrics(df, value_col='portfolio_value'):
    '''
    Calculate performance metrics for a portfolio.

    Parameters:
        df (pd.DataFrame): Dataframe containing portfolio value data.

    Returns:
        dict: Performance metrics including Sharpe ratio, volatility, CAGR, and max drawdown.
    '''
    returns = df[value_col].pct_change().dropna()
    cumulative_returns = (1 + returns).cumprod() - 1
    volatility = returns.std() * np.sqrt(252)  # Assuming daily returns
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    drawdown = (cumulative_returns.cummax() - cumulative_returns) / (cumulative_returns.cummax() + 1)
    max_drawdown = drawdown.max()
    cagr = (df[value_col].iloc[-1] / df[value_col].iloc[0]) ** (1 / (len(df) / 252)) - 1

    metrics = {
        'sharpe_ratio': sharpe_ratio,
        'volatility': volatility,
        'cagr': cagr,
        'max_drawdown': max_drawdown
    }
    return metrics
