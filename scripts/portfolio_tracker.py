import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PortfolioTracker:
    def __init__(self, tickers):
        self.df = pd.DataFrame()
        self.tickers = tickers
        self.dates = []
        self.portfolio_value = []
        self.asset_values = []
        self.asset_pnl = []


    def next(
        self, 
        date,
        portfolio_value: float,
        asset_values: dict, 
        asset_pnl: dict, 
    ):
        self.dates.append(date)
        self.portfolio_value.append(portfolio_value)
        self.asset_values.append(asset_values)
        self.asset_pnl.append(asset_pnl)

    def build_df(self):
        plt.plot(self.portfolio_value)
        plt.show()
        self.df['date'] = self.dates
        self.df['portfolio_value'] = self.portfolio_value
        for ticker in self.tickers:
            self.df['asset_values_' + ticker] = [f[ticker] for f in self.asset_values]
            self.df['asset_pnl_' + ticker] = [f[ticker] for f in self.asset_pnl]
        