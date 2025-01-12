import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import backtrader as bt
from scripts.utils import calculate_performance_metrics
from backtrader import Order


class PortfolioRebalancingStrategy(bt.Strategy):
    params = (
        ('tickers', []),
        ('dataframes', {}),
        ('rebalance_days', 30),
        ('weights', {}),
        ('initial_capital', 100000),
        ('verbose', False),
    )

    def __init__(self):
        self.rebalance_counter = 0
        self.last_rebalance = -1
        self.portfolio_history = []
        self.asset_pnl = {ticker: [] for ticker in self.params.tickers}  # Store PnL for each asset
        self.asset_daily_prices = {ticker: [] for ticker in self.params.tickers}  # Store daily prices for each asset
        self.dates = []  # Track dates
        # self.cheat_on_open = True

    def log(self, txt, dt=None):
        ''' Logging function for this strategy '''
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        if not self.params.verbose:
            return
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Ticker: {order.data._name}, Price: {order.executed.price:.2f}, '
                         f'Size: {order.executed.size}, '
                         f'Cost: {order.executed.value:.2f}, '
                         f'Commission: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Ticker: {order.data._name}, Price: {order.executed.price:.2f}, '
                         f'Size: {order.executed.size}, '
                         f'Cost: {order.executed.value:.2f}, '
                         f'Commission: {order.executed.comm:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order {order.Status[order.status]}')

    def rebalance_portfolio(self):
        
        # Calculate portfolio value
        portfolio_value = self.broker.getvalue()
        if self.params.verbose:
            print("- rebalancing")
            print(f'-- portfolio value: {portfolio_value}')


        for ticker, weight in self.params.weights.items():
            

            # Calculate target value for the asset
            target_value = portfolio_value * weight

            

            # Get the open price of the asset
            data = self.getdatabyname(ticker)
            open_price = data.open[0]

            # Calculate the number of shares to hold
            current_position = self.getposition(data).size
            target_shares = int(target_value / open_price)
            shares_to_trade = target_shares - current_position

            if self.params.verbose:
                print(f'--- {ticker} {weight}')
                print(f'--- target value: {target_value}')
                print(f'--- target shares: {target_shares}')
                print(f'--- current position: {current_position}')
                print(f'--- shares to trade: {shares_to_trade}')
            
            # Execute trades
            if shares_to_trade > 0:
                self.buy(data=data, size=shares_to_trade)
            elif shares_to_trade < 0:
                self.sell(data=data, size=abs(shares_to_trade))

            

    def next(self):

        # Record portfolio value and weights
        portfolio_value = self.broker.getvalue()
        asset_values = {}
        asset_weights = {}
        for ticker in self.params.tickers:
            data = self.getdatabyname(ticker)
            position = self.getposition(data)
            value = position.size * data.close[0]
            asset_values[ticker] = value
            asset_weights[ticker] = value / portfolio_value if portfolio_value > 0 else 0

        

        self.dates.append(self.datas[0].datetime.date(0))

        for ticker in self.params.tickers:
            data = self.getdatabyname(ticker)
            position = self.getposition(data)

            # Store daily close price for the asset
            self.asset_daily_prices[ticker].append(data.close[0])

            # Calculate daily PnL for each asset
            prev_price = self.asset_daily_prices[ticker][-2] if len(self.asset_daily_prices[ticker]) > 1 else data.close[0]
            pnl = position.size * (data.close[0] - prev_price)
            self.asset_pnl[ticker].append(pnl)

        self.portfolio_history.append({
            'datetime': self.datas[0].datetime.datetime(0),
            'portfolio_value': portfolio_value,
            'asset_values': asset_values,
            'asset_weights': asset_weights,
            'cash': self.broker.getcash(),
            'weight_cash': self.broker.getcash() / portfolio_value if portfolio_value > 0 else 0,
        })

        # Check if it is time to rebalance
        if self.rebalance_counter % self.params.rebalance_days == 0:
            self.rebalance_portfolio()
            self.last_rebalance = self.rebalance_counter
        self.rebalance_counter += 1


    def notify_dividends(self, dividend):
        # Add dividends to cash balance
        self.broker.add_cash(dividend.size)


# Main function to run backtest
def backtest_fixed_weight_portfolio(tickers, dataframes, rebalance_days, weights, initial_capital, plot=False, verbose=False):
    cerebro = bt.Cerebro()

    cerebro.broker.setcommission(commission=0.000, margin=0.5)

    # Add strategy
    cerebro.addstrategy(
        PortfolioRebalancingStrategy,
        tickers=tickers,
        dataframes=dataframes,
        rebalance_days=rebalance_days,
        weights=weights,
        initial_capital=initial_capital,
        verbose=verbose
    )

    # Load data
    for ticker, df in dataframes.items():
        data = bt.feeds.PandasData(dataname=df, name=ticker)
        cerebro.adddata(data)

    # Set initial capital
    cerebro.broker.setcash(initial_capital)

    # Run backtest
    strategies = cerebro.run()
    strategy = strategies[0]

    # Plot results with adjusted viewing window
    if plot:
        cerebro.plot()

    # Extract portfolio history
    portfolio_history = pd.DataFrame(strategy.portfolio_history)
    portfolio_history.set_index('datetime', inplace=True)
    portfolio_history['total_cumulative_returns'] = (portfolio_history['portfolio_value'] / initial_capital) - 1
    portfolio_history['pnl'] = portfolio_history['portfolio_value'] - initial_capital
    drawdown = (portfolio_history['total_cumulative_returns'].cummax() - portfolio_history['total_cumulative_returns']) / (portfolio_history['total_cumulative_returns'].cummax() + 1)
    portfolio_history['drawdown'] = drawdown

    # print(portfolio_history)

    for ticker in tickers:
        # Calculate daily and cumulative returns for each asset
        portfolio_history[f'price_{ticker}'] = strategy.asset_daily_prices[ticker]
        portfolio_history[f'daily_returns_{ticker}'] = portfolio_history[f'price_{ticker}'].pct_change()
        portfolio_history[f'cumulative_returns_{ticker}'] = (portfolio_history[f'price_{ticker}'] / portfolio_history[f'price_{ticker}'].iloc[0]) - 1

        portfolio_history[f'value_{ticker}'] = portfolio_history['asset_values'].apply(lambda x: x[ticker])
        portfolio_history[f'weight_{ticker}'] = portfolio_history['asset_weights'].apply(lambda x: x[ticker])

        # Add PnL attributed to each asset
        portfolio_history[f'pnl_{ticker}'] = strategy.asset_pnl[ticker]
        portfolio_history[f'cumulative_pnl_{ticker}'] = portfolio_history[f'pnl_{ticker}'].cumsum()

    # Calculate metrics
    metrics = calculate_performance_metrics(portfolio_history)

    return portfolio_history, metrics