import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import backtrader as bt
from scripts.utils import calculate_performance_metrics
from backtrader import Order
from scripts.portfolio_tracker import PortfolioTracker



class MultiAssetTrendStrategy(bt.Strategy):
    params = (
        ('tickers', []),
        ('initial_capital', 100000),
        ('verbose', False),
        ('rebalance_days', 30),
        ('fast_period', 21),
        ('slow_period', 252),
        ('atr_period', 21),
        ('fast_period_weight', 0.5),
        ('asset_directions', {}),
    )

    def __init__(self):
        self.rebalance_counter = 0
        self.last_rebalance = -1
        self.portfolio_tracker = PortfolioTracker(self.params.tickers)
        self.indicators = {}
        for ticker in self.params.tickers:
            if ticker not in self.params.asset_directions:
                print(f"Asset direction not found for ticker {ticker}, setting to (-1, 1)")
                self.params.asset_directions[ticker] = (-1, 1)

        for ticker in self.params.tickers:
            data = self.getdatabyname(ticker)
            self.indicators[ticker] = {
                'trailing_return_fast': bt.indicators.ROC(
                    data.close, period=self.params.fast_period
                ),
                'trailing_return_slow': bt.indicators.ROC(
                    data.close, period=self.params.slow_period
                ),
                'atr': bt.indicators.ATR(data, period=self.params.atr_period),
            }

        self.history = []
        self.dividends_received = {ticker: 0.0 for ticker in self.params.tickers}
        # self.asset_trade_direction = self.params.asset_params.get('trade_direction', 'long')

    def log(self, txt, dt=None):
        """Logging function controlled by verbose parameter."""
        if self.params.verbose:
            dt = dt or self.datas[0].datetime.datetime(0)
            print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        """Handle order notifications and log execution details."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            self._log_order_execution(order)
        else:
            self.log(f'Order {order.Status[order.status]}')

    def _log_order_execution(self, order):
        """Log details of completed orders."""
        ticker = order.data._name
        executed = order.executed
        if order.isbuy():
            log_msg = (
                f'BUY EXECUTED, Ticker: {ticker}, '
                f'Price: {executed.price:.2f}, Size: {executed.size}, '
                f'Cost: {executed.value:.2f}, Commission: {executed.comm:.2f}'
            )
        else:
            log_msg = (
                f'SELL EXECUTED, Ticker: {ticker}, '
                f'Price: {executed.price:.2f}, Size: {executed.size}, '
                f'Cost: {executed.value:.2f}, Commission: {executed.comm:.2f}'
            )
        self.log(log_msg)

    def next(self):
        """Main method called for each market event."""
        self._handle_rebalance()
        self._record_portfolio_state()

    def _handle_rebalance(self):
        """Check if it's time to rebalance the portfolio."""
        if self.rebalance_counter % self.params.rebalance_days == 0:
            self.trade()
            self.last_rebalance = self.rebalance_counter
        self.rebalance_counter += 1

    def _get_previous_price(self, ticker, default_price):
        """Retrieve previous closing price for PnL calculation."""
        if len(self.asset_daily_prices[ticker]) > 0:
            return self.asset_daily_prices[ticker][-1]
        return default_price

    def trade(self):
        """Execute portfolio rebalancing based on calculated weights."""
        self.log("Rebalancing portfolio")
        portfolio_value = self.broker.getvalue()
        asset_weights = self._calculate_target_weights()
        # print(f"portfolio value: {portfolio_value}")
        for ticker in self.params.tickers:
            data = self.getdatabyname(ticker)
            target_value = int(portfolio_value * asset_weights.get(ticker, 0) / data.open[0])
            position = self.getposition(data)
            delta_position = target_value - position.size
            if delta_position > 0:
                self.buy(data, size=delta_position)
            elif delta_position < 0:
                self.sell(data, size=-delta_position)
            

    def _calculate_target_weights(self):
        """Calculate target weights based on volatility and momentum."""
        atrs = {}
        signals = {}

        for ticker in self.params.tickers:
            ind = self.indicators[ticker]
            atrs[ticker] = 1 / ind['atr'][0]
            if ind['trailing_return_fast'][0] >= 0:
                fast_signal = self.params.asset_directions[ticker][1]
            else:
                fast_signal = self.params.asset_directions[ticker][0]
            if ind['trailing_return_slow'][0] >= 0:
                slow_signal = self.params.asset_directions[ticker][1]
            else:
                slow_signal = self.params.asset_directions[ticker][0]
            signals[ticker] = (
                self.params.fast_period_weight * fast_signal
                + (1 - self.params.fast_period_weight) * slow_signal
            )

            # print(f"{ticker}, fast: {fast_signal}, slow: {slow_signal}, final: {signals[ticker]}")

        total_atr = sum(atrs.values())
        asset_weights = {
            ticker: (atrs[ticker] / total_atr) * signals[ticker] for ticker in self.params.tickers
        }
        # asset_weights = {
        #     ticker: (1 / len(self.params.tickers)) for ticker in self.params.tickers
        # }
        # print(asset_weights)
        # print(f"sum = {sum(asset_weights.values())}")
        return asset_weights

    def notify_dividends(self, dividend):
        """Add dividend payments to cash balance."""
        ticker = dividend.data._name
        shares = self.getposition(dividend.data).size
        self.dividends_received[ticker] += dividend.amount * shares
        self.broker.add_cash(dividend.amount * shares)


    def _record_portfolio_state(self):
        # print('---')
        # print("recording portfolio state")
        """Store daily portfolio state with dividends"""
        portfolio_value = self.broker.getvalue()
        # record asset pnl
        current_asset_pnl = {}
        for ticker in self.params.tickers:
            data = self.getdatabyname(ticker)
            position = self.getposition(data)
            current_price = data.close[0]
            prev_price = data.close[-1]
            current_asset_pnl[ticker] = position.size * (current_price - prev_price)

            # print(ticker, position.size, position.size * current_price / portfolio_value)
            

        record = {
            'date': self.datas[0].datetime.date(0),
            'portfolio_value': self.broker.getvalue(),
            **{f'{ticker}_pnl': current_asset_pnl[ticker] for ticker in self.params.tickers},
            **{f'{ticker}_dividend': self.dividends_received[ticker] for ticker in self.params.tickers},
            **{f'{ticker}_price': self.getdatabyname(ticker).close[0] for ticker in self.params.tickers},
        }
        
        self.history.append(record)
        self.dividends_received = {ticker: 0.0 for ticker in self.params.tickers}

