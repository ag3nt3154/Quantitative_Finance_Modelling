import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from misc import *


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, input_feature_list, **kwargs):
        super(TradingEnv, self).__init__()
        
        self.initial_capital = get_attr(kwargs, 'initial_capital', 1000000)
        
        self.df = df.copy()
        self.input_feature_list = input_feature_list
        assert 'volatility' in input_feature_list
        for f in self.input_feature_list:
            if f not in list(self.df):
                raise Exception('Input feature ' + f + 'not found in dataframe')
        self.input_df = self.df[input_feature_list].to_numpy()


        # Set the action space and observation space
        self.max_price = np.max(self.df['high']) * 2
        self.max_volume = np.max(self.df['volume'])

        self.clean_slate()

        # lookback for time-series data for input
        self.lookback_period = get_attr(kwargs, 'lookback_period', 20)
        # lookforward for hindsight bias on the reward
        self.lookforward_period = get_attr(kwargs, 'lookforward_period', 20)

        self.render_window_size = get_attr(kwargs, 'render_window_size', 20)

        # action_space = limit_order = [order_price, order_quantity]
        self.action_space = spaces.Discrete(
            low=np.array([-5, -5]),
            high=np.array([5, 5]),
            dtype=np.float64
        )

        self.observation_space = spaces.Box(
            low=np.array([-1000000 for f in range(len(self.input_feature_list) + len(self.trader_state))]),
            high=np.array([1000000 for f in range(len(self.input_feature_list) + len(self.trader_state))]),
            dtype=np.float64
        )

        
        # execution mechanism
        self.price = self.df['adjclose'].to_numpy()

        
    def clean_slate(self):
        '''
        Set initial state of environment.
        Called by self.__init__ and self.reset
        '''
        # State variables
        self.cash = self.initial_capital
        self.position = 0
        self.position_value = 0
        self.portfolio_value = self.cash + self.position_value
        self.leverage = self.position_value / self.portfolio_value
        self.portfolio_volatility = 0
        self.portfolio_return = 1
        self.end = False
        self.current_step = 0

        # Records
        self.records = {
            'cash': [],
            'position': [],
            'position_value': [],
            'portfolio_value': [],
            'leverage': [],
            'portfolio_volatility': [],
            'portfolio_return': [],
        }

        self.trade_record = {
            'order_quantity': [],
            'order_price': [],
            'execution_price': [],
            'cost_basis': [],
        }

        self.trader_state = np.array([
            self.cash,
            self.position,
            self.position_value,
            self.portfolio_value,
            self.leverage,
            self.portfolio_volatility,
        ])


    def reset(self):
        self.clean_slate()
        return self.__next_observation()
    

    def __next_observation(self):
        '''
        return input arr and trader_state in 1 vector
        '''
        obs = np.concatenate([self.input_df[self.current_step], self.trader_state])
        return obs
    

    def step(self, action):
        # define terminate condition
        if self.current_step >= len(self.input_df) - 2:
            self.end = True
        if not self.end:
            self.__take_action(action)
            obs = self.__next_observation()
            reward = action[0] * ((self.price[self.current_step] - self.price[self.current_step - 1]))
            return obs, reward, self.end, {}
        else:
            # termination
            # plt.plot(np.array(self.records['portfolio_value']) / 1e6 * 400)
            # plt.plot(self.price)
            # plt.show()
            # plt.plot(self.records['portfolio_volatility'])
            # plt.show()
            obs = self.__next_observation()
            reward = action[0] * (self.price[self.current_step] - self.price[self.current_step - 1])
            return obs, reward, self.end, {}

        
    def __take_action(self, action):
        # execute order
        order_price = action[0]
        execution_price = np.max([action[0], self.price[self.current_step]])
        order_quantity = action[1]
        if order_quantity != 0:
            self.cash -= execution_price * order_quantity
            self.position += order_quantity
        self.position_value = self.position * self.price[self.current_step]
        self.portfolio_value = self.cash + self.position_value
        self.leverage = self.position_value / self.portfolio_value
        self.portfolio_return = self.portfolio_value / self.initial_capital
        self.portfolio_volatility = np.std(
            self.records['portfolio_return'][-self.lookback_period:]
            # self.records['portfolio_return']
            )

        self.trader_state = np.array([
            self.cash,
            self.position,
            self.position_value,
            self.portfolio_value,
            self.leverage,
            self.portfolio_volatility,
        ])

        self.records['cash'].append(self.cash)
        self.records['position'].append(self.position)
        self.records['position_value'].append(self.position_value)
        self.records['portfolio_value'].append(self.portfolio_value)
        self.records['leverage'].append(self.leverage)
        self.records['portfolio_volatility'].append(self.portfolio_volatility)
        self.records['portfolio_return'].append(self.portfolio_return)
        
        self.trade_record['order_price'].append(order_price)
        self.trade_record['order_quantity'].append(order_price)
        self.trade_record['execution_price'].append(execution_price)

        self.current_step += 1
        

    def render(self, mode='human', close=False):
        pass