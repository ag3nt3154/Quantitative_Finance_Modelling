import numpy as np
import pandas as pd
import tqdm
from scipy.stats import gmean

def get_kpi(rt, ch, stock_daily_change):
    '''
    Calculate key performance indicators from daily returns and daily changes of strategy.
    Input: daily returns, daily changes, daily changes
    '''
    kpi_dict = {
            'Total Return': rt[-1] - 1,
            'CAGR': gmean(ch) ** 252 - 1,
            'Volatility': np.std(ch - 1) * np.sqrt(252),
            'Beta': np.corrcoef(stock_daily_change, ch - 1)[0, 1] * np.std(stock_daily_change) * np.std(ch - 1) / np.var(stock_daily_change),
            'Sharpe Ratio': ((np.mean(ch) ** 252 - 1) - 0.03) / (np.std(ch - 1) * np.sqrt(252)),
        }
    
    # Find max return and max drawdown
    max_rt = 0
    max_dd = 0
    for i in range(len(rt)):
        if rt[i] > max_rt:
            max_rt = rt[i]
        else:
            if rt[i] - max_rt < max_dd:
                max_dd = rt[i] - max_rt
    
    kpi_dict['Max Drawdown'] = max_dd
    kpi_dict['RoMaxDD'] = kpi_dict['Total Return'] / (-kpi_dict['Max Drawdown'])
    kpi_dict['Daily Returns'] = rt
    kpi_dict['Daily Changes'] = ch
    
    return kpi_dict


class test:
    def __init__(self, strats_func):
        '''
        strats_func is a function to that runs several strategies at once
        strats_func input: stock price series
        strats_func output: dictionary with strategy names as keys and (daily returns, daily change) as value
        '''
        self.strats_func = strats_func
    
    def kpi_given_stock_price_series(self, stock_price_series, daily_change):
        '''
        Runs strats_func on the stock price series and returns kpi
        '''
        results = self.strats_func(stock_price_series)

        for strat in results:
            rt, ch = results[strat]
            kpi = get_kpi(rt, ch, daily_change)
            results[strat] = kpi

        return results
        
    def single_stock_price_series_expt(self, laplace_params, num_days=252, initial_price=400, display=False):
        '''
        Generates a single stock price series, runs strats_func, and returns kpi
        Statistics and graphs can be displayed
        '''
        # Generate stock price series
        stock_price_series, daily_change = generate_stock(laplace_params, num_days, initial_price)
        
        # Runs strategies and find kpi
        results = self.kpi_given_stock_price_series(stock_price_series, daily_change)

        # Display statistics and graphs
        if display:
            for strat in results:
                kpi = results[strat]

                
                print(strat)
                print('Total Return:          {:.2f} %'.format(kpi['Total Return'] * 100))
                print('CAGR:                  {:.2f} %'.format(kpi['CAGR'] * 100))
                print('Volatility:            {:.2f} %'.format(kpi['Volatility'] * 100))
                print('Beta:                  {:.2f}'.format(kpi['Beta']))
                print('Sharpe Ratio:          {:.2f}'.format(kpi['Sharpe Ratio']))
                print('Max Drawdown:          {:.2f} %'.format(kpi['Max Drawdown'] * 100))
                print('Return over Max DD:    {:.2f}'.format(kpi['RoMaxDD']))
                print('-------------------------------------')

                plt.plot(range(num_days + 1), kpi['Daily Returns'], label=strat)

            plt.xlabel('Time in Days')
            plt.ylabel('Strategy Returns')    
            plt.legend()
            plt.show()
            
        return results

    def multiple_stock_price_series_expt(self, runs, laplace_params, num_days=252, initial_price=400, display=False):
        '''
        Generates multiple stock price series, runs strats_func, and returns average kpi across all runs
        Statistics and graphs can be displayed
        '''
        # Initiate raw results to collect all data
        raw_results = {}


        rng = tqdm(range(runs))

        # Run expt for n times with generated stock price series
        for i in rng:

            # Generate stock price series
            stock_price_series, daily_change = generate_stock(laplace_params, num_days, initial_price)

            # Run strategies and find kpi
            results = self.kpi_given_stock_price_series(stock_price_series, daily_change)

            # Save results
            for strat in results:
                kpi = results[strat]

                # Initialise dict on the 1st run
                if i == 0:
                    raw_results[strat] = []
            
                raw_results[strat].append(kpi)

        # Find average results
        avg_results = {}

        for strat in raw_results:
            avg_results[strat] = {}

            # Average kpi values for all runs
            for key in kpi:
                avg_results[strat][key] = np.mean([f[key] for f in raw_results[strat]])

            # Find winrate of strategy => win = total return at the end of run > 0
            avg_results[strat]['Win-rate'] = np.sum([1 for f in raw_results[strat] if f['Total Return'] > 0]) \
                / runs
            
        
        # display average results
        if display:
            for strat in avg_results:
                print(strat)
                
                print('Total Return:            {:.2f} %'.format(avg_results[strat]['Total Return'] * 100))
                print('CAGR:                    {:.2f} %'.format(avg_results[strat]['CAGR'] * 100))
                print('Volatility:              {:.2f} %'.format(avg_results[strat]['Volatility'] * 100))
                print('Beta:                    {:.2f}'.format(avg_results[strat]['Beta']))
                print('Sharpe Ratio:            {:.2f}'.format(avg_results[strat]['Sharpe Ratio']))
                print('Win-rate:                {:.2f} %'.format(avg_results[strat]['Win-rate'] * 100))
                print('Max Drawdown:            {:.2f} %'.format(avg_results[strat]['Max Drawdown'] * 100))
                print('Return over Max DD:      {:.2f}'.format(avg_results[strat]['RoMaxDD']))
                
                
                print('-------------------------------------')
        
        return avg_results