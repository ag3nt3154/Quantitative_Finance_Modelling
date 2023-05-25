# Quantitative Finance

This repo contains a collection of quantitative finance simulations created by me. These include stock price prediction with machine learning, and option pricing with monte carlo. Each simulation/module can be run independently.

Current simulations include:
1. Stock price prediction with fundamental data
2. Simulation of buy-hold performance vs Dollar Cost Averaging
3. Distribution of daily stock returns
4. Monte Carlo simulation of option strategies
5. Simulation of pairs trading
6. Monte Carlo option pricing

</br>

## 1. Stock Price Prediction with Fundamental Data

We assume that the intrinsic value of a stock is a function of the values of fundamental data such as P/E ratio, EBITDA, cashflow, etc. Therefore, we can use these values as inputs to train a ML model to estimate the intrinsic value of a stock.

Our hypothesis is that market prices of stocks generally reflects the intrinsic values when averaged over the a large number of stocks, i.e. the market prices stocks accurately for the vast majority of companies. Speculators temporarily affect stock prices in random ways, but their net effect is close to 0 when averaged over all stocks.

This suggests that if we train the ML model over many stocks, the model should price stocks fairly according to fundamental data, and we can obtain the intrinsic value of a stock. We can use this fair value to invest in underpriced companies (similar to Warren Buffett).

### Settings

We use the yfinance library https://github.com/ranaroussi/yfinance to obtain fundamental data for stocks. The data collected reflects all stocks in NYSE and NASDAQ exchanges for which yfinance has adequate data. The data collected is stored in (stock_data.csv).

We use pandas to process the data and generate the input vectors according to the factors in (stock_intrinsic_factors.json). The input vectors are normalised to have mean = 0 and variance = 1.

We create the model in TensorFlow and train it using the mean squared error as the loss function. The output is the predicted price of a stock.

### Usage

Usage demonstration is presented in (stock_price_predictor.py). 

</br>

## 2. Simulation of buy-hold performance vs DCA

Dollar cost averaging (DCA) is a investment strategy where one invests a certain amount of money, e.g. $100, at regular intervals regardless of the asset price at the moment.

We investigate the performance of buy-hold vs DCA on SPY.

</br>

## 3. Distribution of daily returns of SPY

While most quantitative finance material treat the distribution of returns of stock prices as a normal distribution for its mathematical qualities. This treatment may not be the most accurate observation.

We find that a better fit may be obtained by fitting a Laplace distribution. More work is required to explore the consequences of this change in distribution in existing models.

</br>

## 4. Monte Carlo simulation of option strategies

We use the black-scholes model to price options and evaluate various strategies on randomly generated stock price series based on factors such as returns, risk, and sharpe ratio.

Strategies evaluated include levered ETFs (UPRO, TQQQ), buy-write (QYLD), and buy-write at different delta and DTE.

</br>

## 5. Simulation of pairs trading

We explore pairs trading, where the weighted combination of two assets creates a stationary times series. We can then do a mean reversion trade on the two assets.

We test for stationarity with the ADF test and measure the cointegration of two assets' prices to create a stationary time series.

</br>

## 6. Monte Carlo option pricing

We investigate option pricing by Monte Carlo methods, randomly generating a set of asset price series and pricing the option based on the expected payoff.

We simulated dynamic delta hedging of vanilla options and noted truncation errors due to hedging discretely (e.g. daily).

We simulated a hypothetical volatility arbitrage trade, where we purchase an underpriced option (20 implied vol) vs a model volatility (30 vol) and hedge it with shares. We investigate if it is better to hedge to market (implied vol) or to the model.



