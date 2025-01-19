class PortfolioTracker:
    def __init__(self, tickers):
        self.df = pd.DataFrame()
        self.tickers = tickers
        self.dates = []
        self.portfolio_value = []
        self.asset_values = {ticker: [] for ticker in self.tickers}
        self.asset_pnl = {ticker: [] for ticker in self.tickers}


    def next(
        self, 
        date,
        portfolio_value: float,
        asset_values: dict, 
        asset_weights: dict, 
        asset_pnl: dict, 
        cash: float,
    ):
        self.dates.append(date)
