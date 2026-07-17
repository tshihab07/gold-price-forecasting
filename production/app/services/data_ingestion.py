from typing import Dict, Optional
from datetime import datetime, timedelta
import yfinance as yf
from abc import ABC, abstractmethod

try:
    from ..config import ASSETS, BASE_FEATURES, TARGET_COLUMN
    from ..utils.logger import get_logger
    from ..utils.validators import validate_market_data, ValidationError

except ImportError:
    from app.config import ASSETS, BASE_FEATURES, TARGET_COLUMN
    from app.utils.logger import get_logger
    from app.utils.validators import validate_market_data, ValidationError

logger = get_logger(__name__)


class MarketDataFetcher:
    # mapping of asset symbols to ticker symbols
    TICKER_MAP = {
        "GLD": "GLD",      # SPDR Gold Trust
        "SPX": "^GSPC",    # S&P 500
        "USO": "USO",      # US Oil Fund
        "SLV": "SLV",      # iShares Silver Trust
        "EURUSD": "EURUSD=X"  # EUR/USD
    }
    

    def __init__(self, retry_attempts: int = 3, timeout: int = 10):
        self.retry_attempts = retry_attempts
        self.timeout = timeout
        self.last_prices: Dict[str, float] = {}
        self.last_update_time: Optional[datetime] = None
        
        logger.info(f"MarketDataFetcher initialized with retry_attempts={retry_attempts}")
    

    def fetch_latest_prices(self, date: Optional[datetime] = None) -> Dict[str, float]:
        if date is None:
            date = datetime.now()
        
        logger.debug(f"Fetching latest prices for date {date.date()}")
        
        prices = {}
        
        for asset, ticker in self.TICKER_MAP.items():
            try:
                for attempt in range(self.retry_attempts):
                    try:
                        # fetch data with a small window around the target date
                        start_date = (date - timedelta(days=2)).strftime('%Y-%m-%d')
                        end_date = date.strftime('%Y-%m-%d')
                        
                        data = yf.download(
                            ticker,
                            start=start_date,
                            end=end_date,
                            progress=False
                        )
                        
                        if data.empty:
                            logger.warning(f"No data returned for {asset} ({ticker})")
                            
                            if attempt == self.retry_attempts - 1:
                                raise RuntimeError(f"Failed to fetch data for {asset}")
                            
                            continue
                        
                        # get the most recent closing price
                        latest_price = float(data['Close'].iloc[-1])
                        prices[asset] = latest_price
                        
                        logger.debug(f"Fetched {asset}: {latest_price:.6f}")
                        break
                        
                    except Exception as e:
                        if attempt == self.retry_attempts - 1:
                            logger.error(f"Failed to fetch {asset} after {self.retry_attempts} attempts: {e}")
                            raise
                        
                        logger.warning(f"Retry {attempt + 1}/{self.retry_attempts} for {asset}: {e}")
                        
            except Exception as e:
                logger.error(f"Error fetching price for {asset}: {e}")
                raise RuntimeError(f"Failed to fetch market data for {asset}: {e}")
        
        self.last_prices = prices
        self.last_update_time = datetime.now()
        
        logger.info(f"Successfully fetched prices for {len(prices)}/{len(self.TICKER_MAP)} assets")
        return prices
    

    def calculate_returns(self, current_prices: Dict[str, float], previous_prices: Dict[str, float]) -> Dict[str, float]:
        logger.debug("Calculating returns from price pairs")
        
        # validate inputs
        validate_market_data(current_prices, list(self.TICKER_MAP.keys()))
        validate_market_data(previous_prices, list(self.TICKER_MAP.keys()))
        
        returns = {}
        
        for asset in self.TICKER_MAP.keys():
            curr_price = current_prices[asset]
            prev_price = previous_prices[asset]
            
            if prev_price == 0:
                logger.warning(f"Previous price for {asset} is 0, cannot calculate return")
                raise ValidationError(f"Previous price for {asset} is 0")
            
            # calculate return
            ret = (curr_price - prev_price) / prev_price
            returns[asset + "_Return"] = float(ret)
            
            logger.debug(f"{asset}_Return: {ret:.6f} (curr: {curr_price:.6f}, prev: {prev_price:.6f})")
        
        logger.info(f"Calculated returns for {len(returns)} assets")
        return returns
    
    def fetch_and_calculate_returns(self, current_date: Optional[datetime] = None, previous_date: Optional[datetime] = None) -> Dict[str, float]:
        if current_date is None:
            current_date = datetime.now()
        
        if previous_date is None:
            previous_date = current_date - timedelta(days=1)
        
        logger.info(f"Fetching prices for {current_date.date()} and {previous_date.date()}")
        
        # fetch current prices
        current_prices = self.fetch_latest_prices(current_date)
        
        # fetch previous prices
        previous_prices = self.fetch_latest_prices(previous_date)
        
        # calculate returns
        returns = self.calculate_returns(current_prices, previous_prices)
        
        return returns
    
    def get_last_update_info(self) -> Dict:

        return {
            "last_prices": self.last_prices,
            "last_update_time": self.last_update_time.isoformat() if self.last_update_time else None,
            "assets": list(self.TICKER_MAP.keys())
        }


def create_sample_returns_from_csv(csv_path: str, rows: int = 10) -> list:

    import pandas as pd
    
    logger.debug(f"Loading sample data from {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # extract required columns
    required_cols = BASE_FEATURES + [TARGET_COLUMN]
    
    samples = []
    for idx in range(min(rows, len(df))):
        row_data = {}
        for col in required_cols:
            row_data[col] = df[col].iloc[idx]
        samples.append(row_data)
    
    logger.debug(f"Extracted {len(samples)} sample rows from CSV")
    return samples