import pandas as pd
import os
import pickle
from datetime import datetime, timedelta
from .data_fetcher import get_stock_data
from .eigenvalue_analyzer import EigenvalueAnalyzer


class StockScreener:
    """Screen stocks based on eigenvalue analysis"""

    def __init__(self, cache_dir="data/cache"):
        self.cache_dir = cache_dir
        self.analyzer = EigenvalueAnalyzer()
        os.makedirs(cache_dir, exist_ok=True)

    def screen_stocks(
        self,
        tickers,
        start_date=None,
        lookback_days=None,
        force_refresh=False,
        analyze_full_history=False,
    ):
        """
        Screen a list of stocks and return ranked results

        Parameters:
        - tickers: List of stock symbols to analyze
        - start_date: Specific start date (datetime) for data fetching
        - lookback_days: Alternative to start_date, days to look back from today
        - force_refresh: Whether to force refresh data from source
        - analyze_full_history: Whether to analyze the entire available data history
        """
        results = []

        # Calculate start date if not provided
        if start_date is None and lookback_days:
            start_date = datetime.now() - timedelta(days=lookback_days)
        elif start_date is None:
            # Default to 1 year lookback if neither is provided
            start_date = datetime.now() - timedelta(days=365)

        end_date = datetime.now()

        for ticker in tickers:
            # Try to load cached data first
            data = self._load_cached_data(ticker) if not force_refresh else None

            if data is not None:
                # Check if cached data covers the requested date range
                if data.index[0].date() > start_date.date():
                    # Need earlier data, refresh
                    data = None

            if data is None:
                # If no cached data or force_refresh, fetch new data
                data = get_stock_data(ticker, start_date, end_date)
                if data is not None:
                    self._cache_data(ticker, data)

            if data is not None:
                # Analyze the stock
                data.name = ticker  # Set the name for reference

                # If analyze_full_history is True, adjust the analyzer params temporarily
                if analyze_full_history:
                    # Store original settings
                    orig_sr_lookback = self.analyzer.params["sr_lookback"]
                    # Use the full dataset length or cap it at a reasonable limit
                    self.analyzer.params["sr_lookback"] = min(len(data), 500)

                analysis = self.analyzer.analyze_stock(data)

                # Restore original settings if needed
                if analyze_full_history:
                    self.analyzer.params["sr_lookback"] = orig_sr_lookback

                if analysis:
                    # Ensure the new metric exists, default to 0 if not
                    if "risk_adjusted_momentum" not in analysis:
                        analysis["risk_adjusted_momentum"] = 0.0
                    results.append(analysis)

        # Rank results by the new risk-adjusted momentum score
        if results:
            ranked_results = sorted(
                results,
                key=lambda x: x["risk_adjusted_momentum"],
                reverse=True,  # Rank by new metric
            )
            return ranked_results
        return []

    def _cache_data(self, ticker, data):
        """Save data to cache"""
        cache_file = os.path.join(self.cache_dir, f"{ticker}.pkl")
        with open(cache_file, "wb") as f:
            pickle.dump({"data": data, "timestamp": datetime.now()}, f)

    def _load_cached_data(self, ticker):
        """Load data from cache if fresh enough"""
        cache_file = os.path.join(self.cache_dir, f"{ticker}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
                # Check if cache is fresh (less than 1 day old)
                if datetime.now() - cached["timestamp"] < timedelta(days=1):
                    return cached["data"]
        return None
