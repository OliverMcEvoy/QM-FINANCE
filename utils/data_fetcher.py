import yfinance as yf
import pandas as pd
import traceback
import os
import pickle
from datetime import datetime, timedelta


def get_stock_data(
    ticker, start_date, end_date, use_cache=True, cache_dir="data/cache"
):
    """Fetches historical stock data with caching."""
    try:
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(
            cache_dir,
            f"{ticker}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl",
        )

        # Try to load from cache if allowed
        if use_cache and os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
                cache_age = datetime.now() - cached["timestamp"]
                # Use cache if it's less than 24 hours old or if end_date is not today
                if (
                    cache_age < timedelta(hours=24)
                    or end_date.date() < datetime.now().date()
                ):
                    print(f"Using cached data for {ticker}")
                    return cached["data"]

        # Make sure we have proper datetime objects
        if isinstance(start_date, datetime):
            start_date_str = start_date.strftime("%Y-%m-%d")
        else:
            start_date_str = start_date

        if isinstance(end_date, datetime):
            end_date_str = end_date.strftime("%Y-%m-%d")
        else:
            end_date_str = end_date

        # Log what we're fetching
        print(f"Fetching {ticker} data from {start_date_str} to {end_date_str}")

        # Fetch fresh data
        data = yf.download(
            ticker, start=start_date_str, end=end_date_str, auto_adjust=False
        )

        if data.empty:
            print(f"No data found for {ticker} between {start_date} and {end_date}")
            return None

        # Standardize column names
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns.values]

        data.columns = [col.capitalize() for col in data.columns]

        # Save to cache
        if use_cache:
            with open(cache_file, "wb") as f:
                pickle.dump({"data": data, "timestamp": datetime.now()}, f)

        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        traceback.print_exc()
        return None
