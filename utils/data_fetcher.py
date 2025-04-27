import yfinance as yf
import pandas as pd
import traceback


def get_stock_data(ticker, start_date, end_date):
    """Fetches historical stock data."""
    try:
        # Use auto_adjust=True to get adjusted prices, simplifies things often
        # Or group_by='ticker' might help if downloading multiple tickers later
        data = yf.download(
            ticker, start=start_date, end=end_date, auto_adjust=False, group_by="column"
        )

        if data.empty:
            print(f"No data found for {ticker} between {start_date} and {end_date}")
            return None

        # --- Modify column flattening ---
        if isinstance(data.columns, pd.MultiIndex):
            print("MultiIndex columns detected. Flattening...")
            # Take the first level of the MultiIndex (e.g., 'Open', 'Close')
            data.columns = [
                col[0] for col in data.columns.values
            ]  # <-- Changed this line
            print(
                f"Columns after flattening: {data.columns.tolist()}"
            )  # Add print for debugging

        # Ensure column names are strings
        data.columns = data.columns.astype(str)
        # --- End of modified section ---

        # Ensure data has standard OHLCV column names for backtrader
        # Make rename case-insensitive by converting fetched columns to lower first
        data.columns = [
            col.lower() for col in data.columns
        ]  # Convert columns to lower case first
        print(
            f"Columns after lowercasing: {data.columns.tolist()}"
        )  # Add print for debugging

        # Define the standard names backtrader expects
        required_cols_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "adj close": "adj_close",  # yfinance often uses 'adj close'
            "volume": "volume",
        }

        # Rename columns based on the map
        data.rename(columns=required_cols_map, inplace=True)
        print(
            f"Columns after renaming: {data.columns.tolist()}"
        )  # Add print for debugging

        # Select only the columns backtrader needs
        final_cols = ["open", "high", "low", "close", "volume"]
        if "adj_close" in data.columns:
            # If you prefer adjusted close for backtesting, rename it to 'close'
            # data.rename(columns={'adj_close': 'close'}, inplace=True)
            # Or just include it if your strategy uses it separately
            final_cols.append("adj_close")

        # Filter data to keep only required columns that actually exist
        cols_to_keep = [col for col in final_cols if col in data.columns]
        missing_core_cols = [
            c
            for c in ["open", "high", "low", "close", "volume"]
            if c not in cols_to_keep
        ]

        if missing_core_cols:
            print(
                f"Error: Core OHLCV columns missing after processing: {missing_core_cols}. Available columns: {data.columns.tolist()}"
            )
            return None

        data = data[cols_to_keep]
        print(
            f"Final columns being passed to backtrader: {data.columns.tolist()}"
        )  # Add print for debugging

        # Ensure the index is a DatetimeIndex
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        print(f"Error fetching or processing data for {ticker}: {e}")
        traceback.print_exc()  # Print full traceback for debugging
        return None
