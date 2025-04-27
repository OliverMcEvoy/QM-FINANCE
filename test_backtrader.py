import backtrader as bt
from datetime import datetime
import traceback

# --- Import local modules ---
# Make sure these paths are correct relative to where you run the script
try:
    from utils.data_fetcher import get_stock_data

    # Assuming quantum_momentum now uses SMA as per previous steps
    from strategies.quantum_momentum import QuantumMomentumStrategy
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print(
        "Ensure you run this script from the /home/olivermcevoy/QuantumEncrypt directory."
    )
    exit()


if __name__ == "__main__":
    print("--- Starting Standalone Backtrader Test ---")
    print(f"Using backtrader version: {bt.__version__}")  # Add version print

    # 1. Fetch Data
    ticker = "AAPL"
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 1, 1)
    print(
        f"Fetching data for {ticker} from {start_date.date()} to {end_date.date()}..."
    )
    data = get_stock_data(ticker, start_date, end_date)

    if data is None or data.empty:
        print("Failed to get data. Exiting.")
        exit()
    print("Data fetched successfully.")

    # 2. Initialize Cerebro
    cerebro = bt.Cerebro(stdstats=True)  # Keep standard stats on
    print("Cerebro initialized.")

    # 3. Add Data Feed
    try:
        data_feed = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(data_feed)
        print("Data feed added.")
    except Exception as e:
        print(f"Error adding data feed: {e}")
        traceback.print_exc()
        exit()

    # 4. Add Strategy (Using current SMA version)
    strategy_params = {
        "sma_period": 10,
        "prob_threshold": 0.6,
    }
    try:
        cerebro.addstrategy(QuantumMomentumStrategy, **strategy_params)
        print("Strategy added.")
    except Exception as e:
        print(f"Error adding strategy: {e}")
        traceback.print_exc()
        exit()

    # 5. Set Initial Cash & Commission
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001)
    print("Broker configured.")

    # 6. Run Backtest
    print("Running Cerebro...")
    try:
        results = cerebro.run()
        print("Cerebro run completed.")
        if results and results[0]:
            print("Strategy executed successfully.")
            print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
        else:
            print("Strategy did not produce results or failed internally.")

    except Exception as e:
        print(f"\n--- An error occurred during cerebro.run() ---")
        traceback.print_exc()
        print("------------------------------------------------")

    print("--- Standalone Backtrader Test Finished ---")

    # Optional: Plot if run was successful and you want to save/view it
    # try:
    #     print("Plotting results...")
    #     # This might open a window or save a file depending on matplotlib backend
    #     cerebro.plot(style='plotly', iplot=False, volume=False)
    # except Exception as e:
    #     print(f"Could not plot results: {e}")
