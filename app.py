import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator, HourLocator, MinuteLocator
import backtrader as bt
import numpy as np
import yfinance as yf
import io

# --- Import local modules ---
from utils.data_fetcher import get_stock_data
from strategies.better_quantum import CustomQuantumStrategy

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide")
st.title("Quantum-Inspired Trading Bot Backtester")

# --- Sidebar Controls ---
st.sidebar.header("Backtest Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", "AAPL")  # Default ticker
start_date_str = st.sidebar.date_input("Start Date", datetime(2018, 4, 25))
end_date_str = st.sidebar.date_input("End Date", datetime(2025, 4, 25))
start_cash = st.sidebar.number_input(
    "Initial Cash", value=10000.0, min_value=100.0, step=1000.0
)

# Strategy Selection (Add more strategies to the dictionary as you create them)
available_strategies = {
    "CustomQuantum": CustomQuantumStrategy,
}
strategy_name = st.sidebar.selectbox(
    "Select Strategy", list(available_strategies.keys())
)
SelectedStrategy = available_strategies[strategy_name]

# Strategy-specific parameters
st.sidebar.subheader(f"{strategy_name} Parameters")
strategy_params = {}
if strategy_name == "CustomQuantum":
    sma_period = st.sidebar.slider("SMA Period", 5, 50, 10)
    prob_threshold = st.sidebar.slider("Probability Threshold", 0.5, 0.95, 0.6, 0.01)
    quantum_factor = st.sidebar.slider("Quantum Factor", 0.1, 1.0, 0.5, 0.05)
    phase_period = st.sidebar.slider("Phase Period", 5, 50, 20)
    uncertainty = st.sidebar.slider("Uncertainty Factor", 0.0, 1.0, 0.2, 0.05)
    wf_components = st.sidebar.slider("Wavefunction Components", 1, 20, 5)
    wf_lookback = st.sidebar.slider("Wavefunction Lookback", 20, 200, 50)

    # Add new parameters for support/resistance eigenvalues
    st.sidebar.subheader("Support/Resistance Parameters")
    sr_lookback = st.sidebar.slider("S/R History Lookback", 60, 250, 120)
    eigenvalue_smoothing = st.sidebar.slider(
        "Eigenvalue Smoothing", 0.0, 1.0, 0.8, 0.05
    )
    price_history_weight = st.sidebar.slider(
        "Price History Weight", 0.0, 1.0, 0.6, 0.05
    )
    eigenvalue_count = st.sidebar.slider("Support/Resistance Levels", 2, 10, 5)

    # New time evolution parameters
    st.sidebar.subheader("Time Evolution Parameters")
    time_evolution_rate = st.sidebar.slider("Time Evolution Rate", 0.0, 1.0, 0.2, 0.05)
    eigenvalue_persistence = st.sidebar.slider(
        "Eigenvalue Persistence", 0.0, 1.0, 0.7, 0.05
    )
    memory_decay = st.sidebar.slider("Memory Decay", 0.0, 0.5, 0.05, 0.01)

    # New eigenvalue trading parameters
    st.sidebar.subheader("Eigenvalue Trading Parameters")
    eigenvalue_buy_threshold = st.sidebar.slider(
        "Eigenvalue Buy Threshold (%)", 0.01, 0.1, 0.02, 0.005, format="%.3f"
    )
    eigenvalue_sell_threshold = st.sidebar.slider(
        "Eigenvalue Sell Threshold (%)", 0.01, 0.1, 0.02, 0.005, format="%.3f"
    )
    eigenvalue_signal_weight = st.sidebar.slider(
        "Eigenvalue Signal Weight", 0.0, 1.0, 0.65, 0.05
    )
    max_position_pct = st.sidebar.slider(
        "Max Position Size (% Cash)", 0.1, 1.0, 0.95, 0.05
    )

    strategy_params = {
        "sma_period": sma_period,
        "prob_threshold": prob_threshold,
        "quantum_factor": quantum_factor,
        "phase_period": phase_period,
        "uncertainty": uncertainty,
        "wf_components": wf_components,
        "wf_lookback": wf_lookback,
        "sr_lookback": sr_lookback,
        "eigenvalue_smoothing": eigenvalue_smoothing,
        "price_history_weight": price_history_weight,
        "eigenvalue_count": eigenvalue_count,
        "time_evolution_rate": time_evolution_rate,
        "eigenvalue_persistence": eigenvalue_persistence,
        "memory_decay": memory_decay,
        "eigenvalue_buy_threshold": eigenvalue_buy_threshold,
        "eigenvalue_sell_threshold": eigenvalue_sell_threshold,
        "eigenvalue_signal_weight": eigenvalue_signal_weight,
        "max_position_pct": max_position_pct,
    }

run_button = st.sidebar.button("Run Backtest")

# --- Main Panel ---
st.header("Backtest Results")

if run_button:
    if not ticker:
        st.error("Please enter a ticker symbol.")
    else:
        # Determine split point
        today = datetime.now()
        week_ago = today - timedelta(days=7)
        start_dt = pd.to_datetime(start_date_str)
        hist_end = min(pd.to_datetime(end_date_str), week_ago)

        # 1) Fetch daily history up to one week ago
        with st.spinner("Fetching historical data..."):
            hist_data = get_stock_data(ticker, start_dt, hist_end)

        if hist_data is None or hist_data.empty:
            st.error(
                f"Could not fetch historical data for {ticker}. Please check the ticker and date range."
            )
            st.stop()

        # 2) Fetch minute bars for the last 7 days
        with st.spinner("Fetching minute data..."):
            minute_data = yf.download(ticker, period="7d", interval="1m")
            minute_data.dropna(inplace=True)

        if minute_data.empty:
            st.error(
                f"Could not fetch minute data for {ticker}. The market might be closed or the ticker invalid."
            )
            st.stop()

        # Rename columns for both feeds to Backtrader standard
        def rename_columns(df):
            # Handle potential MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten MultiIndex: Use the first level (e.g., 'Open', 'Close')
                # Ensure the resulting column names are strings
                df.columns = [str(col[0]) for col in df.columns.values]
            else:
                # Ensure regular column index elements are strings
                df.columns = [str(col) for col in df.columns]

            cols = {
                col.lower(): col for col in df.columns
            }  # Now columns should be strings
            rename_map = {}
            if "open" in cols:
                rename_map[cols["open"]] = "Open"
            if "high" in cols:
                rename_map[cols["high"]] = "High"
            if "low" in cols:
                rename_map[cols["low"]] = "Low"
            if "close" in cols:
                rename_map[cols["close"]] = "Close"
            if "volume" in cols:
                rename_map[cols["volume"]] = "Volume"
            if "adj close" in cols:
                rename_map[cols["adj close"]] = "Adj Close"
            # Note: 'adj_close' handling was already in data_fetcher, might not be needed here
            # but keeping it doesn't hurt if yf changes format for minute data.
            elif "adj_close" in cols:
                rename_map[cols["adj_close"]] = "Adj Close"
            return df.rename(columns=rename_map)

        hist_bt = rename_columns(
            hist_data.copy()
        )  # Use copy to avoid modifying original
        minute_bt = rename_columns(
            minute_data.copy()
        )  # Use copy to avoid modifying original

        # Ensure required columns exist
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in hist_bt.columns for col in required_cols):
            st.error(
                f"Historical data missing required columns: {required_cols}. Available: {hist_bt.columns.tolist()}"
            )
            st.stop()
        if not all(col in minute_bt.columns for col in required_cols):
            st.error(
                f"Minute data missing required columns: {required_cols}. Available: {minute_bt.columns.tolist()}"
            )
            st.stop()

        # --- Historical backtest to init eigenvalues ---
        cerebro_hist = bt.Cerebro()
        # Disable trading for the historical run
        hist_params = strategy_params.copy()
        hist_params["trading_enabled"] = False
        cerebro_hist.addstrategy(SelectedStrategy, **hist_params)
        hist_feed = bt.feeds.PandasData(dataname=hist_bt)
        cerebro_hist.adddata(hist_feed)
        cerebro_hist.broker.setcash(start_cash)
        cerebro_hist.broker.setcommission(commission=0.001)
        hist_strat = None  # Initialize hist_strat
        with st.spinner("Running historical backtest (state initialization)..."):
            try:
                hist_strats = cerebro_hist.run()
                if not hist_strats:
                    st.error("Historical backtest failed to run.")
                    st.stop()
                hist_strat = hist_strats[0]  # Assign hist_strat here
            except Exception as e:
                st.error(f"Error during historical backtest: {e}")
                st.stop()

        # --- Plot Historical Eigenvalues ---
        if hist_strats and hasattr(hist_strat, "get_wavefunction_data"):
            hist_wf_data = hist_strat.get_wavefunction_data()
            hist_dates = hist_wf_data.get("dates", [])
            hist_prices = hist_wf_data.get("prices", [])
            hist_eigenvalue_history = hist_wf_data.get("eigenvalue_history", [])

            # Ensure alignment for historical data
            hist_min_len = min(
                len(hist_dates), len(hist_prices), len(hist_eigenvalue_history)
            )
            hist_dates = hist_dates[:hist_min_len]
            hist_prices = hist_prices[:hist_min_len]
            hist_eigenvalue_history = hist_eigenvalue_history[:hist_min_len]

            if hist_dates and hist_prices and hist_eigenvalue_history:
                st.subheader("Historical Eigenvalue Evolution (Initialization Period)")
                fig_hist_eig, ax_hist_eig = plt.subplots(
                    figsize=(12, 6), facecolor="none"
                )
                ax_hist_eig.set_facecolor("none")
                ax_hist_eig.plot(
                    hist_dates,
                    hist_prices,
                    label="Historical Price",
                    color="grey",
                    linewidth=2.5,  # Changed from 1.5 to 2.5
                    zorder=1,
                    alpha=0.8,
                )

                hist_max_levels = 0
                if hist_eigenvalue_history:
                    # Filter out empty lists before finding max length
                    valid_hist_eigenvalues = [h for h in hist_eigenvalue_history if h]
                    if valid_hist_eigenvalues:
                        hist_max_levels = max(len(h) for h in valid_hist_eigenvalues)
                        hist_max_levels = min(
                            hist_max_levels, strategy_params.get("eigenvalue_count", 5)
                        )  # Use configured count

                if hist_max_levels > 0:
                    for i in range(hist_max_levels):
                        vals, ds = [], []
                        for j, h in enumerate(hist_eigenvalue_history):
                            if h and i < len(h):
                                ds.append(hist_dates[j])
                                vals.append(h[i])
                        if ds:
                            color_val = (
                                i / (hist_max_levels - 1)
                                if hist_max_levels > 1
                                else 0.5
                            )
                            ax_hist_eig.plot(
                                ds,
                                vals,
                                label=f"Hist Level {i+1}",
                                linewidth=1,
                                alpha=0.7,
                                color=plt.cm.viridis(color_val),
                            )

                hist_locator = AutoDateLocator(minticks=5, maxticks=12)
                hist_formatter = DateFormatter("%Y-%m-%d")
                ax_hist_eig.xaxis.set_major_locator(hist_locator)
                ax_hist_eig.xaxis.set_major_formatter(hist_formatter)
                fig_hist_eig.autofmt_xdate()

                for spine in ax_hist_eig.spines.values():
                    spine.set_color("white")
                    spine.set_linewidth(2)
                ax_hist_eig.tick_params(axis="x", colors="white", labelsize=10, width=2)
                ax_hist_eig.tick_params(axis="y", colors="white", labelsize=10, width=2)
                ax_hist_eig.set_xlabel(
                    "Date", fontsize=12, color="white", weight="bold"
                )
                ax_hist_eig.set_ylabel(
                    "Price", fontsize=12, color="white", weight="bold"
                )
                ax_hist_eig.legend(framealpha=0.3, fontsize=8)
                fig_hist_eig.patch.set_alpha(0.0)
                ax_hist_eig.patch.set_alpha(0.0)
                st.pyplot(fig_hist_eig)
            else:
                st.warning(
                    "Could not retrieve sufficient data for historical eigenvalue plot."
                )
        # --- End Historical Eigenvalue Plot ---

        # --- Minute-by-minute simulation for last week ---
        minute_params = strategy_params.copy()
        # Ensure trading is enabled for the minute run (rely on default or explicitly set)
        minute_params["trading_enabled"] = True  # Explicitly enable trading
        # Pass the final state from the historical run
        minute_params.update(
            {
                "initial_smoothed_eigenvalues": getattr(
                    hist_strat, "smoothed_eigenvalues", None
                ),
                "initial_potential_wells": getattr(hist_strat, "potential_wells", []),
                "initial_quantum_state": getattr(
                    hist_strat, "quantum_state", np.array([0.5, 0.5])
                ),
            }
        )

        # Create a unique class name for the minute strategy to avoid conflicts
        MinuteStrategy = type(
            f"{strategy_name}_Minute_{id(minute_params)}", (SelectedStrategy,), {}
        )

        cerebro_min = bt.Cerebro()
        cerebro_min.addstrategy(MinuteStrategy, **minute_params)
        min_feed = bt.feeds.PandasData(dataname=minute_bt)
        cerebro_min.adddata(min_feed)
        # Carry forward cash and commissions from the *initial* state, not the historical run's end value
        cerebro_min.broker.setcash(
            start_cash
        )  # Start minute simulation with initial cash
        cerebro_min.broker.setcommission(commission=0.001)
        strat = None  # Initialize strat
        final_value = start_cash  # Initialize final_value
        with st.spinner(
            "Running minute-by-minute simulation (trading enabled)..."
        ):  # Modified spinner text
            try:
                min_strats = cerebro_min.run()
                if not min_strats:
                    st.error("Minute-by-minute simulation failed to run.")
                    st.stop()
                strat = min_strats[0]
                final_value = cerebro_min.broker.getvalue()
            except Exception as e:
                st.error(f"Error during minute-by-minute simulation: {e}")
                st.stop()

        # --- Display Results ---
        st.success(
            f"Backtest complete for {ticker} from {start_date_str} to {end_date_str} (Minute data for last 7 days)."
        )

        # 4. Display Basic Results
        st.subheader("Performance Metrics")
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-around; font-size: 1.5em; margin-bottom: 20px;">
                <div style="text-align: center;">
                    <strong>Initial Portfolio Value</strong><br>
                    ${start_cash:,.2f}
                </div>
                <div style="text-align: center;">
                    <strong>Final Portfolio Value</strong><br>
                    ${final_value:,.2f}
                </div>
                 <div style="text-align: center;">
                    <strong>Total Return</strong><br>
                    {((final_value - start_cash) / start_cash) * 100:,.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 5. Plot Price with Buy/Sell Signals (Minute Data - Last 7 Days Combined)
        st.subheader("Minute Price Chart with Buy/Sell Signals (Last 7 Days)")
        fig_combined, ax_combined = plt.subplots(figsize=(12, 6), facecolor="none")
        ax_combined.set_facecolor("none")
        ax_combined.grid(False)

        ax_combined.plot(
            minute_bt.index,
            minute_bt["Close"],
            label="Close Price",
            color="steelblue",
            linewidth=2.5,  # Changed from 1.5 to 2.5
            zorder=1,
        )
        locator_combined = AutoDateLocator(
            minticks=5, maxticks=12
        )  # Adjust ticks for minute data
        formatter_combined = DateFormatter("%m-%d %H:%M")  # Format for minute data
        ax_combined.xaxis.set_major_locator(locator_combined)
        ax_combined.xaxis.set_major_formatter(formatter_combined)
        fig_combined.autofmt_xdate()

        if strat is not None and hasattr(strat, "buy_signals") and strat.buy_signals:
            buy_dates, buy_prices, buy_values = zip(*strat.buy_signals)
            ax_combined.scatter(
                buy_dates,
                buy_prices,
                marker="^",
                color="white",
                edgecolor="green",  # Add edge color
                s=70,
                label="Buys",
                zorder=3,
            )

        if strat is not None and hasattr(strat, "sell_signals") and strat.sell_signals:
            sell_dates, sell_prices, sell_values = zip(*strat.sell_signals)
            ax_combined.scatter(
                sell_dates,
                sell_prices,
                marker="v",
                color="white",
                edgecolor="red",  # Add edge color
                s=70,
                label="Sells",
                zorder=3,
            )

        for spine in ax_combined.spines.values():
            spine.set_color("white")
            spine.set_linewidth(2)

        ax_combined.tick_params(
            axis="x", colors="white", labelsize=10, width=2
        )  # Smaller labels
        ax_combined.tick_params(axis="y", colors="white", labelsize=10, width=2)
        ax_combined.set_xlabel("Date / Time", fontsize=12, color="white", weight="bold")
        ax_combined.set_ylabel("Price", fontsize=12, color="white", weight="bold")
        ax_combined.legend(framealpha=0.3)
        fig_combined.patch.set_alpha(0.0)
        ax_combined.patch.set_alpha(0.0)
        st.pyplot(fig_combined)

        # 6. Quantum Visualizations for minute-by-minute simulation
        if (
            strat is not None
            and strategy_name == "CustomQuantum"
            and hasattr(strat, "get_wavefunction_data")
        ):
            wf_data = strat.get_wavefunction_data()

            # Extract data from strategy
            all_dates = wf_data.get("dates", [])
            all_prices = wf_data.get("prices", [])
            all_wavefunction = wf_data.get("wavefunction", [])
            all_eigenvalue_history = wf_data.get("eigenvalue_history", [])
            all_buy_signals = wf_data.get("buy_signals", [])
            all_sell_signals = wf_data.get("sell_signals", [])
            all_trades = wf_data.get("trades", [])

            # Ensure data alignment for minute data
            min_len = min(
                len(all_dates),
                len(all_prices),
                len(all_wavefunction),
                len(all_eigenvalue_history),
            )
            dates = all_dates[:min_len]
            prices = all_prices[:min_len]
            wavefunction = all_wavefunction[:min_len]
            history = all_eigenvalue_history[:min_len]

            # Convert dates to pandas DatetimeIndex for easier filtering
            dates_idx = pd.to_datetime(dates)

            # 6a. Wavefunction vs. Price (Combined 7 days)
            st.subheader("Quantum Wavefunction vs. Price (Minute Data - Last 7 Days)")
            fig2, ax2 = plt.subplots(figsize=(12, 6), facecolor="none")
            ax2.set_facecolor("none")
            ax2.plot(
                dates,
                prices,
                label="Actual Price",
                color="steelblue",
                alpha=0.7,
                linewidth=2.5,  # Changed from 1.5 to 2.5
            )
            ax2.plot(
                dates,
                wavefunction,
                label="Quantum Wavefunction",
                color="#E833FF",
                linewidth=1.5,
            )
            ax2.xaxis.set_major_locator(
                locator_combined
            )  # Use same locator as combined price chart
            ax2.xaxis.set_major_formatter(formatter_combined)  # Use same formatter
            fig2.autofmt_xdate()
            for spine in ax2.spines.values():
                spine.set_color("white")
                spine.set_linewidth(2)
            ax2.tick_params(axis="x", colors="white", labelsize=10, width=2)
            ax2.tick_params(axis="y", colors="white", labelsize=10, width=2)
            ax2.set_xlabel("Date / Time", fontsize=12, color="white", weight="bold")
            ax2.set_ylabel("Value", fontsize=12, color="white", weight="bold")
            ax2.legend(framealpha=0.3)
            fig2.patch.set_alpha(0.0)
            ax2.patch.set_alpha(0.0)
            st.pyplot(fig2)

            # 6b. Eigenvalue Evolution (Combined 7 days)
            st.subheader(
                "Eigenvalue Evolution (Support/Resistance Levels - Minute Data - Last 7 Days)"
            )
            fig4, ax4 = plt.subplots(figsize=(12, 6), facecolor="none")
            ax4.set_facecolor("none")

            # determine max levels dynamically
            max_levels = 0
            valid_history = [h for h in history if h]  # Filter out empty lists
            if valid_history:
                max_levels = max(len(h) for h in valid_history)
                max_levels = min(
                    max_levels, strategy_params.get("eigenvalue_count", 5)
                )  # Use configured count

            if max_levels > 0:
                for i in range(max_levels):
                    vals, ds = [], []
                    for j, h in enumerate(history):
                        if h and i < len(
                            h
                        ):  # Check if history entry exists and has enough levels
                            ds.append(dates[j])
                            vals.append(h[i])
                    if ds:  # Only plot if there's data for this level
                        color_val = i / (max_levels - 1) if max_levels > 1 else 0.5
                        ax4.plot(
                            ds,
                            vals,
                            label=f"Level {i+1}",
                            linewidth=1,
                            alpha=0.8,
                            color=plt.cm.rainbow(color_val),
                        )

            ax4.plot(
                dates,
                prices,
                label="Price",
                color="steelblue",
                linewidth=2.5,
                zorder=1,  # Changed from 1.5 to 2.5
            )
            ax4.xaxis.set_major_locator(locator_combined)
            ax4.xaxis.set_major_formatter(formatter_combined)
            fig4.autofmt_xdate()
            for spine in ax4.spines.values():
                spine.set_color("white")
                spine.set_linewidth(2)
            ax4.tick_params(axis="x", colors="white", labelsize=10, width=2)
            ax4.tick_params(axis="y", colors="white", labelsize=10, width=2)
            ax4.set_xlabel("Date / Time", fontsize=12, color="white", weight="bold")
            ax4.set_ylabel("Price", fontsize=12, color="white", weight="bold")
            ax4.legend(framealpha=0.3)
            fig4.patch.set_alpha(0.0)
            ax4.patch.set_alpha(0.0)
            st.pyplot(fig4)

            # 6c. Combined Quantum Trading Visualization (Combined 7 days)
            st.subheader("Combined Quantum Visualization (Minute Data - Last 7 Days)")
            fig5, ax5 = plt.subplots(figsize=(12, 6), facecolor="none")
            ax5.set_facecolor("none")
            ax5.plot(
                dates,
                prices,
                label="Price",
                color="steelblue",
                linewidth=2.5,  # Changed from 1.5 to 2.5
                alpha=0.9,
                zorder=1,
            )
            ax5.plot(
                dates,
                wavefunction,
                label="Wavefunction",
                color="#E833FF",
                linewidth=1,
                alpha=0.5,
                zorder=2,
            )

            if max_levels > 0:
                for i in range(max_levels):
                    vals, ds = [], []
                    for j, h in enumerate(history):
                        if h and i < len(h):
                            ds.append(dates[j])
                            vals.append(h[i])
                    if ds:
                        color_val = i / (max_levels - 1) if max_levels > 1 else 0.5
                        ax5.plot(
                            ds,
                            vals,
                            linestyle="--",
                            linewidth=1,
                            alpha=0.6,
                            color=plt.cm.rainbow(color_val),
                            zorder=3,
                            label=f"Level {i+1}" if i == 0 else f"_Level {i+1}",
                        )  # Only label first level

            if all_buy_signals:
                bd, bp, _ = zip(*all_buy_signals)
                ax5.scatter(
                    bd,
                    bp,
                    marker="^",
                    s=80,
                    edgecolor="green",
                    facecolor="white",
                    label="Buys",
                    zorder=5,
                )
            if all_sell_signals:
                sd, sp, _ = zip(*all_sell_signals)
                ax5.scatter(
                    sd,
                    sp,
                    marker="v",
                    s=80,
                    edgecolor="red",
                    facecolor="white",
                    label="Sells",
                    zorder=5,
                )

            handles, labels = ax5.get_legend_handles_labels()  # Get handles

            ax5.xaxis.set_major_locator(locator_combined)
            ax5.xaxis.set_major_formatter(formatter_combined)
            fig5.autofmt_xdate()
            for spine in ax5.spines.values():
                spine.set_color("white")
                spine.set_linewidth(2)

            ax5.tick_params(axis="x", colors="white", labelsize=10, width=2)
            ax5.tick_params(axis="y", colors="white", labelsize=10, width=2)
            ax5.set_xlabel("Date / Time", fontsize=12, color="white", weight="bold")
            ax5.set_ylabel("Price / Value", fontsize=12, color="white", weight="bold")
            fig5.patch.set_alpha(0.0)
            ax5.patch.set_alpha(0.0)

            # Combine legends
            ax5.legend(handles, labels, framealpha=0.3, loc="upper left", fontsize=8)
            st.pyplot(fig5)

            # --- START: Daily Plots ---
            st.subheader("Daily Trading Session Visualizations (Last 7 Days)")
            unique_days = dates_idx.normalize().unique()  # Get unique days

            for day in unique_days:
                day_str = day.strftime("%Y-%m-%d")
                st.markdown(f"#### {day_str}")

                # Filter data for the current day
                day_mask = (dates_idx >= day) & (dates_idx < day + timedelta(days=1))
                day_dates = [d for d, m in zip(dates, day_mask) if m]
                day_prices = [p for p, m in zip(prices, day_mask) if m]
                day_wavefunction = [w for w, m in zip(wavefunction, day_mask) if m]
                day_history = [h for h, m in zip(history, day_mask) if m]

                day_buy_signals = [
                    (d, p, v)
                    for d, p, v in all_buy_signals
                    if day <= d < day + timedelta(days=1)
                ]
                day_sell_signals = [
                    (d, p, v)
                    for d, p, v in all_sell_signals
                    if day <= d < day + timedelta(days=1)
                ]

                if not day_dates:  # Skip if no data for this day (e.g., weekend)
                    st.write("No trading data for this day.")
                    continue

                fig_day, ax_day = plt.subplots(
                    figsize=(12, 5), facecolor="none"
                )  # Slightly smaller height
                ax_day.set_facecolor("none")

                ax_day.plot(
                    day_dates,
                    day_prices,
                    label="Price",
                    color="steelblue",
                    linewidth=2.5,  # Changed from 1.5 to 2.5
                    alpha=0.9,
                    zorder=1,
                )
                ax_day.plot(
                    day_dates,
                    day_wavefunction,
                    label="Wavefunction",
                    color="#E833FF",
                    linewidth=1,
                    alpha=0.5,
                    zorder=2,
                )

                day_max_levels = 0
                valid_day_history = [h for h in day_history if h]
                if valid_day_history:
                    day_max_levels = max(len(h) for h in valid_day_history)
                    day_max_levels = min(
                        day_max_levels, strategy_params.get("eigenvalue_count", 5)
                    )

                if day_max_levels > 0:
                    for i in range(day_max_levels):
                        vals, ds = [], []
                        for j, h in enumerate(day_history):
                            if h and i < len(h):
                                ds.append(day_dates[j])
                                vals.append(h[i])
                        if ds:
                            color_val = (
                                i / (day_max_levels - 1) if day_max_levels > 1 else 0.5
                            )
                            ax_day.plot(
                                ds,
                                vals,
                                linestyle="--",
                                linewidth=1,
                                alpha=0.6,
                                color=plt.cm.rainbow(color_val),
                                zorder=3,
                                label=f"Level {i+1}" if i == 0 else f"_Level {i+1}",
                            )

                if day_buy_signals:
                    bd, bp, _ = zip(*day_buy_signals)
                    ax_day.scatter(
                        bd,
                        bp,
                        marker="^",
                        s=80,
                        edgecolor="green",
                        facecolor="white",
                        label="Buys",
                        zorder=5,
                    )
                if day_sell_signals:
                    sd, sp, _ = zip(*day_sell_signals)
                    ax_day.scatter(
                        sd,
                        sp,
                        marker="v",
                        s=80,
                        edgecolor="red",
                        facecolor="white",
                        label="Sells",
                        zorder=5,
                    )

                handles_day, labels_day = ax_day.get_legend_handles_labels()

                # Formatting for daily plots (more granular time)
                ax_day.xaxis.set_major_locator(
                    HourLocator(interval=1)
                )  # Tick every hour
                ax_day.xaxis.set_minor_locator(
                    MinuteLocator(interval=15)
                )  # Minor tick every 15 mins
                ax_day.xaxis.set_major_formatter(
                    DateFormatter("%H:%M")
                )  # Show Hour:Minute
                fig_day.autofmt_xdate()

                for spine in ax_day.spines.values():
                    spine.set_color("white")
                    spine.set_linewidth(1.5)
                ax_day.tick_params(axis="x", colors="white", labelsize=9, width=1.5)
                ax_day.tick_params(axis="y", colors="white", labelsize=9, width=1.5)
                ax_day.set_xlabel(
                    "Time (HH:MM)", fontsize=10, color="white", weight="bold"
                )
                ax_day.set_ylabel(
                    "Price / Value", fontsize=10, color="white", weight="bold"
                )
                fig_day.patch.set_alpha(0.0)
                ax_day.patch.set_alpha(0.0)
                ax_day.legend(
                    handles_day,
                    labels_day,
                    framealpha=0.3,
                    loc="upper left",
                    fontsize=8,
                )
                st.pyplot(fig_day)
                plt.close(fig_day)  # Close the figure to free memory

            # --- END: Daily Plots ---

            # Explanation and Trade Log
            st.subheader("Strategy Explanation & Trade Log")
            st.markdown(
                f"""
                ### Eigenvalue‑Based Trading Strategy

                This strategy uses concepts inspired by quantum mechanics to identify potential trading opportunities.
                - **Wavefunction:** Models the probable distribution of future prices based on recent history ({strategy_params['wf_lookback']} periods).
                - **Hamiltonian:** Represents the 'energy landscape' of the market, combining price momentum (kinetic energy) and price structure (potential energy).
                - **Eigenvalues:** Calculated from the Hamiltonian, these represent stable price levels (support/resistance). The strategy tracks {strategy_params['eigenvalue_count']} levels.
                - **Potential Wells:** Areas of price congestion identified from history ({strategy_params['sr_lookback']} periods) that attract eigenvalues.
                - **Time Evolution:** Eigenvalues and potential wells evolve over time based on market dynamics (Persistence: {strategy_params['eigenvalue_persistence']}, Rate: {strategy_params['time_evolution_rate']}).
                - **Trading Signals:**
                    - *Quantum Probability:* Based on price deviation from SMA ({strategy_params['sma_period']}) and market phase ({strategy_params['phase_period']}), adjusted by uncertainty ({strategy_params['uncertainty']}).
                    - *Eigenvalue Signal:* Generated when the price approaches the lowest eigenvalue (buy signal, threshold: {strategy_params['eigenvalue_buy_threshold']*100:.1f}%) or the highest eigenvalue (sell signal, threshold: {strategy_params['eigenvalue_sell_threshold']*100:.1f}%).
                - **Combined Decision:** The final buy/sell probability blends the quantum probability and the eigenvalue signal (Eigenvalue Weight: {strategy_params['eigenvalue_signal_weight']}). A trade is triggered if the probability exceeds the threshold ({strategy_params['prob_threshold']}).
                - **Position Sizing:** Buys use up to {strategy_params['max_position_pct']*100:.0f}% of available cash. Sells close the entire position.

                ### Trade Log (Minute Simulation - Last 7 Days)
                """
            )

            if all_trades:
                trades_df = pd.DataFrame(all_trades)
                # Ensure dates are timezone-naive before formatting
                trades_df["buy_date"] = pd.to_datetime(
                    trades_df["buy_date"]
                ).dt.tz_localize(None)
                trades_df["sell_date"] = pd.to_datetime(
                    trades_df["sell_date"]
                ).dt.tz_localize(None)

                trades_df["buy_date"] = trades_df["buy_date"].dt.strftime(
                    "%Y-%m-%d %H:%M"
                )
                trades_df["sell_date"] = trades_df["sell_date"].dt.strftime(
                    "%Y-%m-%d %H:%M"
                )

                trades_df["pnl_pct"] = (
                    (trades_df["sell_price"] - trades_df["buy_price"])
                    / trades_df["buy_price"]
                    * 100
                )
                trades_df_display = trades_df[
                    [
                        "buy_date",
                        "buy_price",
                        "buy_value",
                        "sell_date",
                        "sell_price",
                        "sell_value",
                        "shares",
                        "total_pnl",
                        "pnl_pct",
                    ]
                ]
                trades_df_display.columns = [
                    "Buy Date",
                    "Buy Price",
                    "Buy Value ($)",
                    "Sell Date",
                    "Sell Price",
                    "Sell Value ($)",
                    "Shares",
                    "PnL ($)",
                    "PnL (%)",
                ]
                # Format columns
                trades_df_display["Buy Price"] = trades_df_display["Buy Price"].map(
                    "{:,.2f}".format
                )
                trades_df_display["Buy Value ($)"] = trades_df_display[
                    "Buy Value ($)"
                ].map("{:,.2f}".format)
                trades_df_display["Sell Price"] = trades_df_display["Sell Price"].map(
                    "{:,.2f}".format
                )
                trades_df_display["Sell Value ($)"] = trades_df_display[
                    "Sell Value ($)"
                ].map("{:,.2f}".format)
                trades_df_display["PnL ($)"] = trades_df_display["PnL ($)"].map(
                    "{:,.2f}".format
                )
                trades_df_display["PnL (%)"] = trades_df_display["PnL (%)"].map(
                    "{:,.2f}%".format
                )

                # Define coloring function safely
                def color_pnl(val):
                    color = "white"  # Default
                    try:
                        # Check if it's a string representation of a number/percentage
                        if isinstance(val, str):
                            numeric_val_str = val.replace("%", "").replace(",", "")
                            numeric_val = float(numeric_val_str)
                        elif isinstance(val, (int, float)):
                            numeric_val = val
                        else:
                            return (
                                f"color: {color}"  # Return default if not convertible
                            )

                        if numeric_val > 0:
                            color = "lightgreen"
                        elif numeric_val < 0:
                            color = "lightcoral"
                    except (ValueError, TypeError):
                        pass  # Keep default color if conversion fails
                    return f"color: {color}"

                st.dataframe(
                    trades_df_display.style.applymap(
                        color_pnl, subset=["PnL ($)", "PnL (%)"]
                    )
                )
            else:
                st.write("No trades executed during the minute simulation period.")

        elif (
            strat
        ):  # If strat exists but it's not CustomQuantum or doesn't have get_wavefunction_data
            st.warning(
                "Selected strategy does not provide detailed quantum visualization data."
            )
        # else: # If strat is None (simulation failed earlier) - error already shown

else:
    st.info(
        "Configure the backtest parameters in the sidebar and click 'Run Backtest'."
    )

# --- Helper to close figures ---
# (Optional: Explicitly close all figures at the end, though Streamlit usually handles this)
# plt.close('all')
