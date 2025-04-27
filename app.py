import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator
import backtrader as bt
import numpy as np

# --- Import local modules ---
from utils.data_fetcher import get_stock_data
from strategies.quantum_momentum import QuantumMomentumStrategy  # Import the strategy
from strategies.better_quantum import (
    CustomQuantumStrategy,
)  # Import the custom strategy

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

if strategy_name == "QuantumMomentum":
    sma_period = st.sidebar.slider("SMA Period", 5, 50, 10)
    prob_threshold = st.sidebar.slider("Probability Threshold", 0.5, 0.95, 0.6, 0.01)
    entanglement_lookback = st.sidebar.slider("Entanglement Lookback", 5, 50, 20)

    # For phase periods, we'll use a multiselect with default values
    phase_period_options = list(range(5, 51, 5))
    default_phases = [5, 10, 20]
    phase_periods = st.sidebar.multiselect(
        "Phase Periods", options=phase_period_options, default=default_phases
    )

    interference_weight = st.sidebar.slider("Interference Weight", 0.0, 1.0, 0.5, 0.05)
    uncertainty_factor = st.sidebar.slider("Uncertainty Factor", 0.0, 1.0, 0.2, 0.05)
    tunneling_threshold = st.sidebar.slider("Tunneling Threshold", 0.5, 5.0, 2.0, 0.1)
    superposition_count = st.sidebar.slider("Superposition Count", 1, 10, 3, 1)

    strategy_params = {
        "sma_period": sma_period,
        "prob_threshold": prob_threshold,
        "entanglement_lookback": entanglement_lookback,
        "phase_periods": phase_periods,
        "interference_weight": interference_weight,
        "uncertainty_factor": uncertainty_factor,
        "tunneling_threshold": tunneling_threshold,
        "superposition_count": superposition_count,
    }

elif strategy_name == "CustomQuantum":
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
    }

run_button = st.sidebar.button("Run Backtest")

# --- Main Panel ---
st.header("Backtest Results")

if run_button:
    if not ticker:
        st.error("Please enter a ticker symbol.")
    else:
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)

        # 1. Fetch Data
        with st.spinner(f"Fetching data for {ticker}..."):
            data = get_stock_data(ticker, start_date, end_date)

        if data is not None and not data.empty:
            st.success(
                f"Data fetched successfully for {ticker} from {start_date_str} to {end_date_str}."
            )

            # Prepare data for Backtesting.py (needs OHLC columns with specific names)
            # Print the original column names for debugging

            # Ensure we have the right column formats - explicitly rename for clarity
            data_cols = {col.lower(): col for col in data.columns}

            # Create a properly formatted dataframe for backtesting
            bt_data = data.copy()

            # Make sure columns are properly named for Backtesting.py
            rename_map = {}
            if "open" in data_cols:
                rename_map[data_cols["open"]] = "Open"
            if "high" in data_cols:
                rename_map[data_cols["high"]] = "High"
            if "low" in data_cols:
                rename_map[data_cols["low"]] = "Low"
            if "close" in data_cols:
                rename_map[data_cols["close"]] = "Close"
            if "volume" in data_cols:
                rename_map[data_cols["volume"]] = "Volume"
            if "adj_close" in data_cols or "adj close" in data_cols:
                key = data_cols.get("adj_close", data_cols.get("adj close"))
                rename_map[key] = "Adj Close"

            bt_data = bt_data.rename(columns=rename_map)

            # 2. Initialize and run backtrader Cerebro
            # Create a subclass with the strategy parameters
            if strategy_params:
                StrategyClass = type(
                    f"{strategy_name}_Custom", (SelectedStrategy,), strategy_params
                )
            else:
                StrategyClass = SelectedStrategy

            cerebro = bt.Cerebro()
            cerebro.addstrategy(StrategyClass, **strategy_params)
            # Convert pandas DataFrame to backtrader data feed
            data_feed = bt.feeds.PandasData(dataname=bt_data)
            cerebro.adddata(data_feed)
            cerebro.broker.setcash(start_cash)
            cerebro.broker.setcommission(commission=0.001)
            st.write("Running backtest with backtrader...")
            with st.spinner("Backtest in progress..."):
                strats = cerebro.run()
            st.write("Backtest complete.")
            # Extract strategy instance to access buy/sell signals
            strat = strats[0]

            # Get final portfolio value
            final_value = cerebro.broker.getvalue()
            results = {"Equity Final [$]": final_value}

            # 4. Display Basic Results
            st.subheader("Performance Metrics")
            # Use markdown for larger text and better layout control
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
            </div>
            """,
                unsafe_allow_html=True,
            )

            # 5. Plot Price with Buy/Sell Signals
            st.subheader("Price Chart with Buy/Sell Signals")
            # Plot using Matplotlib
            # Apply professional built-in matplotlib style
            # plt.style.use("ggplot")
            # Set transparent background
            fig, ax = plt.subplots(figsize=(12, 6), facecolor="none")
            ax.set_facecolor("none")
            ax.grid(False)  # ← turn off all grid lines

            # Plot closing price with styling
            # Draw close price line with low z-order to sit behind markers
            ax.plot(
                bt_data.index,
                bt_data["Close"],
                label="Close Price",
                color="steelblue",
                linewidth=2,
                zorder=1,
            )
            # Add grid for readability
            # Improve date formatting
            locator = AutoDateLocator()
            formatter = DateFormatter("%Y-%m")
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            fig.autofmt_xdate()

            # Plot buy signals
            if hasattr(strat, "buy_signals") and strat.buy_signals:
                # Unpack all three elements: date, price, value
                buy_dates, buy_prices, buy_values = zip(*strat.buy_signals)
                # Black triangle for buy (drawn above line)
                ax.scatter(
                    buy_dates,
                    buy_prices,  # Use the unpacked prices
                    marker="^",
                    color="white",
                    s=70,
                    label="Buys",
                    zorder=3,
                )

            # Plot sell signals
            if hasattr(strat, "sell_signals") and strat.sell_signals:
                # Unpack all three elements: date, price, value
                sell_dates, sell_prices, sell_values = zip(*strat.sell_signals)
                # Black triangle down for sell (drawn above line)
                ax.scatter(
                    sell_dates,
                    sell_prices,  # Use the unpacked prices
                    marker="v",
                    color="grey",
                    s=70,
                    label="Sells",
                    zorder=3,
                )

            # Annotate profit/loss for each completed trade
            if hasattr(strat, "trades") and strat.trades:
                for trade in strat.trades:
                    dt = trade["sell_date"]
                    price = trade["sell_price"]
                    # Change from trade["pnl"] to trade["total_pnl"]
                    pnl = trade["total_pnl"]
                    text_color = "g" if pnl >= 0 else "r"
                    # Place P&L text above markers with highest z-order
                    ax.annotate(
                        f"${pnl:.2f}",  # Add dollar sign for clarity
                        xy=(dt, price),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha="center",
                        color=text_color,
                        fontsize=12,
                        fontweight="bold",
                        zorder=4,
                    )
                    # make spines white and bold
            for spine in ax.spines.values():
                spine.set_color("white")
                spine.set_linewidth(2)

            # ticks white and bold
            ax.tick_params(axis="x", colors="white", labelsize=12, width=2)
            ax.tick_params(axis="y", colors="white", labelsize=12, width=2)

            # axis labels bold and white
            ax.set_xlabel("Date", fontsize=12, color="white", weight="bold")
            ax.set_ylabel("Price", fontsize=12, color="white", weight="bold")
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Price", fontsize=12)
            ax.legend(framealpha=0.3)
            # Ensure plot background is transparent when saving/displaying
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            st.pyplot(fig)

            # 6. Add Wavefunction Visualization for CustomQuantum Strategy
            if strategy_name == "CustomQuantum" and hasattr(
                strat, "get_wavefunction_data"
            ):
                st.subheader("Quantum Wavefunction Visualization")

                # Get wavefunction data
                wf_data = strat.get_wavefunction_data()

                if len(wf_data["dates"]) > wf_data["wf_lookback"]:
                    # Create wavefunction visualization (price + wavefunction only)
                    fig2, ax2 = plt.subplots(figsize=(12, 6), facecolor="none")
                    ax2.set_facecolor("none")

                    # Plot price data
                    ax2.plot(
                        wf_data["dates"],
                        wf_data["prices"],
                        label="Actual Price",
                        color="steelblue",
                        alpha=0.7,
                        linewidth=2,
                    )

                    # Plot wavefunction
                    ax2.plot(
                        wf_data["dates"],
                        wf_data["wavefunction"],
                        label="Quantum Wavefunction",
                        color="#E833FF",
                        linewidth=2,
                    )

                    # Format plot
                    ax2.xaxis.set_major_locator(locator)
                    ax2.xaxis.set_major_formatter(formatter)
                    fig2.autofmt_xdate()

                    # Style the plot
                    for spine in ax2.spines.values():
                        spine.set_color("white")
                        spine.set_linewidth(2)

                    ax2.tick_params(axis="x", colors="white", labelsize=12, width=2)
                    ax2.tick_params(axis="y", colors="white", labelsize=12, width=2)
                    ax2.set_xlabel("Date", fontsize=12, color="white", weight="bold")
                    ax2.set_ylabel("Price", fontsize=12, color="white", weight="bold")
                    ax2.legend(framealpha=0.3)

                    # Ensure plot background is transparent
                    fig2.patch.set_alpha(0.0)
                    ax2.patch.set_alpha(0.0)

                    st.pyplot(fig2)

                    # Add a visualization of eigenvalue evolution over time without signals
                    if "eigenvalue_history" in wf_data:
                        st.subheader("Eigenvalue Evolution (Support/Resistance Levels)")

                        # Create the clean eigenvalue visualization without signals
                        fig4, ax4 = plt.subplots(figsize=(12, 6), facecolor="none")
                        ax4.set_facecolor("none")

                        # Get eigenvalue history and ensure it's the right length
                        history = wf_data["eigenvalue_history"]
                        full_date_range = wf_data["dates"]

                        # Make sure we're using the full date range
                        if len(history) != len(full_date_range):
                            st.warning(
                                f"Data length mismatch: {len(history)} eigenvalue points vs {len(full_date_range)} dates"
                            )
                            # Truncate to the shorter length to avoid errors
                            min_length = min(len(history), len(full_date_range))
                            history = history[:min_length]
                            full_date_range = full_date_range[:min_length]

                        # Find the first index where we have non-empty eigenvalues
                        start_idx = 0
                        for i, eig_vals in enumerate(history):
                            if eig_vals:  # If not empty
                                start_idx = i
                                break

                        # Plot each eigenvalue's evolution
                        # First, determine the maximum number of eigenvalue levels
                        max_levels = 0
                        for h in history:
                            if len(h) > max_levels:
                                max_levels = len(h)

                        max_levels = min(5, max_levels)  # Limit to 5 levels for clarity

                        # Skip plotting if no eigenvalue data
                        if max_levels > 0:
                            # For each level, plot its evolution
                            for i in range(max_levels):
                                # Create arrays for valid data points
                                valid_dates = []
                                valid_values = []

                                # Collect data points where this level exists
                                for j, h in enumerate(history):
                                    if j >= start_idx and i < len(h):
                                        valid_dates.append(full_date_range[j])
                                        valid_values.append(h[i])

                                # Only plot if we have data
                                if valid_dates:
                                    color = plt.cm.rainbow(i / max(1, max_levels - 1))
                                    ax4.plot(
                                        valid_dates,
                                        valid_values,
                                        label=f"Energy Level {i+1}",
                                        color=color,
                                        linewidth=1,
                                        alpha=0.8,
                                        zorder=2,
                                    )

                        # Always plot price for reference
                        ax4.plot(
                            full_date_range,
                            wf_data["prices"],
                            label="Price",
                            color="steelblue",
                            linewidth=1.5,
                            alpha=1,
                            linestyle="-",
                            zorder=1,
                        )

                        # Set x-axis limits to match the full date range
                        ax4.set_xlim(full_date_range[0], full_date_range[-1])

                        # Style the plot - use the same date formatter as other plots
                        ax4.xaxis.set_major_locator(locator)
                        ax4.xaxis.set_major_formatter(formatter)
                        fig4.autofmt_xdate()

                        # Style the plot
                        for spine in ax4.spines.values():
                            spine.set_color("white")
                            spine.set_linewidth(2)

                        ax4.tick_params(axis="x", colors="white", labelsize=12, width=2)
                        ax4.tick_params(axis="y", colors="white", labelsize=12, width=2)
                        ax4.set_xlabel(
                            "Date", fontsize=12, color="white", weight="bold"
                        )
                        ax4.set_ylabel(
                            "Price", fontsize=12, color="white", weight="bold"
                        )
                        ax4.legend(framealpha=0.3)

                        # Ensure plot background is transparent
                        fig4.patch.set_alpha(0.0)
                        ax4.patch.set_alpha(0.0)

                        st.pyplot(fig4)

                        # Create a final combined visualization with all elements
                        st.subheader("Combined Quantum Trading Visualization")

                        fig5, ax5 = plt.subplots(figsize=(12, 6), facecolor="none")
                        ax5.set_facecolor("none")

                        # Plot price data
                        ax5.plot(
                            full_date_range,
                            wf_data["prices"],
                            label="Price",
                            color="steelblue",
                            linewidth=1.5,
                            alpha=0.9,
                            zorder=1,
                        )

                        # Plot wavefunction (with reduced opacity)
                        ax5.plot(
                            wf_data["dates"],
                            wf_data["wavefunction"],
                            label="Quantum Wavefunction",
                            color="#E833FF",
                            linewidth=1,
                            alpha=0.5,
                            zorder=2,
                        )

                        # Plot eigenvalues (with reduced opacity)
                        if max_levels > 0:
                            for i in range(max_levels):
                                valid_dates = []
                                valid_values = []

                                for j, h in enumerate(history):
                                    if j >= start_idx and i < len(h):
                                        valid_dates.append(full_date_range[j])
                                        valid_values.append(h[i])

                                if valid_dates:
                                    color = plt.cm.rainbow(i / max(1, max_levels - 1))
                                    ax5.plot(
                                        valid_dates,
                                        valid_values,
                                        label=f"Energy Level {i+1}" if i == 0 else None,
                                        color=color,
                                        linewidth=1,
                                        alpha=0.6,
                                        linestyle="--",
                                        zorder=3,
                                    )

                        # Add buy/sell signals to the combined plot
                        if hasattr(strat, "buy_signals") and strat.buy_signals:
                            # Unpack all three elements
                            buy_dates, buy_prices, _ = zip(*strat.buy_signals)
                            ax5.scatter(
                                buy_dates,
                                buy_prices,  # Use unpacked prices
                                marker="^",
                                color="white",
                                edgecolor="green",
                                s=80,
                                label="Buy Signals",
                                zorder=5,
                            )

                        if hasattr(strat, "sell_signals") and strat.sell_signals:
                            # Unpack all three elements
                            sell_dates, sell_prices, _ = zip(*strat.sell_signals)
                            ax5.scatter(
                                sell_dates,
                                sell_prices,  # Use unpacked prices
                                marker="v",
                                color="white",
                                edgecolor="red",
                                s=80,
                                label="Sell Signals",
                                zorder=5,
                            )

                        # Add trading signals overlay if they exist
                        if (
                            "eigenvalue_signals" in wf_data
                            and wf_data["eigenvalue_signals"]
                        ):
                            ax_signals = ax5.twinx()
                            signal_dates, buy_strengths, sell_strengths = zip(
                                *wf_data["eigenvalue_signals"]
                            )

                            # Plot signals as partially transparent areas
                            buy_color = "green"
                            sell_color = "red"
                            alpha = 0.2

                            ax_signals.fill_between(
                                signal_dates,
                                0,
                                buy_strengths,
                                color=buy_color,
                                alpha=alpha,
                                label="Buy Signal Strength",
                            )

                            ax_signals.fill_between(
                                signal_dates,
                                0,
                                sell_strengths,
                                color=sell_color,
                                alpha=alpha,
                                label="Sell Signal Strength",
                            )

                            ax_signals.set_ylim(0, 1.0)
                            ax_signals.set_ylabel("Signal Strength", color="white")
                            ax_signals.tick_params(axis="y", colors="white")
                            ax_signals.set_alpha(0.0)

                        # Style and format the plot
                        ax5.xaxis.set_major_locator(locator)
                        ax5.xaxis.set_major_formatter(formatter)
                        fig5.autofmt_xdate()

                        for spine in ax5.spines.values():
                            spine.set_color("white")
                            spine.set_linewidth(2)

                        ax5.tick_params(axis="x", colors="white", labelsize=12, width=2)
                        ax5.tick_params(axis="y", colors="white", labelsize=12, width=2)
                        ax5.set_xlabel(
                            "Date", fontsize=12, color="white", weight="bold"
                        )
                        ax5.set_ylabel(
                            "Price", fontsize=12, color="white", weight="bold"
                        )

                        # Create comprehensive legend
                        handles, labels = ax5.get_legend_handles_labels()
                        if (
                            "eigenvalue_signals" in wf_data
                            and wf_data["eigenvalue_signals"]
                        ):
                            sig_handles, sig_labels = (
                                ax_signals.get_legend_handles_labels()
                            )
                            handles.extend(sig_handles)
                            labels.extend(sig_labels)

                        ax5.legend(
                            handles=handles,
                            labels=labels,
                            framealpha=0.3,
                            loc="upper left",
                        )

                        # Ensure plot background is transparent
                        fig5.patch.set_alpha(0.0)
                        ax5.patch.set_alpha(0.0)

                        st.pyplot(fig5)

                        # Add eigenvalue trading signals plot (original one)
                        st.subheader("Eigenvalue Trading Signals")

                        # Create trading signals visualization
                        fig6, ax6 = plt.subplots(figsize=(12, 6), facecolor="none")
                        ax6.set_facecolor("none")

                        # Reuse the eigenvalue plotting code
                        if max_levels > 0:
                            for i in range(max_levels):
                                valid_dates = []
                                valid_values = []

                                for j, h in enumerate(history):
                                    if j >= start_idx and i < len(h):
                                        valid_dates.append(full_date_range[j])
                                        valid_values.append(h[i])

                                if valid_dates:
                                    color = plt.cm.rainbow(i / max(1, max_levels - 1))
                                    ax6.plot(
                                        valid_dates,
                                        valid_values,
                                        label=f"Energy Level {i+1}",
                                        color=color,
                                        linewidth=1,
                                        alpha=0.8,
                                        zorder=2,
                                    )

                        # Plot price
                        ax6.plot(
                            full_date_range,
                            wf_data["prices"],
                            label="Price",
                            color="steelblue",
                            linewidth=1.5,
                            alpha=1,
                            linestyle="-",
                            zorder=1,
                        )

                        # Add eigenvalue trading signals to the plot
                        if (
                            "eigenvalue_signals" in wf_data
                            and wf_data["eigenvalue_signals"]
                        ):
                            # Extract signal data
                            signal_dates, buy_strengths, sell_strengths = zip(
                                *wf_data["eigenvalue_signals"]
                            )

                            # Create a secondary y-axis for signal strength
                            ax_signals = ax6.twinx()

                            # Plot signals as partially transparent areas
                            buy_color = "green"
                            sell_color = "red"
                            alpha = 0.3

                            # Plot buy signals
                            ax_signals.fill_between(
                                signal_dates,
                                0,
                                buy_strengths,
                                color=buy_color,
                                alpha=alpha,
                                label="Buy Signal",
                            )

                            # Plot sell signals
                            ax_signals.fill_between(
                                signal_dates,
                                0,
                                sell_strengths,
                                color=sell_color,
                                alpha=alpha,
                                label="Sell Signal",
                            )

                            # Configure secondary axis
                            ax_signals.set_ylim(0, 1.0)
                            ax_signals.set_ylabel("Signal Strength", color="white")
                            ax_signals.tick_params(axis="y", colors="white")

                            # Add legend for signals
                            handles, labels = ax_signals.get_legend_handles_labels()
                            ax6.legend(
                                handles=handles,
                                labels=labels,
                                loc="upper right",
                                framealpha=0.0,
                            )

                        # Set x-axis limits to match the full date range
                        ax6.set_xlim(full_date_range[0], full_date_range[-1])

                        # Style the plot
                        ax6.xaxis.set_major_locator(locator)
                        ax6.xaxis.set_major_formatter(formatter)
                        fig6.autofmt_xdate()

                        for spine in ax6.spines.values():
                            spine.set_color("white")
                            spine.set_linewidth(2)

                        ax6.tick_params(axis="x", colors="white", labelsize=12, width=2)
                        ax6.tick_params(axis="y", colors="white", labelsize=12, width=2)
                        ax6.set_xlabel(
                            "Date", fontsize=12, color="white", weight="bold"
                        )
                        ax6.set_ylabel(
                            "Price", fontsize=12, color="white", weight="bold"
                        )
                        ax6.legend(framealpha=0.3)

                        # Ensure plot background is transparent
                        fig6.patch.set_alpha(0.0)
                        ax6.patch.set_alpha(0.0)

                    # Add explanation about eigenvalue trading
                    st.markdown(
                        """
                    ### Eigenvalue-Based Trading Strategy
                    
                    The strategy now uses eigenvalues as quantum support and resistance levels:
                    
                    - **Buy signals** are generated when price approaches or falls below the lowest eigenvalue (quantum support)
                    - **Sell signals** are generated when price approaches or rises above the highest eigenvalue (quantum resistance)
                    - Signal strength is combined with the quantum probability to make final trading decisions
                    
                    This approach treats eigenvalues as energy states that price may naturally gravitate towards or bounce away from,
                    similar to how electrons in atoms can only occupy certain discrete energy levels.
                    """
                    )

                    # ...existing explanation code...

        elif data is None:
            st.error(
                f"Could not fetch data for {ticker}. Please check the symbol and date range."
            )
        else:  # Data is empty
            st.warning(f"No data available for {ticker} in the specified date range.")

# --- How to Run ---
st.sidebar.markdown("---")
st.sidebar.info("To run: `streamlit run app.py` in your terminal.")
st.sidebar.info("Make sure to install backtrader package: `pip install backtrader`")
