import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import os
import yfinance as yf
from matplotlib.dates import DateFormatter
import matplotlib as mpl
import json

from utils.stock_screener import StockScreener
from utils.eigenvalue_analyzer import EigenvalueAnalyzer

# Set default matplotlib style for all plots
plt.style.use("dark_background")
mpl.rcParams["axes.facecolor"] = "none"
mpl.rcParams["figure.facecolor"] = "none"
mpl.rcParams["text.color"] = "white"
mpl.rcParams["axes.labelcolor"] = "white"
mpl.rcParams["xtick.color"] = "white"
mpl.rcParams["ytick.color"] = "white"
mpl.rcParams["axes.edgecolor"] = "white"
mpl.rcParams["grid.color"] = "#555555"

# Default stocks list
DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "NVDA",
    "AMD",
    "INTC",
    "NFLX",
]

# File to store user's preferred ticker list
SAVED_TICKERS_FILE = "data/saved_tickers.json"
CACHE_DIR = "data/cache"

# Ensure directories exist
os.makedirs(os.path.dirname(SAVED_TICKERS_FILE), exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# Function to load saved tickers
def load_saved_tickers():
    if os.path.exists(SAVED_TICKERS_FILE):
        try:
            with open(SAVED_TICKERS_FILE, "r") as f:
                return json.load(f)
        except:
            return DEFAULT_TICKERS
    return DEFAULT_TICKERS


# Function to save tickers
def save_tickers(tickers):
    with open(SAVED_TICKERS_FILE, "w") as f:
        json.dump(tickers, f)


# Function to clear cache
def clear_cache():
    count = 0
    for file in os.listdir(CACHE_DIR):
        if file.endswith(".pkl"):
            os.remove(os.path.join(CACHE_DIR, file))
            count += 1
    return count


# App configuration
st.set_page_config(layout="wide", page_title="Eigenvalue Stock Screener")

# Apply custom CSS for transparent background
st.markdown(
    """
<style>
.stApp {
    background-color: transparent;
}
div.block-container {
    padding-top: 2rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Quantum Eigenvalue Stock Screener")

# Sidebar controls
st.sidebar.header("Screener Configuration")

# Cache control section
st.sidebar.subheader("Cache Control")
if st.sidebar.button("Clear Cached Stock Data"):
    files_removed = clear_cache()
    st.sidebar.success(f"Cleared {files_removed} cached stock files")

# User stock list management
st.sidebar.subheader("Manage Your Stock List")
saved_tickers = load_saved_tickers()

# Show current saved list
st.sidebar.write("Your Saved Stocks:")
saved_tickers_text = st.sidebar.text_area(
    "Edit your saved stock list (one ticker per line)",
    "\n".join(saved_tickers),
    height=150,
)

# Save button for ticker list
if st.sidebar.button("Save Stock List"):
    new_tickers = [
        ticker.strip().upper()
        for ticker in saved_tickers_text.split("\n")
        if ticker.strip()
    ]
    save_tickers(new_tickers)
    st.sidebar.success(f"Saved {len(new_tickers)} tickers to your list")
    saved_tickers = new_tickers

# Choose which ticker list to use
st.sidebar.subheader("Select Ticker Source")
ticker_source = st.sidebar.radio(
    "Which stocks to analyze?", ["Saved Stock List", "Default Stocks", "Custom Input"]
)

custom_tickers_input = ""
if ticker_source == "Custom Input":
    custom_tickers_input = st.sidebar.text_area(
        "Enter ticker symbols for this session only (one per line)", ""
    )

# Replace lookback days with date selection
default_start_date = datetime.now() - timedelta(days=3650)
start_date = st.sidebar.date_input(
    "Start Date",
    value=default_start_date,
    min_value=datetime(2010, 1, 1),
    max_value=datetime.now() - timedelta(days=1),
)

# Advanced parameters
st.sidebar.subheader("Eigenvalue Parameters")
eigenvalue_count = st.sidebar.slider("Support/Resistance Levels", 2, 10, 5)
eigenvalue_buy_threshold = st.sidebar.slider(
    "Buy Threshold (%)", 0.01, 0.1, 0.02, 0.005, format="%.3f"
)
eigenvalue_sell_threshold = st.sidebar.slider(
    "Sell Threshold (%)", 0.01, 0.1, 0.02, 0.005, format="%.3f"
)

# Option to force refresh data
# force_refresh = st.sidebar.checkbox("Force Data Refresh")

# Run button
run_button = st.sidebar.button("Run Screener")

# Main content
st.header("Attractive Stocks Based on Eigenvalue Analysis")

if run_button:
    # Determine which ticker list to use
    if ticker_source == "Saved Stock List":
        tickers_to_use = saved_tickers
    elif ticker_source == "Default Stocks":
        tickers_to_use = DEFAULT_TICKERS
    else:  # Custom Input
        tickers_to_use = [
            ticker.strip().upper()
            for ticker in custom_tickers_input.split("\n")
            if ticker.strip()
        ]
        if not tickers_to_use:
            st.warning("No tickers entered. Using default tickers instead.")
            tickers_to_use = DEFAULT_TICKERS

    # Initialize screener
    screener = StockScreener(cache_dir=CACHE_DIR)

    # Rest of your existing code continues here...
    # Set screener parameters
    screener.analyzer.params["eigenvalue_count"] = eigenvalue_count
    screener.analyzer.params["eigenvalue_buy_threshold"] = eigenvalue_buy_threshold
    screener.analyzer.params["eigenvalue_sell_threshold"] = eigenvalue_sell_threshold

    # Convert start_date to datetime with time component
    start_datetime = datetime.combine(start_date, datetime.min.time())

    # Run screening with specific start date
    with st.spinner(f"Analyzing {len(tickers_to_use)} stocks..."):
        results = screener.screen_stocks(
            tickers_to_use,
            start_date=start_datetime,
            force_refresh=False,
            analyze_full_history=True,
        )
    if results:
        # Display results as a table
        st.subheader("Ranked Stocks by Breakout Potential")

        # Prepare data for the table
        table_data = []
        for res in results:
            table_data.append(
                {
                    "Ticker": res["ticker"],
                    "Current Price": f"${res['current_price']:.2f}",
                    "Breakout Potential": f"{res['breakout_potential']:.2f}",
                    "Buy Signal": f"{res['buy_signal']:.2f}",
                    "Sell Signal": f"{res['sell_signal']:.2f}",
                    "Nearest Support": f"${res['support_level']:.2f}",
                    "Nearest Resistance": f"${res['resistance_level']:.2f}",
                    "Distance to Level (%)": f"{res['distance_to_nearest']:.2f}%",
                }
            )

        df_results = pd.DataFrame(table_data)
        st.table(df_results)

        # Display detailed analysis for top stocks
        st.subheader("Detailed Analysis of Top Stocks")

        # Select top stocks to show (one per row with full width)
        top_stocks = results[:10]
        for i, stock in enumerate(top_stocks):
            st.markdown(f"### {stock['ticker']}")

            # Create visualization of price and evolving eigenvalues - full width
            fig, ax = plt.subplots(
                figsize=(20, 8)
            )  # Much wider figure for full screen width

            # Get data for this stock to plot
            ticker_data = screener._load_cached_data(stock["ticker"])

            if (
                ticker_data is not None
                and "eigenvalue_history" in stock
                and "dates" in stock
            ):
                # Get the subset of price data that aligns with eigenvalue history
                if len(stock["eigenvalue_history"]) > 0:
                    # Plot price history - show more data points for longer history
                    display_window = len(ticker_data)
                    date_subset = ticker_data.index[-display_window:]
                    price_subset = ticker_data["Close"].values[-display_window:]

                    # Plot price with enhanced styling
                    ax.plot(
                        date_subset,
                        price_subset,
                        label="Price",
                        color="#00BFFF",
                        linewidth=2.5,
                    )

                    # Get eigenvalue data
                    ev_dates = stock["dates"][
                        -min(display_window, len(stock["dates"])) :
                    ]
                    ev_history = stock["eigenvalue_history"][
                        -min(display_window, len(stock["eigenvalue_history"])) :
                    ]

                    # Plot evolving eigenvalues over time
                    if len(ev_history) > 0:
                        num_eigenvalues = len(ev_history[0])
                        colors = plt.cm.plasma(np.linspace(0, 1, num_eigenvalues))

                        # For each eigenvalue level (e.g., support/resistance)
                        for j in range(num_eigenvalues):
                            # Extract this eigenvalue's evolution over time
                            eigenvalue_series = [ev[j] for ev in ev_history]
                            ax.plot(
                                ev_dates,
                                eigenvalue_series,
                                "--",
                                color=colors[j],
                                alpha=0.85,
                                linewidth=1.8,
                                label=f"Level {j+1}",
                            )

                    # Mark current price with more visible marker
                    ax.scatter(
                        [ticker_data.index[-1]],
                        [stock["current_price"]],
                        color="white",
                        edgecolor="#00FF7F",
                        s=180,
                        zorder=5,
                        marker="o",
                        linewidth=2.5,
                    )

                    # Format dates on x-axis
                    ax.xaxis.set_major_formatter(DateFormatter("%b %d %Y"))
                    plt.xticks(rotation=45)

                    # Add grid for better readability
                    ax.grid(True, linestyle="--", alpha=0.3)

                    # Add title and labels
                    ax.set_title(
                        f"{stock['ticker']} - Eigenvalue Analysis",
                        color="white",
                        fontsize=16,
                        pad=20,
                    )
                    ax.set_ylabel("Price ($)", color="white", fontsize=12)

                    # Add legend with transparent background
                    legend = ax.legend(loc="upper left", framealpha=0.3, fontsize=12)
                    plt.setp(legend.get_texts(), color="white")

                    # Make sure everything fits
                    plt.tight_layout()

                    # Display the full-width chart
                    st.pyplot(fig)

                # Show key metrics in 4 columns for better layout
                metrics_cols = st.columns(4)

                with metrics_cols[0]:
                    st.metric("Current Price", f"${stock['current_price']:.2f}")
                with metrics_cols[1]:
                    st.metric(
                        "Breakout Potential", f"{stock['breakout_potential']:.2f}"
                    )
                with metrics_cols[2]:
                    st.metric("Buy Signal", f"{stock['buy_signal']:.2f}")
                with metrics_cols[3]:
                    st.metric(
                        "Distance to Level", f"{stock['distance_to_nearest']:.2f}%"
                    )

                metrics_cols2 = st.columns(4)
                with metrics_cols2[0]:
                    st.metric("Nearest Support", f"${stock['support_level']:.2f}")
                with metrics_cols2[1]:
                    st.metric("Nearest Resistance", f"${stock['resistance_level']:.2f}")

            st.markdown("---")  # Add separator between stocks
    else:
        st.error(
            "No results found or analysis failed. Try different parameters or check your internet connection."
        )
else:
    st.info(
        "Configure the screener parameters in the sidebar and click 'Run Screener'."
    )
