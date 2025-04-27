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
from utils.data_fetcher import get_stock_data

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
    "CSCO",
    "ADBE",
    "CRM",
    "PYPL",
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
        except json.JSONDecodeError:
            return DEFAULT_TICKERS  # Return default if file is corrupted
    return DEFAULT_TICKERS


# Function to save tickers
def save_tickers(tickers):
    with open(SAVED_TICKERS_FILE, "w") as f:
        json.dump(tickers, f)


# Function to clear cache
def clear_cache():
    files_removed = 0
    for filename in os.listdir(CACHE_DIR):
        file_path = os.path.join(CACHE_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                files_removed += 1
        except Exception as e:
            st.error(f"Failed to delete {file_path}. Reason: {e}")
    return files_removed


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
# force_refresh = st.sidebar.checkbox("Force Data Refresh") # Consider adding this back

# Run button
run_button = st.sidebar.button("Run Screener")

# --- Create Tabs ---
tab1, tab2, tab3 = st.tabs(["Screener Results", "About & Mathematics", "Market Trends"])

with tab1:
    st.header("Stock Ranking by Risk-Adjusted Momentum")  # Update header

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

        # Set screener parameters
        screener.analyzer.params["eigenvalue_count"] = eigenvalue_count
        screener.analyzer.params["eigenvalue_buy_threshold"] = eigenvalue_buy_threshold
        screener.analyzer.params["eigenvalue_sell_threshold"] = (
            eigenvalue_sell_threshold
        )

        # Convert start_date to datetime with time component
        start_datetime = datetime.combine(start_date, datetime.min.time())

        # Run screening with specific start date
        with st.spinner(f"Analyzing {len(tickers_to_use)} stocks..."):
            results = screener.screen_stocks(
                tickers_to_use,
                start_date=start_datetime,
                force_refresh=False,  # Consider adding checkbox for this
                analyze_full_history=True,
            )
        if results:
            # Display results table
            st.subheader("Ranked Stocks")
            table_data = []
            for res in results:
                table_data.append(
                    {
                        "Ticker": res["ticker"],
                        "RiskAdj Momentum": f"{res.get('risk_adjusted_momentum', 0.0):.3f}",
                        "Current Price": f"${res['current_price']:.2f}",
                        "Buy Signal": f"{res['buy_signal']:.2f}",
                        "Sell Signal": f"{res['sell_signal']:.2f}",
                        "Breakout Potential": f"{res['breakout_potential']:.2f}",
                        "Nearest Support": f"${res['support_level']:.2f}",
                        "Nearest Resistance": f"${res['resistance_level']:.2f}",
                        "Dist to Level (%)": f"{res['distance_to_nearest']:.2f}%",
                        "Sharpe Ratio": f"{res.get('sharpe_ratio', 0.0):.3f}",
                        "Max Drawdown": f"{res.get('max_drawdown', 0.0):.3f}",
                    }
                )
            df_results = pd.DataFrame(table_data)
            st.dataframe(df_results)  # Use dataframe for better layout

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
                        legend = ax.legend(
                            loc="upper left", framealpha=0.3, fontsize=12
                        )
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
                        "RiskAdj Momentum",
                        f"{stock.get('risk_adjusted_momentum', 0.0):.3f}",
                    )  # Show new metric
                with metrics_cols[2]:
                    st.metric("Buy Signal", f"{stock['buy_signal']:.2f}")
                with metrics_cols[3]:
                    st.metric("Sell Signal", f"{stock['sell_signal']:.2f}")

                metrics_cols2 = st.columns(4)
                with metrics_cols2[0]:
                    st.metric(
                        "Breakout Potential", f"{stock['breakout_potential']:.2f}"
                    )
                with metrics_cols2[1]:
                    st.metric("Dist to Level", f"{stock['distance_to_nearest']:.2f}%")
                with metrics_cols2[2]:
                    st.metric("Nearest Support", f"${stock['support_level']:.2f}")
                with metrics_cols2[3]:
                    st.metric("Nearest Resistance", f"${stock['resistance_level']:.2f}")

                # New performance metrics display
                metrics_cols3 = st.columns(2)
                with metrics_cols3[0]:
                    st.metric("Sharpe Ratio", f"{stock.get('sharpe_ratio', 0.0):.3f}")
                with metrics_cols3[1]:
                    st.metric("Max Drawdown", f"{stock.get('max_drawdown', 0.0):.3f}")

                st.markdown("---")  # Add separator between stocks

            # Get the bottom 3 stocks (lowest risk-adjusted momentum)
            st.subheader("Bottom Ranked Stocks")
            bottom_stocks = (
                results[-3:] if len(results) >= 3 else results[-len(results) :]
            )
            bottom_stocks.reverse()  # Show worst first

            # Create a 3-column layout for the bottom stocks
            cols = st.columns(3)

            for i, stock in enumerate(bottom_stocks):
                with cols[i]:
                    st.markdown(f"### {stock['ticker']}")
                    st.metric(
                        "RiskAdj Momentum",
                        f"{stock.get('risk_adjusted_momentum', 0.0):.3f}",
                    )  # Show new metric
                    st.metric("Current Price", f"${stock['current_price']:.2f}")
                    st.metric(
                        "Breakout Potential", f"{stock['breakout_potential']:.2f}"
                    )
                    st.metric("Buy Signal", f"{stock['buy_signal']:.2f}")
                    st.metric("Sell Signal", f"{stock['sell_signal']:.2f}")

                    # Create small chart showing price and eigenvalues
                    fig, ax = plt.subplots(
                        figsize=(6, 4)
                    )  # Keep smaller size for columns

                    # Get data for this stock
                    ticker_data = screener._load_cached_data(stock["ticker"])

                    if ticker_data is not None and len(ticker_data) > 0:
                        # Plot recent price history (last 60 days)
                        display_window = min(60, len(ticker_data))
                        date_subset = ticker_data.index[-display_window:]
                        price_subset = ticker_data["Close"].values[-display_window:]

                        # Plot price with consistent styling
                        ax.plot(
                            date_subset,
                            price_subset,
                            label="Price",
                            color="#00BFFF",  # Match top stock color
                            linewidth=2.0,  # Slightly thinner for smaller chart
                        )

                        # Add current eigenvalues as horizontal lines with consistent styling
                        if "eigenvalues" in stock and stock["eigenvalues"]:
                            num_eigenvalues = len(stock["eigenvalues"])
                            colors = plt.cm.plasma(np.linspace(0, 1, num_eigenvalues))
                            for j, eigenvalue in enumerate(
                                sorted(stock["eigenvalues"])
                            ):  # Sort for consistent coloring order
                                ax.axhline(
                                    y=eigenvalue,
                                    linestyle="--",
                                    color=colors[j],
                                    alpha=0.85,  # Match top stock alpha
                                    linewidth=1.5,  # Slightly thinner
                                    label=(
                                        f"Level {j+1}" if i == 0 else ""
                                    ),  # Only label once per column set potentially
                                )

                        # Mark current price with consistent marker
                        ax.scatter(
                            [date_subset[-1]],
                            [stock["current_price"]],
                            color="white",
                            edgecolor="#00FF7F",
                            s=100,  # Smaller marker size
                            zorder=5,
                            marker="o",
                            linewidth=2.0,  # Match top stock linewidth
                        )

                        # Format the chart consistently
                        ax.set_title(
                            f"{stock['ticker']} Recent Price",
                            color="white",
                            fontsize=12,
                        )
                        ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
                        plt.xticks(rotation=45)
                        ax.grid(True, linestyle="--", alpha=0.3)  # Match top stock grid
                        plt.tight_layout()

                        st.pyplot(fig)
                    else:
                        st.write("No data available for chart")

        else:
            st.error(
                "No results found or analysis failed. Try different parameters or check your internet connection."
            )

    else:
        st.info(
            "Configure the screener parameters in the sidebar and click 'Run Screener'."
        )

with tab2:
    st.header("About & Mathematical Concepts")
    st.markdown(
        r"""
        This application employs concepts inspired by quantum mechanics to analyze stock price dynamics.
        It models market behavior using a time-dependent Hamiltonian operator, whose eigenvalues are
        interpreted as dynamic support and resistance levels.

        ---

        ### Enhanced Eigenvalue Computation

        • For larger Hamiltonian matrices (dim > 50), we leverage ARPACK's sparse solver (`scipy.sparse.linalg.eigsh`) with configurable tolerance and iteration limits to compute the lowest eigenvalues.
        • Each computed eigenpair can be further refined via Rayleigh quotient iteration, sharpening accuracy at the cost of extra compute.
        • All eigenvalue operations—including potential update and time evolution—run in parallel across symbols using Python's `concurrent.futures`.
        • The Hamiltonian itself now includes a term influenced by recent overall market trend (e.g. S&P 500 mean return).

        ### 1. The Market Hamiltonian Operator ($\hat{H}$)

        The core of the analysis lies in the Hamiltonian operator, analogous to the total energy operator in quantum mechanics. It is constructed as the sum of a kinetic energy term ($\hat{K}$) and a potential energy term ($\hat{V}$):

        $$
        \hat{H}(t) = -\hat{K}(t) + \hat{V}(t)
        $$

        This operator acts on a state space representing recent price history. We discretize the price history over a lookback window of $N$ points (parameter `wf_lookback`).

        #### 1.1 Kinetic Energy Term ($\hat{K}$)

        The kinetic term models the 'flow' or momentum of price changes. It's represented by a tridiagonal matrix, approximating the second derivative (Laplacian) in the continuous case. Its elements depend on recent price momentum:

        $$
        K_{ij}(t) =
        \begin{cases}
            -k_f(t) & \text{if } |i-j| = 1 \\
            2k_f(t) & \text{if } i = j \\
            0 & \text{otherwise}
        \end{cases}
        $$

        The time-dependent kinetic factor $k_f(t)$ is modulated by the normalized price momentum $m(t)$:

        $$
        k_f(t) = k_{base} \cdot \left(1 + \alpha \cdot |m(t)|\right)
        $$

        where $k_{base}$ is the base `kinetic_factor`, $\alpha$ is a scaling constant (e.g., 0.5), and $m(t)$ is:

        $$
        m(t) = \frac{p(t) - p(t-1)}{\sigma_p(t)}
        $$

        Here, $p(t)$ is the price at time $t$, and $\sigma_p(t)$ is the standard deviation of prices over the lookback window. Increased momentum enhances the kinetic 'energy'.

        #### 1.2 Potential Energy Term ($\hat{V}$)

        The potential term represents the market structure, price 'landscape', and tendency towards certain levels. It's a diagonal matrix $V_{ij} = V_i \delta_{ij}$, where $V_i$ is the potential at the $i$-th point in the lookback window.

        $$
        V_i(t) = V_{\text{base}, i}(t) + V_{\text{wells}, i}(t)
        $$

        *   **Base Potential ($V_{\text{base}}$):** A harmonic oscillator potential centered around the mean price $\bar{p}(t)$, encouraging mean reversion:

            $$
            V_{\text{base}, i}(t) = p_f \cdot \left( \frac{p_i - \bar{p}(t)}{\sigma_p(t)} \right)^2
            $$
            where $p_f$ is the `potential_factor`.

        *   **Potential Wells ($V_{\text{wells}}$):** These represent price congestion zones (support/resistance) identified via histogram analysis of recent price data (lookback `sr_lookback`). Each well $w$ adds a negative Gaussian potential:

            $$
            V_{\text{wells}, i}(t) = - \sum_{w} d_w(t) \cdot \exp\left( -\frac{(p_i - c_w(t))^2}{2 \sigma_{w}^2(t)} \right)
            $$
            where $c_w(t)$, $d_w(t)$, and $\sigma_{w}(t)$ are the center (price level), depth (strength), and width of the $w$-th potential well at time $t$. These wells evolve dynamically based on market activity, governed by `time_evolution_rate` and `eigenvalue_persistence`.

        ---

        ### 2. Eigenvalues ($E_n$) as Support/Resistance

        Solving the time-independent Schrödinger equation for the Hamiltonian at time $t$:

        $$
        \hat{H}(t) |\psi_n(t)\rangle = E_n(t) |\psi_n(t)\rangle
        $$

        yields the eigenvalues $E_n(t)$ and eigenvectors $|\psi_n(t)\rangle$. This is achieved numerically using `scipy.linalg.eigh`.

        The eigenvalues $E_n(t)$ represent the stable 'energy states' of the market model. We interpret a subset of these (parameter `eigenvalue_count`), particularly the lower ones, as dynamic support and resistance levels.

        *   **Mapping:** Raw eigenvalues are mapped from the abstract energy space back to the price domain based on the price range within the lookback window.
        *   **Time Evolution:** The calculated eigenvalues are smoothed over time using an exponential moving average approach, incorporating `eigenvalue_persistence` and attraction towards the centers of identified `potential_wells`.

            $$
            E_{n, \text{smooth}}(t) = \gamma E_{n, \text{smooth}}(t-1) + (1-\gamma) E_{n, \text{mapped}}(t) + \delta (C_{n, \text{closest}} - E_{n, \text{smooth}}(t-1))
            $$
            where $\gamma$ is `eigenvalue_persistence`, $\delta$ relates to `time_evolution_rate`, and $C_{n, \text{closest}}$ is the center of the nearest potential well.

        ---

        ### 3. Signal Generation

        Signals are derived by comparing the current price $p(t)$ to the calculated eigenvalue levels.

        #### 3.1 Buy/Sell Signals ($S_{\text{buy}}, S_{\text{sell}}$)

        These signals quantify the proximity of the current price to the nearest support ($E_{\text{support}}$) and resistance ($E_{\text{resistance}}$) levels, adjusted by dynamic thresholds ($\theta_{\text{buy}}, \theta_{\text{sell}}$) based on recent volatility (e.g., ATR).

        *   **Buy Signal:** Increases as $p(t)$ approaches or falls below $E_{\text{support}}$.
            $$ S_{\text{buy}} \approx \max\left(0, 1 - \frac{\max(0, p(t) - E_{\text{support}})}{p(t) \cdot \theta_{\text{buy}}}\right) $$
        *   **Sell Signal:** Increases as $p(t)$ approaches or exceeds $E_{\text{resistance}}$.
            $$ S_{\text{sell}} \approx \max\left(0, 1 - \frac{\max(0, E_{\text{resistance}} - p(t))}{p(t) \cdot \theta_{\text{sell}}}\right) $$
        *(Simplified representation)*

        #### 3.2 Breakout Potential

        Estimates the likelihood of price breaking through a level, considering momentum and volatility:

        $$
        \text{Potential}_{\text{breakout}} \approx f(\text{Proximity}, \text{Momentum}, \text{Volatility})
        $$

        Higher potential is assigned when price is near a level with strong momentum towards it and elevated volatility.

        ---

        ### 4. Risk-Adjusted Momentum Score (Ranking Metric)

        This metric provides a composite score for ranking stocks, combining momentum, risk, volatility, and market structure:

        $$
        \text{Score} = \tanh \left( \frac{(\text{Momentum}) \times (1 + w_R \cdot \text{Risk}) \times (1 + w_S \cdot \text{Spread})}{1 + w_V \cdot \text{Volatility}} \right)
        $$

        Where:
        *   **Momentum:** $S_{\text{buy}} - S_{\text{sell}}$
        *   **Risk:** Normalized distance to the nearest eigenvalue level. $ \text{Risk} = \frac{\min_n |p(t) - E_n(t)|}{p(t)} $
        *   **Spread:** Normalized difference between nearest resistance and support. $ \text{Spread} = \frac{E_{\text{resistance}} - E_{\text{support}}}{p(t)} $.
            *   *Note on Support/Resistance for this metric:*
                *   `Support` is the highest eigenvalue strictly *below* the current price. If none exists below, it's the lowest eigenvalue strictly *above* the current price (or the lowest overall as a fallback).
                *   `Resistance` is the lowest eigenvalue strictly *above* the determined `Support` level (or the highest overall as a fallback).
        *   **Volatility:** Standard deviation of recent price changes relative to the mean recent price. $ \text{Volatility} = \frac{\sigma(\text{recent prices})}{\bar{p}(\text{recent prices})} $
        *   $w_R, w_S, w_V$: Weighting factors (e.g., 5, 1, 2 respectively in the code).
        *   $\tanh$: Hyperbolic tangent function to bound the score between -1 and 1.

        This score favors stocks with strong net buy signals, larger distances to the nearest level (lower immediate reversal risk for momentum), wider gaps between support/resistance (room to move), and moderate volatility.

        ---

        ### Disclaimer
        This tool utilizes mathematical models inspired by physics for market analysis. It is **not** financial advice. All trading involves significant risk. Perform your own due diligence before making any investment decisions.
        """
    )

with tab3:
    st.header("General Market & Sector Trends")
    # Fetch index data for S&P 500, Nasdaq, Dow
    index_map = {"S&P 500": "^GSPC", "Nasdaq Composite": "^IXIC", "Dow Jones": "^DJI"}
    # Use sidebar start date as chart start
    chart_start = datetime.combine(start_date, datetime.min.time())
    chart_end = datetime.now()
    trend_series = {}
    for name, idx in index_map.items():
        idx_data = get_stock_data(idx, chart_start, chart_end)
        if idx_data is not None and "Close" in idx_data:
            trend_series[name] = idx_data["Close"]
    if trend_series:
        df_trends = pd.DataFrame(trend_series)
        st.line_chart(df_trends)
        # Show performance metrics
        st.subheader("Index Performance")
        perf = {}
        for name, series in trend_series.items():
            ret = series.pct_change().dropna()
            perf[name] = {
                "Total Return (%)": f"{(series.iloc[-1]/series.iloc[0]-1)*100:.2f}",
                "Annualized Vol (%)": f"{ret.std()*np.sqrt(252)*100:.2f}",
            }
        st.dataframe(pd.DataFrame(perf).T)
    else:
        st.write("Market index data unavailable.")
