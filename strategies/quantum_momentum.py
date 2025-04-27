import backtrader as bt
import numpy as np
from scipy import fft
from scipy.stats import norm


class QuantumMomentumStrategy(bt.Strategy):
    # Define strategy parameters using backtrader params
    params = (
        ("sma_period", 10),
        ("prob_threshold", 0.6),
        ("entanglement_lookback", 20),
        ("phase_periods", [5, 10, 20]),  # Multiple periods for quantum phase analysis
        ("interference_weight", 0.5),  # Weight for interference pattern
        ("uncertainty_factor", 0.2),  # Heisenberg uncertainty factor
        ("tunneling_threshold", 2.0),  # Quantum tunneling threshold
        ("superposition_count", 3),  # Number of superposition states to consider
    )

    def __init__(self, *args, **kwargs):
        # Initialize base strategy with params
        super().__init__(*args, **kwargs)
        # Lists to record buy and sell signals (datetime, price)
        self.buy_signals = []
        self.sell_signals = []
        # Track individual trades and current open trade
        self.trades = []  # Each trade: dict with buy/sell info and pnl
        self.current_trade = None

        # Quantum indicators
        self.phase_indicators = {}
        for period in self.p.phase_periods:
            self.phase_indicators[period] = bt.indicators.SMA(
                self.data.close, period=period
            )

        # Entanglement matrix (correlation of price with lagged versions)
        self.entanglement_matrix = np.zeros(
            (self.p.entanglement_lookback, self.p.entanglement_lookback)
        )

        # Quantum state vector (probabilities for different states)
        self.quantum_state = (
            np.ones(self.p.superposition_count) / self.p.superposition_count
        )

    def next(self):
        # Get current close price (latest bar)
        price = self.data.close[0]
        dt = self.data.datetime.datetime(0)

        # Ensure enough data points for calculations
        if len(self.data) < max(self.p.sma_period, self.p.entanglement_lookback):
            return

        # Calculate simple moving average for last sma_period bars, including current
        closes = [self.data.close[-i] for i in range(self.p.sma_period)]
        sma = np.mean(closes)

        # Quantum-inspired probability using z-score relative to volatility
        vol = np.std(closes) if np.std(closes) > 0 else 1
        z_score = (price - sma) / vol

        # Update entanglement matrix (correlation structure of price history)
        self._update_entanglement_matrix()

        # Calculate quantum phase estimation (detect market cycles)
        phase_estimation = self._quantum_phase_estimation()

        # Apply quantum wave function (amplitude calculation with interference)
        amplitude = self._calculate_quantum_amplitude(z_score, phase_estimation)

        # Apply quantum superposition (evaluate multiple states)
        superposition_states = self._quantum_superposition(amplitude)

        # Calculate final probabilities using quantum measurement
        buy_probability, sell_probability = self._quantum_measurement(
            superposition_states
        )

        # Apply Heisenberg uncertainty principle (trade-off between timing and price prediction)
        buy_probability, sell_probability = self._apply_uncertainty_principle(
            buy_probability, sell_probability
        )

        # Random draw for quantum-inspired decisions with tunneling effect
        decision = self._quantum_decision(buy_probability, sell_probability)

        # Trading logic with quantum tunneling for breakthrough moments

        if not self.position:
            if decision == "BUY" or self._quantum_tunneling(price, closes, "BUY"):
                # Calculate maximum number of shares to buy with all available cash
                # Accounting for a small cash buffer (0.5%) to avoid potential rounding issues
                cash = self.broker.getcash() * 0.995
                shares = int(cash / price)

                if shares > 0:
                    self.buy(size=shares)
                    # Record buy signal and start a new trade
                    self.buy_signals.append((dt, price))
                    self.current_trade = {
                        "buy_date": dt,
                        "buy_price": price,
                        "shares": shares,
                    }
        else:
            if decision == "SELL" or self._quantum_tunneling(price, closes, "SELL"):
                # Close all open positions
                self.close()

                # Record sell signal
                self.sell_signals.append((dt, price))
                # Complete the trade and compute P&L
                if self.current_trade is not None:
                    buy_info = self.current_trade
                    pnl = price - buy_info["buy_price"]
                    total_pnl = pnl * buy_info.get("shares", 1)  # Calculate total P&L
                    trade = {
                        "buy_date": buy_info["buy_date"],
                        "buy_price": buy_info["buy_price"],
                        "sell_date": dt,
                        "sell_price": price,
                        "shares": buy_info.get("shares", 1),
                        "pnl_per_share": pnl,
                        "total_pnl": total_pnl,
                    }
                    self.trades.append(trade)
                self.current_trade = None

    def _update_entanglement_matrix(self):
        """Update the quantum entanglement matrix (correlation structure)"""
        lookback = self.p.entanglement_lookback
        price_history = np.array([self.data.close[-i] for i in range(lookback)])
        price_history = (price_history - np.mean(price_history)) / (
            np.std(price_history) if np.std(price_history) > 0 else 1
        )

        for i in range(lookback):
            for j in range(lookback):
                # Calculate correlation (entanglement) between different time points
                if i + j < lookback:
                    self.entanglement_matrix[i, j] = price_history[i] * price_history[j]

    def _quantum_phase_estimation(self):
        """Quantum phase estimation to detect market cycles using Quantum Fourier Transform"""
        # Get price history for quantum phase analysis
        lookback = max(self.p.phase_periods) * 2
        if len(self.data) < lookback:
            return 0

        price_history = np.array([self.data.close[-i] for i in range(lookback)])
        # Apply Fast Fourier Transform (QFT-inspired)
        fft_result = fft.fft(price_history)
        # Get power spectrum
        power = np.abs(fft_result) ** 2
        # Find dominant frequency (largest amplitude)
        dominant_freq = np.argmax(power[1 : len(power) // 2]) + 1
        # Convert to phase
        phase = np.angle(fft_result[dominant_freq])
        # Normalize phase to [0, 1]
        return (phase + np.pi) / (2 * np.pi)

    def _calculate_quantum_amplitude(self, z_score, phase_estimation):
        """Calculate quantum amplitude with interference pattern"""
        # Basic amplitude from z-score using tanh (as before)
        base_amplitude = np.tanh(z_score)

        # Add interference from phase estimation
        # Constructive interference when phase aligns with z_score direction
        phase_factor = 2 * phase_estimation - 1  # Convert [0,1] to [-1,1]
        interference = self.p.interference_weight * phase_factor * np.sign(z_score)

        # Combine with interference effect
        combined_amplitude = base_amplitude + interference
        # Ensure amplitude stays in [-1, 1] range
        return np.clip(combined_amplitude, -1, 1)

    def _quantum_superposition(self, base_amplitude):
        """Create quantum superposition of multiple trading states"""
        # Create slightly varied amplitudes representing superposition of states
        noise_scale = 0.2
        superposition = []
        for i in range(self.p.superposition_count):
            # Add quantum noise to create superposition
            noise = noise_scale * (np.random.rand() - 0.5)
            # Ensure amplitude stays within [-1, 1]
            state_amplitude = np.clip(base_amplitude + noise, -1, 1)
            superposition.append(state_amplitude)

        # Update quantum state probabilities using soft voting
        self.quantum_state = np.array([(a + 1) / 2 for a in superposition])
        self.quantum_state /= np.sum(self.quantum_state)  # Normalize

        return superposition

    def _quantum_measurement(self, superposition_states):
        """Perform quantum measurement to collapse superposition to probabilities"""
        # Weighted average of superposition states using quantum state vector
        avg_amplitude = np.average(superposition_states, weights=self.quantum_state)
        # Convert to probabilities
        buy_probability = 0.5 * (avg_amplitude + 1)
        sell_probability = 1.0 - buy_probability
        return buy_probability, sell_probability

    def _apply_uncertainty_principle(self, buy_probability, sell_probability):
        """Apply Heisenberg uncertainty principle to trading decisions"""
        # More certain about price direction = less certain about timing
        certainty = abs(buy_probability - 0.5) * 2  # How far from 50/50
        uncertainty_adjustment = self.p.uncertainty_factor * certainty

        # Reduce extreme probabilities to model timing uncertainty
        if buy_probability > 0.5:
            buy_probability -= uncertainty_adjustment
            sell_probability += uncertainty_adjustment
        else:
            buy_probability += uncertainty_adjustment
            sell_probability -= uncertainty_adjustment

        return buy_probability, sell_probability

    def _quantum_decision(self, buy_probability, sell_probability):
        """Make a quantum-based decision with random draw"""
        random_draw = np.random.rand()

        if random_draw < buy_probability and buy_probability > self.p.prob_threshold:
            return "BUY"
        elif (
            random_draw < sell_probability and sell_probability > self.p.prob_threshold
        ):
            return "SELL"
        else:
            return "HOLD"

    def _quantum_tunneling(self, price, closes, direction):
        """Model quantum tunneling effect for breakthrough moments"""
        # Calculate recent volatility as the barrier height
        volatility = np.std(closes)
        if volatility == 0:
            return False

        # Calculate momentum as recent price changes
        momentum = (price - closes[1]) / volatility

        # Quantum tunneling probability (simplified model)
        # Higher momentum = higher probability of tunneling through resistance/support
        tunnel_probability = norm.cdf(abs(momentum) / self.p.tunneling_threshold)

        # Higher momentum in the right direction increases tunneling probability
        if (direction == "BUY" and momentum > 0) or (
            direction == "SELL" and momentum < 0
        ):
            # Random tunneling based on probability
            return np.random.rand() < tunnel_probability
        return False
