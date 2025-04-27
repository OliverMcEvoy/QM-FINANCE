import backtrader as bt
import numpy as np
from scipy import fft
from scipy.stats import norm
from scipy import linalg


class CustomQuantumStrategy(bt.Strategy):
    """
    A simplified template for quantum-inspired trading strategies.
    Focuses on core quantum concepts with minimal parameters.
    """

    # Simplified parameter set
    params = (
        # Core parameters
        ("sma_period", 10),  # Period for moving average calculation
        ("prob_threshold", 0.58),  # Decision threshold (lowered slightly)
        # Essential quantum parameters
        ("quantum_factor", 0.5),  # Controls overall quantum effect strength
        ("phase_period", 20),  # Period for phase analysis
        ("uncertainty", 0.2),  # Uncertainty principle factor
        # Wavefunction parameters
        ("wf_components", 5),  # Number of wavefunction components to fit
        ("wf_lookback", 50),  # Lookback period for wavefunction fitting
        # Hamiltonian parameters
        ("potential_factor", 0.5),  # Weight of potential energy term
        ("kinetic_factor", 0.5),  # Weight of kinetic energy term
        ("eigenvalue_count", 5),  # Number of eigenvalues to calculate
        # Support/Resistance eigenvalue parameters
        ("sr_lookback", 120),  # Lookback for support/resistance detection
        (
            "eigenvalue_smoothing",
            0.8,
        ),  # Smoothing factor for eigenvalues over time (0-1)
        ("price_history_weight", 0.6),  # Weight given to historical price levels
        # Time evolution parameters
        ("time_evolution_rate", 0.2),  # Rate of eigenvalue time evolution (0-1)
        ("eigenvalue_persistence", 0.7),  # How persistent eigenvalues are over time
        ("memory_decay", 0.05),  # Rate at which old price memory decays
        # New eigenvalue trading parameters
        (
            "eigenvalue_buy_threshold",
            0.02,
        ),  # Buy when price within this % of lowest eigenvalue
        (
            "eigenvalue_sell_threshold",
            0.02,
        ),  # Sell when price within this % of highest eigenvalue
        (
            "eigenvalue_signal_weight",
            0.65,  # Increased weight for eigenvalue signals
        ),
        # Variable sizing parameter
        ("max_position_pct", 0.95),  # Max % of cash to use on a max strength buy signal
    )

    def __init__(self, *args, **kwargs):
        # ...existing code...
        super().__init__(*args, **kwargs)
        # Trade tracking (now storing dollar amounts)
        self.buy_signals = []  # Stores (datetime, price, buy_value)
        self.sell_signals = []  # Stores (datetime, price, sell_value)
        self.trades = []  # Stores dict with buy/sell info including values
        self.current_trade = None

        # Moving average indicator
        self.sma = bt.indicators.SMA(self.data.close, period=self.p.sma_period)

        # Quantum state (simplified)
        self.quantum_state = np.array([0.5, 0.5])  # Initial 50/50 state

        # Wavefunction storage
        self.wavefunction_data = []
        self.price_data = []
        self.dates = []

        # Hamiltonian and eigenvalues storage
        self.eigenvalues = []
        self.hamiltonian_data = []
        self.last_hamiltonian = None
        self.smoothed_eigenvalues = None  # For storing time-smoothed eigenvalues

        # Time evolution tracking - create with default values
        self.eigenvalue_history = []  # Track eigenvalues over time
        self.time_evolution_matrix = None  # Time evolution operator
        self.potential_wells = []  # Dynamic potential wells
        self.price_memory = []  # Store price history with decaying importance

        # Initialize default eigenvalues based on params
        self.default_eigenvalues = []  # Will be populated in first next() call

        # Track eigenvalue-based signals for visualization
        self.eigenvalue_signals = []

    def next(self):
        # Skip if not enough data
        if len(self.data) < self.p.sma_period:
            return

        # 1. MARKET STATE CALCULATION
        price = self.data.close[0]
        dt = self.data.datetime.datetime(0)

        # ...existing code for date/price storage and memory update...
        self.dates.append(dt)
        self.price_data.append(price)

        # If this is the first call (or default_eigenvalues are still empty),
        # initialize default eigenvalues around the first price
        if not self.default_eigenvalues:
            spacing = price * 0.01  # 1% spacing between levels
            self.default_eigenvalues = [
                price + (i - self.p.eigenvalue_count // 2) * spacing
                for i in range(self.p.eigenvalue_count)
            ]

        # Update price memory with decay
        if self.price_memory:
            # Apply decay to existing memory
            self.price_memory = [
                p * (1 - self.p.memory_decay) for p in self.price_memory
            ]
        # Add current price with full weight
        self.price_memory.append(price)

        # Calculate z-score (normalized distance from moving average)
        # ...existing code...
        closes = [self.data.close[-i] for i in range(self.p.sma_period)]
        sma = np.mean(closes)
        vol = np.std(closes) if np.std(closes) > 0 else 1
        z_score = (price - sma) / vol

        # 2. QUANTUM ANALYSIS
        # ...existing code...
        phase = self._calculate_phase()

        # Calculate quantum amplitude (-1 to 1 representing market direction)
        amplitude = np.tanh(z_score) + self.p.quantum_factor * (
            2 * phase - 1
        ) * np.sign(z_score)
        amplitude = np.clip(amplitude, -1, 1)

        # Calculate buy/sell probabilities with uncertainty principle
        buy_prob = 0.5 * (amplitude + 1)
        sell_prob = 1.0 - buy_prob

        # Apply uncertainty (stronger signals have more timing uncertainty)
        certainty = abs(buy_prob - 0.5) * 2
        adjustment = self.p.uncertainty * certainty

        if buy_prob > 0.5:
            buy_prob -= adjustment
        else:
            buy_prob += adjustment

        # Fit wavefunction and calculate eigenvalues
        current_eigenvalues = []

        if len(self.data) >= self.p.wf_lookback:
            # ...existing code for wavefunction fitting...
            wf_value = self._fit_wavefunction()
            self.wavefunction_data.append(wf_value)

            # Calculate time-evolving Hamiltonian and eigenvalues
            hamiltonian = self._calculate_time_dependent_hamiltonian()
            if hamiltonian is not None:
                self.last_hamiltonian = hamiltonian
                eig_vals, time_evolution = self._calculate_evolving_eigenvalues(
                    hamiltonian
                )
                self.eigenvalues.append(eig_vals)
                self.hamiltonian_data.append(hamiltonian)
                current_eigenvalues = eig_vals.copy()

                # Store time evolution operator
                self.time_evolution_matrix = time_evolution

                # Track eigenvalue history for visualization
                self.eigenvalue_history.append(eig_vals)
            else:
                # If we can't calculate a new Hamiltonian, use default eigenvalues
                # ...existing code for default eigenvalues...
                self.hamiltonian_data.append(np.zeros((2, 2)))
                # Use previous eigenvalues if available, otherwise default ones
                if (
                    self.smoothed_eigenvalues is not None
                    and len(self.smoothed_eigenvalues) > 0
                ):
                    empty_eig_vals = self.smoothed_eigenvalues.copy()
                    current_eigenvalues = empty_eig_vals
                else:
                    # Use default eigenvalues centered around current price
                    base_level = price
                    spacing = price * 0.01  # 1% spacing between levels
                    empty_eig_vals = [
                        base_level + (i - self.p.eigenvalue_count // 2) * spacing
                        for i in range(self.p.eigenvalue_count)
                    ]
                    current_eigenvalues = empty_eig_vals

                self.eigenvalues.append(empty_eig_vals)
                # Still append to history to maintain alignment with dates
                self.eigenvalue_history.append(empty_eig_vals)
        else:
            # For early data points, use default eigenvalues centered around current price
            # ...existing code for default eigenvalue creation...
            self.wavefunction_data.append(price)
            self.hamiltonian_data.append(np.zeros((2, 2)))

            # Create simple eigenvalues spaced around the current price
            base_level = price
            spacing = price * 0.01  # 1% spacing between levels
            default_eigs = [
                base_level + (i - self.p.eigenvalue_count // 2) * spacing
                for i in range(self.p.eigenvalue_count)
            ]
            current_eigenvalues = default_eigs

            self.eigenvalues.append(default_eigs)
            # Also append to history to maintain alignment with dates
            self.eigenvalue_history.append(default_eigs)

        # 3. EIGENVALUE-BASED SIGNALS
        # Calculate trading signals based on current price relative to eigenvalues
        eigenvalue_buy_signal, eigenvalue_sell_signal = (
            self._evaluate_eigenvalue_signals(price, current_eigenvalues)
        )

        # 4. COMBINED DECISION MAKING
        # Blend quantum probability signals with eigenvalue signals
        final_buy_prob = (
            buy_prob * (1 - self.p.eigenvalue_signal_weight)
            + eigenvalue_buy_signal * self.p.eigenvalue_signal_weight
        )
        final_sell_prob = (
            sell_prob * (1 - self.p.eigenvalue_signal_weight)
            + eigenvalue_sell_signal * self.p.eigenvalue_signal_weight
        )

        # Store signal for visualization
        self.eigenvalue_signals.append(
            (dt, eigenvalue_buy_signal, eigenvalue_sell_signal)
        )

        # 5. TRADING DECISION
        if not self.position:  # Not in market
            if final_buy_prob > self.p.prob_threshold:
                # --- MODIFIED BUY SIZING ---
                # Always use the maximum allowed percentage of cash
                available_cash = self.broker.getcash()
                cash_to_use = (
                    available_cash * self.p.max_position_pct
                )  # Use max allowed cash

                # Calculate shares and actual buy value
                shares = int(cash_to_use / price)
                if shares > 0:
                    buy_value = shares * price
                    self.buy(size=shares)
                    self.buy_signals.append((dt, price, buy_value))  # Store buy value
                    self.current_trade = {
                        "buy_date": dt,
                        "buy_price": price,
                        "shares": shares,
                        "buy_value": buy_value,  # Store buy value in trade log
                    }
                # --- END MODIFIED BUY SIZING ---
        else:  # In market
            if final_sell_prob > self.p.prob_threshold:
                # Calculate sell value (based on current position size)
                sell_value = self.position.size * price
                # Sell all shares (already closes the entire position)
                self.close()
                self.sell_signals.append((dt, price, sell_value))  # Store sell value

                # Record trade
                if self.current_trade:
                    pnl = price - self.current_trade["buy_price"]
                    total_pnl = pnl * self.current_trade["shares"]
                    self.trades.append(
                        {
                            "buy_date": self.current_trade["buy_date"],
                            "buy_price": self.current_trade["buy_price"],
                            "buy_value": self.current_trade[
                                "buy_value"
                            ],  # Add buy value
                            "sell_date": dt,
                            "sell_price": price,
                            "sell_value": sell_value,  # Add sell value
                            "shares": self.current_trade["shares"],
                            "total_pnl": total_pnl,
                        }
                    )
                    self.current_trade = None

    def _evaluate_eigenvalue_signals(self, price, eigenvalues):
        """
        Generate buy/sell signals based on price position relative to eigenvalues
        Returns: (buy_signal_strength, sell_signal_strength) both between 0-1
        """
        if not eigenvalues or len(eigenvalues) == 0:
            return 0.0, 0.0

        # Sort eigenvalues to find lowest and highest
        sorted_eigenvalues = sorted(eigenvalues)
        lowest_eigenvalue = sorted_eigenvalues[0]
        highest_eigenvalue = sorted_eigenvalues[-1]

        # Calculate relative position of price within eigenvalue range
        eig_range = highest_eigenvalue - lowest_eigenvalue
        if eig_range <= 0:
            rel_position = 0.5  # Default to middle if all eigenvalues are the same
        else:
            rel_position = (price - lowest_eigenvalue) / eig_range

        # Calculate distance to nearest eigenvalues as percentage
        distances = [abs(price - eig) / price for eig in eigenvalues]
        min_distance = min(distances) if distances else 1.0

        # Buy signal: stronger when price is near or below lowest eigenvalue (support)
        # Linear signal strength that increases as price approaches or drops below support
        buy_threshold = self.p.eigenvalue_buy_threshold
        buy_signal_strength = 0.0

        if price <= lowest_eigenvalue:
            # Price is below lowest eigenvalue (strong support signal)
            distance_factor = min(1.0, min_distance / buy_threshold)
            buy_signal_strength = 1.0 - (
                distance_factor * 0.5
            )  # Start at 0.5, go to 1.0 as gets closer
        elif price <= lowest_eigenvalue * (1 + buy_threshold):
            # Price is within threshold of lowest eigenvalue
            proximity = (lowest_eigenvalue * (1 + buy_threshold) - price) / (
                lowest_eigenvalue * buy_threshold
            )
            buy_signal_strength = max(0.0, proximity * 0.8)  # Scale to 0.0-0.8

        # Sell signal: stronger when price is near or above highest eigenvalue (resistance)
        sell_threshold = self.p.eigenvalue_sell_threshold
        sell_signal_strength = 0.0

        if price >= highest_eigenvalue:
            # Price is above highest eigenvalue (strong resistance signal)
            distance_factor = min(1.0, min_distance / sell_threshold)
            sell_signal_strength = 1.0 - (
                distance_factor * 0.5
            )  # Start at 0.5, go to 1.0 as gets closer
        elif price >= highest_eigenvalue * (1 - sell_threshold):
            # Price is within threshold of highest eigenvalue
            proximity = (price - highest_eigenvalue * (1 - sell_threshold)) / (
                highest_eigenvalue * sell_threshold
            )
            sell_signal_strength = max(0.0, proximity * 0.8)  # Scale to 0.0-0.8

        return buy_signal_strength, sell_signal_strength

    def _calculate_phase(self):
        """Calculate market phase using simplified quantum phase estimation"""
        # Get price history
        lookback = self.p.phase_period * 2
        if len(self.data) < lookback:
            return 0.5  # Default to middle of cycle

        price_history = np.array([self.data.close[-i] for i in range(lookback)])

        # Simple Fourier transform to find dominant cycle
        fft_result = fft.fft(price_history)
        power = np.abs(fft_result) ** 2
        dominant_freq = np.argmax(power[1 : len(power) // 2]) + 1
        phase = np.angle(fft_result[dominant_freq])

        # Normalize to [0,1]
        return (phase + np.pi) / (2 * np.pi)

    def _fit_wavefunction(self):
        """Fit a quantum-inspired wavefunction to recent price history"""
        # Get price history for wavefunction fitting
        lookback = min(len(self.data), self.p.wf_lookback)
        if lookback < 10:  # Need minimum data points
            return self.data.close[0]

        price_history = np.array([self.data.close[-i] for i in range(lookback)])
        # Normalize price history
        mean_price = np.mean(price_history)
        std_price = np.std(price_history) if np.std(price_history) > 0 else 1
        normalized_price = (price_history - mean_price) / std_price

        # Apply FFT to get frequency components
        fft_result = fft.fft(normalized_price)

        # Get top frequency components (by amplitude)
        amplitudes = np.abs(fft_result[: lookback // 2])
        phases = np.angle(fft_result[: lookback // 2])

        # Sort by amplitude and get top components
        top_indices = np.argsort(amplitudes)[-self.p.wf_components :]

        # Reconstruct wavefunction using top components
        t = np.arange(lookback)
        wavefunction = np.zeros(lookback)

        for idx in top_indices:
            if idx > 0:  # Skip DC component
                freq = idx / lookback
                amplitude = amplitudes[idx] / lookback
                phase = phases[idx]
                wavefunction += amplitude * np.cos(2 * np.pi * freq * t + phase)

        # Scale back to price range and add mean
        wavefunction = (wavefunction * std_price) + mean_price

        # Return current fitted value (can be used for prediction)
        return wavefunction[0]

    def _calculate_time_dependent_hamiltonian(self):
        """Calculate a time-dependent Hamiltonian that evolves with market conditions"""
        lookback = min(len(self.data), self.p.wf_lookback)
        if lookback < 10:  # Need minimum data points
            return None

        # Get price history
        price_history = np.array([self.data.close[-i] for i in range(lookback)])

        # Get current and previous prices for momentum
        current_price = self.data.close[0]
        prev_price = self.data.close[-1] if len(self.data) > 1 else current_price
        price_momentum = current_price - prev_price

        # Normalize price history
        mean_price = np.mean(price_history)
        std_price = np.std(price_history) if np.std(price_history) > 0 else 1
        normalized_price = (price_history - mean_price) / std_price

        # Calculate first differences (momentum/velocity)
        price_momentum_series = np.diff(normalized_price, prepend=normalized_price[0])

        # TIME-DEPENDENT KINETIC TERM
        # The kinetic energy operator changes based on recent price momentum
        kinetic_term = np.zeros((lookback, lookback))
        momentum_factor = abs(price_momentum) / (std_price + 1e-10)

        # Enhanced kinetic term with time-dependence
        for i in range(1, lookback - 1):
            # Momentum weights the neighboring interactions
            k_factor = self.p.kinetic_factor * (1 + 0.5 * momentum_factor)
            kinetic_term[i, i - 1] = k_factor
            kinetic_term[i, i] = -2 * k_factor
            kinetic_term[i, i + 1] = k_factor

        kinetic_term[0, 0] = -2 * self.p.kinetic_factor
        kinetic_term[0, 1] = self.p.kinetic_factor
        kinetic_term[-1, -2] = self.p.kinetic_factor
        kinetic_term[-1, -1] = -2 * self.p.kinetic_factor

        # TIME-DEPENDENT POTENTIAL TERM
        # Dynamic potential wells that evolve with market conditions
        potential_term = np.zeros((lookback, lookback))

        # Update potential wells based on price history
        self._update_potential_wells(price_history, mean_price, std_price)

        # Apply potential wells to the potential term
        for i in range(lookback):
            price_point = price_history[i]

            # Base potential (quadratic from mean)
            base_potential = self.p.potential_factor * (normalized_price[i] ** 2)

            # Add contribution from each potential well
            well_contribution = 0
            for well in self.potential_wells:
                # Calculate distance from this price point to well center
                distance = abs(price_point - well["center"]) / std_price
                # Well strength decreases with distance (Gaussian wells)
                well_effect = well["depth"] * np.exp(-(distance**2) / well["width"])
                well_contribution += well_effect

            # Final potential combines base and wells (wells reduce potential)
            potential_term[i, i] = max(0.01, base_potential - well_contribution)

        # Complete time-dependent Hamiltonian
        hamiltonian = -kinetic_term + potential_term

        return hamiltonian

    def _update_potential_wells(self, price_history, mean_price, std_price):
        """Update the time-evolving potential wells based on price congestion"""
        # Calculate price histogram to identify congestion zones
        hist_data = price_history
        if len(self.data) >= self.p.sr_lookback:
            # Use longer history for support/resistance detection
            hist_data = np.array(
                [self.data.close[-i] for i in range(self.p.sr_lookback)]
            )

        # Create histogram with dynamic bins
        price_range = np.max(hist_data) - np.min(hist_data)
        bin_count = min(20, max(5, int(len(hist_data) / 10)))
        hist, bin_edges = np.histogram(hist_data, bins=bin_count)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Normalize histogram to get probability density
        hist = hist / np.sum(hist)

        # Identify significant price levels (potential wells)
        threshold = np.mean(hist) * 1.5
        significant_levels = []

        for i, count in enumerate(hist):
            if count > threshold:
                # This is a significant price level (congestion zone)
                significant_levels.append(
                    {
                        "center": bin_centers[i],
                        "strength": count / np.max(hist),  # Normalized strength
                        "width": price_range / bin_count / 2,  # Well width
                    }
                )

        # Time evolution of wells
        if not self.potential_wells:
            # First time - initialize wells
            self.potential_wells = [
                {
                    "center": level["center"],
                    "depth": level["strength"] * 2.0,  # Initial depth
                    "width": level["width"] * 2.0,  # Initial width
                }
                for level in significant_levels
            ]
        else:
            # Evolve existing wells and add new ones
            evolution_rate = self.p.time_evolution_rate
            persistence = self.p.eigenvalue_persistence

            # Gradually fade out existing wells
            for well in self.potential_wells:
                well["depth"] *= persistence
                well["width"] *= 1 + 0.1 * (
                    1 - persistence
                )  # Wells get wider as they fade

            # Remove wells that are too weak
            self.potential_wells = [
                well for well in self.potential_wells if well["depth"] > 0.05
            ]

            # Add or strengthen wells from current significant levels
            for level in significant_levels:
                # Check if this level is near an existing well
                found = False
                for well in self.potential_wells:
                    if abs(well["center"] - level["center"]) < well["width"]:
                        # Strengthen and adjust existing well
                        well["depth"] = (
                            well["depth"] * (1 - evolution_rate)
                            + level["strength"] * 2.0 * evolution_rate
                        )
                        well["center"] = (
                            well["center"] * (1 - evolution_rate)
                            + level["center"] * evolution_rate
                        )
                        well["width"] = max(
                            level["width"], well["width"] * 0.9
                        )  # Keep wells from growing too wide
                        found = True
                        break

                if not found:
                    # Add new well
                    self.potential_wells.append(
                        {
                            "center": level["center"],
                            "depth": level["strength"]
                            * 1.0,  # Start with moderate depth
                            "width": level["width"] * 2.0,
                        }
                    )

    def _calculate_evolving_eigenvalues(self, hamiltonian):
        """Calculate eigenvalues that evolve over time based on the time-dependent Hamiltonian"""
        try:
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = linalg.eigh(hamiltonian)

            # Get price history for scaling
            sr_lookback = min(len(self.data), self.p.sr_lookback)
            price_history = np.array([self.data.close[-i] for i in range(sr_lookback)])

            # Get price statistics
            current_price = self.data.close[0]
            mean_price = np.mean(price_history)
            std_price = np.std(price_history) if np.std(price_history) > 0 else 1
            min_price = np.min(price_history)
            max_price = np.max(price_history)
            price_range = max_price - min_price

            # Calculate potential wells as price levels
            well_prices = [well["center"] for well in self.potential_wells]

            # Select eigenvalues to track
            n = min(self.p.eigenvalue_count, len(eigenvalues))

            # Create a weighted blend of:
            # 1. Previous eigenvalues (evolving with time)
            # 2. Current eigenvalues from Hamiltonian
            # 3. Potential well centers (congestion points)

            # Get evenly spaced eigenvalues from the spectrum
            indices = np.linspace(0, len(eigenvalues) - 1, n).astype(int)
            raw_eigenvalues = eigenvalues[indices]

            # Initialize time evolution operator (identity at start)
            if self.time_evolution_matrix is None:
                time_evolution = np.eye(n)
            else:
                time_evolution = self.time_evolution_matrix

            # Create the new eigenvalues
            evolved_eigenvalues = []

            # If we have previous values, include them in evolution
            if self.smoothed_eigenvalues is not None:
                # Mix previous and current eigenvalues
                for i in range(n):
                    if i < len(self.smoothed_eigenvalues):
                        # Normalize raw eigenvalue to 0-1 range
                        norm_eig = (raw_eigenvalues[i] - np.min(eigenvalues)) / (
                            np.max(eigenvalues) - np.min(eigenvalues) + 1e-10
                        )

                        # Map to price range
                        price_eig = min_price + norm_eig * price_range

                        # Time evolution: blend previous and new
                        evolved_val = self.smoothed_eigenvalues[
                            i
                        ] * self.p.eigenvalue_persistence + price_eig * (
                            1 - self.p.eigenvalue_persistence
                        )

                        # Include influence from potential wells
                        if well_prices:
                            # Find closest well
                            closest_well = min(
                                well_prices, key=lambda x: abs(x - evolved_val)
                            )
                            # Pull eigenvalue toward well based on distance and time evolution rate
                            well_pull = (
                                closest_well - evolved_val
                            ) * self.p.time_evolution_rate
                            evolved_val += well_pull

                        evolved_eigenvalues.append(evolved_val)
                    else:
                        # For any new eigenvalues, map directly from Hamiltonian
                        norm_eig = (raw_eigenvalues[i] - np.min(eigenvalues)) / (
                            np.max(eigenvalues) - np.min(eigenvalues) + 1e-10
                        )
                        evolved_eigenvalues.append(min_price + norm_eig * price_range)
            else:
                # First time: initialize from current eigenvalues
                for i in range(n):
                    # Normalize and map to price range
                    norm_eig = (raw_eigenvalues[i] - np.min(eigenvalues)) / (
                        np.max(eigenvalues) - np.min(eigenvalues) + 1e-10
                    )
                    evolved_eigenvalues.append(min_price + norm_eig * price_range)

            # Ensure eigenvalues are within price range
            evolved_eigenvalues = [
                max(min_price * 0.9, min(max_price * 1.1, val))
                for val in evolved_eigenvalues
            ]

            # Sort eigenvalues for consistent ordering
            evolved_eigenvalues = sorted(evolved_eigenvalues)

            # Update the time evolution operator
            new_time_evolution = np.eye(n)
            if self.smoothed_eigenvalues is not None:
                # Calculate how eigenvalues moved between steps
                for i in range(n):
                    if i < len(self.smoothed_eigenvalues):
                        prev_val = self.smoothed_eigenvalues[i]
                        # Find which current eigenvalue is closest to prev_val
                        closest_idx = min(
                            range(n),
                            key=lambda j: abs(evolved_eigenvalues[j] - prev_val),
                        )
                        # Update the evolution matrix
                        new_time_evolution[i, closest_idx] = 1.0

            # Store for next iteration
            self.smoothed_eigenvalues = evolved_eigenvalues.copy()

            return evolved_eigenvalues, new_time_evolution

        except Exception as e:
            print(f"Error calculating evolving eigenvalues: {e}")
            # Return previous values or empty list
            return (
                self.smoothed_eigenvalues if self.smoothed_eigenvalues else []
            ), np.eye(self.p.eigenvalue_count)

    def get_wavefunction_data(self):
        """Return the fitted wavefunction data along with dates, prices, and trade signals"""
        # ...existing code...

        # Include eigenvalue signals for visualization
        eigenvalue_signal_data = (
            self.eigenvalue_signals if self.eigenvalue_signals else []
        )

        # Ensure buy/sell signals are included (they now contain dollar amounts)
        buy_signal_data = self.buy_signals if self.buy_signals else []
        sell_signal_data = self.sell_signals if self.sell_signals else []

        return {
            "dates": self.dates,
            "prices": self.price_data,
            "wavefunction": self.wavefunction_data,
            "wf_lookback": self.p.wf_lookback,
            "eigenvalues": (
                self.smoothed_eigenvalues
                if hasattr(self, "smoothed_eigenvalues")
                else []
            ),
            "eigenvalue_history": (
                self.eigenvalue_history if hasattr(self, "eigenvalue_history") else []
            ),
            "potential_wells": (
                self.potential_wells if hasattr(self, "potential_wells") else []
            ),
            "eigenvalue_signals": eigenvalue_signal_data,
            "buy_signals": buy_signal_data,  # Include buy signals with values
            "sell_signals": sell_signal_data,  # Include sell signals with values
            "trades": self.trades,  # Include completed trade details with values
        }
