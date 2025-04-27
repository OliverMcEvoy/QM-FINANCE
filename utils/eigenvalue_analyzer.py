import numpy as np
from scipy import linalg, fft
import pandas as pd


class EigenvalueAnalyzer:
    """Core eigenvalue analysis for stock screening"""

    def __init__(self, params=None):
        self.params = params or {
            "potential_factor": 0.5,
            "kinetic_factor": 0.5,
            "eigenvalue_count": 5,
            "eigenvalue_smoothing": 0.8,
            "sr_lookback": 120,
            "wf_lookback": 50,
            "eigenvalue_buy_threshold": 0.02,
            "eigenvalue_sell_threshold": 0.02,
            "time_evolution_rate": 0.2,
            "eigenvalue_persistence": 0.7,
        }
        self.potential_wells = []
        self.smoothed_eigenvalues = None
        self.time_evolution_matrix = None
        self.eigenvalue_history = []

    def _calculate_risk_adjusted_momentum(
        self,
        current_price,
        eigenvalues,
        price_history,
        buy_signal,
        sell_signal,
        support_level,
        resistance_level,
        distance_to_nearest,
    ):
        """Calculate a risk-adjusted momentum score."""
        if not eigenvalues or len(eigenvalues) == 0 or current_price == 0:
            return 0.0

        # Momentum component
        momentum = buy_signal - sell_signal

        # Risk component (normalized distance to nearest level) - lower is riskier
        # We use the raw distance here, not percentage
        risk = (distance_to_nearest / current_price) if current_price else 0.0

        # Volatility component (using recent price history)
        lookback = min(20, len(price_history))
        if lookback < 5:
            volatility = 0.0
        else:
            recent_prices = price_history[-lookback:]
            mean_recent = np.mean(recent_prices)
            std_recent = np.std(recent_prices)
            volatility = (std_recent / mean_recent) if mean_recent else 0.0

        # Eigenvalue spread component (normalized)
        eigenvalue_spread = (
            ((resistance_level - support_level) / current_price)
            if current_price
            else 0.0
        )

        # Combine components
        # We want high momentum, low risk (being far from levels is less risky for momentum),
        # wider spread (more room to move), adjusted by volatility.
        # Formula: Momentum * (1 + Risk) * Eigenvalue_Spread / (1 + Volatility)
        # Using (1+Risk) because higher distance (lower risk) should amplify momentum score.
        score = (
            momentum
            * (1 + risk * 5)  # Amplify effect of distance
            * (1 + eigenvalue_spread)  # Amplify effect of spread
            / (1 + volatility * 2)  # Dampen score by volatility
        )

        # Normalize roughly (can exceed +/- 1 but centers around 0)
        return np.tanh(score)  # Use tanh to keep score bounded (-1 to 1)

    def analyze_stock(self, price_data):
        """Analyze a stock's price history and return eigenvalue metrics"""
        if len(price_data) < max(
            self.params["sr_lookback"], self.params["wf_lookback"]
        ):  # Ensure enough data for both lookbacks
            return None

        close_prices = price_data["Close"].values
        current_price = close_prices[-1]

        # Calculate eigenvalues over time
        eigenvalue_history, dates = self._calculate_eigenvalue_history(price_data)

        if not eigenvalue_history or len(eigenvalue_history) == 0:
            return None

        # Use the latest set of eigenvalues
        current_eigenvalues = eigenvalue_history[-1]
        if not current_eigenvalues or len(current_eigenvalues) == 0:
            return None  # No eigenvalues calculated for the latest point

        # --- New Support/Resistance Logic ---
        sorted_eigenvalues = sorted(
            list(set(current_eigenvalues))
        )  # Ensure sorted unique values
        n_eigenvalues = len(sorted_eigenvalues)

        if n_eigenvalues == 0:
            return None  # Should not happen if check above passed, but safety first

        # Determine Support Level based on user's definition
        support_candidates_below = [e for e in sorted_eigenvalues if e < current_price]
        if support_candidates_below:
            support_level = max(support_candidates_below)
        else:
            # No eigenvalues strictly below price. Find the lowest one strictly above.
            support_candidates_above = [
                e for e in sorted_eigenvalues if e > current_price
            ]
            if support_candidates_above:
                support_level = min(support_candidates_above)
            else:
                # Price might be equal to all, or outside range. Fallback to lowest.
                support_level = sorted_eigenvalues[0]

        # Determine Resistance Level based on user's definition
        resistance_candidates_above_support = [
            e for e in sorted_eigenvalues if e > support_level
        ]
        if resistance_candidates_above_support:
            resistance_level = min(resistance_candidates_above_support)
        else:
            # Support level might be the highest eigenvalue. Fallback to highest.
            resistance_level = sorted_eigenvalues[-1]
            # Ensure resistance is not below support in edge cases
            if resistance_level < support_level:
                resistance_level = support_level
        # --- End of New Support/Resistance Logic ---

        # Calculate other metrics using original methods but potentially passing sorted eigenvalues
        # Note: _evaluate_eigenvalue_signals and _calculate_breakout_potential have their own internal
        # logic for finding relevant levels based on <= and >= which we keep for now.
        buy_signal, sell_signal = self._evaluate_eigenvalue_signals(
            current_price, sorted_eigenvalues
        )

        breakout_potential = self._calculate_breakout_potential(
            current_price, sorted_eigenvalues, close_prices
        )

        # Calculate distance to nearest using the standard definition (closest absolute distance)
        raw_distance_to_nearest = self._distance_to_nearest_eigenvalue(
            current_price, sorted_eigenvalues
        )
        percent_distance_to_nearest = (
            (raw_distance_to_nearest / current_price * 100) if current_price else 0.0
        )

        # Calculate the risk-adjusted momentum using the NEWLY defined support/resistance levels
        risk_adjusted_momentum = self._calculate_risk_adjusted_momentum(
            current_price,
            sorted_eigenvalues,  # Pass all eigenvalues for context if needed
            close_prices,
            buy_signal,  # Use signals from original method
            sell_signal,  # Use signals from original method
            support_level,  # Use NEW definition
            resistance_level,  # Use NEW definition
            raw_distance_to_nearest,  # Use standard distance definition for risk component
        )

        return {
            "ticker": price_data.name if hasattr(price_data, "name") else "",
            "current_price": current_price,
            "eigenvalues": current_eigenvalues,  # Return original (potentially unsorted/non-unique) list from history
            "eigenvalue_history": eigenvalue_history,
            "dates": dates,
            "buy_signal": buy_signal,
            "sell_signal": sell_signal,
            "breakout_potential": breakout_potential,
            "support_level": support_level,  # Return NEW definition
            "resistance_level": resistance_level,  # Return NEW definition
            "distance_to_nearest": percent_distance_to_nearest,  # Return standard distance metric
            "risk_adjusted_momentum": risk_adjusted_momentum,  # Calculated with NEW levels
        }

    def _calculate_eigenvalue_history(self, price_data):
        """Calculate eigenvalues over time for the entire price history"""
        # Reset state variables for fresh calculation
        self.potential_wells = []
        self.smoothed_eigenvalues = None
        self.time_evolution_matrix = None
        self.eigenvalue_history = []

        # Want to analyze as much data as reasonably possible
        total_points = len(price_data)
        wf_lookback = self.params["wf_lookback"]

        # Ensure we have enough data
        if total_points < wf_lookback:
            return [], []

        # Determine a reasonable number of points to analyze (without overloading)
        # Either analyze all points or sample to get approximately 200 analysis points
        step = max(1, (total_points - wf_lookback) // 200)

        # Keep track of all dates and eigenvalues
        all_dates = []
        eigenvalue_history = []

        # Calculate eigenvalues for windows across the entire price history
        for i in range(wf_lookback, total_points, step):
            # Get window ending at this point
            price_window = price_data["Close"].values[i - wf_lookback : i]

            # Calculate the Hamiltonian and eigenvalues for this window
            hamiltonian = self._calculate_hamiltonian(price_window)

            if hamiltonian is not None:
                eigenvalues, _ = self._calculate_eigenvalues(hamiltonian, price_window)
                if eigenvalues:
                    eigenvalue_history.append(eigenvalues)
                    all_dates.append(
                        price_data.index[i - 1]
                    )  # Use the date at the end of the window

        return eigenvalue_history, all_dates

    def _calculate_hamiltonian(self, price_history):
        """Calculate a time-dependent Hamiltonian based on price history"""
        lookback = min(len(price_history), self.params["wf_lookback"])
        if lookback < 10:  # Need minimum data points
            return None

        # Get current and previous prices for momentum
        current_price = price_history[-1]
        prev_price = price_history[-2] if len(price_history) > 1 else current_price
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
            k_factor = self.params["kinetic_factor"] * (1 + 0.5 * momentum_factor)
            kinetic_term[i, i - 1] = k_factor
            kinetic_term[i, i] = -2 * k_factor
            kinetic_term[i, i + 1] = k_factor

        kinetic_term[0, 0] = -2 * self.params["kinetic_factor"]
        kinetic_term[0, 1] = self.params["kinetic_factor"]
        kinetic_term[-1, -2] = self.params["kinetic_factor"]
        kinetic_term[-1, -1] = -2 * self.params["kinetic_factor"]

        # TIME-DEPENDENT POTENTIAL TERM
        # Dynamic potential wells that evolve with market conditions
        potential_term = np.zeros((lookback, lookback))

        # Update potential wells based on price history
        self._update_potential_wells(price_history, mean_price, std_price)

        # Apply potential wells to the potential term
        for i in range(lookback):
            price_point = price_history[i]

            # Base potential (quadratic from mean)
            base_potential = self.params["potential_factor"] * (
                normalized_price[i] ** 2
            )

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
        if len(price_history) >= self.params["sr_lookback"]:
            # Use longer history for support/resistance detection if available
            hist_data = price_history[-self.params["sr_lookback"] :]

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
            evolution_rate = self.params["time_evolution_rate"]
            persistence = self.params["eigenvalue_persistence"]

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

    def _calculate_eigenvalues(self, hamiltonian, price_history):
        """Calculate eigenvalues from Hamiltonian and map them to price space"""
        try:
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = linalg.eigh(hamiltonian)

            # Get price statistics
            mean_price = np.mean(price_history)
            std_price = np.std(price_history) if np.std(price_history) > 0 else 1
            min_price = np.min(price_history)
            max_price = np.max(price_history)
            price_range = max_price - min_price

            # Calculate potential wells as price levels
            well_prices = [well["center"] for well in self.potential_wells]

            # Select eigenvalues to track
            n = min(self.params["eigenvalue_count"], len(eigenvalues))

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
                        evolved_val = self.smoothed_eigenvalues[i] * self.params[
                            "eigenvalue_persistence"
                        ] + price_eig * (1 - self.params["eigenvalue_persistence"])

                        # Include influence from potential wells
                        if well_prices:
                            # Find closest well
                            closest_well = min(
                                well_prices, key=lambda x: abs(x - evolved_val)
                            )
                            # Pull eigenvalue toward well based on distance and time evolution rate
                            well_pull = (closest_well - evolved_val) * self.params[
                                "time_evolution_rate"
                            ]
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
            print(f"Error calculating eigenvalues: {e}")
            # Return previous values or empty list
            return (
                self.smoothed_eigenvalues if self.smoothed_eigenvalues else []
            ), np.eye(self.params["eigenvalue_count"])

    def _evaluate_eigenvalue_signals(self, price, eigenvalues):
        """
        Generate buy/sell signals based on price position relative to eigenvalues
        Returns: (buy_signal_strength, sell_signal_strength) both between 0-1
        """
        if not eigenvalues or len(eigenvalues) == 0:
            return 0.0, 0.0

        # Use volatility for adaptive thresholds
        std_price = np.std(eigenvalues) if np.std(eigenvalues) > 0 else price * 0.01
        atr_factor = min(0.05, std_price / price)  # Cap at 5% of price

        # Dynamic thresholds based on volatility
        buy_threshold = max(self.params["eigenvalue_buy_threshold"], atr_factor)
        sell_threshold = max(self.params["eigenvalue_sell_threshold"], atr_factor)

        # Sort eigenvalues to find lowest and highest
        sorted_eigenvalues = sorted(eigenvalues)
        lowest_eigenvalue = sorted_eigenvalues[0]
        highest_eigenvalue = sorted_eigenvalues[-1]

        # Find nearest eigenvalues (support below, resistance above)
        support_levels = [e for e in sorted_eigenvalues if e <= price]
        resistance_levels = [e for e in sorted_eigenvalues if e >= price]

        nearest_support = max(support_levels) if support_levels else lowest_eigenvalue
        nearest_resistance = (
            min(resistance_levels) if resistance_levels else highest_eigenvalue
        )

        # Buy signal: stronger when price is near support
        buy_signal_strength = 0.0
        if price <= nearest_support:
            # Price is at or below support
            distance_factor = min(
                1.0, abs(price - nearest_support) / (price * buy_threshold)
            )
            buy_signal_strength = 1.0 - (distance_factor * 0.5)
        elif price <= nearest_support * (1 + buy_threshold):
            # Price is within threshold of support
            proximity = (nearest_support * (1 + buy_threshold) - price) / (
                nearest_support * buy_threshold
            )
            buy_signal_strength = max(0.0, proximity * 0.8)

        # Sell signal: stronger when price is near resistance
        sell_signal_strength = 0.0
        if price >= nearest_resistance:
            # Price is at or above resistance
            distance_factor = min(
                1.0, abs(price - nearest_resistance) / (price * sell_threshold)
            )
            sell_signal_strength = 1.0 - (distance_factor * 0.5)
        elif price >= nearest_resistance * (1 - sell_threshold):
            # Price is within threshold of resistance
            proximity = (price - nearest_resistance * (1 - sell_threshold)) / (
                nearest_resistance * sell_threshold
            )
            sell_signal_strength = max(0.0, proximity * 0.8)

        return buy_signal_strength, sell_signal_strength

    def _calculate_breakout_potential(self, current_price, eigenvalues, price_history):
        """Calculate the potential for a stock to break out based on eigenvalues and price action"""
        if not eigenvalues or len(eigenvalues) == 0:
            return 0.0

        # Sort eigenvalues
        sorted_eigenvalues = sorted(eigenvalues)

        # Calculate distances to nearest support and resistance
        supports = [e for e in sorted_eigenvalues if e <= current_price]
        resistances = [e for e in sorted_eigenvalues if e >= current_price]

        nearest_support = max(supports) if supports else sorted_eigenvalues[0]
        nearest_resistance = min(resistances) if resistances else sorted_eigenvalues[-1]

        # Calculate price momentum (recent trend strength)
        lookback = min(20, len(price_history))
        if lookback < 5:
            return 0.0

        recent_prices = price_history[-lookback:]
        price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

        # Calculate volatility
        price_volatility = np.std(recent_prices) / np.mean(recent_prices)

        # Calculate distance ratios
        support_distance = (current_price - nearest_support) / current_price
        resistance_distance = (nearest_resistance - current_price) / current_price

        # Breakout potential is high when:
        # 1. Price is close to resistance but with strong upward momentum
        # 2. Or price is showing strong momentum likely to overcome resistance
        # 3. Higher volatility increases breakout potential

        if price_trend > 0:
            # Bullish momentum - potential upside breakout
            breakout_score = (
                (1.0 - resistance_distance)
                * (1.0 + price_trend * 10)
                * (1.0 + price_volatility * 2)
            )
            # Proximity bonus when very close to resistance
            if resistance_distance < 0.01:  # Within 1%
                breakout_score *= 1.5
        else:
            # Bearish momentum - potential downside breakout
            breakout_score = (
                (1.0 - support_distance)
                * (1.0 + abs(price_trend) * 10)
                * (1.0 + price_volatility * 2)
            )
            # Proximity bonus when very close to support
            if support_distance < 0.01:  # Within 1%
                breakout_score *= 1.5

        # Normalize score to 0-1 range
        return min(1.0, breakout_score)

    def _distance_to_nearest_eigenvalue(self, price, eigenvalues):
        """Calculate the distance to the nearest eigenvalue support/resistance level"""
        if not eigenvalues or len(eigenvalues) == 0:
            return float("inf")

        distances = [abs(price - ev) for ev in eigenvalues]
        return min(distances)
