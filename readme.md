# The Mathematics of Eigenvalue Trading Strategy

## Table of Contents

1. [Introduction](#introduction)
2. [Quantum Mechanical Foundation](#quantum-mechanical-foundation)
3. [The Hamiltonian Operator](#the-hamiltonian-operator)
4. [Eigenvalues and Market Support/Resistance](#eigenvalues-and-market-supportresistance)
5. [Time-Dependent Hamiltonians](#time-dependent-hamiltonians)
6. [Potential Wells as Price Attractors](#potential-wells-as-price-attractors)
7. [Wave Function Representation](#wave-function-representation)
8. [Eigenvalue Signal Generation](#eigenvalue-signal-generation)
9. [Time Evolution of Eigenvalues](#time-evolution-of-eigenvalues)
10. [Mathematical Implementation](#mathematical-implementation)
11. [Trading Decision Framework](#trading-decision-framework)
12. [References](#references)

## Introduction

The Eigenvalue Trading Strategy applies concepts from quantum mechanics to financial markets, particularly using eigenvalues of time-dependent Hamiltonians to identify potential support and resistance levels. This document provides a comprehensive mathematical treatment of the strategy's foundation and implementation.

## Quantum Mechanical Foundation

In quantum mechanics, physical systems are described by wave functions $\psi(x,t)$ whose evolution is governed by the Schrödinger equation:

$$i\hbar\frac{\partial\psi(x,t)}{\partial t} = \hat{H}\psi(x,t)$$

where $\hat{H}$ is the Hamiltonian operator, representing the total energy of the system.

For financial markets, we create an analog where price movements can be modeled as a quantum system, with prices following wave-like patterns influenced by "energy landscapes" of market psychology and technical levels.

## The Hamiltonian Operator

Our market Hamiltonian consists of two primary components:

$$\hat{H} = \hat{T} + \hat{V}$$

where:

- $\hat{T}$ is the kinetic energy operator, related to price momentum
- $\hat{V}$ is the potential energy operator, representing market structure

In matrix form, we construct a discretized Hamiltonian for N price points:

$$H_{ij} = -K_{ij} + V_{ij}$$

where:

- $K$ is the kinetic energy matrix with tridiagonal structure
- $V$ is the diagonal potential energy matrix

### Kinetic Energy Matrix

The kinetic term describes how prices "flow" between levels:

$$
K_{ij} =
\begin{cases}
k_f & \text{if } j = i+1 \\
-2k_f & \text{if } j = i \\
k_f & \text{if } j = i-1 \\
0 & \text{otherwise}
\end{cases}
$$

where $k_f$ is the kinetic factor parameter controlling the strength of momentum effects.

### Potential Energy Matrix

The potential term encodes the "energy landscape" of prices:

$$
V_{ij} =
\begin{cases}
p_f \cdot (p_i - \bar{p})^2 - \sum_{w \in \text{wells}} d_w \cdot e^{-\frac{(p_i-c_w)^2}{w_w}} & \text{if } i = j \\
0 & \text{otherwise}
\end{cases}
$$

where:

- $p_f$ is the potential factor parameter
- $p_i$ is the price at position $i$
- $\bar{p}$ is the mean price
- The sum term represents potential wells (support/resistance zones)
- $d_w$, $c_w$, and $w_w$ are the depth, center, and width of each well

## Eigenvalues and Market Support/Resistance

Solving the eigenvalue equation:

$$\hat{H}\psi_n = E_n\psi_n$$

yields eigenvalues $E_n$ and eigenvectors $\psi_n$.

These eigenvalues represent "energy states" that prices naturally gravitate toward or bounce away from, analogous to electron energy levels in atoms. In market terms:

- Lower eigenvalues represent support levels
- Higher eigenvalues represent resistance levels

## Time-Dependent Hamiltonians

Markets evolve over time, requiring a time-dependent Hamiltonian:

$$\hat{H}(t) = \hat{T}(t) + \hat{V}(t)$$

The kinetic term varies with recent price momentum:

$$k_f(t) = k_0 \cdot (1 + 0.5 \cdot m(t))$$

where $m(t)$ is the normalized price momentum:

$$m(t) = \frac{p(t) - p(t-1)}{\sigma_p}$$

The potential term evolves as potential wells emerge, deepen, fade, or shift based on price congestion patterns.

## Potential Wells as Price Attractors

Potential wells represent price levels where trading activity concentrates. Each well is characterized by:

1. Center ($c_w$): The price level of the well
2. Depth ($d_w$): The strength of the support/resistance
3. Width ($w_w$): The price range influenced by the well

The well contribution to the potential at price $p$ is:

$$V_{\text{well}}(p) = d_w \cdot e^{-\frac{(p-c_w)^2}{w_w}}$$

Wells evolve over time according to:

$$d_w(t+1) = d_w(t) \cdot \gamma + d_{\text{new}} \cdot (1-\gamma)$$
$$c_w(t+1) = c_w(t) \cdot \gamma + c_{\text{new}} \cdot (1-\gamma)$$

where $\gamma$ is the eigenvalue persistence parameter.

## Wave Function Representation

We construct a market wave function by analyzing price history:

$$\psi_{\text{market}}(p,t) = \sum_{j=1}^{N_c} A_j e^{i(2\pi f_j \cdot t + \phi_j)}$$

where:

- $N_c$ is the number of components (wf_components parameter)
- $A_j$, $f_j$, and $\phi_j$ are the amplitude, frequency, and phase of each component
- These parameters are derived from the Fourier transform of price history

The probability density of finding prices at level $p$ is given by:

$$P(p) = |\psi_{\text{market}}(p)|^2$$

## Eigenvalue Signal Generation

Trading signals are generated based on the relationship between current price and eigenvalues:

For a price $p$ and eigenvalues $\{E_1, E_2, ..., E_n\}$ (sorted in ascending order):

### Buy Signal Strength

$$
S_{\text{buy}}(p) =
\begin{cases}
1 - 0.5 \cdot \min\left(1, \frac{\min_j|p-E_j|/p}{\theta_{\text{buy}}}\right) & \text{if } p \leq E_1 \\
\max\left(0, 0.8 \cdot \frac{E_1(1+\theta_{\text{buy}}) - p}{E_1 \cdot \theta_{\text{buy}}}\right) & \text{if } p \leq E_1(1+\theta_{\text{buy}}) \\
0 & \text{otherwise}
\end{cases}
$$

### Sell Signal Strength

$$
S_{\text{sell}}(p) =
\begin{cases}
1 - 0.5 \cdot \min\left(1, \frac{\min_j|p-E_j|/p}{\theta_{\text{sell}}}\right) & \text{if } p \geq E_n \\
\max\left(0, 0.8 \cdot \frac{p - E_n(1-\theta_{\text{sell}})}{E_n \cdot \theta_{\text{sell}}}\right) & \text{if } p \geq E_n(1-\theta_{\text{sell}}) \\
0 & \text{otherwise}
\end{cases}
$$

where $\theta_{\text{buy}}$ and $\theta_{\text{sell}}$ are threshold parameters controlling signal sensitivity.

## Time Evolution of Eigenvalues

Eigenvalues evolve over time through:

1. **Hamiltonian Evolution**: Recalculating eigenvalues from the updated Hamiltonian
2. **Eigenvalue Smoothing**: Blending new and old eigenvalues
3. **Well Attraction**: Eigenvalues are drawn toward potential wells

The mathematical form is:

$$E_i(t+1) = \gamma \cdot E_i(t) + (1-\gamma) \cdot E'_i(t) + \delta_t \cdot (C_i - E_i(t))$$

where:

- $E_i(t)$ is the $i$-th eigenvalue at time $t$
- $\gamma$ is the eigenvalue persistence parameter
- $E'_i(t)$ is the raw eigenvalue from the current Hamiltonian
- $\delta_t$ is the time evolution rate parameter
- $C_i$ is the center of the closest potential well to $E_i$

## Mathematical Implementation

### Hamiltonian Construction Algorithm

1. Calculate price statistics: $\bar{p}$, $\sigma_p$, $p_{\min}$, $p_{\max}$
2. Normalize price history: $\tilde{p}_i = (p_i - \bar{p})/\sigma_p$
3. Calculate price momentum: $m = (p_0 - p_1)/\sigma_p$
4. Construct kinetic matrix $K$ with momentum-adjusted factor $k_f \cdot (1 + 0.5m)$
5. Identify price congestion zones from histogram analysis
6. Update potential wells based on congestion and previous wells
7. Construct potential matrix $V$ using wells and normalized prices
8. Form Hamiltonian: $H = -K + V$

### Eigenvalue Calculation and Evolution

1. Solve eigenvalue equation: $H\psi = E\psi$
2. Map eigenvalues to price space: $E_p = p_{\min} + \frac{E - E_{\min}}{E_{\max} - E_{\min}} \cdot (p_{\max} - p_{\min})$
3. Apply time evolution: blend previous eigenvalues with current ones
4. Apply well attraction: pull eigenvalues toward nearby potential wells
5. Sort eigenvalues to maintain consistent ordering

## Trading Decision Framework

The final decision combines:

1. Quantum probability signals ($P_{\text{buy}}$, $P_{\text{sell}}$)
2. Eigenvalue-based signals ($S_{\text{buy}}$, $S_{\text{sell}}$)

The combined probabilities are:

$$P'_{\text{buy}} = w \cdot P_{\text{buy}} + (1-w) \cdot S_{\text{buy}}$$
$$P'_{\text{sell}} = w \cdot P_{\text{sell}} + (1-w) \cdot S_{\text{sell}}$$

where $w$ is the eigenvalue signal weight parameter.

Trading decisions are made when these probabilities exceed the threshold:

$$
\text{Decision} =
\begin{cases}
\text{BUY} & \text{if } P'_{\text{buy}} > \theta_p \\
\text{SELL} & \text{if } P'_{\text{sell}} > \theta_p \\
\text{HOLD} & \text{otherwise}
\end{cases}
$$

where $\theta_p$ is the probability threshold parameter.

## Complete Hamiltonian and Solution

The complete Hamiltonian matrix $H$ combines the kinetic ($K$) and potential ($V$) terms:

$$H = -K + V$$

For a system discretized into $N$ price points (or time steps in the lookback window), the Hamiltonian is an $N \times N$ matrix. Its elements $H_{ij}$ are:

$$
H_{ij} =
\begin{cases}
-k_f(t) & \text{if } j = i+1 \text{ or } j = i-1 \\
2k_f(t) + V_{ii} & \text{if } j = i \\
0 & \text{otherwise}
\end{cases}
$$

where $k_f(t)$ is the time-dependent kinetic factor and $V_{ii}$ is the diagonal potential energy at point $i$:

$$V_{ii} = p_f \cdot (p_i - \bar{p})^2 - \sum_{w \in \text{wells}} d_w \cdot e^{-\frac{(p_i-c_w)^2}{w_w}}$$

Explicitly, the matrix looks like:

$$
H =
\begin{pmatrix}
2k_f + V_{11} & -k_f & 0 & \cdots & 0 \\
-k_f & 2k_f + V_{22} & -k_f & \cdots & 0 \\
0 & -k_f & 2k_f + V_{33} & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & 2k_f + V_{NN}
\end{pmatrix}
$$

The implementation uses a standard finite difference approximation to solve this matix

### Solving the Eigenvalue Problem

The core of the strategy involves solving the time-independent Schrödinger equation for this Hamiltonian at each time step:

$$H \psi_n = E_n \psi_n$$

This is a standard eigenvalue problem for a real, symmetric (Hermitian) matrix $H$. We seek the eigenvalues $E_n$ (representing energy levels or support/resistance) and the corresponding eigenvectors $\psi_n$ (representing the probability distribution of price states).

Numerically, this is solved using standard linear algebra libraries, such as `scipy.linalg.eigh` in Python. This function efficiently computes the eigenvalues and eigenvectors of a Hermitian matrix. The function returns:

1.  **Eigenvalues ($E_n$)**: A sorted array of the energy levels. The strategy typically focuses on a subset of these, particularly the lowest and highest, to represent primary support and resistance.
2.  **Eigenvectors ($\psi_n$)**: A matrix where each column is an eigenvector corresponding to an eigenvalue. While not directly used for signal generation in this simplified strategy, they represent the "shape" of the market state at each energy level.

The calculated eigenvalues $E_n$ are then mapped back to the price domain and used for generating the $S_{\text{buy}}$ and $S_{\text{sell}}$ signals as described previously. The time evolution aspect is handled by recalculating this Hamiltonian and its eigenvalues at each time step, incorporating the evolving potential wells and kinetic factor.

## References

1. Quantum Mechanics, Claude Cohen-Tannoudji, Bernard Diu, Frank Laloë
2. Principles of Quantum Mechanics, R. Shankar
3. Path Integrals in Quantum Mechanics, Jean Zinn-Justin
4. Quantum Physics for Beginners, Z. Schechter
5. Quantum Trading, Fabio Oreste
6. Market Microstructure Theory, M. O'Hara
