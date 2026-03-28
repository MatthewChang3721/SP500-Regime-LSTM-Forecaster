# SP500 Market Regime Detection (Phase 1: GMM)

## 📖 Project Overview
This repository represents the foundational phase of a larger hybrid quantitative trading model (GMM + LSTM). The current focus is on **Unsupervised Market Regime Detection**. 

Financial markets operate in different latent states or "regimes" (e.g., low-volatility bull markets, high-volatility bear markets). By clustering historical S&P 500 data based on statistical features like log returns and rolling volatility, we can mathematically identify these underlying regimes. 

**Key Technical Highlight:** Instead of relying on standard libraries like `scikit-learn`, the core Gaussian Mixture Model (GMM) is implemented completely from scratch using **TensorFlow**. This ensures high-performance tensor computations and demonstrates a deep understanding of the underlying Expectation-Maximization (EM) algorithm.

## 📂 Repository Structure

The project is modularized into four key components:

### 1. `Data.ipynb` (Data Pipeline & Feature Engineering)
* **Purpose:** Handles the ETL (Extract, Transform, Load) process.
* **Details:** Fetches historical S&P 500 data via `yfinance`. Cleans the raw price data and engineers critical financial features essential for regime detection, primarily focusing on **Log Returns** and **Annualized Rolling Volatility**.

### 2. `GMM.py` (The Core Engine)
* **Purpose:** Custom Machine Learning implementation.
* **Details:** A pure **TensorFlow** implementation of the Gaussian Mixture Model. It builds the computational graph for calculating multivariate Gaussian probabilities and iteratively optimizes the parameters (means, covariances, and mixture weights) using the EM algorithm. 

### 3. `GMM_demo.ipynb` (Algorithm Demonstration)
* **Purpose:** Proof of Concept & Unit Testing.
* **Details:** A sandbox environment designed to demonstrate the mathematical principles behind GMM. It tests the custom `GMM.py` function on generated synthetic data (e.g., clear 2D Gaussian blobs) to verify convergence and accuracy before applying it to noisy financial data.

### 4. `GMM.ipynb` (Market Classifier & Optimization)
* **Purpose:** Model Training & Hyperparameter Tuning.
* **Details:** The primary analysis notebook. It imports the engineered S&P 500 data and applies the custom TensorFlow GMM. A key focus of this notebook is determining the **Optimal K** (number of market regimes) using evaluation metrics, and ultimately classifying historical market data into distinct states.
**Regime Profiling:** To ensure model stability and eliminate the Unsupervised Learning "Label Switching" problem, the identified regimes are strictly anchored and sorted by the 
**VIX index (from lowest to highest)**. The model successfully isolates four distinct macroeconomic environments:
    * **Regime 1 (Low Volatility Bull Market):** Characterized by extremely stable market sentiment, suppressed volatility, and steady growth.
    * **Regime 2 (Upward Oscillation / Slow Bull):** The most frequent market state, featuring normal market "breathing" and a gradual upward trend.
    * **Regime 3 (Downward Oscillation / Inflection Point):** Marked by rising VIX and widening credit spreads, indicating market divergence and potential trend reversals.
    * **Regime 4 (Extreme Crisis / Panic):** Precisely captures historical market crashes and liquidity crises (e.g., the 2008 GFC and the 2020 COVID-19 crash) characterized by exploding VIX and aggressive sell-offs.

# 🚀 LSTM Expert Training & Regime Integration (Phase 2)
## Overview
In this stage, the market regime labels (latent states) identified by the GMM pipeline are integrated as categorical features into a specialized LSTM Neural Network. The objective is to enable the model to switch its "predictive logic" based on the detected macro environment (e.g., high-volatility crash vs. low-volatility growth). While the system is fully operational, the high noise-to-signal ratio of the S&P 500 presents significant challenges in convergence and stability.

## 🛠 Technical Highlights
### 1. Class Weight Equilibrium (Addressing Bullish Bias)
The S&P 500 exhibits a natural long-term upward drift. Without intervention, models often collapse into a "perpetual bull" state, achieving high accuracy by simply guessing "UP" every day.

The Solution: We identified a critical equilibrium point using class_weight at {0: 1.05, 1: 0.95}.

Impact: This subtle 5% shift increases the penalty for missing a downturn, forcing the LSTM to actively look for bearish signals rather than relying on the market's natural drift.

### 2. The "Golden Moment" Early Stopping
Financial time series are non-stationary and prone to rapid overfitting. We observed that the model captures genuine Alpha signals early, before it begins "memorizing" specific historical noise.

Logic: We implement EarlyStopping with restore_best_weights=True.

The Strategy: We intercept the "Golden Moment" between Epoch 10 and 25. Training beyond this window typically leads to a divergence where training loss continues to fall while validation loss surges, indicating a loss of generalizability.

### 3. Systematic Grid Search
To move beyond manual heuristics, we utilized a Grid Search approach to optimize the window_size. This ensures that the look-back period is not just an arbitrary choice but a statistically backed hyperparameter.

Optimal Window: The search identified 10 days as the peak performance window for the current feature set.

## 📊 Performance Results
Peak Test Accuracy: 53.72%

---
*Built with Python, TensorFlow, and Financial Intuition.*