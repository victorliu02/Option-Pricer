"""
European Option Monte Carlo Pricer and Black Scholes Check
"""

import numpy as np
from scipy.stats import norm



# Closed-form Black-Scholes

def black_scholes(S0, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return price


# Monte Carlo

def mc_european_option(
    S0, K, T, r, sigma,
    n_sims=1_000_000,
    option_type="call",
    seed=None,   #argyuments 
):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_sims)  #generate Z's

    # Terminal price under risk-neutral geometric brownian motion
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)  
    # taking the stock's fixed volatility, scaling it for the time horizon (np.sqrt(T)), and then multiplying by that particular simulation's random z-score (Z) to get the actual random dollar-magnitude swing.
    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type invalid, must be 'call' or 'put'")

    discounted = np.exp(-r * T) * payoff

    price = discounted.mean()
    stderr = discounted.std(ddof=1) / np.sqrt(len(discounted))
    ci95 = (price - 1.96 * stderr, price + 1.96 * stderr)

    return price, stderr, ci95

if __name__ == "__main__":
    # Example parameters
    S0 = 100      # spot price
    K = 105       # strike
    T = 1.0       # years to expiry
    r = 0.05      # risk-free rate
    sigma = 0.20  # volatility
    n_sims = 1_000_000

    print(f"{'':10}{'MC Price':>12}{'Std Err':>12}{'95% CI':>26}{'BS Price':>12}")
    print("-" * 72)

    for option_type in ("call", "put"):
        price, se, ci = mc_european_option(
            S0, K, T, r, sigma,
            n_sims=n_sims,
            option_type=option_type,
            seed=42,
        )
        bs_price = black_scholes(S0, K, T, r, sigma, option_type)
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        print(f"{option_type:10}{price:12.4f}{se:12.4f}{ci_str:>26}{bs_price:12.4f}")