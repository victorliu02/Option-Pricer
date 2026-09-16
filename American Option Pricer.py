import numpy as np
from scipy.stats import norm
from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kwargs, te - ts))
        return result
    return wrap


def american_option_pricer(K, T, S0, r, N, u, d, opttype='P'):
    dt = T / N
    q = (np.exp(r * dt) - d) / (u - d)   # risk-neutral probability of up move
    disc = np.exp(-r * dt)               # discount factor per time step

    # stock prices at maturity (final layer of the tree)
    S = S0 * d ** (np.arange(N, -1, -1)) * u ** (np.arange(0, N + 1, 1))

    # option payoff at maturity
    if opttype == 'C':
        C = np.maximum(0, S - K)
    else:
        C = np.maximum(0, K - S)

    # step backward through the tree
    for i in np.arange(N - 1, -1, -1):
        S = S0 * d ** (np.arange(i, -1, -1)) * u ** (np.arange(0, i + 1, 1))
        C[:i + 1] = disc * (q * C[1:i + 2] + (1 - q) * C[0:i + 1]) # continuation value.

        """C[0] is the node with least value. For example at down, it inherits value from future C1 and C0,future C[1] is one more up move, and future c[0] is one more down move. 
        Current C[x] factors in value from future possible value"""

        C = C[:i + 1] #when i = 0, C[0]
        if opttype == 'C':
            C = np.maximum(C, S - K)   # early exercise check
        else:
            C = np.maximum(C, K - S)

    return C[0]


def black_scholes(S0, K, T, r, sigma, opttype='P'):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if opttype == 'C':
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


if __name__ == "__main__":
    # Parameters
    S0 = 100      # spot price
    K = 100       # strike
    T = 1.0       # years to expiry
    r = 0.05      # risk-free rate
    sigma = 0.20  # volatility
    N = 1000       # binomial steps

    u = np.exp(sigma * np.sqrt(T / N))  # up factor (CRR)
    d = 1 / u                            # down factor (recombining tree)

    print(f"{'':10}{'American':>12}{'European (BS)':>16}{'Premium':>12}")
    print("-" * 50)

    for opttype in ("C", "P"):
        label = "call" if opttype == "C" else "put"
        american_price = american_option_pricer(K, T, S0, r, N, u, d, opttype)
        european_price = black_scholes(S0, K, T, r, sigma, opttype)
        premium = american_price - european_price
        print(f"{label:10}{american_price:12.4f}{european_price:16.4f}{premium:12.4f}")