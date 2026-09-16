Option Pricing Models

Python implementations of option pricing models
The project was built to explore derivative pricing, risk-neutral valuation, stochastic modeling, and numerical methods used in quantitative finance

Black-Scholes — Closed-form pricing for European calls and puts.
Monte Carlo — Simulates terminal stock prices under the risk-neutral geometric Brownian motion framework and estimates option values.
Binomial Tree (CRR) — Prices American options using a recombining Cox-Ross-Rubinstein tree and accounts for early exercise.

Features

* European call and put pricing
* American call and put pricing
* Risk-neutral valuation
* Monte Carlo standard error and 95% confidence intervals
* Early-exercise logic for American options
* Comparison of numerical methods against Black-Scholes prices

The Monte Carlo model reports the estimated option price, standard error, and confidence interval, while the binomial model can be compared against the corresponding Black-Scholes price.



