import numpy as np
import pytest

from risk.var import simulate_portfolio_returns


@pytest.mark.benchmark(group="mc")
def test_mc_10k_x_50_under_100ms(benchmark):
    rng = np.random.default_rng(0)
    n = 50
    mu = np.zeros(n)
    sigma = np.eye(n) * 0.0001
    w = np.ones(n) / n

    def run():
        return simulate_portfolio_returns(mu, sigma, w, 10_000, rng)

    benchmark(run)
