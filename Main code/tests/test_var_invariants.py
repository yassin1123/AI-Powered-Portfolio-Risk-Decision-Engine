import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from risk.var import historical_var_cvar, simulate_portfolio_returns


@given(
    st.lists(
        st.floats(min_value=-0.05, max_value=0.05, allow_nan=False),
        min_size=30,
        max_size=200,
    )
)
@settings(max_examples=30)
def test_var_monotonicity(returns):
    x = np.array(returns, dtype=float)
    v95, _ = historical_var_cvar(x, 0.95)
    v99, _ = historical_var_cvar(x, 0.99)
    assert v99 >= v95 - 1e-9


@given(
    st.lists(
        st.floats(min_value=-0.05, max_value=0.05, allow_nan=False),
        min_size=30,
        max_size=200,
    )
)
@settings(max_examples=30)
def test_cvar_ge_var(returns):
    x = np.array(returns, dtype=float)
    v, c = historical_var_cvar(x, 0.95)
    assert c + 1e-9 >= v


def test_mc_simulation_shape():
    rng = np.random.default_rng(42)
    n = 5
    mu = np.zeros(n)
    sigma = np.eye(n) * 0.0001
    w = np.ones(n) / n
    p = simulate_portfolio_returns(mu, sigma, w, 1000, rng)
    assert len(p) == 1000
