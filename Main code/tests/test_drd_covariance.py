import numpy as np

from risk.garch import conditional_covariance_drd, dcc_R_from_epsilon


def test_sigma_drd_matches_formula():
    n = 4
    rng = np.random.default_rng(0)
    eps = rng.standard_normal((200, n))
    r = dcc_R_from_epsilon(eps)
    d = np.abs(rng.standard_normal(n)) * 0.01 + 0.005
    s = conditional_covariance_drd(d, r)
    dmat = np.diag(d)
    np.testing.assert_allclose(s, dmat @ r @ dmat, rtol=1e-5, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(s) >= -1e-8)
