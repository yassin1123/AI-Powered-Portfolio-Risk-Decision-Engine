from data.universe import get_tickers


def test_at_least_50_tickers_full():
    assert len(get_tickers("full")) >= 50


def test_core_universe_smaller_but_usable():
    assert len(get_tickers("core")) >= 40
    assert len(get_tickers("core")) < len(get_tickers("full"))
