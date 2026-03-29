import numpy as np

from reports.generator import basel_traffic_light


def test_basel_green():
    r = np.zeros(100)
    v = np.ones(100) * 0.1
    assert basel_traffic_light(r, v) == "GREEN"


def test_basel_red_many_breaches():
    n = 100
    r = np.full(n, -0.5)
    v = np.full(n, 0.01)
    assert basel_traffic_light(r, v) == "RED"
