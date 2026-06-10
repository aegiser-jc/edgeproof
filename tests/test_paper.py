from edgeproof.paper import PaperPortfolio


def test_rebalance_to_full_long_no_cost():
    pf = PaperPortfolio(cost_rate=0.0, cash=10_000, initial_equity=10_000)
    tr = pf.rebalance_to(1.0, price=100.0, when="t0")
    assert tr is not None and tr["side"] == "buy"
    assert pf.position_units == 100.0
    assert abs(pf.cash) < 1e-9
    assert abs(pf.equity(100.0) - 10_000) < 1e-9
    assert abs(pf.position_fraction(100.0) - 1.0) < 1e-9


def test_costs_are_charged_on_rebalance():
    pf = PaperPortfolio(cost_rate=0.001, cash=10_000, initial_equity=10_000)
    tr = pf.rebalance_to(1.0, price=100.0, when="t0")
    assert abs(tr["cost"] - 10.0) < 1e-9          # 10000 notional * 0.001
    assert pf.equity(100.0) < 10_000              # cost eats into equity


def test_tiny_rebalance_is_skipped():
    pf = PaperPortfolio(cost_rate=0.0, cash=10_000, initial_equity=10_000)
    pf.rebalance_to(1.0, 100.0, "t0")
    none = pf.rebalance_to(1.0005, 100.0, "t1")   # delta well under min trade frac
    assert none is None
    assert len(pf.trades) == 1


def test_rebalance_to_flat_closes_position():
    pf = PaperPortfolio(cost_rate=0.0, cash=10_000, initial_equity=10_000)
    pf.rebalance_to(1.0, 100.0, "t0")
    pf.rebalance_to(0.0, 110.0, "t1")             # exit after a +10% move
    assert abs(pf.position_units) < 1e-9
    assert pf.equity(110.0) > 10_000              # profit realised to cash


def test_short_position_and_roundtrip():
    pf = PaperPortfolio(cost_rate=0.0, cash=10_000, initial_equity=10_000)
    pf.rebalance_to(-1.0, 100.0, "t0")
    assert pf.position_units < 0
    d = pf.to_dict()
    pf2 = PaperPortfolio.from_dict(d)
    assert pf2.position_units == pf.position_units
    assert pf2.cash == pf.cash
