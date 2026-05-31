"""
Testy poprawionej formuły Deflated Sharpe Ratio (audyt 3.3).

Uruchom: python trading_system/agents/test_validation.py

Dowodzimy:
- SR w członie korekcyjnym jest PER-OKRES (nie zannualizowany),
- kurtoza jest SUROWA (pandas.kurt() + 3),
- SE(SR) zgadza się z ręcznym wzorem Mertens/Lo,
- więcej prób (n_trials) => wyższy benchmark => niższy dsr_stat => wyższe p-value,
- dobra strategia istotna, losowa o zerowej średniej nieistotna.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validation_agent import deflated_sharpe_ratio


def _se_manual(r, sr_periodic):
    T = len(r)
    skew = float(pd.Series(r).skew())
    kurt_raw = float(pd.Series(r).kurt()) + 3.0
    inner = (1 - skew * sr_periodic + (kurt_raw - 1) / 4 * sr_periodic**2) / (T - 1)
    return np.sqrt(inner)


def test_periodic_sr_and_raw_kurtosis():
    rng = np.random.default_rng(1)
    r = rng.normal(0.3, 1.4, 500)
    d = deflated_sharpe_ratio(r, n_trials=50, periods_per_year=252)

    sr_p_manual = r.mean() / r.std(ddof=1)
    np.testing.assert_allclose(d["sr_periodic"], round(sr_p_manual, 5), atol=1e-5)
    # annualized = per-okres × √252
    np.testing.assert_allclose(d["sr"], round(sr_p_manual * np.sqrt(252), 3), atol=1e-3)
    # kurtoza surowa ≈ 3 dla normalnych
    assert 2.3 < d["kurt"] < 3.7, f"kurt_raw poza zakresem normalności: {d['kurt']}"


def test_correction_uses_periodic_not_annualized():
    """dsr_stat liczy się z SE opartym na SR per-okres, nie na SR×√252."""
    rng = np.random.default_rng(2)
    r = rng.normal(0.5, 1.0, 600)   # wyraźny SR, by różnica SE była jednoznaczna
    d = deflated_sharpe_ratio(r, n_trials=100)

    sr_p = r.mean() / r.std(ddof=1)
    se_expected = _se_manual(r, sr_p)            # poprawne: per-okres
    se_wrong = _se_manual(r, sr_p * np.sqrt(252))  # błędne: annualizowane

    # Odtwórz dsr_stat z benchmarkiem zwróconym przez funkcję:
    dsr_recomputed = (sr_p - d["sr_benchmark"]) / se_expected
    np.testing.assert_allclose(d["dsr_stat"], round(dsr_recomputed, 3), atol=1e-3)
    # SE poprawne i błędne muszą się znacząco różnić (dowód, że to ma znaczenie):
    # człon sr² z SR annualizowanym (×√252) zawyża SE ~2.5× tutaj.
    assert se_wrong > se_expected * 2, "annualizacja powinna zawyżać SE — test bez sensu inaczej"


def test_more_trials_lowers_significance():
    rng = np.random.default_rng(3)
    r = rng.normal(0.2, 1.0, 800)
    d_few = deflated_sharpe_ratio(r, n_trials=2)
    d_many = deflated_sharpe_ratio(r, n_trials=1000)
    assert d_many["sr_benchmark"] > d_few["sr_benchmark"], "więcej prób => wyższy benchmark"
    assert d_many["dsr_stat"] < d_few["dsr_stat"], "więcej prób => niższy dsr_stat"
    assert d_many["pvalue"] >= d_few["pvalue"], "więcej prób => wyższe p-value"


def test_good_strategy_significant_random_not():
    rng = np.random.default_rng(4)
    good = rng.normal(0.4, 1.0, 1000)   # wyraźny edge
    noise = rng.normal(0.0, 1.0, 1000)  # zerowa średnia
    dg = deflated_sharpe_ratio(good, n_trials=50)
    dn = deflated_sharpe_ratio(noise, n_trials=50)
    assert dg["is_significant"], f"dobra strategia powinna być istotna: p={dg['pvalue']}"
    assert not dn["is_significant"], f"szum nie powinien być istotny: p={dn['pvalue']}"


def test_short_sample_guarded():
    d = deflated_sharpe_ratio(np.array([0.1, 0.2, -0.1]), n_trials=10)
    assert d["is_significant"] is False
    assert d["pvalue"] == 1.0


if __name__ == "__main__":
    tests = [test_periodic_sr_and_raw_kurtosis,
             test_correction_uses_periodic_not_annualized,
             test_more_trials_lowers_significance,
             test_good_strategy_significant_random_not,
             test_short_sample_guarded]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  ❌ {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} testów przeszło")
    sys.exit(1 if failed else 0)
