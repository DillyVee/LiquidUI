"""Tests for the anti-overfitting validation toolkit"""

import numpy as np
import pytest

from optimization.psr_composite import PSRCalculator
from optimization.validation import (
    cscv_pbo,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    validate_candidate,
)

ANN = 252.0


def _sharpe(r):
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(ANN))


@pytest.fixture
def noise_trials():
    """50 pure-noise strategies: any 'edge' in the best one is luck.
    1500 bars keeps the Sharpe standard error (~0.41 annualized) small
    enough that the gates separate luck from edge decisively."""
    rng = np.random.default_rng(42)
    return [rng.normal(0.0, 0.01, 1500) for _ in range(50)]


def test_expected_max_sharpe_grows_with_trials():
    rng = np.random.default_rng(0)
    sharpes = list(rng.normal(0, 1.0, 1000))
    few = expected_max_sharpe(sharpes[:10])
    many = expected_max_sharpe(sharpes)
    assert 0 < few < many, "more trials must raise the luck benchmark"


def test_dsr_punishes_best_of_noise(noise_trials):
    """The luckiest of 50 noise strategies fools PSR but not DSR"""
    trial_sharpes = [_sharpe(r) for r in noise_trials]
    best = noise_trials[int(np.argmax(trial_sharpes))]

    psr = PSRCalculator.calculate_psr(best, 0.0, ANN)
    dsr = deflated_sharpe_ratio(best, trial_sharpes, ANN)

    assert psr > 0.85, f"plain PSR should be fooled by lucky noise (got {psr:.3f})"
    assert dsr < 0.75, f"DSR should not be fooled (got {dsr:.3f})"
    assert dsr < psr - 0.15, "deflation should bite hard on lucky noise"


def test_dsr_passes_genuine_edge(noise_trials):
    """A real edge clears the deflated benchmark despite many trials"""
    rng = np.random.default_rng(7)
    real = rng.normal(0.002, 0.01, 1500)  # ~3.2 annualized Sharpe
    trial_sharpes = [_sharpe(r) for r in noise_trials] + [_sharpe(real)]
    dsr = deflated_sharpe_ratio(real, trial_sharpes, ANN)
    assert dsr > 0.90, f"genuine edge should pass DSR (got {dsr:.3f})"


def test_cscv_pbo_high_for_noise():
    """With all-noise configs the in-sample winner is random OOS: PBO ~ 0.5"""
    rng = np.random.default_rng(3)
    matrix = rng.normal(0, 0.01, (800, 12))
    pbo = cscv_pbo(matrix, n_blocks=10)
    assert 0.3 <= pbo <= 0.7, f"noise PBO should hover near 0.5, got {pbo:.2f}"


def test_cscv_pbo_low_for_genuine_winner():
    """One config with a real edge keeps winning OOS: PBO small"""
    rng = np.random.default_rng(4)
    noise = rng.normal(0, 0.01, (800, 11))
    edge = rng.normal(0.0025, 0.01, (800, 1))  # strong, persistent edge
    matrix = np.hstack([edge, noise])
    pbo = cscv_pbo(matrix, n_blocks=10)
    assert pbo < 0.25, f"persistent edge should give low PBO, got {pbo:.2f}"


def test_cscv_pbo_input_validation():
    with pytest.raises(ValueError):
        cscv_pbo(np.zeros(10))  # 1-D
    with pytest.raises(ValueError):
        cscv_pbo(np.zeros((100, 1)))  # single config


def _make_simulate_fn(returns_by_key):
    """simulate_fn keyed on params['id'] -> equity curve from returns"""

    def simulate(params):
        r = returns_by_key[params["id"]]
        # Plenty of trades so the effective-sample-size correction doesn't
        # dominate (it is tested separately below)
        return 1000.0 * np.cumprod(1 + np.concatenate([[0.0], r])), 200

    return simulate


def test_validate_candidate_passes_edge_fails_noise(noise_trials):
    rng = np.random.default_rng(11)
    real = rng.normal(0.002, 0.01, 1500)
    trial_sharpes = [_sharpe(r) for r in noise_trials]
    best_noise = noise_trials[int(np.argmax(trial_sharpes))]

    returns_by_key = {"real": real, "lucky": best_noise}
    for i, r in enumerate(noise_trials[:8]):
        returns_by_key[f"rival{i}"] = r
    rivals = [{"id": f"rival{i}"} for i in range(8)]
    simulate = _make_simulate_fn(returns_by_key)

    good = validate_candidate(
        simulate, {"id": "real"}, rivals, trial_sharpes + [_sharpe(real)], ANN
    )
    assert good.passed, f"genuine edge should pass: {good.reasons}"
    assert good.pbo is not None and good.pbo < 0.5

    bad = validate_candidate(
        simulate, {"id": "lucky"}, rivals, trial_sharpes, ANN
    )
    assert not bad.passed, "lucky noise should fail the gates"
    assert any("DSR" in r or "PBO" in r for r in bad.reasons)


def test_validate_candidate_min_trades_and_oos_gates():
    rng = np.random.default_rng(2)
    r = rng.normal(0.002, 0.01, 400)

    def simulate(params):
        return 1000.0 * np.cumprod(1 + np.concatenate([[0.0], r])), 5  # 5 trades

    rep = validate_candidate(simulate, {}, [], [1.0], ANN, min_trades=20)
    assert not rep.passed and any("trades" in x for x in rep.reasons)

    def simulate2(params):
        return 1000.0 * np.cumprod(1 + np.concatenate([[0.0], r])), 60

    rep2 = validate_candidate(
        simulate2, {}, [], [1.0], ANN, oos_sharpe=-0.5, min_oos_sharpe=0.0
    )
    assert not rep2.passed and any("OOS" in x for x in rep2.reasons)

    report_text = rep2.summary()
    assert "FAILED" in report_text and "OOS" in report_text
