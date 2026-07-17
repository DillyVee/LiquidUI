"""
Probabilistic Sharpe Ratio machinery used by the Deflated Sharpe Ratio
anti-overfit gate (optimization.validation).

Key properties:
1. Correct PSR variance scaling for annualized Sharpe
2. Effective sample size (trade count) support
3. Conservative fallback for negative variance
4. Adaptive z-score clipping based on sample size
"""

import numpy as np
from scipy import stats


class PSRCalculator:
    """
    Probabilistic Sharpe Ratio Calculator
    - Correct variance scaling for annualized Sharpe
    - Properly handles low trade counts
    - Conservative fallback for negative variance
    - Adaptive confidence clipping
    """

    @staticmethod
    def calculate_psr(
        returns: np.ndarray,
        benchmark_sharpe: float = 0.0,
        annualization_factor: float = 252.0,
        trade_count: int = None,
    ) -> float:
        """Calculate PSR with correct scaling and trade-count awareness"""
        if returns is None or len(returns) == 0:
            return 0.5

        returns = returns[~(np.isnan(returns) | np.isinf(returns))]
        if len(returns) < 30:
            return 0.5

        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        if std_ret == 0 or std_ret < 1e-12:
            return 0.95 if mean_ret > 0 else 0.05

        # per-period Sharpe (not annualized)
        s_p = mean_ret / std_ret

        # Annualized Sharpe for reporting and comparison
        observed_sharpe = s_p * np.sqrt(annualization_factor)
        observed_sharpe = np.clip(observed_sharpe, -10.0, 10.0)

        try:
            skew = stats.skew(returns, bias=False)
            # RAW kurtosis (normal = 3): the Lopez de Prado variance formula
            # below uses (kurt - 1)/4 with raw kurtosis, not Fisher excess
            kurt = stats.kurtosis(returns, bias=False, fisher=False)
            skew = np.clip(skew, -10.0, 10.0)
            kurt = np.clip(kurt, 1.0, 50.0)
        except Exception:
            skew = 0.0
            kurt = 3.0  # kurtosis of a normal distribution

        n = len(returns)
        if trade_count is not None and trade_count < n / 10:
            effective_n = max(int(trade_count), 10)
        else:
            effective_n = n

        try:
            # Corrected variance computation
            numerator = 1.0 - (skew * s_p) + (((kurt - 1.0) / 4.0) * (s_p**2))
            denom = max(effective_n - 1.0, 1.0)
            var_sp = numerator / denom
            variance_annual_sharpe = annualization_factor * var_sp

            if variance_annual_sharpe <= 0 or not np.isfinite(variance_annual_sharpe):
                base_se = 1.0 / np.sqrt(max(effective_n - 1.0, 1.0))
                skew_penalty = 1.0 + min(5.0, abs(skew)) * 0.3
                kurt_penalty = 1.0 + min(10.0, abs(kurt - 3.0)) * 0.15
                sharpe_std = base_se * skew_penalty * kurt_penalty * np.sqrt(annualization_factor)
            else:
                sharpe_std = np.sqrt(variance_annual_sharpe)

            if sharpe_std < 1e-10 or not np.isfinite(sharpe_std):
                sharpe_std = 0.2

            z_score = (observed_sharpe - benchmark_sharpe) / sharpe_std

            if effective_n < 30:
                max_z = 2.0
            elif effective_n < 100:
                max_z = 2.5
            else:
                max_z = 3.0

            z_score = np.clip(z_score, -max_z, max_z)
            psr = stats.norm.cdf(z_score)
            psr = np.clip(psr, 0.001, 0.999)

        except Exception:
            psr = 0.75 if observed_sharpe > benchmark_sharpe else 0.25

        return float(psr)

    @staticmethod
    def calculate_sharpe_from_equity(
        equity_curve: np.ndarray, annualization_factor: float = 252.0
    ) -> float:
        """Calculate annualized Sharpe ratio from equity curve"""
        if len(equity_curve) < 2:
            return 0.0

        returns = np.diff(equity_curve) / equity_curve[:-1]
        returns = returns[~(np.isnan(returns) | np.isinf(returns))]

        if len(returns) < 2:
            return 0.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        if std_ret == 0 or std_ret < 1e-10:
            return 0.0

        sharpe = mean_ret / std_ret * np.sqrt(annualization_factor)
        return np.clip(sharpe, -5, 10)
