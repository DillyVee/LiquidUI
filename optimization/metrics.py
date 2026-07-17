"""
Performance Metrics Calculation

`calculate_metrics` produces the full institutional metric suite from an
equity curve (and, when available, per-trade returns): total return, CAGR,
annualized Sharpe/Sortino/Calmar, volatility, max drawdown, Ulcer index,
VaR/CVaR, and trade-level statistics (win rate, profit factor, expectancy,
payoff ratio). Every optimization objective reads from this one dict so a
metric can never mean two different things in two places.
"""

from typing import Dict, Optional

import numpy as np

# Annualized ratios explode when the deviation denominator is nearly zero
# (e.g. one tiny losing bar); clip so ranking stays meaningful
_RATIO_CAP = 100.0
_PF_CAP = 1000.0


class PerformanceMetrics:
    """Calculate trading performance metrics"""

    @staticmethod
    def calculate_metrics(
        eq_curve: np.ndarray,
        annualization_factor: float = 252.0,
        trade_returns: Optional[np.ndarray] = None,
        exposure_frac: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Calculate comprehensive performance metrics.

        Args:
            eq_curve: Equity curve as numpy array (starts at 1000.0)
            annualization_factor: Periods per year for the equity curve's
                bar frequency (252 daily, 252*6.5 hourly, 252*6.5*12 5-min)
            trade_returns: Optional per-trade percent changes (e.g. [1.2,
                -0.4, ...] in %). Enables trade-level statistics and a
                trade-based profit factor.
            exposure_frac: Optional fraction of bars spent in the market
                (0..1, from the simulation). Adds "Exposure_%": a Sharpe
                earned while invested 8% of the time has a very different
                capacity, regime and risk profile than one earned at 80%,
                and per-bar risk metrics are diluted by flat bars.

        Returns:
            Dictionary of metrics or None if invalid
        """
        if eq_curve is None or len(eq_curve) == 0:
            return None

        final_equity = eq_curve[-1]
        if final_equity <= 0 or np.isnan(final_equity) or np.isinf(final_equity):
            return None

        # Percent gain
        initial_equity = 1000.0
        percent_gain = (final_equity - initial_equity) / initial_equity * 100

        # Returns calculation
        returns = np.diff(eq_curve) / eq_curve[:-1]
        valid_mask = ~(np.isnan(returns) | np.isinf(returns))
        returns = returns[valid_mask]

        if len(returns) == 0:
            return None

        years = len(eq_curve) / float(annualization_factor)
        if years > 0 and final_equity > 0:
            cagr = ((final_equity / initial_equity) ** (1.0 / years) - 1.0) * 100
        else:
            cagr = 0.0

        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
        volatility_ann = std_return * np.sqrt(annualization_factor) * 100

        # Sharpe ratio (annualized, rf = 0)
        sharpe = (
            (avg_return / std_return) * np.sqrt(annualization_factor)
            if std_return > 1e-12
            else 0.0
        )
        sharpe = float(np.clip(sharpe, -_RATIO_CAP, _RATIO_CAP))

        # Sortino ratio, annualized to the bar frequency. Downside deviation
        # is the target semideviation (root lower partial moment of order 2,
        # MAR = 0, averaged over ALL bars - Sortino & van der Meer 1991).
        # NOT the std of losing bars about their own mean: that rewards
        # loss UNIFORMITY (identical -1% losses -> std ~ 0 -> ratio
        # explodes), which an optimizer maximizing Sortino will exploit.
        downside_dev = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)))
        sortino = (
            (avg_return / downside_dev) * np.sqrt(annualization_factor)
            if downside_dev > 1e-10
            else 0.0
        )
        sortino = float(np.clip(sortino, -_RATIO_CAP, _RATIO_CAP))

        # Maximum drawdown
        peak = np.maximum.accumulate(eq_curve)
        drawdown = (eq_curve - peak) / peak
        max_drawdown = np.min(drawdown) * 100

        # Calmar ratio: CAGR over max drawdown magnitude
        if abs(max_drawdown) > 1e-9:
            calmar = cagr / abs(max_drawdown)
        else:
            calmar = cagr if cagr > 0 else 0.0
        calmar = float(np.clip(calmar, -_RATIO_CAP, _RATIO_CAP))

        # Ulcer index: RMS of the drawdown-percentage series (depth AND
        # duration of drawdowns, unlike max drawdown which is depth only)
        ulcer = float(np.sqrt(np.mean((drawdown * 100) ** 2)))

        # Historical VaR / CVaR at 95%, in percent per bar (losses positive)
        if len(returns) >= 20:
            var_95 = float(-np.percentile(returns, 5) * 100)
            tail = returns[returns <= np.percentile(returns, 5)]
            cvar_95 = float(-np.mean(tail) * 100) if len(tail) else var_95
        else:
            var_95 = 0.0
            cvar_95 = 0.0

        # Profit factor from per-bar equity changes (fallback when no trade
        # log is available; overridden by the trade-based version below)
        equity_changes = np.diff(eq_curve)
        gross_profit = np.sum(equity_changes[equity_changes > 0])
        gross_loss = np.abs(np.sum(equity_changes[equity_changes < 0]))
        if gross_loss > 1e-12:
            profit_factor = min(gross_profit / gross_loss, _PF_CAP)
        else:
            profit_factor = _PF_CAP if gross_profit > 0 else 1.0

        metrics = {
            "Final_Equity_$": round(final_equity, 2),
            "Percent_Gain_%": round(percent_gain, 2),
            "CAGR_%": round(cagr, 2),
            "Sharpe_Ratio": round(sharpe, 3),
            "Sortino_Ratio": round(sortino, 3),
            "Calmar_Ratio": round(calmar, 3),
            "Volatility_Ann_%": round(volatility_ann, 2),
            "Max_Drawdown_%": round(max_drawdown, 2),
            "Ulcer_Index": round(ulcer, 3),
            "VaR_95_%": round(var_95, 3),
            "CVaR_95_%": round(cvar_95, 3),
            "Profit_Factor": round(profit_factor, 3),
        }

        if exposure_frac is not None and np.isfinite(exposure_frac):
            metrics["Exposure_%"] = round(
                float(np.clip(exposure_frac, 0.0, 1.0)) * 100.0, 2
            )

        # Trade-level statistics (industry-standard definitions: computed
        # over closed trades, not bars)
        if trade_returns is not None:
            tr = np.asarray(trade_returns, dtype=np.float64)
            tr = tr[np.isfinite(tr)]
            if len(tr) > 0:
                wins = tr[tr > 0]
                losses = tr[tr < 0]
                win_rate = len(wins) / len(tr) * 100
                avg_win = float(np.mean(wins)) if len(wins) else 0.0
                avg_loss = float(np.mean(losses)) if len(losses) else 0.0
                gross_win = float(np.sum(wins))
                gross_lose = float(abs(np.sum(losses)))
                if gross_lose > 1e-12:
                    pf_trades = min(gross_win / gross_lose, _PF_CAP)
                else:
                    pf_trades = _PF_CAP if gross_win > 0 else 1.0
                payoff = (
                    min(avg_win / abs(avg_loss), _PF_CAP) if abs(avg_loss) > 1e-12 else 0.0
                )
                metrics.update(
                    {
                        "Win_Rate_%": round(win_rate, 2),
                        "Profit_Factor": round(pf_trades, 3),
                        "Avg_Win_%": round(avg_win, 3),
                        "Avg_Loss_%": round(avg_loss, 3),
                        "Payoff_Ratio": round(payoff, 3),
                        "Expectancy_%": round(float(np.mean(tr)), 4),
                        "Best_Trade_%": round(float(np.max(tr)), 2),
                        "Worst_Trade_%": round(float(np.min(tr)), 2),
                    }
                )

        return metrics

    @staticmethod
    def compute_rsi_vectorized(prices: np.ndarray, length: int) -> np.ndarray:
        """
        Vectorized RSI computation (3x faster than pandas rolling)

        Args:
            prices: Array of prices
            length: RSI period

        Returns:
            RSI values as numpy array
        """
        length = max(1, int(length))
        delta = np.diff(prices, prepend=prices[0])
        gain = np.maximum(delta, 0)
        loss = np.maximum(-delta, 0)

        # Use cumsum for faster rolling mean
        gain_sum = np.cumsum(gain)
        loss_sum = np.cumsum(loss)

        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)

        # Rolling window
        avg_gain[length - 1 :] = (
            gain_sum[length - 1 :] - np.concatenate([[0], gain_sum[:-length]])
        ) / length
        avg_loss[length - 1 :] = (
            loss_sum[length - 1 :] - np.concatenate([[0], loss_sum[:-length]])
        ) / length

        # First window
        avg_gain[:length] = gain_sum[:length] / np.arange(1, length + 1)
        avg_loss[:length] = loss_sum[:length] / np.arange(1, length + 1)

        # Calculate RSI
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        rsi = 100 - 100 / (1 + rs)

        # With zero average loss the divide above leaves rs=0, which would
        # invert the signal: no losses means RSI=100 (or 50 when flat)
        no_loss = avg_loss == 0
        rsi[no_loss & (avg_gain > 0)] = 100.0
        rsi[no_loss & (avg_gain == 0)] = 50.0

        return rsi

    @staticmethod
    def smooth_vectorized(arr: np.ndarray, length: int) -> np.ndarray:
        """
        Vectorized smoothing (faster than pandas rolling)

        Args:
            arr: Array to smooth
            length: Smoothing period

        Returns:
            Smoothed array
        """
        length = max(1, int(length))
        cumsum = np.cumsum(arr)
        result = np.zeros_like(arr)

        # Rolling window
        result[length - 1 :] = (
            cumsum[length - 1 :] - np.concatenate([[0], cumsum[:-length]])
        ) / length

        # First window
        result[:length] = cumsum[:length] / np.arange(1, length + 1)

        return result

    @staticmethod
    def cycle_anchor_index(datetimes: np.ndarray, timeframe: str) -> np.ndarray:
        """
        Calendar-anchored bar index for the time-cycle signal.

        Anchoring the cycle to time-since-epoch (instead of the bar's position
        in whatever data slice happens to be loaded) makes the cycle phase
        identical across full-history backtests, walk-forward windows, and
        live trading. Timestamps must be timezone-naive and use the same
        convention everywhere (exchange time for stocks, UTC for crypto).
        """
        dt = np.asarray(datetimes).astype("datetime64[ns]")
        if timeframe == "daily":
            return dt.astype("datetime64[D]").astype(np.int64)
        if timeframe == "hourly":
            return dt.astype("datetime64[h]").astype(np.int64)
        if timeframe == "5min":
            return dt.astype("datetime64[m]").astype(np.int64) // 5
        # 1min and any other intraday timeframe
        return dt.astype("datetime64[m]").astype(np.int64)

    @staticmethod
    def calculate_buyhold_return(prices: np.ndarray) -> float:
        """Calculate buy and hold return percentage"""
        if len(prices) < 2:
            return 0.0
        return (prices[-1] / prices[0] - 1) * 100

    @staticmethod
    def calculate_buyhold_metrics(
        prices: np.ndarray, annualization_factor: float = 252.0
    ) -> Optional[Dict[str, float]]:
        """
        Buy & hold benchmark metrics over a price series, on the same scale
        as calculate_metrics (equity normalized to a 1000.0 start).
        """
        prices = np.asarray(prices, dtype=np.float64)
        prices = prices[np.isfinite(prices)]
        if len(prices) < 2 or prices[0] <= 0:
            return None
        equity = 1000.0 * prices / prices[0]
        return PerformanceMetrics.calculate_metrics(
            equity, annualization_factor=annualization_factor
        )
