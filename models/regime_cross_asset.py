"""
Cross-Asset Regime Analysis
Global Risk-On/Risk-Off Detection

Features:
1. Multi-asset regime detection
2. Cross-asset correlation analysis
3. Global risk regime identification
4. Regime synchronization metrics
5. Asset class divergence detection
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import numpy as np
import pandas as pd

from models.regime_detection import MarketRegime, MarketRegimeDetector, RegimeState


class GlobalRegime(Enum):
    """Global market regimes across all assets"""

    RISK_ON = "risk_on"  # Equities up, bonds down, vol low
    RISK_OFF = "risk_off"  # Equities down, bonds up, vol high
    MIXED = "mixed"  # Divergent signals across assets
    CRISIS = "crisis"  # All assets stressed
    RECOVERY = "recovery"  # Emerging from crisis


@dataclass
class AssetRegimeState:
    """Regime state for specific asset"""

    asset_name: str
    asset_class: str  # 'equity', 'bond', 'commodity', 'crypto'
    regime_state: RegimeState
    correlation_to_spy: float  # Correlation to equity benchmark


@dataclass
class GlobalRegimeAnalysis:
    """Global regime across all asset classes"""

    global_regime: GlobalRegime
    confidence: float  # 0-1
    asset_regimes: Dict[str, AssetRegimeState]
    regime_synchronization: float  # 0-1, how aligned are regimes?
    divergence_pairs: List[tuple]  # Assets with divergent regimes
    risk_on_score: float  # -1 to +1, negative=risk-off, positive=risk-on
    interpretation: str


class CrossAssetRegimeAnalyzer:
    """
    Analyzes regimes across multiple asset classes

    Key Insight: True regime shifts show up across assets
    - Risk-on: Equities up, bonds down, commodities up, VIX down
    - Risk-off: Equities down, bonds up, commodities down, VIX up
    - Crisis: Everything down except safe havens (gold, treasuries)
    """

    def __init__(self):
        """Initialize cross-asset analyzer"""
        self.detector = MarketRegimeDetector()
        self.asset_data = {}

    def add_asset(
        self,
        asset_name: str,
        asset_class: str,
        prices: pd.Series,
        returns: pd.Series,
    ):
        """
        Add asset to analysis

        Args:
            asset_name: Asset identifier (e.g., 'SPY', 'TLT', 'GLD')
            asset_class: Asset class ('equity', 'bond', 'commodity', 'crypto')
            prices: Price series
            returns: Return series
        """
        self.asset_data[asset_name] = {
            "asset_class": asset_class,
            "prices": prices,
            "returns": returns,
        }

    def analyze_global_regime(
        self, spy_returns: pd.Series = None
    ) -> GlobalRegimeAnalysis:
        """
        Analyze global regime across all assets

        Args:
            spy_returns: SPY returns for correlation calculation (optional)

        Returns:
            GlobalRegimeAnalysis with cross-asset insights
        """
        if len(self.asset_data) < 2:
            raise ValueError("Need at least 2 assets for cross-asset analysis")

        print(f"\n{'='*70}")
        print(f"CROSS-ASSET REGIME ANALYSIS")
        print(f"{'='*70}")
        print(f"Analyzing {len(self.asset_data)} assets...")

        # 1. Detect regime for each asset
        asset_regimes = {}

        for asset_name, data in self.asset_data.items():
            regime_state = self.detector.detect_regime(data["prices"], data["returns"])

            # Calculate correlation to SPY if provided
            if spy_returns is not None and len(data["returns"]) == len(spy_returns):
                correlation = data["returns"].corr(spy_returns)
            else:
                correlation = 0.0

            asset_regimes[asset_name] = AssetRegimeState(
                asset_name=asset_name,
                asset_class=data["asset_class"],
                regime_state=regime_state,
                correlation_to_spy=correlation,
            )

            print(
                f"   {asset_name:8s} ({data['asset_class']:10s}): "
                f"{regime_state.current_regime.value:10s} "
                f"(conf: {regime_state.confidence:.1%})"
            )

        # 2. Calculate risk-on/risk-off score
        risk_on_score = self._calculate_risk_score(asset_regimes)

        # 3. Determine global regime
        global_regime, confidence = self._determine_global_regime(
            asset_regimes, risk_on_score
        )

        # 4. Calculate synchronization
        synchronization = self._calculate_synchronization(asset_regimes)

        # 5. Identify divergences
        divergence_pairs = self._identify_divergences(asset_regimes)

        # 6. Generate interpretation
        interpretation = self._interpret_global_regime(
            global_regime, risk_on_score, synchronization, divergence_pairs
        )

        print(f"\n🌍 GLOBAL REGIME: {global_regime.value.upper()}")
        print(f"   Confidence: {confidence:.1%}")
        print(
            f"   Risk-On Score: {risk_on_score:+.2f} "
            f"({'Risk-On' if risk_on_score > 0 else 'Risk-Off'})"
        )
        print(f"   Synchronization: {synchronization:.1%}")

        if divergence_pairs:
            print(f"\n⚠️  DIVERGENCES DETECTED:")
            for asset1, asset2 in divergence_pairs[:3]:  # Top 3
                print(f"   • {asset1} vs {asset2}")

        print(f"\n{'='*70}\n")

        return GlobalRegimeAnalysis(
            global_regime=global_regime,
            confidence=confidence,
            asset_regimes=asset_regimes,
            regime_synchronization=synchronization,
            divergence_pairs=divergence_pairs,
            risk_on_score=risk_on_score,
            interpretation=interpretation,
        )

    def _calculate_risk_score(
        self, asset_regimes: Dict[str, AssetRegimeState]
    ) -> float:
        """
        Calculate risk-on/risk-off score

        Risk-on indicators:
        - Equities in BULL (+1)
        - Bonds in BEAR (+0.5) [bonds down when equities up]
        - Commodities in BULL (+0.5)
        - Crypto in BULL (+0.3)

        Risk-off indicators:
        - Equities in BEAR (-1)
        - Bonds in BULL (-0.5) [bonds up when equities down]
        - Crisis regimes (-2)

        Args:
            asset_regimes: Dict of asset regime states

        Returns:
            Risk score (-3 to +3, positive=risk-on, negative=risk-off)
        """
        score = 0.0

        for asset_name, asset_state in asset_regimes.items():
            regime = asset_state.regime_state.current_regime
            asset_class = asset_state.asset_class
            confidence = asset_state.regime_state.confidence

            # Weight by confidence
            weight = confidence

            if asset_class == "equity":
                if regime == MarketRegime.BULL:
                    score += 1.0 * weight
                elif regime == MarketRegime.BEAR:
                    score -= 1.0 * weight
                elif regime == MarketRegime.CRISIS:
                    score -= 2.0 * weight

            elif asset_class == "bond":
                # Inverse relationship: bonds up = risk-off
                if regime == MarketRegime.BULL:
                    score -= 0.5 * weight
                elif regime == MarketRegime.BEAR:
                    score += 0.5 * weight

            elif asset_class == "commodity":
                if regime == MarketRegime.BULL:
                    score += 0.5 * weight
                elif regime == MarketRegime.BEAR:
                    score -= 0.3 * weight

            elif asset_class == "crypto":
                if regime == MarketRegime.BULL:
                    score += 0.3 * weight
                elif regime == MarketRegime.BEAR:
                    score -= 0.3 * weight

        # Normalize to -3 to +3 range
        return np.clip(score, -3.0, 3.0)

    def _determine_global_regime(
        self, asset_regimes: Dict[str, AssetRegimeState], risk_score: float
    ) -> tuple:
        """
        Determine global regime from asset regimes

        Args:
            asset_regimes: Asset regime states
            risk_score: Risk-on/risk-off score

        Returns:
            (GlobalRegime, confidence)
        """
        # Count crisis regimes
        crisis_count = sum(
            1
            for state in asset_regimes.values()
            if state.regime_state.current_regime == MarketRegime.CRISIS
        )

        # Average confidence
        avg_confidence = np.mean(
            [state.regime_state.confidence for state in asset_regimes.values()]
        )

        # Decision logic
        if crisis_count >= len(asset_regimes) * 0.5:
            # Half or more in crisis
            return GlobalRegime.CRISIS, avg_confidence

        elif risk_score > 1.0:
            # Strong risk-on
            return GlobalRegime.RISK_ON, avg_confidence

        elif risk_score < -1.0:
            # Strong risk-off
            return GlobalRegime.RISK_OFF, avg_confidence

        elif risk_score > 0:
            # Mild risk-on (possibly recovery)
            if crisis_count > 0:
                return GlobalRegime.RECOVERY, avg_confidence * 0.8
            else:
                return GlobalRegime.RISK_ON, avg_confidence * 0.7

        else:
            # Mixed or uncertain
            return GlobalRegime.MIXED, avg_confidence * 0.6

    def _calculate_synchronization(
        self, asset_regimes: Dict[str, AssetRegimeState]
    ) -> float:
        """
        Calculate how synchronized regimes are across assets

        High synchronization = assets agree on regime
        Low synchronization = divergent signals

        Args:
            asset_regimes: Asset regime states

        Returns:
            Synchronization score (0-1)
        """
        # Count most common regime
        regime_counts = {}
        for state in asset_regimes.values():
            regime = state.regime_state.current_regime
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        max_count = max(regime_counts.values())
        total_assets = len(asset_regimes)

        synchronization = max_count / total_assets

        return synchronization

    def _identify_divergences(
        self, asset_regimes: Dict[str, AssetRegimeState]
    ) -> List[tuple]:
        """
        Identify asset pairs with divergent regimes

        Divergence = assets in opposite regimes
        E.g., one in BULL, another in BEAR

        Args:
            asset_regimes: Asset regime states

        Returns:
            List of (asset1, asset2) tuples with divergent regimes
        """
        divergences = []

        asset_list = list(asset_regimes.items())

        for i, (name1, state1) in enumerate(asset_list):
            for name2, state2 in asset_list[i + 1 :]:
                regime1 = state1.regime_state.current_regime
                regime2 = state2.regime_state.current_regime

                # Check for opposite regimes
                is_divergent = (
                    regime1 == MarketRegime.BULL and regime2 == MarketRegime.BEAR
                ) or (regime1 == MarketRegime.BEAR and regime2 == MarketRegime.BULL)

                if is_divergent:
                    divergences.append((name1, name2))

        return divergences

    def _interpret_global_regime(
        self,
        global_regime: GlobalRegime,
        risk_score: float,
        synchronization: float,
        divergences: List[tuple],
    ) -> str:
        """
        Generate interpretation of global regime

        Args:
            global_regime: Detected global regime
            risk_score: Risk-on/risk-off score
            synchronization: Synchronization metric
            divergences: Divergent asset pairs

        Returns:
            Interpretation string
        """
        if global_regime == GlobalRegime.CRISIS:
            return (
                "🚨 GLOBAL CRISIS: Multiple asset classes in crisis regime. "
                "Extreme risk-off environment. Preserve capital."
            )

        elif global_regime == GlobalRegime.RISK_ON:
            if synchronization > 0.7:
                return (
                    "✅ STRONG RISK-ON: All asset classes aligned in bullish/growth regime. "
                    "High confidence environment for risk assets."
                )
            else:
                return (
                    "⚠️  MODERATE RISK-ON: Generally positive but some divergences. "
                    "Monitor for regime shift signals."
                )

        elif global_regime == GlobalRegime.RISK_OFF:
            if synchronization > 0.7:
                return (
                    "❌ STRONG RISK-OFF: Widespread defensive positioning across assets. "
                    "Reduce risk exposure significantly."
                )
            else:
                return (
                    "⚠️  MODERATE RISK-OFF: Generally defensive but not uniform. "
                    "Selective opportunities may exist."
                )

        elif global_regime == GlobalRegime.RECOVERY:
            return (
                "📈 RECOVERY: Emerging from crisis/risk-off. "
                "Early stage risk-on signals. Gradual re-entry opportunity."
            )

        else:  # MIXED
            return (
                "🔀 MIXED SIGNALS: Asset classes showing divergent regimes. "
                f"{len(divergences)} divergence(s) detected. "
                "Exercise caution until clarity emerges."
            )

    def generate_cross_asset_report(self, analysis: GlobalRegimeAnalysis) -> str:
        """
        Generate formatted cross-asset report

        Args:
            analysis: GlobalRegimeAnalysis result

        Returns:
            Formatted report
        """
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           CROSS-ASSET REGIME ANALYSIS                        ║
╚══════════════════════════════════════════════════════════════╝

🌍 GLOBAL REGIME: {analysis.global_regime.value.upper()}
   Confidence: {analysis.confidence:.1%}
   Risk-On Score: {analysis.risk_on_score:+.2f}
   Synchronization: {analysis.regime_synchronization:.1%}

📊 ASSET BREAKDOWN:
"""

        # Group by asset class
        by_class = {}
        for name, state in analysis.asset_regimes.items():
            asset_class = state.asset_class
            if asset_class not in by_class:
                by_class[asset_class] = []
            by_class[asset_class].append((name, state))

        for asset_class, assets in sorted(by_class.items()):
            report += f"\n   {asset_class.upper()}:\n"
            for name, state in assets:
                regime = state.regime_state.current_regime.value
                confidence = state.regime_state.confidence
                corr = state.correlation_to_spy

                report += f"      • {name:8s}: {regime:10s} "
                report += f"(conf: {confidence:.1%}, corr: {corr:+.2f})\n"

        if analysis.divergence_pairs:
            report += f"\n⚠️  DIVERGENCES:\n"
            for asset1, asset2 in analysis.divergence_pairs:
                regime1 = analysis.asset_regimes[
                    asset1
                ].regime_state.current_regime.value
                regime2 = analysis.asset_regimes[
                    asset2
                ].regime_state.current_regime.value
                report += f"   • {asset1} ({regime1}) ↔ {asset2} ({regime2})\n"

        report += f"\n💡 INTERPRETATION:\n   {analysis.interpretation}\n"

        report += f"\n{'═'*62}\n"

        return report


def load_multi_asset_data(tickers: Dict[str, str]) -> Dict:
    """
    Load data for multiple assets

    Args:
        tickers: Dict mapping ticker to asset_class
                Example: {'SPY': 'equity', 'TLT': 'bond', 'GLD': 'commodity'}

    Returns:
        Dict with loaded data
    """
    from data import DataLoader

    loaded_data = {}

    print(f"Loading {len(tickers)} assets...")

    for ticker, asset_class in tickers.items():
        try:
            df_dict, error = DataLoader.load_yfinance_data(ticker)

            if error or not df_dict:
                print(f"   ⚠️  Failed to load {ticker}: {error}")
                continue

            # Use daily data
            df = df_dict.get("daily")
            if df is None or len(df) < 100:
                print(f"   ⚠️  Insufficient data for {ticker}")
                continue

            prices = df["Close"]
            returns = prices.pct_change().dropna()

            loaded_data[ticker] = {
                "asset_class": asset_class,
                "prices": prices,
                "returns": returns,
            }

            print(f"   ✅ {ticker} loaded: {len(prices)} days")

        except Exception as e:
            print(f"   ❌ Error loading {ticker}: {e}")

    return loaded_data
