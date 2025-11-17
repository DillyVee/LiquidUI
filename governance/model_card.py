"""
Model Card Generator - ML/AI Governance
Documents model intent, limitations, performance, and ethical considerations
Following Google's Model Card framework
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from infrastructure.logger import quant_logger

logger = quant_logger.get_logger("governance")


@dataclass
class ModelDetails:
    """Basic information about the model"""

    name: str
    version: str
    description: str
    model_type: str  # 'quant_strategy', 'ml_predictor', 'risk_model', etc.
    developed_by: str
    developed_date: str
    license: str = "Proprietary"
    contact: str = ""
    references: List[str] = field(default_factory=list)


@dataclass
class IntendedUse:
    """Intended use cases and users"""

    primary_uses: List[str]
    primary_users: List[str]
    out_of_scope_uses: List[str]


@dataclass
class Factors:
    """Factors that affect model performance"""

    relevant_factors: List[str]  # Market regime, volatility, liquidity, etc.
    evaluation_factors: List[str]


@dataclass
class Metrics:
    """Performance metrics"""

    model_performance_measures: Dict[str, float]
    decision_thresholds: Dict[str, float]
    variation_approaches: List[str]  # How performance varies across factors


@dataclass
class TrainingData:
    """Information about training data"""

    dataset_description: str
    start_date: str
    end_date: str
    data_sources: List[str]
    preprocessing: List[str]
    known_issues: List[str] = field(default_factory=list)


@dataclass
class QuantitativeAnalyses:
    """Quantitative analyses and results"""

    performance_by_regime: Optional[Dict[str, Any]] = None
    robustness_tests: Optional[Dict[str, Any]] = None
    capacity_analysis: Optional[Dict[str, Any]] = None
    transaction_costs: Optional[Dict[str, Any]] = None


@dataclass
class EthicalConsiderations:
    """Ethical considerations and risks"""

    sensitive_data: List[str]
    risks_and_harms: List[str]
    use_cases_to_avoid: List[str]
    mitigation_strategies: List[str] = field(default_factory=list)


@dataclass
class Caveats:
    """Caveats and recommendations"""

    limitations: List[str]
    tradeoffs: List[str]
    recommendations: List[str]


class ModelCard:
    """
    Model Card for ML/Quant Strategy Governance

    Comprehensive documentation of model characteristics, performance,
    and responsible use guidelines
    """

    def __init__(
        self,
        model_details: ModelDetails,
        intended_use: IntendedUse,
        factors: Factors,
        metrics: Metrics,
        training_data: TrainingData,
        quantitative_analyses: QuantitativeAnalyses,
        ethical_considerations: EthicalConsiderations,
        caveats: Caveats,
    ):
        self.model_details = model_details
        self.intended_use = intended_use
        self.factors = factors
        self.metrics = metrics
        self.training_data = training_data
        self.quantitative_analyses = quantitative_analyses
        self.ethical_considerations = ethical_considerations
        self.caveats = caveats

        self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert model card to dictionary"""
        return {
            "model_card_version": "1.0",
            "created_at": self.created_at,
            "model_details": asdict(self.model_details),
            "intended_use": asdict(self.intended_use),
            "factors": asdict(self.factors),
            "metrics": asdict(self.metrics),
            "training_data": asdict(self.training_data),
            "quantitative_analyses": asdict(self.quantitative_analyses),
            "ethical_considerations": asdict(self.ethical_considerations),
            "caveats": asdict(self.caveats),
        }

    def to_json(self, filepath: Path):
        """Save model card as JSON"""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Model card saved to {filepath}")

    def to_markdown(self) -> str:
        """Generate model card as Markdown"""
        md_lines = [
            f"# Model Card: {self.model_details.name}",
            "",
            f"**Version:** {self.model_details.version}  ",
            f"**Date:** {self.model_details.developed_date}  ",
            f"**Developer:** {self.model_details.developed_by}  ",
            "",
            "---",
            "",
            "## Model Details",
            "",
            f"**Type:** {self.model_details.model_type}  ",
            f"**Description:** {self.model_details.description}  ",
            f"**License:** {self.model_details.license}  ",
            "",
            "### References",
            "",
        ]

        for ref in self.model_details.references:
            md_lines.append(f"- {ref}")

        md_lines.extend(["", "---", "", "## Intended Use", "", "### Primary Uses", ""])

        for use in self.intended_use.primary_uses:
            md_lines.append(f"- {use}")

        md_lines.extend(["", "### Primary Users", ""])

        for user in self.intended_use.primary_users:
            md_lines.append(f"- {user}")

        md_lines.extend(["", "### Out-of-Scope Uses", ""])

        for use in self.intended_use.out_of_scope_uses:
            md_lines.append(f"- ❌ {use}")

        md_lines.extend(
            [
                "",
                "---",
                "",
                "## Performance Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
            ]
        )

        for metric, value in self.metrics.model_performance_measures.items():
            md_lines.append(f"| {metric} | {value} |")

        md_lines.extend(
            [
                "",
                "---",
                "",
                "## Training Data",
                "",
                f"**Description:** {self.training_data.dataset_description}  ",
                f"**Period:** {self.training_data.start_date} to {self.training_data.end_date}  ",
                "",
                "### Data Sources",
                "",
            ]
        )

        for source in self.training_data.data_sources:
            md_lines.append(f"- {source}")

        md_lines.extend(["", "### Preprocessing Steps", ""])

        for step in self.training_data.preprocessing:
            md_lines.append(f"1. {step}")

        if self.training_data.known_issues:
            md_lines.extend(["", "### Known Data Issues", ""])
            for issue in self.training_data.known_issues:
                md_lines.append(f"- ⚠️ {issue}")

        md_lines.extend(
            ["", "---", "", "## Ethical Considerations", "", "### Risks and Harms", ""]
        )

        for risk in self.ethical_considerations.risks_and_harms:
            md_lines.append(f"- ⚠️ {risk}")

        if self.ethical_considerations.mitigation_strategies:
            md_lines.extend(["", "### Mitigation Strategies", ""])
            for strategy in self.ethical_considerations.mitigation_strategies:
                md_lines.append(f"- ✅ {strategy}")

        md_lines.extend(
            ["", "---", "", "## Limitations & Caveats", "", "### Limitations", ""]
        )

        for limitation in self.caveats.limitations:
            md_lines.append(f"- {limitation}")

        md_lines.extend(["", "### Recommendations", ""])

        for rec in self.caveats.recommendations:
            md_lines.append(f"- ✅ {rec}")

        md_lines.extend(["", "---", "", f"*Model card generated on {self.created_at}*"])

        return "\n".join(md_lines)

    def save_markdown(self, filepath: Path):
        """Save model card as Markdown"""
        md_content = self.to_markdown()

        with open(filepath, "w") as f:
            f.write(md_content)

        logger.info(f"Model card (Markdown) saved to {filepath}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCard":
        """Create ModelCard from dictionary"""
        return cls(
            model_details=ModelDetails(**data["model_details"]),
            intended_use=IntendedUse(**data["intended_use"]),
            factors=Factors(**data["factors"]),
            metrics=Metrics(**data["metrics"]),
            training_data=TrainingData(**data["training_data"]),
            quantitative_analyses=QuantitativeAnalyses(**data["quantitative_analyses"]),
            ethical_considerations=EthicalConsiderations(
                **data["ethical_considerations"]
            ),
            caveats=Caveats(**data["caveats"]),
        )

    @classmethod
    def from_json(cls, filepath: Path) -> "ModelCard":
        """Load ModelCard from JSON"""
        with open(filepath, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)


# ============================================================================
# Example: Generate Model Card for a Quant Strategy
# ============================================================================


def generate_example_strategy_model_card() -> ModelCard:
    """Generate an example model card for a quant strategy"""

    model_details = ModelDetails(
        name="RSI Mean Reversion Strategy",
        version="1.0.0",
        description="Mean reversion strategy using RSI indicator. Buys when RSI < 30, sells when RSI > 70.",
        model_type="quant_strategy",
        developed_by="Quant Research Team",
        developed_date="2025-11-17",
        license="Proprietary",
        contact="quant@example.com",
        references=[
            "Wilder, J. W. (1978). New Concepts in Technical Trading Systems",
            "Almgren & Chriss (2001). Optimal execution of portfolio transactions",
        ],
    )

    intended_use = IntendedUse(
        primary_uses=[
            "Mean reversion trading in liquid equities",
            "Portfolio diversification component",
            "Short-term tactical allocation",
        ],
        primary_users=[
            "Quantitative portfolio managers",
            "Systematic trading desks",
            "Risk-managed algorithmic execution",
        ],
        out_of_scope_uses=[
            "High-frequency trading (<1 second holding periods)",
            "Illiquid or low-volume securities",
            "Non-equity asset classes without validation",
            "Leveraged trading beyond 1.5x without risk approval",
        ],
    )

    factors = Factors(
        relevant_factors=[
            "Market volatility regime (low/medium/high)",
            "Market liquidity conditions",
            "Sector rotation patterns",
            "News/event-driven price movements",
        ],
        evaluation_factors=[
            "Volatility regime",
            "Trading volume",
            "Market cap category",
            "Sector classification",
        ],
    )

    metrics = Metrics(
        model_performance_measures={
            "sharpe_ratio": 1.85,
            "sortino_ratio": 2.34,
            "max_drawdown": -0.12,
            "annual_return": 0.18,
            "volatility": 0.15,
            "win_rate": 0.58,
            "total_trades": 347,
        },
        decision_thresholds={
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "min_volume": 100000,
            "max_position_size": 100000,
        },
        variation_approaches=[
            "Performance segmented by volatility regime",
            "Separate analysis for market cap categories",
            "Walk-forward validation across time periods",
        ],
    )

    training_data = TrainingData(
        dataset_description="Daily OHLCV data for S&P 500 constituents",
        start_date="2020-01-01",
        end_date="2024-12-31",
        data_sources=[
            "Yahoo Finance API",
            "Corporate actions database (splits, dividends)",
        ],
        preprocessing=[
            "Adjusted for stock splits and dividends",
            "Removed delisted stocks (survivorship bias mitigation)",
            "Filled missing data with forward-fill (max 3 days)",
            "Calculated RSI(14) using standard Wilder method",
            "Filtered for minimum volume threshold (>100k shares/day)",
        ],
        known_issues=[
            "March 2020 COVID crash: extreme volatility may not represent future conditions",
            "Low liquidity periods may have inflated slippage estimates",
        ],
    )

    quantitative_analyses = QuantitativeAnalyses(
        performance_by_regime={
            "low_volatility": {"sharpe": 1.45, "annual_return": 0.12},
            "medium_volatility": {"sharpe": 2.10, "annual_return": 0.21},
            "high_volatility": {"sharpe": 0.95, "annual_return": 0.15},
        },
        robustness_tests={
            "walk_forward_degradation": "15% (within acceptable range)",
            "monte_carlo_confidence": "95% CI: [1.5, 2.2] Sharpe",
            "parameter_stability": "Stable within ±20% of optimal RSI thresholds",
        },
        capacity_analysis={
            "estimated_capacity_usd": 50_000_000,
            "max_participation_rate": 0.10,
            "transaction_cost_breakeven": 12_000_000,
        },
        transaction_costs={
            "avg_slippage_bps": 2.5,
            "commission_bps": 1.0,
            "total_cost_bps": 3.5,
            "annual_turnover": 450,
        },
    )

    ethical_considerations = EthicalConsiderations(
        sensitive_data=[],
        risks_and_harms=[
            "Market impact: Large trades may move prices adversely",
            "Liquidity risk: Strategy may underperform during market stress",
            "Crowding risk: RSI is widely used; may be crowded trade",
            "Technology risk: Execution latency can erode alpha",
        ],
        use_cases_to_avoid=[
            "Trading during circuit breaker events",
            "Application to penny stocks or illiquid securities",
            "Use without proper risk controls (kill switches, limits)",
        ],
        mitigation_strategies=[
            "Real-time position limits enforced by risk system",
            "Kill switch activated on 10% daily loss",
            "Participation rate capped at 10% of average daily volume",
            "Pre-trade checks for liquidity and volatility filters",
        ],
    )

    caveats = Caveats(
        limitations=[
            "Strategy is statistically trained on US equity data only",
            "Performance degrades in trending markets (designed for mean reversion)",
            "Relies on timely and accurate market data",
            "Does not account for extreme black swan events",
            "Requires continuous monitoring and parameter updates",
        ],
        tradeoffs=[
            "Higher Sharpe ratio vs lower capacity",
            "Faster signals vs higher transaction costs",
            "Tighter stops vs more whipsaws",
        ],
        recommendations=[
            "Monitor performance weekly; trigger review if Sharpe < 1.0 for 3 months",
            "Run daily data quality checks before trading",
            "Maintain 20% cash buffer for risk management",
            "Review and update parameters quarterly",
            "Conduct regime analysis monthly to detect shifts",
        ],
    )

    return ModelCard(
        model_details=model_details,
        intended_use=intended_use,
        factors=factors,
        metrics=metrics,
        training_data=training_data,
        quantitative_analyses=quantitative_analyses,
        ethical_considerations=ethical_considerations,
        caveats=caveats,
    )


if __name__ == "__main__":
    # Generate example
    card = generate_example_strategy_model_card()

    # Save as JSON
    card.to_json(Path("governance/model_cards/rsi_strategy_v1.json"))

    # Save as Markdown
    card.save_markdown(Path("governance/model_cards/rsi_strategy_v1.md"))
