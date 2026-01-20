"""Planning visualization utilities for CJE budget optimization.

This module provides the planning dashboard for budget optimization.

All functions return matplotlib Figure objects and optionally save to file.

IMPORTANT: cost_model is REQUIRED for all functions. There are no meaningful
defaults - the optimal allocation depends critically on your actual costs.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Import planning utilities
from ..diagnostics.planning import (
    FittedVarianceModel,
    CostModel,
    plan_evaluation,
    plan_for_mde,
)


def _compute_adaptive_budget_range(
    variance_model: FittedVarianceModel,
    cost_model: CostModel,
    target_mde_range: Tuple[float, float] = (0.01, 0.10),
) -> Tuple[float, float]:
    """Compute budget range that spans meaningful MDE values.

    Args:
        variance_model: Fitted variance model.
        cost_model: Cost model for optimization.
        target_mde_range: (min_mde, max_mde) to span. Default: 1% to 10%.

    Returns:
        (budget_low, budget_high) tuple.
    """
    # Budget for high MDE (cheap) and low MDE (expensive)
    plan_high_mde = plan_for_mde(target_mde_range[1], variance_model, cost_model)
    plan_low_mde = plan_for_mde(target_mde_range[0], variance_model, cost_model)

    budget_low = plan_high_mde.total_cost
    budget_high = plan_low_mde.total_cost

    # Ensure reasonable range (at least 5x spread)
    if budget_high < budget_low * 5:
        budget_high = budget_low * 10

    return (budget_low, budget_high)


def _compute_adaptive_highlight_budgets(
    variance_model: FittedVarianceModel,
    cost_model: CostModel,
    target_mdes: Tuple[float, ...] = (0.02, 0.05, 0.10),
) -> List[float]:
    """Compute meaningful budget points to highlight.

    Args:
        variance_model: Fitted variance model.
        cost_model: Cost model for optimization.
        target_mdes: MDEs to mark (default: 2%, 5%, 10%).

    Returns:
        List of budgets that achieve the target MDEs.
    """
    budgets = []
    for mde in target_mdes:
        plan = plan_for_mde(mde, variance_model, cost_model)
        budgets.append(plan.total_cost)
    return budgets


def plot_planning_dashboard(
    variance_model: FittedVarianceModel,
    cost_model: CostModel,
    budget_range: Optional[Tuple[float, float]] = None,
    highlight_budgets: Optional[List[float]] = None,
    figsize: Tuple[float, float] = (14, 4.5),
    save_path: Optional[Path] = None,
) -> Figure:
    """Create 3-panel planning dashboard: MDE vs Budget, Power Curve, Cost Sensitivity.

    This is the primary user-facing visualization for budget planning.

    Args:
        variance_model: Fitted variance model from fit_variance_model().
        cost_model: Cost model for optimization. REQUIRED - no default.
        budget_range: (min, max) budget to plot. If None, computed adaptively
            to span 1%-10% MDE.
        highlight_budgets: Budgets to mark with dots. If None, computed as
            budgets for 2%, 5% MDE.
        figsize: Figure size (width, height).
        save_path: Optional path to save figure.

    Returns:
        matplotlib Figure with 3 panels.

    Example:
        >>> from cje.diagnostics.planning import FittedVarianceModel, CostModel
        >>> model = FittedVarianceModel(sigma2_eval=0.008, sigma2_cal=0.004, r_squared=0.95, n_measurements=12)
        >>> cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)  # Actual dollar costs
        >>> fig = plot_planning_dashboard(model, cost)
        >>> fig.savefig("planning_dashboard.png")
    """
    # Compute adaptive defaults if not provided
    if budget_range is None:
        budget_range = _compute_adaptive_budget_range(variance_model, cost_model)
    if highlight_budgets is None:
        highlight_budgets = _compute_adaptive_highlight_budgets(
            variance_model, cost_model, target_mdes=(0.02, 0.05)
        )

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: MDE vs Budget
    ax = axes[0]
    budgets = np.linspace(budget_range[0], budget_range[1], 50)
    mdes = [
        plan_evaluation(
            budget=b, variance_model=variance_model, cost_model=cost_model
        ).mde
        * 100
        for b in budgets
    ]

    ax.plot(budgets / 1000, mdes, "b-", linewidth=2.5)
    ax.axhline(
        y=2,
        color="#e74c3c",
        linestyle="--",
        alpha=0.8,
        linewidth=1.5,
        label="2% target",
    )
    ax.axhline(
        y=5,
        color="#f39c12",
        linestyle="--",
        alpha=0.8,
        linewidth=1.5,
        label="5% target",
    )

    # Mark highlighted budgets
    colors = ["#27ae60", "#8e44ad", "#3498db", "#e67e22"]
    for i, budget in enumerate(highlight_budgets):
        if budget_range[0] <= budget <= budget_range[1]:
            plan = plan_evaluation(
                budget=budget, variance_model=variance_model, cost_model=cost_model
            )
            color = colors[i % len(colors)]
            ax.scatter([budget / 1000], [plan.mde * 100], color=color, s=80, zorder=5)
            ax.annotate(
                f"${budget/1000:.0f}K → {plan.mde*100:.1f}%",
                xy=(budget / 1000, plan.mde * 100),
                xytext=(
                    budget / 1000 + (budget_range[1] - budget_range[0]) / 20000,
                    plan.mde * 100 + 0.3,
                ),
                fontsize=9,
                color=color,
            )

    ax.set_xlabel("Budget ($K)", fontsize=11)
    ax.set_ylabel("Minimum Detectable Effect (%)", fontsize=11)
    ax.set_title("MDE vs Budget", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(budget_range[0] / 1000, budget_range[1] / 1000)
    ax.set_ylim(0, max(mdes) * 1.1)

    # Panel 2: Power Curve for first highlighted budget
    ax = axes[1]
    ref_budget = highlight_budgets[0] if highlight_budgets else budget_range[0] * 2
    plan = plan_evaluation(
        budget=ref_budget, variance_model=variance_model, cost_model=cost_model
    )

    # Adaptive effect size range based on actual MDE
    max_effect = max(plan.mde * 3, 0.08)
    effect_sizes = np.linspace(plan.mde * 0.2, max_effect, 100)
    powers = [plan.power_to_detect(e) for e in effect_sizes]

    ax.plot(np.array(effect_sizes) * 100, powers, "b-", linewidth=2.5)
    ax.fill_between(np.array(effect_sizes) * 100, powers, alpha=0.15, color="blue")
    ax.axhline(
        y=0.8,
        color="#e74c3c",
        linestyle="--",
        alpha=0.8,
        linewidth=1.5,
        label="80% power",
    )
    ax.axvline(
        x=plan.mde * 100,
        color="#27ae60",
        linestyle="--",
        alpha=0.8,
        linewidth=1.5,
        label=f"MDE = {plan.mde*100:.1f}%",
    )

    ax.set_xlabel("Effect Size (%)", fontsize=11)
    ax.set_ylabel("Statistical Power", fontsize=11)
    ax.set_title(
        f"Power Curve (${ref_budget/1000:.0f}K: n={plan.n_samples:,}, m={plan.m_oracle})",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max_effect * 100)

    # Panel 3: Cost Sensitivity
    ax = axes[2]
    # Adaptive cost range centered on user's cost
    base_cost = cost_model.oracle_cost
    oracle_costs = [
        base_cost / 4,
        base_cost / 2,
        base_cost,
        base_cost * 2,
        base_cost * 4,
    ]
    colors_cost = plt.get_cmap("plasma")(np.linspace(0.15, 0.85, len(oracle_costs)))

    for i, oc in enumerate(oracle_costs):
        cm = CostModel(surrogate_cost=cost_model.surrogate_cost, oracle_cost=oc)
        budgets_sens = np.linspace(budget_range[0], budget_range[1] * 0.75, 30)
        mdes_sens = [
            plan_evaluation(budget=b, variance_model=variance_model, cost_model=cm).mde
            * 100
            for b in budgets_sens
        ]
        multiplier = oc / cost_model.surrogate_cost
        ax.plot(
            budgets_sens / 1000,
            mdes_sens,
            color=colors_cost[i],
            linewidth=2,
            label=f"{multiplier:.0f}×",
        )

    ax.set_xlabel("Budget ($K)", fontsize=11)
    ax.set_ylabel("Minimum Detectable Effect (%)", fontsize=11)
    ax.set_title("Cost Sensitivity", fontsize=12, fontweight="bold")
    ax.legend(title="Oracle Cost", loc="upper right", fontsize=8, title_fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    return fig
