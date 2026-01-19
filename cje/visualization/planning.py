"""Planning visualization utilities for CJE budget optimization.

This module provides plotting functions for:
- Budget planning dashboards (MDE vs budget, power curves)
- Variance model fit quality visualization
- Oracle fraction sensitivity analysis
- Optimality proof visualizations

All functions return matplotlib Figure objects and optionally save to file.

IMPORTANT: cost_model is REQUIRED for all functions. There are no meaningful
defaults - the optimal allocation depends critically on your actual costs.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
        >>> cost = CostModel(oracle_cost=16.0)  # Your actual cost ratio
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


def plot_variance_model_fit(
    measurements: List[Tuple[int, int, float]],
    variance_model: FittedVarianceModel,
    cost_model: CostModel,
    budget: Optional[float] = None,
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[Path] = None,
) -> Figure:
    """Plot variance model fit quality: measured vs predicted + SE contours.

    Left panel shows scatter of measured vs predicted variance with R².
    Right panel shows SE contours with budget lines and optimal point.

    Args:
        measurements: List of (n, m, variance) tuples from variance fitting.
        variance_model: Fitted variance model.
        cost_model: Cost model. REQUIRED - no default.
        budget: Budget for marking optimal point. If None, uses budget for 2% MDE.
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        matplotlib Figure with 2 panels.
    """
    # Compute adaptive budget if not provided
    if budget is None:
        budget = plan_for_mde(0.02, variance_model, cost_model).total_cost

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: Measured vs Predicted variance
    ax = axes[0]

    measured = np.array([m[2] for m in measurements])
    predicted = np.array(
        [variance_model.predict_variance(m[0], m[1]) for m in measurements]
    )

    ax.scatter(
        measured * 1e4,
        predicted * 1e4,
        s=60,
        alpha=0.7,
        color="#3498db",
        edgecolors="white",
    )

    # Perfect fit line
    max_val = max(max(measured), max(predicted)) * 1e4
    ax.plot(
        [0, max_val], [0, max_val], "k--", linewidth=1.5, alpha=0.5, label="Perfect fit"
    )

    # R² annotation
    ax.annotate(
        f"R² = {variance_model.r_squared:.3f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.set_xlabel("Measured Variance (×10⁻⁴)", fontsize=11)
    ax.set_ylabel("Predicted Variance (×10⁻⁴)", fontsize=11)
    ax.set_title(
        f"Variance Model Fit\n(n={variance_model.n_measurements} measurements)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_val * 1.1)
    ax.set_ylim(0, max_val * 1.1)
    ax.legend(loc="lower right", fontsize=9)

    # Panel 2: SE Contours with budget lines
    ax = axes[1]

    # Compute adaptive grid range based on budget and costs
    plan = plan_evaluation(
        budget=budget, variance_model=variance_model, cost_model=cost_model
    )
    n_max = min(plan.n_samples * 3, budget / cost_model.surrogate_cost)
    m_max = min(plan.m_oracle * 4, budget / cost_model.oracle_cost)

    n_grid = np.linspace(max(50, plan.n_samples / 5), n_max, 60)
    m_grid = np.linspace(max(10, plan.m_oracle / 5), m_max, 60)
    N, M = np.meshgrid(n_grid, m_grid)
    SE = np.sqrt(variance_model.sigma2_eval / N + variance_model.sigma2_cal / M) * 100

    contour = ax.contourf(N, M, SE, levels=20, cmap="viridis_r")
    plt.colorbar(contour, ax=ax, label="SE (%)", shrink=0.9)

    # Budget lines
    for b in [budget / 2, budget, budget * 2]:
        n_line = np.linspace(
            n_grid[0], min(b - m_grid[0] * cost_model.oracle_cost, n_max), 100
        )
        m_line = (b - n_line * cost_model.surrogate_cost) / cost_model.oracle_cost
        valid = (m_line >= m_grid[0]) & (m_line <= m_max)
        if np.any(valid):
            ax.plot(n_line[valid], m_line[valid], "w--", alpha=0.9, linewidth=2)
            idx = np.sum(valid) // 3
            ax.annotate(
                f"${b/1000:.0f}K",
                (n_line[valid][idx], m_line[valid][idx]),
                color="white",
                fontsize=10,
                fontweight="bold",
            )

    # Mark optimal point
    ax.scatter(
        [plan.n_samples],
        [plan.m_oracle],
        color="#e74c3c",
        s=100,
        zorder=5,
        edgecolors="white",
        linewidths=2,
        marker="*",
    )
    ax.annotate(
        f"Optimal\n({plan.n_samples}, {plan.m_oracle})",
        xy=(plan.n_samples, plan.m_oracle),
        xytext=(plan.n_samples + n_max / 10, plan.m_oracle + m_max / 10),
        color="white",
        fontsize=9,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
    )

    ax.set_xlabel("n (evaluation samples)", fontsize=11)
    ax.set_ylabel("m (oracle labels)", fontsize=11)
    ax.set_title(
        f"SE Contours with Budget Lines\n({cost_model.oracle_cost:.0f}× oracle cost)",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    return fig


def plot_oracle_sensitivity(
    variance_model: FittedVarianceModel,
    cost_model: CostModel,
    budgets: Optional[List[float]] = None,
    figsize: Tuple[float, float] = (8, 5),
    save_path: Optional[Path] = None,
) -> Figure:
    """Plot MDE vs oracle fraction at different budgets.

    Shows how MDE changes with oracle fraction, revealing the optimal
    allocation and diminishing returns beyond it.

    Args:
        variance_model: Fitted variance model.
        cost_model: Cost model. REQUIRED - no default.
        budgets: List of budgets to show. If None, computed adaptively.
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        matplotlib Figure.
    """
    # Compute adaptive budgets if not provided
    if budgets is None:
        budgets = _compute_adaptive_highlight_budgets(
            variance_model, cost_model, target_mdes=(0.02, 0.05, 0.10)
        )

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.get_cmap("viridis")(np.linspace(0.2, 0.8, len(budgets)))

    # Compute optimal fraction to set x-axis range
    plan_ref = plan_evaluation(
        budget=budgets[len(budgets) // 2],
        variance_model=variance_model,
        cost_model=cost_model,
    )
    optimal_frac = plan_ref.oracle_fraction

    # Adaptive oracle fraction range (centered on optimal, with margin)
    frac_min = max(0.02, optimal_frac / 4)
    frac_max = min(0.80, optimal_frac * 4)
    oracle_fracs = np.linspace(frac_min, frac_max, 50)

    for i, budget in enumerate(budgets):
        mdes = []
        for frac in oracle_fracs:
            # Solve for n given budget and oracle fraction
            n = budget / (cost_model.surrogate_cost + frac * cost_model.oracle_cost)
            m = frac * n

            if n >= 50 and m >= 10:  # Minimum viable allocation
                se = variance_model.predict_se(int(n), int(m))
                # MDE = (z_alpha + z_beta) * sqrt(2) * SE for 80% power
                mde = 2.8 * np.sqrt(2) * se * 100  # as percentage
                mdes.append(mde)
            else:
                mdes.append(np.nan)

        ax.plot(
            np.array(oracle_fracs) * 100,
            mdes,
            color=colors[i],
            linewidth=2.5,
            label=f"${budget/1000:.0f}K",
        )

        # Mark optimal for this budget
        plan = plan_evaluation(
            budget=budget, variance_model=variance_model, cost_model=cost_model
        )
        opt_frac = plan.oracle_fraction
        opt_mde = plan.mde * 100
        ax.scatter(
            [opt_frac * 100],
            [opt_mde],
            color=colors[i],
            s=80,
            zorder=5,
            edgecolors="white",
            linewidths=2,
        )

    # Mark optimal fraction line
    ax.axvline(
        x=optimal_frac * 100,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Optimal ≈ {optimal_frac*100:.0f}%",
    )

    ax.set_xlabel("Oracle Fraction (%)", fontsize=11)
    ax.set_ylabel("Minimum Detectable Effect (%)", fontsize=11)
    ax.set_title(
        "MDE vs Oracle Fraction\n(dots mark optimal allocation per budget)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(frac_min * 100, frac_max * 100)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    return fig


def plot_optimality_proof(
    variance_model: FittedVarianceModel,
    cost_model: CostModel,
    budget: Optional[float] = None,
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[Path] = None,
) -> Figure:
    """Plot optimality proof: marginal ratio and curves crossing.

    Left panel shows marginal benefit ratio at different oracle fractions.
    Right panel shows marginal curves crossing at optimal allocation.

    Args:
        variance_model: Fitted variance model.
        cost_model: Cost model. REQUIRED - no default.
        budget: Budget for reference. If None, uses budget for 2% MDE.
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        matplotlib Figure with 2 panels.
    """
    # Compute adaptive budget if not provided
    if budget is None:
        budget = plan_for_mde(0.02, variance_model, cost_model).total_cost

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Get optimal oracle fraction
    plan = plan_evaluation(
        budget=budget, variance_model=variance_model, cost_model=cost_model
    )
    optimal_frac = plan.oracle_fraction

    # Panel 1: Marginal ratio at different oracle fractions
    ax = axes[0]

    # Adaptive n range based on plan
    n_min = max(100, plan.n_samples // 5)
    n_max = plan.n_samples * 3
    n_range = np.linspace(n_min, n_max, 100)

    oracle_fracs = [
        max(0.02, optimal_frac / 3),
        optimal_frac / 1.5,
        optimal_frac,
        optimal_frac * 1.5,
        min(0.60, optimal_frac * 3),
    ]
    colors = ["#e74c3c", "#f39c12", "#27ae60", "#3498db", "#9b59b6"]
    labels = [
        f"{oracle_fracs[0]*100:.0f}% (too few)",
        f"{oracle_fracs[1]*100:.0f}%",
        f"{optimal_frac*100:.1f}% (optimal)",
        f"{oracle_fracs[3]*100:.0f}%",
        f"{oracle_fracs[4]*100:.0f}% (too many)",
    ]

    for frac, color, label in zip(oracle_fracs, colors, labels):
        m_range = frac * n_range

        marginal_n = (
            variance_model.sigma2_eval / (n_range**2) / cost_model.surrogate_cost
        )
        marginal_m = variance_model.sigma2_cal / (m_range**2) / cost_model.oracle_cost

        ratio = marginal_n / marginal_m

        lw = 3 if abs(frac - optimal_frac) < 0.01 else 1.5
        ax.plot(n_range, ratio, color=color, linewidth=lw, label=label)

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.annotate(
        "Optimal: ratio = 1",
        xy=(n_max * 0.7, 1.0),
        xytext=(n_max * 0.7, 1.15),
        fontsize=10,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
    )

    ax.set_xlabel("Sample Size (n)", fontsize=11)
    ax.set_ylabel("Ratio: (∂Var/∂$ on n) / (∂Var/∂$ on m)", fontsize=11)
    ax.set_title("Marginal Benefit Ratio", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2.5)
    ax.set_xlim(n_min, n_max)

    ax.text(
        n_min + (n_max - n_min) * 0.05,
        2.1,
        "> 1: Spend more on n",
        fontsize=9,
        color="#e74c3c",
    )
    ax.text(
        n_min + (n_max - n_min) * 0.05,
        0.4,
        "< 1: Spend more on m",
        fontsize=9,
        color="#9b59b6",
    )

    # Panel 2: Marginal curves crossing at optimal
    ax = axes[1]

    n_fixed = plan.n_samples
    m_min_plot = max(10, plan.m_oracle // 4)
    m_max_plot = min(n_fixed, plan.m_oracle * 4)
    m_values = np.linspace(m_min_plot, m_max_plot, 100)
    oracle_fracs_shown = m_values / n_fixed

    marginal_n_fixed = (
        variance_model.sigma2_eval / (n_fixed**2) / cost_model.surrogate_cost
    )
    marginal_m_curve = (
        variance_model.sigma2_cal / (m_values**2) / cost_model.oracle_cost
    )

    scale = 1 / marginal_n_fixed  # Normalize so marginal_n = 1
    ax.plot(
        oracle_fracs_shown * 100,
        np.ones_like(m_values),
        "b-",
        linewidth=2.5,
        label="∂Var/∂$ on n (fixed)",
    )
    ax.plot(
        oracle_fracs_shown * 100,
        marginal_m_curve * scale,
        "r-",
        linewidth=2.5,
        label="∂Var/∂$ on m (varies)",
    )

    # Mark optimal
    ax.axvline(
        x=optimal_frac * 100, color="#27ae60", linestyle="--", linewidth=2, alpha=0.8
    )
    ax.scatter([optimal_frac * 100], [1.0], color="#27ae60", s=100, zorder=5)
    ax.annotate(
        f"Optimal\n({optimal_frac*100:.1f}%)",
        xy=(optimal_frac * 100, 1.0),
        xytext=(
            optimal_frac * 100
            + (m_max_plot / n_fixed - m_min_plot / n_fixed) * 100 * 0.15,
            1.3,
        ),
        fontsize=10,
        color="#27ae60",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5),
    )

    # Shade regions
    under_opt = oracle_fracs_shown < optimal_frac
    over_opt = oracle_fracs_shown > optimal_frac
    if np.any(under_opt):
        ax.fill_between(
            oracle_fracs_shown[under_opt] * 100,
            marginal_m_curve[under_opt] * scale,
            1.0,
            alpha=0.2,
            color="red",
        )
    if np.any(over_opt):
        ax.fill_between(
            oracle_fracs_shown[over_opt] * 100,
            1.0,
            marginal_m_curve[over_opt] * scale,
            alpha=0.2,
            color="blue",
        )

    ax.set_xlabel("Oracle Fraction (m/n) %", fontsize=11)
    ax.set_ylabel("Marginal Variance Reduction per $ (normalized)", fontsize=11)
    ax.set_title(
        f"Marginal Curves at n={n_fixed:,}\n(curves cross at optimal)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(m_min_plot / n_fixed * 100, m_max_plot / n_fixed * 100)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    return fig


def generate_canonical_planning_figures(
    variance_model: FittedVarianceModel,
    cost_model: CostModel,
    measurements: Optional[List[Tuple[int, int, float]]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Figure]:
    """Generate all canonical planning figures.

    Args:
        variance_model: Fitted variance model.
        cost_model: Cost model. REQUIRED - no default.
        measurements: Optional measurements for fit quality plot.
        output_dir: Optional directory to save figures.

    Returns:
        Dict mapping figure names to Figure objects.
    """
    figures = {}

    # Figure 1: Planning Dashboard
    save_path = Path(output_dir) / "dashboard.png" if output_dir else None
    figures["dashboard"] = plot_planning_dashboard(
        variance_model, cost_model, save_path=save_path
    )

    # Figure 2: Variance Model Fit (if measurements provided)
    if measurements:
        save_path = Path(output_dir) / "variance_fit.png" if output_dir else None
        figures["variance_fit"] = plot_variance_model_fit(
            measurements, variance_model, cost_model, save_path=save_path
        )

    # Figure 3a: Oracle Sensitivity
    save_path = Path(output_dir) / "oracle_sensitivity.png" if output_dir else None
    figures["oracle_sensitivity"] = plot_oracle_sensitivity(
        variance_model, cost_model, save_path=save_path
    )

    # Figure 3b: Optimality Proof
    save_path = Path(output_dir) / "optimality.png" if output_dir else None
    figures["optimality"] = plot_optimality_proof(
        variance_model, cost_model, save_path=save_path
    )

    return figures
