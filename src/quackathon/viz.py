"""Plots for exploring a pilot: the region, the demand filter, the QUBO and its solutions.

Colours follow one fixed role mapping so every figure reads the same way:
blue = served / selected, orange = far-from-grid but unserved, grey = context.
"""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from quackathon.classical import all_bitstrings, qubo_energies
from quackathon.pipeline import Pilot
from quackathon.problem import MicrogridProblem

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
CONTEXT = "#c9c8c3"
SERVED = "#2a78d6"
UNSERVED = "#eb6834"

BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b52f2e"])


def use_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": CONTEXT,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#ebeae6",
        "grid.linewidth": 0.8,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "legend.frameon": False,
        "legend.labelcolor": INK_SECONDARY,
        "lines.linewidth": 2,
    })


def _map_axes(ax, gdf, title):
    xmin, ymin, xmax, ymax = gdf.total_bounds
    pad = 0.03 * max(xmax - xmin, ymax - ymin)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title)


def plot_municipality(wards, buildings, grid, pilot_ward: int | None = None, ax=None):
    """Wards shaded by building count, with the mapped grid and the pilot ward outlined."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9))
    counts = buildings.sjoin(wards[["WardNo", "geometry"]], predicate="within")["WardNo"].value_counts()
    wards = wards.assign(buildings=wards["WardNo"].map(counts).fillna(0))
    wards.plot(ax=ax, column="buildings", cmap=BLUES, edgecolor=SURFACE, linewidth=1.5,
               legend=True, legend_kwds={"label": "buildings per ward", "shrink": 0.6})
    grid.clip(wards.buffer(5_000).union_all()).plot(ax=ax, color=INK, linewidth=1.2)
    if pilot_ward is not None:
        wards[wards["WardNo"] == pilot_ward].boundary.plot(ax=ax, color=UNSERVED, linewidth=2.5)
    for _, w in wards.iterrows():
        p = w.geometry.representative_point()
        ax.annotate(str(w["WardNo"]), (p.x, p.y), ha="center", va="center", fontsize=8, color=INK,
                    bbox={"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": "none", "alpha": 0.8})
    ax.plot([], [], color=INK, linewidth=1.2, label="mapped grid")
    if pilot_ward is not None:
        ax.plot([], [], color=UNSERVED, linewidth=2.5, label=f"pilot ward {pilot_ward}")
    ax.legend(loc="lower left")
    _map_axes(ax, wards, "Ntabankulu (EC444): buildings per ward")
    return ax


def plot_grid_distance(pilot: Pilot, ax=None):
    """Distribution of building distance to the grid, with the demand cut-off."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.5))
    d_km = pilot.buildings.distance(pilot.grid.union_all()).to_numpy() / 1000
    cut_km = pilot.cfg.grid_distance_m / 1000
    bins = np.linspace(0, max(d_km.max(), cut_km) * 1.02, 50)
    ax.hist(d_km[d_km <= cut_km], bins=bins, color=CONTEXT, rwidth=0.85, label="within cut-off (excluded)")
    ax.hist(d_km[d_km > cut_km], bins=bins, color=UNSERVED, rwidth=0.85, label="beyond cut-off (demand)")
    ax.axvline(cut_km, color=INK, linewidth=1, linestyle="--")
    share = (d_km > cut_km).mean()
    ax.annotate(f"grid_distance_m = {pilot.cfg.grid_distance_m:,.0f}\n{share:.0%} of buildings beyond",
                (cut_km, ax.get_ylim()[1] * 0.95), xytext=(6, 0), textcoords="offset points",
                va="top", fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel("distance to nearest mapped grid line (km)")
    ax.set_ylabel("buildings")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax.set_title(f"Ward {pilot.cfg.ward_no}: how far buildings are from the grid")
    return ax


def plot_pilot(pilot: Pilot, x: np.ndarray | None = None, ax=None, title: str | None = None):
    """Ward map: buildings, grid, demand nodes, candidate sites and (optionally) a selection."""
    cfg, problem = pilot.cfg, pilot.problem
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9))
    pilot.ward.plot(ax=ax, color="#f4f3f0", edgecolor=INK_SECONDARY, linewidth=1)
    pilot.buildings.plot(ax=ax, color=CONTEXT, markersize=0.4)
    pilot.grid.clip(pilot.ward.buffer(2_000).union_all()).plot(ax=ax, color=INK, linewidth=1.5)

    served = (problem.coverage[:, np.asarray(x, bool)].any(axis=1) if x is not None
              else np.zeros(len(pilot.nodes), bool))
    sizes = 8 + 3 * pilot.nodes["demand_kwh_day"].to_numpy()
    if x is not None:
        chosen = pilot.sites[np.asarray(x, bool)]
        chosen.buffer(cfg.service_radius_m).plot(ax=ax, color=SERVED, alpha=0.10, edgecolor=SERVED, linewidth=1)
    for mask, colour in [(~served, UNSERVED), (served, SERVED)]:
        if not mask.any():
            continue
        pilot.nodes[mask].plot(ax=ax, color=colour, markersize=sizes[mask], edgecolor=SURFACE, linewidth=0.5)

    pilot.sites.plot(ax=ax, marker="^", color=SURFACE, edgecolor=INK_SECONDARY, markersize=70, linewidth=1)
    if x is not None:
        pilot.sites[np.asarray(x, bool)].plot(ax=ax, marker="^", color=SERVED, edgecolor=SURFACE,
                                              markersize=140, linewidth=1)
    for j, s in pilot.sites.iterrows():
        ax.annotate(str(j), (s.geometry.x, s.geometry.y), xytext=(7, 5), textcoords="offset points",
                    fontsize=8, color=INK, zorder=10,
                    bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.85})

    handles = [
        plt.Line2D([], [], color=INK, linewidth=1.5, label="mapped grid"),
        plt.Line2D([], [], marker="o", linestyle="", color=UNSERVED, label="far-from-grid demand, unserved"),
        plt.Line2D([], [], marker="^", linestyle="", markerfacecolor=SURFACE, markeredgecolor=INK_SECONDARY,
                   label="candidate site (qubit index)"),
    ]
    if x is not None:
        handles[2:2] = [plt.Line2D([], [], marker="o", linestyle="", color=SERVED, label="served demand")]
        handles.append(plt.Line2D([], [], marker="^", linestyle="", color=SERVED, markersize=10,
                                  label=f"selected site ({cfg.service_radius_m:,.0f} m reach)"))
    ax.legend(handles=handles, loc="lower left", fontsize=9)
    if title is None:
        title = f"Ward {cfg.ward_no}: choose {problem.n_select} of {problem.n_sites} sites"
        if x is not None:
            title += f", {problem.served_demand(x):.0f} of {problem.demand.sum():.0f} kWh/day served"
    _map_axes(ax, pilot.ward, title)
    return ax


def plot_qubo_matrix(problem: MicrogridProblem, include_penalty: bool = False, ax=None):
    """Q as a symmetric heatmap: diagonal = linear terms, off-diagonal = pairwise couplings.

    The cardinality penalty adds the same constant to every entry of each kind, which
    hides the problem structure, so it is left out by default.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))
    q, _ = problem.to_qubo(None if include_penalty else 0.0)
    sym = (q + q.T) / 2
    sym[np.diag_indices_from(sym)] = np.diag(q)
    lim = np.abs(sym).max()
    im = ax.imshow(sym, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim))
    ax.figure.colorbar(im, ax=ax, shrink=0.8, label="coefficient")
    ax.set_xticks(range(problem.n_sites))
    ax.set_yticks(range(problem.n_sites))
    ax.grid(False)
    ax.set_xlabel("site j")
    ax.set_ylabel("site i")
    if include_penalty:
        ax.set_title("QUBO matrix Q (with penalty)")
    else:
        ax.set_title("QUBO matrix Q (without penalty)")
    return ax


def plot_energy_landscape(problem: MicrogridProblem, n_lowest: int = 150, ax=None):
    """The lowest-energy bitstrings of the QUBO: the states QAOA should concentrate on."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.5))
    q, offset = problem.to_qubo()
    energies = qubo_energies(q, offset)
    feasible = all_bitstrings(problem.n_sites).sum(axis=1) == problem.n_select
    order = np.argsort(energies)[:n_lowest]
    rank = np.arange(len(order))
    for mask, colour, label in [(~feasible[order], CONTEXT, f"infeasible (≠ {problem.n_select} sites)"),
                                (feasible[order], SERVED, f"feasible (= {problem.n_select} sites)")]:
        ax.scatter(rank[mask], energies[order][mask], s=14, color=colour, label=label, zorder=2)
    best = order[0]
    bits = "".join(map(str, all_bitstrings(problem.n_sites)[best]))
    ax.annotate(f"ground state x = {bits}", (0, energies[best]), xytext=(10, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)
    ax.set_xlabel(f"state rank (lowest {n_lowest} of {2**problem.n_sites:,})")
    ax.set_ylabel("QUBO energy")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Energy landscape: lowest-energy states")
    return ax


def plot_qubo_vs_exact(problem: MicrogridProblem, ax=None):
    """For every feasible selection, QUBO-approximated vs true served demand."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5))
    q, offset = problem.to_qubo()
    approx, exact = [], []
    for chosen in combinations(range(problem.n_sites), problem.n_select):
        x = np.zeros(problem.n_sites)
        x[list(chosen)] = 1
        # On feasible points the penalty is zero, so -energy is the approximate value.
        approx.append(-(x @ q @ x + offset - problem.site_cost @ x) / problem.value_per_kwh_day)
        exact.append(problem.served_demand(x))
    approx, exact = np.array(approx), np.array(exact)
    lo, hi = min(approx.min(), exact.min()), max(approx.max(), exact.max())
    ax.plot([lo, hi], [lo, hi], color=CONTEXT, linewidth=1, zorder=1)
    ax.scatter(exact, approx, s=16, color=SERVED, edgecolor=SURFACE, linewidth=0.5, zorder=2)
    ib = int(np.argmax(exact))
    ax.scatter(exact[ib], approx[ib], s=60, facecolor="none", edgecolor=INK, linewidth=1.5, zorder=3)
    ax.annotate("true optimum", (exact[ib], approx[ib]), xytext=(-8, 8), textcoords="offset points",
                ha="right", fontsize=9, color=INK)
    ax.set_xlabel("true served demand (kWh/day)")
    ax.set_ylabel("QUBO approximation (kWh/day)")
    exact_share = np.isclose(approx, exact).mean()
    ax.set_title(f"Approximation quality: exact on {exact_share:.0%} of selections")
    return ax


def plot_sweep(values, served, total, xlabel: str, title: str, ax=None):
    """Served share of far-from-grid demand across one config parameter.

    `served` maps a series name to one value per entry in `values`.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    # Exact is drawn wide underneath so the QUBO line stays visible where they coincide.
    styles = {"exact optimum": dict(color=SERVED, linewidth=6, alpha=0.35, marker="o", markersize=13),
              "QUBO optimum": dict(color=UNSERVED, linewidth=2, marker="o", markersize=6)}
    for name, ys in served.items():
        share = 100 * np.asarray(ys) / np.asarray(total)
        ax.plot(values, share, markeredgecolor=SURFACE, label=name,
                **styles.get(name, dict(color=INK_SECONDARY, marker="o")))
    ax.set_ylim(0, 105)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("far-from-grid demand served (%)")
    ax.legend(loc="lower right")
    ax.set_title(title)
    return ax
