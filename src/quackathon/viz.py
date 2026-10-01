"""Plots for the siting problem: the region, the demand filter, the QUBO and its solutions.

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


def _plot_grid(ax, grid, clip_to, linewidth=1.5):
    """Mapped lines solid, gridfinder-predicted lines dashed. Returns legend handles."""
    grid = grid.clip(clip_to)
    predicted = grid["source"].eq("gridfinder") if "source" in grid else None
    handles = [plt.Line2D([], [], color=INK, linewidth=linewidth, label="grid (mapped)")]
    if predicted is None or not predicted.any():
        grid.plot(ax=ax, color=INK, linewidth=linewidth)
        return handles
    grid[~predicted].plot(ax=ax, color=INK, linewidth=linewidth)
    grid[predicted].plot(ax=ax, color=INK, linewidth=linewidth, linestyle=(0, (3, 2)))
    handles.append(plt.Line2D([], [], color=INK, linewidth=linewidth, linestyle=(0, (3, 2)),
                              label="grid (predicted, gridfinder)"))
    return handles


def _map_axes(ax, gdf, title):
    xmin, ymin, xmax, ymax = gdf.total_bounds
    pad = 0.03 * max(xmax - xmin, ymax - ymin)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title)


def plot_municipality(wards, buildings, grid, pilot_ward: int | None = None, ax=None):
    """Wards shaded by building count, with the grid and the selected ward outlined."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9))
    counts = buildings.sjoin(wards[["WardNo", "geometry"]], predicate="within")["WardNo"].value_counts()
    wards = wards.assign(buildings=wards["WardNo"].map(counts).fillna(0))
    wards.plot(ax=ax, column="buildings", cmap=BLUES, edgecolor=SURFACE, linewidth=1.5,
               legend=True, legend_kwds={"label": "buildings per ward", "shrink": 0.6})
    handles = _plot_grid(ax, grid, wards.buffer(5_000).union_all(), linewidth=1.2)
    if pilot_ward is not None:
        wards[wards["WardNo"] == pilot_ward].boundary.plot(ax=ax, color=UNSERVED, linewidth=2.5)
    for _, w in wards.iterrows():
        p = w.geometry.representative_point()
        ax.annotate(str(w["WardNo"]), (p.x, p.y), ha="center", va="center", fontsize=8, color=INK,
                    bbox={"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": "none", "alpha": 0.8})
    if pilot_ward is not None:
        handles.append(plt.Line2D([], [], color=UNSERVED, linewidth=2.5, label=f"selected ward {pilot_ward}"))
    ax.legend(handles=handles, loc="lower left")
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
    ax.set_xlabel(f"distance to nearest grid line (km, {pilot.cfg.grid_source})")
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
    grid_handles = _plot_grid(ax, pilot.grid, pilot.ward.buffer(2_000).union_all())

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

    handles = grid_handles + [
        plt.Line2D([], [], marker="o", linestyle="", color=UNSERVED, label="far-from-grid demand, unserved"),
        plt.Line2D([], [], marker="^", linestyle="", markerfacecolor=SURFACE, markeredgecolor=INK_SECONDARY,
                   label="candidate site (qubit index)"),
    ]
    if x is not None:
        handles.insert(len(grid_handles) + 1,
                       plt.Line2D([], [], marker="o", linestyle="", color=SERVED, label="served demand"))
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


def plot_qaoa_distribution(result, n_top: int = 60, ax=None):
    """Output probability of the tuned circuit over feasible selections, best energy first.

    A good QAOA run piles probability on the left; the dashed line is random guessing.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.5))
    idx = np.flatnonzero(result.feasible)
    idx = idx[np.argsort(result.energies[idx])][:n_top]
    probs = result.probabilities[idx]
    colours = [SERVED if i == result.ground_state_index else CONTEXT for i in idx]
    ax.bar(np.arange(len(idx)), probs, color=colours, width=0.8)
    ax.axhline(result.p_ground_random, color=INK, linewidth=1, linestyle="--")
    ax.annotate("random guess", (len(idx) - 1, result.p_ground_random), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=9, color=INK_SECONDARY)
    ax.annotate(f"optimum: {result.p_ground:.1%}", (0, result.p_ground), xytext=(6, 2),
                textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel(f"feasible selections ranked by energy (best {len(idx)} of {result.feasible.sum()})")
    ax.set_ylabel("probability")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"QAOA output distribution ({result.mixer} mixer, p={result.reps})")
    return ax


def plot_depth_sweep(sweeps: dict[str, list], metric: str = "p_ground", ax=None):
    """One metric against QAOA depth, one line per mixer.

    metric: "p_ground" (P(optimum), with each mixer's random-guess baseline dashed),
    "approximation_ratio" or "p_feasible".
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    labels = {"p_ground": "P(optimum)", "approximation_ratio": "approximation ratio",
              "p_feasible": "P(feasible)"}
    colours = {"xy": SERVED, "x": UNSERVED}
    names = {"xy": "XY mixer (Dicke start)", "x": "X mixer (penalty)"}
    for mixer, results in sweeps.items():
        colour = colours.get(mixer, INK_SECONDARY)
        depths = [r.reps for r in results]
        ys = [getattr(r, metric) for r in results]
        ax.plot(depths, ys, color=colour, marker="o", markersize=8, markeredgecolor=SURFACE,
                label=names.get(mixer, mixer))
        fmt = "{:.1%}" if metric != "approximation_ratio" else "{:.2f}"
        ax.annotate(fmt.format(ys[-1]), (depths[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_SECONDARY)
        if metric == "p_ground":
            ax.axhline(results[0].p_ground_random, color=colour, linewidth=1, linestyle="--")
    if metric == "p_ground":
        ax.plot([], [], color=INK_SECONDARY, linewidth=1, linestyle="--", label="random guess (same colour)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
        ax.set_ylim(bottom=0)
    if metric == "p_feasible":
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_ylim(0, 1.05)
    ax.set_xticks(depths)
    ax.set_xlim(depths[0] - 0.3, depths[-1] + 0.6)
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel(labels[metric])
    ax.legend(loc="best", fontsize=9)
    ax.set_title(labels[metric] + " vs depth")
    return ax


def plot_convergence(result, ax=None):
    """Expected energy at each optimiser evaluation for the final depth."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(result.history, color=SERVED, linewidth=1.5)
    ax.axhline(result.ground_energy, color=INK, linewidth=1, linestyle="--")
    ax.annotate("optimum energy", (len(result.history) - 1, result.ground_energy), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel("optimiser evaluation")
    ax.set_ylabel("expected energy ⟨E⟩")
    ax.set_title(f"COBYLA convergence (p={result.reps})")
    return ax


def plot_ward_screening(screen, ax=None):
    """Per ward: far-from-grid demand, the best any K sites could serve, and the candidate optimum."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))
    s = screen.sort_values(["served", "far_demand"], ascending=False, na_position="last")
    pos = np.arange(len(s))
    ax.bar(pos, s["far_demand"], width=0.8, color=CONTEXT, edgecolor=SURFACE, linewidth=2,
           label="far-from-grid demand")
    ax.bar(pos, s["full_served"].fillna(0), width=0.8, color=UNSERVED, edgecolor=SURFACE, linewidth=2,
           label="best K sites, any demand node")
    ax.bar(pos, s["served"].fillna(0), width=0.8, color=SERVED, edgecolor=SURFACE, linewidth=2,
           label="best K of the candidate sites")
    for x, (_, r) in zip(pos, s.iterrows()):
        if not r["eligible"]:
            ax.annotate("too few\nnodes", (x, r["far_demand"]), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color=INK_SECONDARY)
    ax.set_axisbelow(True)
    ax.set_xticks(pos, [str(w) for w in s.index])
    ax.set_xlabel("ward (best first)")
    ax.set_ylabel("kWh/day")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax.set_title("Demand that each ward can serve")
    return ax


def plot_ward_choropleth(wards, values, label: str, title: str, highlight: int | None = None, ax=None):
    """Wards shaded by `values` (a Series indexed by WardNo); wards without a value are grey."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    wards = wards.assign(value=wards["WardNo"].map(values))
    wards[wards["value"].isna()].plot(ax=ax, color="#f0efec", edgecolor=SURFACE, linewidth=1.5)
    wards[wards["value"].notna()].plot(ax=ax, column="value", cmap=BLUES, edgecolor=SURFACE, linewidth=1.5,
                                       legend=True, legend_kwds={"label": label, "shrink": 0.6})
    if highlight is not None:
        wards[wards["WardNo"] == highlight].boundary.plot(ax=ax, color=UNSERVED, linewidth=2.5)
    for _, w in wards.iterrows():
        p = w.geometry.representative_point()
        ax.annotate(str(w["WardNo"]), (p.x, p.y), ha="center", va="center", fontsize=8, color=INK,
                    bbox={"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": "none", "alpha": 0.8})
    _map_axes(ax, wards, title)
    return ax


def plot_rank_heatmap(ranks, ax=None, title: str = "Ward rank across configs"):
    """Wards (rows) by config (columns), coloured by rank: dark = best. Blank = not eligible.

    Wards that are never eligible are left out.
    """
    ranks = ranks.dropna(how="all")
    ranks = ranks.loc[ranks.astype(float).median(axis=1).sort_values(na_position="last").index]
    values = ranks.to_numpy(float)
    if ax is None:
        _, ax = plt.subplots(figsize=(1 + 0.75 * values.shape[1], 0.4 * values.shape[0] + 1.5))
    cmap = BLUES.reversed().copy()
    cmap.set_bad("#f0efec")
    vmax = np.nanmax(values)
    ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=1, vmax=vmax, aspect="auto")
    for (i, j), v in np.ndenumerate(values):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color=SURFACE if v <= vmax / 2 else INK)
    cols = ranks.columns
    labels = ["\n".join(f"{v:,.0f}" for v in (c if isinstance(c, tuple) else (c,))) for c in cols]
    ax.set_xticks(range(len(cols)), labels, fontsize=8)
    ax.set_yticks(range(len(ranks)), [f"ward {w}" for w in ranks.index], fontsize=8)
    names = cols.names if getattr(cols, "names", None) else [cols.name]
    ax.set_xlabel(" / ".join(str(n) for n in names))
    ax.grid(False)
    ax.set_title(title)
    return ax


def plot_allocation(wards, grid, nodes, sites, radius_m: float, ax=None, title: str | None = None):
    """Municipality map: far-from-grid demand nodes and the chosen sites with their reach."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9))
    wards.plot(ax=ax, color="#f4f3f0", edgecolor=CONTEXT, linewidth=1)
    handles = _plot_grid(ax, grid, wards.buffer(5_000).union_all(), linewidth=1)
    reach = sites.buffer(radius_m)
    served = nodes.within(reach.union_all()).to_numpy()
    sizes = 4 + 1.5 * nodes["demand_kwh_day"].to_numpy()
    reach.plot(ax=ax, color=SERVED, alpha=0.12, edgecolor=SERVED, linewidth=1)
    for mask, colour in [(~served, UNSERVED), (served, SERVED)]:
        nodes[mask].plot(ax=ax, color=colour, markersize=sizes[mask], edgecolor=SURFACE, linewidth=0.4)
    sites.plot(ax=ax, marker="^", color=SERVED, edgecolor=SURFACE, markersize=120, linewidth=1)
    for _, w in wards.iterrows():
        p = w.geometry.representative_point()
        ax.annotate(str(w["WardNo"]), (p.x, p.y), ha="center", va="center", fontsize=8, color=INK_SECONDARY)
    handles += [
        plt.Line2D([], [], marker="o", linestyle="", color=SERVED, label="served demand"),
        plt.Line2D([], [], marker="o", linestyle="", color=UNSERVED, label="far-from-grid demand, unserved"),
        plt.Line2D([], [], marker="^", linestyle="", color=SERVED, markersize=10,
                   label=f"chosen site ({radius_m:,.0f} m reach)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    _map_axes(ax, wards, title or f"Best {len(sites)} sites anywhere in the municipality")
    return ax
