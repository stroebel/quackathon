"""Map of a pilot ward and a site selection."""

import matplotlib.pyplot as plt
import numpy as np

from quackathon.pipeline import Pilot


def plot_pilot(pilot: Pilot, x: np.ndarray | None = None, ax=None):
    cfg = pilot.cfg
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9))
    pilot.ward.boundary.plot(ax=ax, color="black", linewidth=1)
    xmin, ymin, xmax, ymax = pilot.ward.total_bounds
    pilot.grid.plot(ax=ax, color="tab:red", linewidth=1.5, label="grid")
    pilot.buildings.plot(ax=ax, color="lightgrey", markersize=0.5)

    served = pilot.problem.coverage[:, np.asarray(x, bool)].any(axis=1) if x is not None else None
    sizes = 5 + 2 * pilot.nodes["demand_kwh_day"]
    pilot.all_nodes.plot(ax=ax, color="tab:grey", markersize=4, alpha=0.4)
    pilot.nodes.plot(ax=ax, color=np.where(served, "tab:green", "tab:orange") if served is not None else "tab:orange",
                     markersize=sizes, label=f"demand > {cfg.grid_distance_m:.0f} m from grid")

    pilot.sites.plot(ax=ax, marker="^", color="white", edgecolor="tab:blue", markersize=80, label="candidate sites")
    if x is not None:
        chosen = pilot.sites[np.asarray(x, bool)]
        chosen.buffer(cfg.service_radius_m).plot(ax=ax, color="tab:blue", alpha=0.12)
        chosen.plot(ax=ax, marker="^", color="tab:blue", markersize=120, label="selected")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_axis_off()
    ax.legend(loc="lower left")
    ax.set_title(f"Ntabankulu ward {cfg.ward_no}: {pilot.problem.n_select} of {pilot.problem.n_sites} sites")
    return ax
