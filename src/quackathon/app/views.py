"""Dear PyGui front end: settings, run history, and map / sites / compare / QAOA / wards tabs.

All Dear PyGui calls happen on the main thread. Slow work goes through `JobRunner`, and
`App.poll()` (called once per frame) picks up progress and results.
"""

import re
import time
from concurrent.futures import Future
from math import comb

import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd
import shapely

from quackathon.app import jobs
from quackathon.app.jobs import QAOAOptions, Run
from quackathon.app.runner import JobRunner
from quackathon.classical import all_bitstrings, qubo_energies
from quackathon.config import PilotConfig
from quackathon.quantum import QAOAResult

DEFAULT_WARD = 4  # chosen in notebooks/ward_choice.ipynb
N_TOP_STATES = 60
REDRAW_EVERY_S = 0.3
CLICK_TOLERANCE_PX = 5

# Colours (RGBA) on Dear PyGui's dark theme.
WARD = (200, 200, 200, 255)
GRID_MAPPED = (235, 235, 235, 255)
GRID_PREDICTED = (130, 140, 170, 255)
BUILDING = (140, 140, 140, 90)
UNSERVED = (225, 95, 80, 255)
SERVED = (70, 195, 145, 255)
CANDIDATE = (230, 230, 230, 255)
CHOSEN = (250, 200, 60, 255)
REFERENCE = (160, 160, 160, 255)


def _series_theme(color, marker=None, size=None, weight=None, fill=None):
    with dpg.theme() as theme, dpg.theme_component(dpg.mvAll):
        cat = dpg.mvThemeCat_Plots
        dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=cat)
        dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, color, category=cat)
        dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, fill or color, category=cat)
        dpg.add_theme_color(dpg.mvPlotCol_Fill, fill or color, category=cat)
        if marker is not None:
            dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, marker, category=cat)
        if size is not None:
            dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, size, category=cat)
        if weight is not None:
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, weight, category=cat)
    return theme


def _lines_xy(geoms) -> tuple[list[float], list[float]]:
    """Coordinates of every line or polygon ring, NaN-separated for one `skip_nan` line series."""
    xs, ys = [], []
    for geom in geoms:
        for part in shapely.get_parts(geom):
            rings = [part.exterior, *part.interiors] if part.geom_type == "Polygon" else [part]
            for ring in rings:
                c = np.asarray(ring.coords)
                xs += [*c[:, 0], np.nan]
                ys += [*c[:, 1], np.nan]
    return xs, ys


def _circles_xy(xy: np.ndarray, radius: float) -> tuple[list[float], list[float]]:
    t = np.linspace(0, 2 * np.pi, 73)
    xs, ys = [], []
    for x, y in xy:
        xs += [*(x + radius * np.cos(t)), np.nan]
        ys += [*(y + radius * np.sin(t)), np.nan]
    return xs, ys


def _sites_label(x: np.ndarray) -> str:
    return ", ".join(map(str, np.flatnonzero(x))) or "none"


class App:
    def __init__(self, runner: JobRunner, wards: list[int]):
        self.runner = runner
        self.wards = wards
        self.runs: dict[int, Run] = {}
        self.futures: dict[int, Future] = {}  # run id -> future of its current stage
        self.active: int | None = None
        self.plan = "Exact"
        self.manual: np.ndarray | None = None
        self.sample: tuple[int, np.ndarray] | None = None  # (rank, x) picked on the QAOA distribution
        self.qaoa_depth: int | None = None                   # depth shown on the QAOA tab (None = deepest)
        self.screen_future: Future | None = None
        self.screen: pd.DataFrame | None = None
        self.screen_sort: tuple[str, bool] = ("rank", True)
        self._next_id = 1
        self._press = None
        self._unlock_axes_at: int | None = None
        self._last_redraw = 0.0
        self._dirty: set[str] = set()
        self._map_layers: dict[int, dict] = {}
        self._ground: dict[int, float] = {}
        self._run_rows: dict[int, dict] = {}
        self._annotations: list[int] = []
        self._message = ""

    # ---------------------------------------------------------------- layout

    def build(self) -> None:
        self.themes = {
            "ward": _series_theme(WARD, weight=1.5),
            "grid_mapped": _series_theme(GRID_MAPPED, weight=2.0),
            "grid_predicted": _series_theme(GRID_PREDICTED, weight=1.5),
            "building": _series_theme(BUILDING, marker=dpg.mvPlotMarker_Circle, size=1.0),
            "candidate": _series_theme(CANDIDATE, marker=dpg.mvPlotMarker_Up, size=6, fill=(0, 0, 0, 0)),
            "chosen": _series_theme(CHOSEN, marker=dpg.mvPlotMarker_Up, size=9),
            "reach": _series_theme(CHOSEN, weight=1.5),
            "reference": _series_theme(REFERENCE, weight=1.0),
            "bar": _series_theme(SERVED),
            "optimum": _series_theme(CHOSEN),
            **{f"{kind}{size}": _series_theme(color, marker=dpg.mvPlotMarker_Circle, size=size)
               for kind, color in (("served", SERVED), ("unserved", UNSERVED)) for size in (2.5, 4.0, 6.0)},
        }
        defaults = PilotConfig()

        with dpg.window(tag="main"):
            with dpg.group(horizontal=True):
                with dpg.child_window(width=380, border=False):
                    self._build_settings(defaults)
                    dpg.add_separator()
                    self._build_run_table()
                with dpg.child_window(border=False):
                    dpg.add_text("", tag="status")
                    with dpg.tab_bar():
                        with dpg.tab(label="Map"):
                            self._build_map_tab()
                        with dpg.tab(label="Sites"):
                            dpg.add_text("", tag="sites_summary")
                            dpg.add_table(tag="sites_table", header_row=True, row_background=True,
                                          borders_innerH=True, scrollY=True, policy=dpg.mvTable_SizingStretchProp)
                        with dpg.tab(label="Compare"):
                            dpg.add_text("Click a plan to show it on the map. QUBO - true is the error of the "
                                         "pairwise (inclusion-exclusion) approximation.", wrap=900)
                            dpg.add_table(tag="compare_table", header_row=True, row_background=True,
                                          borders_innerH=True, policy=dpg.mvTable_SizingStretchProp)
                        with dpg.tab(label="QAOA"):
                            self._build_qaoa_tab()
                        with dpg.tab(label="Wards"):
                            self._build_wards_tab()

        with dpg.handler_registry():
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=self._on_mouse_down)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=self._on_mouse_up)

        self._update_size_hint()
        self._draw_wards()
        self._refresh_run_table()
        self._refresh_active()

    def _build_settings(self, d: PilotConfig) -> None:
        w = 150
        with dpg.collapsing_header(label="Problem", default_open=True):
            dpg.add_combo([str(x) for x in self.wards], label="ward", tag="ward", default_value=str(DEFAULT_WARD),
                          width=w)
            dpg.add_combo(["overture", "synthetic", "osm"], label="buildings", tag="buildings",
                          default_value="overture", width=w)
            dpg.add_combo(["gridfinder", "osm"], label="grid source", tag="grid_source",
                          default_value=d.grid_source, width=w)
            dpg.add_input_float(label="grid distance (m)", tag="grid_distance", default_value=d.grid_distance_m,
                                step=250, format="%.0f", width=w)
            dpg.add_input_int(label="sites to build (K)", tag="n_sites", default_value=d.n_sites, width=w,
                              min_value=1, min_clamped=True, callback=self._update_size_hint)
            dpg.add_input_int(label="candidates (qubits)", tag="n_candidates", default_value=d.n_candidates,
                              width=w, min_value=1, min_clamped=True, max_value=20, max_clamped=True,
                              callback=self._update_size_hint)
            dpg.add_input_float(label="service radius (m)", tag="radius", default_value=d.service_radius_m,
                                step=250, format="%.0f", width=w)
            dpg.add_input_float(label="households / building", tag="hh_per_building",
                                default_value=d.households_per_building, step=0.05, format="%.2f", width=w)
            dpg.add_input_float(label="kWh / household / day", tag="kwh_per_hh",
                                default_value=d.kwh_per_household_day, step=0.1, format="%.2f", width=w)
        o = QAOAOptions()
        with dpg.collapsing_header(label="QAOA", default_open=True):
            dpg.add_checkbox(label="run QAOA", tag="qaoa_on", default_value=True)
            dpg.add_combo(["xy", "x"], label="mixer", tag="mixer", default_value=o.mixer, width=w)
            dpg.add_input_int(label="depth p", tag="reps", default_value=o.reps, min_value=1, min_clamped=True,
                              max_value=8, max_clamped=True, width=w)
            dpg.add_input_int(label="shots", tag="shots", default_value=o.shots, step=1024, width=w)
            dpg.add_input_int(label="COBYLA maxiter", tag="maxiter", default_value=o.maxiter, step=50, width=w)
            dpg.add_input_int(label="restarts", tag="restarts", default_value=o.restarts, min_value=1,
                              min_clamped=True, width=w)
            dpg.add_input_int(label="seed", tag="seed", default_value=o.seed, width=w)
        dpg.add_text("", tag="size_hint", wrap=360)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Run", width=80, callback=self.submit_run)
            dpg.add_button(label="Cancel selected", callback=self.cancel_active)
            dpg.add_button(label="Load settings", callback=self.load_settings)

    def _build_run_table(self) -> None:
        dpg.add_text("Runs")
        with dpg.table(tag="run_table", header_row=True, row_background=True, borders_innerH=True,
                       scrollY=True, policy=dpg.mvTable_SizingStretchProp):
            for label in ("#", "ward", "K/N/R", "QAOA", "status", "exact", "QAOA", "time"):
                dpg.add_table_column(label=label)

    def _build_map_tab(self) -> None:
        with dpg.group(horizontal=True):
            dpg.add_combo([], label="plan", tag="plan", width=160, callback=lambda s, a: self.set_plan(a))
            dpg.add_checkbox(label="buildings", tag="show_buildings", default_value=True,
                             callback=lambda s, a: self._draw_map(static=True))
            dpg.add_button(label="Fit", callback=self._fit_map)
            dpg.add_button(label="Export plan (GeoJSON)", callback=self.export_plan)
        dpg.add_text("", tag="plan_summary")
        dpg.add_text("Click a candidate site to add or remove it (starts a Manual plan from the current one). "
                     "Drag to pan, scroll to zoom.", color=(150, 150, 150))
        with dpg.plot(tag="map", width=-1, height=-1, equal_aspects=True, no_title=True):
            dpg.add_plot_legend(location=dpg.mvPlot_Location_SouthWest)
            dpg.add_plot_axis(dpg.mvXAxis, tag="map_x", no_tick_labels=True)
            dpg.add_plot_axis(dpg.mvYAxis, tag="map_y", no_tick_labels=True)

    def _build_qaoa_tab(self) -> None:
        with dpg.group(horizontal=True):
            dpg.add_combo([], label="depth", tag="qaoa_depth", width=80,
                          callback=lambda s, a: self._set_qaoa_depth(int(a.removeprefix("p="))))
            dpg.add_text("", tag="qaoa_summary")
        dpg.add_text("", tag="qaoa_sample")
        with dpg.plot(label="Output probability of each valid plan, best energy first (click a bar to map it)",
                      tag="dist", width=-1, height=300):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="plan rank by QUBO energy", tag="dist_x")
            dpg.add_plot_axis(dpg.mvYAxis, label="probability", tag="dist_y")
        with dpg.group(horizontal=True):
            with dpg.plot(label="Optimiser trace", tag="conv", width=600, height=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="evaluation", tag="conv_x")
                dpg.add_plot_axis(dpg.mvYAxis, label="<E>", tag="conv_y")
            with dpg.plot(label="Quality by depth", tag="depth_plot", width=-1, height=-1):
                dpg.add_plot_legend(location=dpg.mvPlot_Location_SouthEast)
                dpg.add_plot_axis(dpg.mvXAxis, label="depth p", tag="depth_x")
                dpg.add_plot_axis(dpg.mvYAxis, label="P(optimum)", tag="depth_y")
                dpg.add_plot_axis(dpg.mvYAxis2, label="P(feasible), approx. ratio", tag="depth_y2", opposite=True)

    def _build_wards_tab(self) -> None:
        with dpg.group(horizontal=True):
            dpg.add_button(label="Screen all wards", callback=self.submit_screen)
            dpg.add_text("Solves every ward classically with the Problem settings (ward is ignored). "
                         "Click a ward to load it into the settings.", color=(150, 150, 150))
        dpg.add_text("", tag="screen_status")
        dpg.add_table(tag="ward_table", header_row=True, row_background=True, borders_innerH=True,
                      sortable=True, scrollY=True, policy=dpg.mvTable_SizingStretchProp,
                      callback=self._sort_wards)

    # ---------------------------------------------------------------- settings

    def _update_size_hint(self, *_):
        n, k = dpg.get_value("n_candidates"), dpg.get_value("n_sites")
        hint = f"{2**n:,} states, {comb(n, k):,} valid plans" if 1 <= k <= n else "need K <= candidates"
        if n > 14:
            hint += ". Above 14 qubits QAOA takes many minutes."
        dpg.set_value("size_hint", hint)

    def _read_settings(self) -> tuple[PilotConfig, str, QAOAOptions | None]:
        cfg = PilotConfig(
            ward_no=int(dpg.get_value("ward")),
            grid_distance_m=dpg.get_value("grid_distance"),
            grid_source=dpg.get_value("grid_source"),
            n_sites=dpg.get_value("n_sites"),
            n_candidates=dpg.get_value("n_candidates"),
            service_radius_m=dpg.get_value("radius"),
            households_per_building=dpg.get_value("hh_per_building"),
            kwh_per_household_day=dpg.get_value("kwh_per_hh"),
        )
        opts = None
        if dpg.get_value("qaoa_on"):
            opts = QAOAOptions(reps=dpg.get_value("reps"), mixer=dpg.get_value("mixer"),
                               shots=max(1, dpg.get_value("shots")), maxiter=max(1, dpg.get_value("maxiter")),
                               restarts=dpg.get_value("restarts"), seed=dpg.get_value("seed"))
        return cfg, dpg.get_value("buildings"), opts

    def load_settings(self, *_):
        run = self.runs.get(self.active)
        if run is None:
            return
        c = run.cfg
        for tag, value in [("ward", str(c.ward_no)), ("buildings", run.buildings_source),
                           ("grid_source", c.grid_source), ("grid_distance", c.grid_distance_m),
                           ("n_sites", c.n_sites), ("n_candidates", c.n_candidates), ("radius", c.service_radius_m),
                           ("hh_per_building", c.households_per_building), ("kwh_per_hh", c.kwh_per_household_day),
                           ("qaoa_on", run.opts is not None)]:
            dpg.set_value(tag, value)
        if run.opts:
            o = run.opts
            for tag, value in [("mixer", o.mixer), ("reps", o.reps), ("shots", o.shots),
                               ("maxiter", o.maxiter), ("restarts", o.restarts), ("seed", o.seed)]:
                dpg.set_value(tag, value)
        self._update_size_hint()

    # ---------------------------------------------------------------- jobs

    def submit_run(self, *_):
        try:
            cfg, source, opts = self._read_settings()
        except ValueError as e:
            self._status(f"Invalid settings: {e}")
            return
        run = Run(self._next_id, cfg, source, opts)
        self._next_id += 1
        self.runs[run.id] = run
        self.futures[run.id] = self.runner.submit(("build", run.id), jobs.build_run, cfg, source)
        self.select_run(run.id)

    def cancel_active(self, *_):
        run, future = self.runs.get(self.active), self.futures.get(self.active)
        if run is None or future is None:
            return
        if self.runner.cancel(future):
            del self.futures[run.id]
            run.status, run.finished = "cancelled", time.perf_counter()
        else:
            run.cancelled, run.status = True, "cancelling"
            self._status(f"Run {run.id} is already running; it is discarded when it finishes.")
        self._dirty.add("runs")

    def submit_screen(self, *_):
        if self.screen_future is not None:
            return
        try:
            cfg, _, _ = self._read_settings()
        except ValueError as e:
            self._status(f"Invalid settings: {e}")
            return
        self.screen_future = self.runner.submit("screen", jobs.screen_run, cfg)
        dpg.set_value("screen_status", f"Screening wards (radius {cfg.service_radius_m:.0f} m, "
                                       f"grid distance {cfg.grid_distance_m:.0f} m, K={cfg.n_sites}, "
                                       f"N={cfg.n_candidates})...")

    def poll(self) -> None:
        """Once per frame: apply progress messages and finished jobs."""
        messages, done = self.runner.poll()
        for kind, run_id, depth, _, energies in messages:
            if kind == "progress" and run_id in self.runs:
                self.runs[run_id].live.setdefault(depth, []).extend(energies)
                self._dirty.add("runs")
                if run_id == self.active:
                    self._dirty.add("qaoa")

        for key, future in done:
            if key == "screen":
                self._finish_screen(future)
            else:
                self._finish_stage(*key, future)

        for run_id, future in self.futures.items():
            run = self.runs[run_id]
            if future.running() and run.status in ("queued", "qaoa queued"):
                run.status = "building" if run.status == "queued" else "qaoa"
                self._dirty.add("runs")

        if self._unlock_axes_at is not None and dpg.get_frame_count() >= self._unlock_axes_at:
            dpg.set_axis_limits_auto("map_x")
            dpg.set_axis_limits_auto("map_y")
            self._unlock_axes_at = None

        now = time.perf_counter()
        if self.futures:
            self._dirty.add("runs")  # elapsed time
        if self._dirty and now - self._last_redraw > REDRAW_EVERY_S:
            self._last_redraw = now
            if "runs" in self._dirty:
                self._refresh_run_table()
            if "qaoa" in self._dirty:
                self._draw_qaoa()
            self._dirty.clear()
            self._status()

    def _finish_stage(self, stage: str, run_id: int, future: Future) -> None:
        run = self.runs[run_id]
        self.futures.pop(run_id, None)
        self._dirty.add("runs")
        if future.cancelled():
            return
        if run.cancelled:
            run.status, run.finished = "cancelled", time.perf_counter()
            return
        if (exc := future.exception()) is not None:
            run.status, run.error, run.finished = "failed", f"{type(exc).__name__}: {exc}", time.perf_counter()
            self._status(f"Run {run_id} failed: {run.error}")
            if run_id == self.active:
                self._refresh_active()
            return
        if stage == "build":
            run.build = future.result()
            if run.opts is not None:
                run.status = "qaoa queued"
                self.futures[run_id] = self.runner.submit(("qaoa", run_id), jobs.qaoa_run, run.problem,
                                                          run.opts, run_id, self.runner.queue)
            else:
                run.status, run.finished = "done", time.perf_counter()
        else:
            run.sweep = future.result()
            run.status, run.finished = "done", time.perf_counter()
        if run_id == self.active:
            self._refresh_active()

    def _finish_screen(self, future: Future) -> None:
        self.screen_future = None
        if (exc := future.exception()) is not None:
            dpg.set_value("screen_status", f"Screening failed: {type(exc).__name__}: {exc}")
            return
        self.screen = future.result()
        dpg.set_value("screen_status", f"{len(self.screen)} wards screened.")
        self._draw_wards()

    def _status(self, message: str | None = None) -> None:
        if message is not None:
            self._message = message
        jobs_text = f"Workers: {self.runner.n_running} running, {self.runner.n_queued} queued"
        dpg.set_value("status", jobs_text + (f"  |  {self._message}" if self._message else ""))

    # ---------------------------------------------------------------- run selection

    def _refresh_run_table(self) -> None:
        """Rebuild rows when runs are added, otherwise update cells in place (so clicks are not lost)."""
        if set(self._run_rows) != set(self.runs):
            dpg.delete_item("run_table", children_only=True, slot=1)
            self._run_rows = {}
            for run in sorted(self.runs.values(), key=lambda r: -r.id):
                with dpg.table_row(parent="run_table"):
                    cells = {"select": dpg.add_selectable(label=str(run.id), span_columns=True,
                                                          callback=lambda s, a, u: self.select_run(u),
                                                          user_data=run.id)}
                    for key in ("ward", "size", "qaoa", "status", "exact", "qaoa_served", "time"):
                        cells[key] = dpg.add_text("")
                self._run_rows[run.id] = cells
        for run_id, cells in self._run_rows.items():
            run = self.runs[run_id]
            c, p = run.cfg, run.problem
            dpg.set_value(cells["select"], run_id == self.active)
            dpg.set_value(cells["ward"], str(c.ward_no))
            dpg.set_value(cells["size"], f"{c.n_sites}/{c.n_candidates}/{c.service_radius_m / 1000:g}k")
            dpg.set_value(cells["qaoa"], f"{run.opts.mixer} p{run.opts.reps}" if run.opts else "-")
            dpg.set_value(cells["status"], run.progress)
            dpg.configure_item(cells["status"], color=UNSERVED if run.status == "failed" else (255, 255, 255))
            dpg.set_value(cells["exact"], f"{p.served_demand(run.build.exact.x):.0f}" if run.build else "")
            dpg.set_value(cells["qaoa_served"], f"{p.served_demand(run.sweep[-1].x):.0f}" if run.sweep else "")
            dpg.set_value(cells["time"], f"{run.elapsed_s:.0f}s")

    def select_run(self, run_id: int) -> None:
        if run_id != self.active:
            self.active = run_id
            self.plan, self.manual, self.sample, self.qaoa_depth = "Exact", None, None, None
        self._refresh_run_table()
        self._refresh_active()
        run = self.runs[run_id]
        self._status(f"Run {run_id} failed: {run.error}" if run.error else "")

    def _refresh_active(self) -> None:
        self._draw_map(static=True, fit=True)
        self._draw_sites()
        self._draw_compare()
        self._draw_qaoa()

    # ---------------------------------------------------------------- plans

    def plans(self, run: Run | None) -> dict[str, np.ndarray]:
        if run is None or run.build is None:
            return {}
        b = run.build
        out = {"Exact": b.exact.x, "Greedy": b.greedy.x, "QUBO": b.qubo.x}
        for r in run.sweep:
            out[f"QAOA p={r.reps}"] = r.x
        if self.sample is not None:
            out[f"QAOA #{self.sample[0]}"] = self.sample[1]
        if self.manual is not None:
            out["Manual"] = self.manual
        return out

    def current_plan(self) -> np.ndarray | None:
        return self.plans(self.runs.get(self.active)).get(self.plan)

    def set_plan(self, name: str) -> None:
        self.plan = name
        self._draw_map()
        self._draw_sites()

    def toggle_site(self, j: int) -> None:
        run = self.runs.get(self.active)
        if run is None or run.build is None:
            return
        if self.plan != "Manual" or self.manual is None:
            x = self.current_plan()
            self.manual = (x if x is not None else np.zeros(run.problem.n_sites)).astype(np.int8).copy()
        self.manual[j] ^= 1
        self.plan = "Manual"
        self._draw_map()
        self._draw_sites()
        self._draw_compare()

    # ---------------------------------------------------------------- map

    def _layers(self, run: Run) -> dict:
        """Static map geometry for a run, computed once."""
        if run.id not in self._map_layers:
            pilot = run.build.pilot
            grid = pilot.grid.clip(pilot.ward.buffer(2_000).union_all())
            predicted = grid["source"].eq("gridfinder") if "source" in grid else pd.Series(False, grid.index)
            nodes_xy = np.column_stack([pilot.nodes.geometry.x, pilot.nodes.geometry.y])
            demand = pilot.nodes["demand_kwh_day"].to_numpy()
            self._map_layers[run.id] = {
                "ward": _lines_xy(pilot.ward.geometry),
                "grid_mapped": _lines_xy(grid.geometry[~predicted]),
                "grid_predicted": _lines_xy(grid.geometry[predicted]),
                "buildings": (pilot.buildings.geometry.x.tolist(), pilot.buildings.geometry.y.tolist()),
                "nodes": nodes_xy,
                "node_bin": np.digitize(demand, np.quantile(demand, [1 / 3, 2 / 3])),
                "sites": np.column_stack([pilot.sites.geometry.x, pilot.sites.geometry.y]),
            }
        return self._map_layers[run.id]

    def _draw_map(self, static: bool = False, fit: bool = False) -> None:
        run = self.runs.get(self.active)
        plans = self.plans(run)
        dpg.configure_item("plan", items=list(plans))
        if self.plan not in plans:
            self.plan = "Exact"
        dpg.set_value("plan", self.plan if plans else "")

        if static:
            dpg.delete_item("map_y", children_only=True)
            for item in self._annotations:
                dpg.delete_item(item)
            self._annotations = []
        else:  # keep the static layers, redraw the plan
            for item in dpg.get_item_children("map_y", slot=1) or []:
                if dpg.get_item_user_data(item) == "plan":
                    dpg.delete_item(item)

        if run is None or run.build is None:
            msg = "No run selected." if run is None else f"Run {run.id}: {run.error or run.status}"
            dpg.set_value("plan_summary", msg)
            return

        layers, p, cfg = self._layers(run), run.problem, run.cfg
        x = plans[self.plan]
        if static:
            for key, label in [("ward", "ward"), ("grid_mapped", "grid (mapped)"),
                               ("grid_predicted", "grid (predicted, gridfinder)")]:
                if layers[key][0]:
                    dpg.add_line_series(*layers[key], label=label, parent="map_y", skip_nan=True)
                    dpg.bind_item_theme(dpg.last_item(), self.themes[key])
            if dpg.get_value("show_buildings"):
                dpg.add_scatter_series(*layers["buildings"], label="buildings", parent="map_y")
                dpg.bind_item_theme(dpg.last_item(), self.themes["building"])
            sites = layers["sites"]
            dpg.add_scatter_series(sites[:, 0].tolist(), sites[:, 1].tolist(), label="candidate site",
                                   parent="map_y")
            dpg.bind_item_theme(dpg.last_item(), self.themes["candidate"])
            for j, (sx, sy) in enumerate(sites):
                self._annotations.append(dpg.add_plot_annotation(
                    label=str(j), default_value=(sx, sy), offset=(9, -9), color=(40, 40, 40, 200), parent="map"))

        served = p.coverage[:, x.astype(bool)].any(axis=1)
        nodes = layers["nodes"]
        for b, size in enumerate((2.5, 4.0, 6.0)):
            for kind, mask in (("unserved", ~served), ("served", served)):
                m = mask & (layers["node_bin"] == b)
                if m.any():
                    dpg.add_scatter_series(nodes[m, 0].tolist(), nodes[m, 1].tolist(), label=f"{kind} demand",
                                           parent="map_y", user_data="plan")
                    dpg.bind_item_theme(dpg.last_item(), self.themes[f"{kind}{size}"])
        chosen = layers["sites"][x.astype(bool)]
        if len(chosen):
            dpg.add_line_series(*_circles_xy(chosen, cfg.service_radius_m), parent="map_y", skip_nan=True,
                                label=f"reach ({cfg.service_radius_m:,.0f} m)", user_data="plan")
            dpg.bind_item_theme(dpg.last_item(), self.themes["reach"])
            dpg.add_scatter_series(chosen[:, 0].tolist(), chosen[:, 1].tolist(), label="chosen site",
                                   parent="map_y", user_data="plan")
            dpg.bind_item_theme(dpg.last_item(), self.themes["chosen"])

        dpg.set_value("plan_summary", self._plan_summary(run, x))
        if fit:
            self._fit_map()

    def _plan_summary(self, run: Run, x: np.ndarray) -> str:
        p = run.problem
        served, total = p.served_demand(x), p.demand.sum()
        best = p.served_demand(run.build.exact.x)
        text = (f"Ward {run.cfg.ward_no}, {self.plan}: sites {_sites_label(x)}. "
                f"Serves {served:.1f} of {total:.1f} kWh/day ({served / total:.0%}), "
                f"{best - served:.1f} kWh/day below the optimum.")
        if not p.is_feasible(x):
            text += f" Not a valid plan: {int(x.sum())} sites, need {p.n_select}."
        return text

    def _fit_map(self, *_):
        dpg.fit_axis_data("map_x")
        dpg.fit_axis_data("map_y")

    def _centre_map(self, j: int) -> None:
        run = self.runs.get(self.active)
        if run is None or run.build is None:
            return
        sx, sy = self._layers(run)["sites"][j]
        xlo, xhi = dpg.get_axis_limits("map_x")
        ylo, yhi = dpg.get_axis_limits("map_y")
        half_w = 2.5 * run.cfg.service_radius_m
        half_h = half_w * (yhi - ylo) / max(xhi - xlo, 1e-9)
        dpg.set_axis_limits("map_x", sx - half_w, sx + half_w)
        dpg.set_axis_limits("map_y", sy - half_h, sy + half_h)
        self._unlock_axes_at = dpg.get_frame_count() + 2  # then let the user pan and zoom again

    def export_plan(self, *_):
        run, x = self.runs.get(self.active), self.current_plan()
        if run is None or x is None:
            return
        pilot, p = run.build.pilot, run.problem
        chosen = np.flatnonzero(x)
        # Demand reachable from several chosen sites is credited to the nearest one.
        dist = np.where(p.coverage[:, chosen], p.distance[:, chosen], np.inf)
        covered, nearest = np.isfinite(dist).any(axis=1), np.argmin(dist, axis=1)
        sites = pilot.sites.iloc[chosen].assign(
            site=chosen, serves_kwh_day=[float(p.demand[covered & (nearest == k)].sum()) for k in range(len(chosen))])
        plan = re.sub(r"\W+", "_", self.plan).strip("_").lower()
        path = run.cfg.data_dir / "exports" / f"run{run.id}_ward{run.cfg.ward_no}_{plan}.geojson"
        path.parent.mkdir(parents=True, exist_ok=True)
        sites.to_crs("EPSG:4326").to_file(path, driver="GeoJSON")
        self._status(f"Exported {len(chosen)} sites to {path}")

    # ---------------------------------------------------------------- mouse

    def _on_mouse_down(self, *_):
        for plot in ("map", "dist"):
            if dpg.is_item_hovered(plot):
                self._press = (plot, dpg.get_mouse_pos(local=False), dpg.get_plot_mouse_pos())
                return
        self._press = None

    def _on_mouse_up(self, *_):
        press, self._press = self._press, None
        if press is None:
            return
        plot, screen_xy, (px, py) = press
        if np.hypot(*np.subtract(dpg.get_mouse_pos(local=False), screen_xy)) > CLICK_TOLERANCE_PX:
            return  # a drag (pan), not a click
        run = self.runs.get(self.active)
        if run is None or run.build is None:
            return
        if plot == "map":
            sites = self._layers(run)["sites"]
            xlo, xhi = dpg.get_axis_limits("map_x")
            d = np.hypot(sites[:, 0] - px, sites[:, 1] - py)
            if d.min() <= 0.02 * (xhi - xlo):
                self.toggle_site(int(np.argmin(d)))
        elif plot == "dist" and (res := self._shown_result(run)) is not None:
            order = self._feasible_order(res)
            rank = int(round(px))
            if 0 <= rank < min(N_TOP_STATES, len(order)):
                self.sample = (rank, all_bitstrings(run.problem.n_sites)[order[rank]])
                self.plan = f"QAOA #{rank}"
                self._draw_map()
                self._draw_sites()
                self._draw_qaoa_sample(run, res, rank)

    # ---------------------------------------------------------------- sites and compare

    def _draw_sites(self) -> None:
        table = "sites_table"
        dpg.delete_item(table, children_only=True)
        run, x = self.runs.get(self.active), self.current_plan()
        if run is None or run.build is None or x is None:
            dpg.set_value("sites_summary", "")
            return
        pilot, p = run.build.pilot, run.problem
        qaoa_x = run.sweep[-1].x if run.sweep else np.zeros(p.n_sites, np.int8)
        # Each candidate site is a demand node; its grid distance is that node's.
        node_of_site = np.argmin(p.distance, axis=0)
        grid_d = pilot.nodes["grid_distance_m"].to_numpy()[node_of_site]
        base = p.served_demand(x)
        dpg.set_value("sites_summary", f"{self.plan}: {_sites_label(x)}. Click a site number to zoom the map; "
                                       f"tick 'in plan' to edit a Manual plan. Delta = change in served "
                                       f"demand if this site is toggled.")
        for label in ("site", "reachable kWh/day", "nodes in reach", "grid distance (m)", "delta kWh/day",
                      "exact", "QAOA", "in plan"):
            dpg.add_table_column(label=label, parent=table)
        for j in range(p.n_sites):
            toggled = x.copy()
            toggled[j] ^= 1
            with dpg.table_row(parent=table):
                dpg.add_selectable(label=str(j), user_data=j,
                                   callback=lambda s, a, u: (self._centre_map(u), dpg.set_value(s, False)))
                dpg.add_text(f"{pilot.sites['reachable_kwh_day'].iloc[j]:.1f}")
                dpg.add_text(str(int(p.coverage[:, j].sum())))
                dpg.add_text(f"{grid_d[j]:,.0f}")
                delta = p.served_demand(toggled) - base
                dpg.add_text(f"{delta:+.1f}", color=SERVED if delta > 0 else (UNSERVED if delta < 0 else REFERENCE))
                dpg.add_text("yes" if run.build.exact.x[j] else "")
                dpg.add_text("yes" if qaoa_x[j] else "")
                dpg.add_checkbox(default_value=bool(x[j]), callback=lambda s, a, u: self.toggle_site(u), user_data=j)

    def _draw_compare(self) -> None:
        table = "compare_table"
        dpg.delete_item(table, children_only=True)
        run = self.runs.get(self.active)
        plans = self.plans(run)
        if not plans:
            return
        p = run.problem
        best, total = p.served_demand(run.build.exact.x), p.demand.sum()
        for label in ("plan", "sites", "served kWh/day", "share", "below optimum", "valid",
                      "QUBO energy", "true objective", "QUBO - true"):
            dpg.add_table_column(label=label, parent=table)
        for name, x in plans.items():
            served, qubo, true = p.served_demand(x), p.qubo_energy(x), p.objective(x)
            with dpg.table_row(parent=table):
                dpg.add_selectable(label=name, span_columns=True, default_value=name == self.plan,
                                   callback=lambda s, a, u: (self.set_plan(u), self._draw_compare()), user_data=name)
                dpg.add_text(_sites_label(x))
                dpg.add_text(f"{served:.1f}")
                dpg.add_text(f"{served / total:.0%}")
                dpg.add_text(f"{best - served:.1f}")
                dpg.add_text("yes" if p.is_feasible(x) else "no")
                dpg.add_text(f"{qubo:.1f}")
                dpg.add_text(f"{true:.1f}")
                dpg.add_text(f"{qubo - true:+.1f}" if p.is_feasible(x) else "(penalty)")

    # ---------------------------------------------------------------- QAOA

    def _set_qaoa_depth(self, depth: int) -> None:
        self.qaoa_depth, self.sample = depth, None
        self._draw_qaoa()

    def _shown_result(self, run: Run) -> QAOAResult | None:
        if not run.sweep:
            return None
        by_depth = {r.reps: r for r in run.sweep}
        return by_depth.get(self.qaoa_depth, run.sweep[-1])

    @staticmethod
    def _feasible_order(res: QAOAResult) -> np.ndarray:
        """Indices of feasible bitstrings, lowest energy first."""
        idx = np.flatnonzero(res.feasible)
        return idx[np.argsort(res.energies[idx], kind="stable")]

    def _ground_energy(self, run: Run) -> float:
        """Lowest QUBO energy over valid plans, in the energy QAOA optimises (no penalty for xy)."""
        if run.id not in self._ground:
            p = run.problem
            energies = qubo_energies(*p.to_qubo(0.0 if run.opts.mixer == "xy" else None))
            feasible = all_bitstrings(p.n_sites).sum(axis=1) == p.n_select
            self._ground[run.id] = float(energies[feasible].min())
        return self._ground[run.id]

    def _draw_qaoa(self) -> None:
        for axis in ("dist_y", "conv_y", "depth_y", "depth_y2"):
            dpg.delete_item(axis, children_only=True)
        run = self.runs.get(self.active)
        dpg.configure_item("qaoa_depth", items=[f"p={r.reps}" for r in run.sweep] if run else [])
        if run is None or run.opts is None or run.build is None:
            dpg.set_value("qaoa_summary", "No QAOA for this run." if run else "")
            dpg.set_value("qaoa_depth", "")
            dpg.set_value("qaoa_sample", "")
            return

        res = self._shown_result(run)
        ground = self._ground_energy(run)
        if res is None:  # still running: live trace of the depth in progress
            depth = max(run.live) if run.live else 1
            trace = run.live.get(depth, [])
            dpg.set_value("qaoa_summary", f"{run.progress}. Trace shows every evaluation of depth {depth} "
                                          f"(grid search and all restarts).")
            dpg.set_value("qaoa_depth", "")
            dpg.set_value("qaoa_sample", "")
        else:
            trace = res.history
            p = run.problem
            dpg.set_value("qaoa_depth", f"p={res.reps}")
            dpg.set_value("qaoa_summary",
                          f"{res.mixer} mixer, p={res.reps}: P(optimum) {res.p_ground:.3f} vs random "
                          f"{res.p_ground_random:.3f}, P(feasible) {res.p_feasible:.2f}, approximation ratio "
                          f"{res.approximation_ratio:.2f}. Best of {res.shots} shots serves "
                          f"{p.served_demand(res.x):.1f} kWh/day. Trace: best restart.")
            order = self._feasible_order(res)[:N_TOP_STATES]
            probs = res.probabilities[order]
            dpg.add_bar_series(np.arange(1.0, len(order)).tolist(), probs[1:].tolist(), weight=0.7, label="valid plan",
                               parent="dist_y")
            dpg.bind_item_theme(dpg.last_item(), self.themes["bar"])
            dpg.add_bar_series([0.0], [float(probs[0])], weight=0.7, label="optimum", parent="dist_y")
            dpg.bind_item_theme(dpg.last_item(), self.themes["optimum"])
            dpg.add_inf_line_series([res.p_ground_random], horizontal=True, label="random guess", parent="dist_y")
            dpg.bind_item_theme(dpg.last_item(), self.themes["reference"])
            dpg.fit_axis_data("dist_x")
            dpg.fit_axis_data("dist_y")
            if self.sample is not None:
                self._draw_qaoa_sample(run, res, self.sample[0])
            else:
                dpg.set_value("qaoa_sample", "Click a bar to see that plan on the map.")

        if trace:
            dpg.add_line_series(list(range(1, len(trace) + 1)), list(trace), label="<E>", parent="conv_y")
            dpg.bind_item_theme(dpg.last_item(), self.themes["bar"])
        dpg.add_inf_line_series([ground], horizontal=True, label="optimum", parent="conv_y")
        dpg.bind_item_theme(dpg.last_item(), self.themes["optimum"])
        dpg.fit_axis_data("conv_x")
        dpg.fit_axis_data("conv_y")

        if run.sweep:
            depths = [float(r.reps) for r in run.sweep]
            for axis, label, values, theme in [
                ("depth_y", "P(optimum)", [r.p_ground for r in run.sweep], "optimum"),
                ("depth_y2", "P(feasible)", [r.p_feasible for r in run.sweep], "reference"),
                ("depth_y2", "approx. ratio", [r.approximation_ratio for r in run.sweep], "bar"),
            ]:
                dpg.add_line_series(depths, values, label=label, parent=axis)
                dpg.bind_item_theme(dpg.last_item(), self.themes[theme])
                dpg.add_scatter_series(depths, values, parent=axis)
                dpg.bind_item_theme(dpg.last_item(), self.themes[theme])
            dpg.add_inf_line_series([run.sweep[0].p_ground_random], horizontal=True, label="P(optimum), random",
                                    parent="depth_y")
            dpg.bind_item_theme(dpg.last_item(), self.themes["reference"])
            dpg.set_axis_limits("depth_x", 0.5, len(depths) + 0.5)
            dpg.set_axis_limits("depth_y2", 0, 1.05)
            dpg.fit_axis_data("depth_y")

    def _draw_qaoa_sample(self, run: Run, res: QAOAResult, rank: int) -> None:
        p, x = run.problem, self.sample[1]
        index = self._feasible_order(res)[rank]
        dpg.set_value("qaoa_sample", f"Plan #{rank}: sites {_sites_label(x)}, serves {p.served_demand(x):.1f} "
                                     f"kWh/day, P = {res.probabilities[index]:.4f}, "
                                     f"{res.counts.get(format(index, f'0{p.n_sites}b'), 0)} of {res.shots} shots. "
                                     f"Shown on the map as '{self.plan}'.")

    # ---------------------------------------------------------------- wards

    WARD_COLUMNS = [("ward", "ward", "{}"), ("rank", "rank", "{}"), ("buildings", "buildings", "{:,}"),
                    ("far_nodes", "far nodes", "{}"), ("far_demand", "far demand", "{:.1f}"),
                    ("served", "served (candidates)", "{:.1f}"), ("full_served", "served (all nodes)", "{:.1f}"),
                    ("served_share", "share", "{:.0%}"), ("candidate_gap", "candidate gap", "{:.1%}"),
                    ("greedy_gap", "greedy gap", "{:.1%}")]

    def _sort_wards(self, sender, sort_specs):
        if not sort_specs:
            return
        column, direction = sort_specs[0]
        key = dpg.get_item_user_data(column)
        self.screen_sort = (key, direction > 0)
        self._draw_wards()

    def _draw_wards(self) -> None:
        table = "ward_table"
        if not dpg.get_item_children(table, slot=0):
            for key, label, _ in self.WARD_COLUMNS:
                dpg.add_table_column(label=label, parent=table, user_data=key)
        dpg.delete_item(table, children_only=True, slot=1)
        if self.screen is None:
            return
        key, ascending = self.screen_sort
        df = self.screen.reset_index().sort_values(key, ascending=ascending, na_position="last")
        for _, row in df.iterrows():
            with dpg.table_row(parent=table):
                for i, (col, _, fmt) in enumerate(self.WARD_COLUMNS):
                    value = row[col]
                    text = "-" if pd.isna(value) else fmt.format(int(value) if fmt in ("{}", "{:,}") else value)
                    if i == 0:
                        dpg.add_selectable(label=text, span_columns=True, user_data=int(row["ward"]),
                                           callback=lambda s, a, u: self._use_ward(s, u))
                    else:
                        dpg.add_text(text)

    def _use_ward(self, sender, ward: int) -> None:
        dpg.set_value(sender, False)
        dpg.set_value("ward", str(ward))
        self._status(f"Ward {ward} loaded into the settings. Press Run to solve it.")
