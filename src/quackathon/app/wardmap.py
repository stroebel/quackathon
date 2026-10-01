"""Municipality map geometry for the GUI: ward raster, line layers and grid distances.

Nothing here imports dearpygui, so it can be tested and used in worker processes.
Dear PyGui cannot fill concave polygons, so the wards are drawn as an image: a raster of
ward indices, coloured per ward through a lookup table.
"""

from dataclasses import dataclass, replace

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from quackathon.config import PilotConfig
from quackathon.grid import load_grid

SIMPLIFY_M = 25.0
RASTER_WIDTH = 1000

# Sequential ramp for the dark theme, low to high (RGBA, 0-1).
RAMP = np.array([[0.13, 0.20, 0.32], [0.16, 0.33, 0.55], [0.20, 0.47, 0.78], [0.43, 0.65, 0.93],
                 [0.80, 0.89, 0.98]])
NO_VALUE = (0.22, 0.22, 0.22, 1.0)
OUTSIDE = (0.0, 0.0, 0.0, 0.0)


def lines_xy(geoms) -> tuple[list[float], list[float]]:
    """Coordinates of every line or polygon ring, NaN-separated for one line series
    (ImPlot breaks a line at NaN; its `skip_nan` flag would join the pieces instead)."""
    xs, ys = [], []
    for geom in geoms:
        for part in shapely.get_parts(geom):
            rings = [part.exterior, *part.interiors] if part.geom_type == "Polygon" else [part]
            for ring in rings:
                c = np.asarray(ring.coords)
                xs += [*c[:, 0], np.nan]
                ys += [*c[:, 1], np.nan]
    return xs, ys


def circles_xy(xy: np.ndarray, radius: float) -> tuple[list[float], list[float]]:
    t = np.linspace(0, 2 * np.pi, 73)
    xs, ys = [], []
    for x, y in xy:
        xs += [*(x + radius * np.cos(t)), np.nan]
        ys += [*(y + radius * np.sin(t)), np.nan]
    return xs, ys


def dash_segments(geoms, dash_m: float = 400.0, gap_m: float = 250.0) -> tuple[list[float], list[float]]:
    """Dashes along every line, as consecutive endpoint pairs for a `segments=True` line series."""
    xs, ys = [], []
    for geom in geoms:
        for line in shapely.get_parts(geom):
            length = line.length
            starts = np.arange(0.0, length, dash_m + gap_m)
            if not len(starts):
                continue
            ends = np.minimum(starts + dash_m, length)
            a = shapely.get_coordinates(shapely.line_interpolate_point(line, starts))
            b = shapely.get_coordinates(shapely.line_interpolate_point(line, ends))
            pairs = np.stack([a, b], axis=1).reshape(-1, 2)
            xs += pairs[:, 0].tolist()
            ys += pairs[:, 1].tolist()
    return xs, ys


def is_predicted(grid: gpd.GeoDataFrame) -> np.ndarray:
    """Gridfinder-predicted lines (the rest are mapped in OSM)."""
    if "source" not in grid:
        return np.zeros(len(grid), bool)
    return grid["source"].eq("gridfinder").to_numpy()


def grid_distances(points: gpd.GeoSeries, grid: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each point to the nearest mapped and the nearest predicted line (inf if none)."""
    predicted = is_predicted(grid)
    out = []
    for mask in (~predicted, predicted):
        lines = grid.geometry[mask]
        out.append(points.distance(lines.union_all()).to_numpy() if mask.any()
                   else np.full(len(points), np.inf))
    return out[0], out[1]


def ward_raster(wards: gpd.GeoDataFrame, width: int = RASTER_WIDTH) -> tuple[np.ndarray, tuple]:
    """Index of the ward under each pixel (-1 outside), top row first, and the raster bounds."""
    x0, y0, x1, y1 = wards.total_bounds
    height = max(1, round(width * (y1 - y0) / (x1 - x0)))
    xs = x0 + (np.arange(width) + 0.5) * (x1 - x0) / width
    ys = y1 - (np.arange(height) + 0.5) * (y1 - y0) / height
    X, Y = np.meshgrid(xs, ys)
    index = np.full(X.shape, -1, np.int16)
    for i, geom in enumerate(wards.geometry):
        a, b, c, d = geom.bounds
        m = (X >= a) & (X <= c) & (Y >= b) & (Y <= d)
        index[m] = np.where(shapely.contains_xy(geom, X[m], Y[m]), i, index[m])
    return index, (x0, y0, x1, y1)


def ramp(t: np.ndarray) -> np.ndarray:
    """RGB for values in [0, 1]."""
    stops = np.linspace(0, 1, len(RAMP))
    return np.column_stack([np.interp(t, stops, RAMP[:, c]) for c in range(3)])


def ward_colours(values: np.ndarray, highlight: np.ndarray | None = None, reverse: bool = False) -> np.ndarray:
    """RGBA per ward: the ramp over the finite values, NO_VALUE for NaN, brighter where highlighted."""
    values = np.asarray(values, float)
    colours = np.tile(NO_VALUE, (len(values), 1))
    ok = np.isfinite(values)
    if ok.any():
        lo, hi = values[ok].min(), values[ok].max()
        t = (values[ok] - lo) / (hi - lo) if hi > lo else np.full(ok.sum(), 0.5)
        colours[ok, :3] = ramp(1 - t if reverse else t)
    if highlight is not None:
        colours[highlight, :3] = 0.55 * colours[highlight, :3] + 0.45
    return colours


def colour_raster(index: np.ndarray, colours: np.ndarray) -> np.ndarray:
    """Flat RGBA float32 texture data for a ward-index raster and per-ward colours."""
    lut = np.vstack([colours, OUTSIDE]).astype(np.float32)  # index -1 picks the last row
    return lut[index].ravel()


def ward_at(wards: gpd.GeoDataFrame, x: float, y: float) -> int | None:
    """WardNo of the ward containing (x, y), or None."""
    hit = np.flatnonzero(shapely.contains_xy(wards.geometry.values, x, y))
    return int(wards["WardNo"].iloc[hit[0]]) if len(hit) else None


@dataclass
class MuniLayers:
    """Everything the municipality map draws that does not depend on a run."""
    wards: gpd.GeoDataFrame          # simplified, with WardNo, WardID, area_km2, buildings, label_x/y
    grid: gpd.GeoDataFrame           # gridfinder: mapped and predicted lines, with `source`
    ward_index: np.ndarray
    bounds: tuple
    outline_xy: tuple
    grid_mapped_xy: tuple
    grid_predicted_xy: tuple

    @property
    def ward_numbers(self) -> list[int]:
        return sorted(int(w) for w in self.wards["WardNo"])

    def row(self, ward_no: int) -> pd.Series:
        return self.wards.set_index("WardNo").loc[ward_no]


def load_layers(cfg: PilotConfig) -> MuniLayers:
    """Ward map layers. Always uses gridfinder, the only source with predicted lines."""
    from quackathon.screening import Municipality

    muni = Municipality.load(replace(cfg, grid_source="gridfinder"))
    wards = muni.wards[["WardNo", "WardID", "geometry"]].copy()
    wards["area_km2"] = wards.geometry.area / 1e6
    wards["buildings"] = wards["WardNo"].map(muni.buildings["WardNo"].value_counts()).fillna(0).astype(int)
    wards["geometry"] = wards.geometry.simplify(SIMPLIFY_M)
    label = wards.geometry.representative_point()
    wards["label_x"], wards["label_y"] = label.x, label.y
    wards = wards.sort_values("WardNo").reset_index(drop=True)

    grid = load_grid(wards, replace(cfg, grid_source="gridfinder"))
    grid = grid.clip(wards.buffer(5_000).union_all())
    predicted = is_predicted(grid)
    index, bounds = ward_raster(wards)
    return MuniLayers(wards, grid, index, bounds, lines_xy(wards.geometry),
                      lines_xy(grid.geometry[~predicted]), dash_segments(grid.geometry[predicted]))
