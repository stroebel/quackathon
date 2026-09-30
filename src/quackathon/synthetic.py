"""Synthetic buildings inside a ward, for offline development and tests."""

import geopandas as gpd
import numpy as np
from shapely.geometry import Point


def synthetic_buildings(
    area: gpd.GeoDataFrame,
    n_villages: int = 25,
    buildings_per_village: tuple[int, int] = (10, 120),
    village_spread_m: float = 250.0,
    seed: int = 0,
) -> gpd.GeoDataFrame:
    """Gaussian blobs of buildings around random village centres within `area`."""
    rng = np.random.default_rng(seed)
    shape = area.union_all()
    minx, miny, maxx, maxy = shape.bounds

    centres: list[tuple[float, float]] = []
    while len(centres) < n_villages:
        p = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if shape.contains(Point(p)):
            centres.append(p)

    pts = np.vstack([
        rng.normal(c, village_spread_m, size=(rng.integers(*buildings_per_village), 2))
        for c in centres
    ])
    buildings = gpd.GeoDataFrame(geometry=gpd.points_from_xy(pts[:, 0], pts[:, 1]), crs=area.crs)
    return buildings[buildings.within(shape)].reset_index(drop=True)
