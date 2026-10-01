import json

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from quackathon.app.wardmap import (NO_VALUE, colour_raster, dash_segments, grid_distances, ward_at, ward_colours,
                                    ward_raster)
from quackathon.config import METRIC_CRS, PilotConfig
from quackathon.osm import _bbox, fetch_facilities, fetch_roads

HAS_SHAPEFILE = (PilotConfig().shapefile_dir / "MDBWards2026").exists()


def toy_wards() -> gpd.GeoDataFrame:
    """An L-shaped (concave) ward 1 and a square ward 2 beside it."""
    l_shape = Polygon([(0, 0), (2000, 0), (2000, 1000), (1000, 1000), (1000, 2000), (0, 2000)])
    square = Polygon([(2000, 0), (3000, 0), (3000, 1000), (2000, 1000)])
    return gpd.GeoDataFrame({"WardNo": [1, 2]}, geometry=[l_shape, square], crs=METRIC_CRS)


def test_ward_raster_and_ward_at_agree_on_a_concave_ward():
    wards = toy_wards()
    index, bounds = ward_raster(wards, width=300)
    assert bounds == tuple(wards.total_bounds)
    assert index.shape == (200, 300)
    x0, _, x1, y1 = bounds
    px = (x1 - x0) / 300

    def pixel(x, y):
        return index[int((y1 - y) / px), int((x - x0) / px)]

    assert pixel(500, 1500) == 0 and ward_at(wards, 500, 1500) == 1
    assert pixel(2500, 500) == 1 and ward_at(wards, 2500, 500) == 2
    # The notch of the L is inside the bounding box but outside every ward.
    assert pixel(2000, 1500) == -1 and ward_at(wards, 2000, 1500) is None


def test_colour_raster_maps_indices_through_ward_colours():
    colours = ward_colours(np.array([1.0, np.nan]))
    assert np.allclose(colours[1], NO_VALUE)
    index = np.array([[0, 1], [-1, 0]], np.int16)
    data = colour_raster(index, colours).reshape(2, 2, 4)
    assert data.dtype == np.float32
    assert np.allclose(data[0, 0], colours[0]) and np.allclose(data[0, 1], NO_VALUE)
    assert data[1, 0, 3] == 0  # outside is transparent


def test_ward_colours_reverse_and_highlight():
    plain = ward_colours(np.array([0.0, 1.0]))
    reverse = ward_colours(np.array([0.0, 1.0]), reverse=True)
    assert np.allclose(plain[0], reverse[1])
    lit = ward_colours(np.array([0.0, 1.0]), highlight=np.array([True, False]))
    assert (lit[0, :3] > plain[0, :3]).all() and np.allclose(lit[1], plain[1])


def test_dash_segments_cover_the_dash_fraction_of_each_line():
    xs, ys = dash_segments([LineString([(0, 0), (6500, 0)])], dash_m=400, gap_m=250)
    starts, ends = np.array(xs[0::2]), np.array(xs[1::2])
    assert len(xs) == 2 * len(starts) == 20
    assert np.allclose(ends - starts, 400)
    assert set(ys) == {0.0}


def test_grid_distances_split_mapped_and_predicted():
    grid = gpd.GeoDataFrame({"source": ["openstreetmap", "gridfinder"]},
                            geometry=[LineString([(0, 0), (0, 10_000)]), LineString([(5_000, 0), (5_000, 10_000)])],
                            crs=METRIC_CRS)
    points = gpd.GeoSeries([Point(1_000, 500), Point(4_500, 500)], crs=METRIC_CRS)
    mapped, predicted = grid_distances(points, grid)
    assert np.allclose(mapped, [1_000, 4_500]) and np.allclose(predicted, [4_000, 500])
    mapped, predicted = grid_distances(points, grid.drop(columns="source"))
    assert np.allclose(mapped, [1_000, 500]) and np.isinf(predicted).all()


def test_osm_roads_and_facilities_parse_the_overpass_cache(tmp_path):
    area = gpd.GeoDataFrame(geometry=[Point(29.3, -30.9).buffer(0.05)], crs="EPSG:4326").to_crs(METRIC_CRS)
    bbox = _bbox(area, 0)
    roads = [{"type": "way", "id": 1, "tags": {"highway": "track"},
              "geometry": [{"lon": 29.30, "lat": -30.90}, {"lon": 29.31, "lat": -30.90}]},
             {"type": "way", "id": 2, "tags": {"highway": "track"}, "geometry": [{"lon": 29.3, "lat": -30.9}]}]
    facilities = [{"type": "node", "id": 3, "lon": 29.3, "lat": -30.9, "tags": {"amenity": "school", "name": "A"}},
                  {"type": "way", "id": 4, "center": {"lon": 29.31, "lat": -30.91}, "tags": {"amenity": "clinic"}},
                  {"type": "relation", "id": 5, "tags": {"healthcare": "clinic"}}]  # no centre: skipped
    (tmp_path / f"roads_{bbox}.json").write_text(json.dumps({"elements": roads}))
    (tmp_path / f"facilities_{bbox}.json").write_text(json.dumps({"elements": facilities}))

    r = fetch_roads(area, tmp_path)
    assert list(r["osm_id"]) == [1] and r.crs == METRIC_CRS
    assert 900 < r.length.iloc[0] < 1_000  # 0.01 degrees of longitude at 31 S
    f = fetch_facilities(area, tmp_path)
    assert list(f["kind"]) == ["school", "health"] and list(f["name"]) == ["A", ""]


@pytest.mark.skipif(not HAS_SHAPEFILE, reason="ward shapefile missing")
def test_screen_reports_demand_excluded_only_by_predicted_lines():
    from quackathon.screening import Municipality, screen_wards

    cfg = PilotConfig()
    df = screen_wards(Municipality.load(cfg), cfg)
    assert (df["predicted_excluded_demand"] >= -1e-9).all()
    assert df["predicted_excluded_demand"].sum() > 0  # gridfinder adds lines in this municipality
    assert np.allclose(df["far_households"], df["far_buildings"] * cfg.households_per_building)


@pytest.mark.skipif(not HAS_SHAPEFILE, reason="ward shapefile missing")
def test_load_layers_labels_fall_in_their_own_ward():
    from quackathon.app.wardmap import load_layers

    m = load_layers(PilotConfig())
    assert m.ward_numbers == list(range(1, 20))
    for ward, x, y in zip(m.wards["WardNo"], m.wards["label_x"], m.wards["label_y"]):
        assert ward_at(m.wards, x, y) == ward
    assert m.grid_mapped_xy[0] and m.grid_predicted_xy[0]
