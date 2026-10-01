"""OpenStreetMap extracts via Overpass, cached to disk.

OSM coverage of the distribution network in rural Eastern Cape is incomplete, so
"distance from grid" here means distance from the *mapped* grid.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, Point

from quackathon.config import METRIC_CRS

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def _overpass(query: str, cache: Path, attempts: int = 3) -> list[dict]:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(_overpass_request(query, attempts)))
    return json.loads(cache.read_text())["elements"]


def _overpass_request(query: str, attempts: int) -> dict:
    """Public Overpass servers often time out under load; retry across mirrors."""
    data = urllib.parse.urlencode({"data": query}).encode()
    errors = []
    for attempt in range(attempts):
        for url in OVERPASS_URLS:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "quackathon/0.1"})
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    return json.load(resp)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                errors.append(f"{url}: {e}")
        time.sleep(10 * (attempt + 1))
    raise RuntimeError("Overpass unavailable:\n" + "\n".join(errors))


def _bbox(area: gpd.GeoDataFrame, buffer_m: float) -> str:
    west, south, east, north = area.buffer(buffer_m).to_crs("EPSG:4326").total_bounds
    return f"{south},{west},{north},{east}"


def fetch_power_lines(area: gpd.GeoDataFrame, cache_dir: Path, buffer_m: float = 20_000) -> gpd.GeoDataFrame:
    """Power lines around `area`. The buffer catches grid just outside the boundary."""
    bbox = _bbox(area, buffer_m)
    query = f'[out:json][timeout:180];way["power"~"^(line|minor_line|cable)$"]({bbox});out geom;'
    elements = _overpass(query, cache_dir / f"power_{bbox}.json")
    rows = [
        {"osm_id": e["id"], "power": e["tags"].get("power"), "voltage": e["tags"].get("voltage"),
         "geometry": LineString([(p["lon"], p["lat"]) for p in e["geometry"]])}
        for e in elements if e["type"] == "way" and len(e.get("geometry", [])) >= 2
    ]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326").to_crs(METRIC_CRS)


def fetch_buildings(area: gpd.GeoDataFrame, cache_dir: Path) -> gpd.GeoDataFrame:
    """Building centroids within `area`."""
    bbox = _bbox(area, 0)
    query = f'[out:json][timeout:240];way["building"]({bbox});out center;'
    elements = _overpass(query, cache_dir / f"buildings_{bbox}.json")
    pts = gpd.GeoDataFrame(
        {"osm_id": [e["id"] for e in elements]},
        geometry=[Point(e["center"]["lon"], e["center"]["lat"]) for e in elements],
        crs="EPSG:4326",
    ).to_crs(METRIC_CRS)
    return pts[pts.within(area.union_all())].reset_index(drop=True)


ROAD_CLASSES = "trunk|primary|secondary|tertiary|unclassified|track"


def fetch_roads(area: gpd.GeoDataFrame, cache_dir: Path) -> gpd.GeoDataFrame:
    """Roads and tracks within the bounding box of `area`, for site access."""
    bbox = _bbox(area, 0)
    query = f'[out:json][timeout:240];way["highway"~"^({ROAD_CLASSES})$"]({bbox});out geom;'
    elements = _overpass(query, cache_dir / f"roads_{bbox}.json")
    rows = [
        {"osm_id": e["id"], "highway": e["tags"].get("highway"),
         "geometry": LineString([(p["lon"], p["lat"]) for p in e["geometry"]])}
        for e in elements if e["type"] == "way" and len(e.get("geometry", [])) >= 2
    ]
    return gpd.GeoDataFrame(rows, columns=["osm_id", "highway", "geometry"], geometry="geometry",
                            crs="EPSG:4326").to_crs(METRIC_CRS)


def fetch_facilities(area: gpd.GeoDataFrame, cache_dir: Path) -> gpd.GeoDataFrame:
    """Clinics, hospitals and schools within the bounding box of `area`: possible anchor loads."""
    bbox = _bbox(area, 0)
    query = (f'[out:json][timeout:180];(nwr["amenity"~"^(clinic|hospital|doctors|school)$"]({bbox});'
             f'nwr["healthcare"]({bbox}););out center;')
    elements = _overpass(query, cache_dir / f"facilities_{bbox}.json")
    rows = []
    for e in elements:
        point = (e["lon"], e["lat"]) if e["type"] == "node" else (e.get("center", {}).get("lon"),
                                                                    e.get("center", {}).get("lat"))
        if None in point:
            continue
        tags = e.get("tags", {})
        kind = tags.get("amenity") or tags.get("healthcare")
        rows.append({"osm_id": e["id"], "kind": "school" if kind == "school" else "health",
                     "name": tags.get("name", ""), "geometry": Point(point)})
    return gpd.GeoDataFrame(rows, columns=["osm_id", "kind", "name", "geometry"], geometry="geometry",
                            crs="EPSG:4326").to_crs(METRIC_CRS)
