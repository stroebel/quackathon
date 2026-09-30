"""Building footprints from Overture Maps, queried in place on S3 with DuckDB.

Overture merges OSM, Microsoft ML and Google Open Buildings footprints, which gives
far better rural coverage here than OSM alone (~12k vs ~0.5k buildings in ward 1).
"""

from pathlib import Path

import duckdb
import geopandas as gpd

from quackathon.config import METRIC_CRS

OVERTURE_RELEASE = "2026-09-23.1"
BUILDINGS_URL = f"s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}/theme=buildings/type=building/*"


def fetch_building_centroids(area: gpd.GeoDataFrame, cache_dir: Path) -> gpd.GeoDataFrame:
    """Centroids of every building in `area`'s bounding box, cached as Parquet.

    The remote scan takes minutes, so callers should fetch a large area (e.g. the
    whole municipality) once and clip locally.
    """
    west, south, east, north = area.to_crs("EPSG:4326").total_bounds
    cache = cache_dir / f"overture_buildings_{OVERTURE_RELEASE}_{west:.3f}_{south:.3f}_{east:.3f}_{north:.3f}.parquet"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect()
        con.sql("SET enable_progress_bar = false; INSTALL spatial; LOAD spatial; "
                "INSTALL httpfs; LOAD httpfs; SET s3_region = 'us-west-2';")
        con.sql(f"""
            COPY (
                SELECT id,
                       (bbox.xmin + bbox.xmax) / 2 AS lon,
                       (bbox.ymin + bbox.ymax) / 2 AS lat,
                       ST_Area_Spheroid(ST_FlipCoordinates(geometry)) AS area_m2,
                       sources[1].dataset AS source
                FROM read_parquet('{BUILDINGS_URL}', hive_partitioning = 1)
                WHERE bbox.xmin BETWEEN {west} AND {east}
                  AND bbox.ymin BETWEEN {south} AND {north}
            ) TO '{cache}' (FORMAT parquet)
        """)
    df = duckdb.sql(f"SELECT * FROM read_parquet('{cache}')").df()
    return gpd.GeoDataFrame(
        df.drop(columns=["lon", "lat"]), geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
    ).to_crs(METRIC_CRS)
