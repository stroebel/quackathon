"""Ward boundaries from the Municipal Demarcation Board 2026 shapefile."""

import geopandas as gpd

from quackathon.config import METRIC_CRS, MUNICIPALITY_CODE, PilotConfig


def load_wards(cfg: PilotConfig) -> gpd.GeoDataFrame:
    shp = cfg.shapefile_dir / "MDBWards2026" / "MDBWards2026.shp"
    wards = gpd.read_file(shp)
    wards = wards[wards["CAT_B"] == MUNICIPALITY_CODE]
    return wards.to_crs(METRIC_CRS).reset_index(drop=True)


def load_ward(cfg: PilotConfig) -> gpd.GeoDataFrame:
    wards = load_wards(cfg)
    ward = wards[wards["WardNo"] == cfg.ward_no]
    if ward.empty:
        raise ValueError(f"ward {cfg.ward_no} not in {MUNICIPALITY_CODE} (1-{len(wards)})")
    return ward.reset_index(drop=True)
