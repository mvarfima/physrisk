

import os
import numpy as np
import pandas as pd
from typing import List

import physrisk.api.v1.common
import physrisk.data.static.world as wd
from physrisk import calculate_impacts
from physrisk.kernel import Asset, PowerGeneratingAsset
from physrisk.kernel.assets import IndustrialActivity, RealEstateAsset
from physrisk.kernel.hazard_model import HazardEventDataResponse
from physrisk.models.power_generating_asset_models import InundationModel
from physrisk.utils.lazy import lazy_import

pd = lazy_import("pandas")








def api_assets(assets: List[Asset]):
    items = [
        physrisk.api.v1.common.Asset(
            asset_class=type(a).__name__,
            type=getattr(a, "type") if hasattr(a, "type") else None,
            location=getattr(a, "location") if hasattr(a, "location") else None,
            latitude=a.latitude,
            longitude=a.longitude,
        )
        for a in assets
    ]
    return physrisk.api.v1.common.Assets(items=items)









base_path_exp = os.path.join(os.getenv("physical_risk_database"), 'exposure')




exp_df_name = 'global_power_plant_database_v_1_3/global_power_plant_database.csv'
exp_df = pd.read_csv(os.path.join(base_path_exp, exp_df_name))
exp_df = exp_df[exp_df.country == 'ESP']
exp_df = exp_df[exp_df.latitude > 25]

longitudes = np.array(exp_df["longitude"])
latitudes = np.array(exp_df["latitude"])
primary_fuel = np.array(exp_df["primary_fuel"])
generation = np.array(exp_df["estimated_generation_gwh_2017"])

_, continents = wd.get_countries_and_continents(latitudes=latitudes, longitudes=longitudes)

# Power generating assets that are of interest
assets = [
    PowerGeneratingAsset(lat, lon, generation=gen, primary_fuel=prim_fuel, location=continent, type=prim_fuel)
    for lon, lat, gen, prim_fuel, continent in zip(longitudes, latitudes, generation, primary_fuel, continents)
]




detailed_results = calculate_impacts(assets, scenario="ssp585", year=2030)
keys = list(detailed_results.keys())
# detailed_results[keys[0]].impact.to_exceedance_curve()
means = np.array([detailed_results[key].impact.mean_impact() for key in keys])
interesting = [k for (k, m) in zip(keys, means) if m > 0]
assets_out = api_assets(item[0] for item in interesting[0:10])
with open("assets_example_power_generating_small.json", "w") as f:
    f.write(assets_out.json(indent=4))