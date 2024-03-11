
import os
import numpy as np
import physrisk.data.static.world as wd
from physrisk.kernel import calculation
from physrisk.kernel.assets import ThermalPowerGeneratingAsset
from physrisk.kernel.impact import calculate_impacts
import pandas as pd
from physrisk.vulnerability_models.thermal_power_generation_models_ECB import (
    ThermalPowerGenerationCoastalInundationModel,
    ThermalPowerGenerationRiverineInundationModel,
    SevereConvectiveWindstormModel,
    HighFireModel,
    WaterstressModel,
    LandslideModel,
    SubsidenceModel,
)

cache_folder = os.getcwd()
asset_list = pd.read_csv(os.path.join(cache_folder, "wri-all.csv"))
filtered = asset_list.loc[asset_list["primary_fuel"].isin(["Coal", "Gas", "Nuclear", "Oil"])]
filtered = filtered[-60 < filtered["latitude"]]

longitudes = np.array(filtered["longitude"])
latitudes = np.array(filtered["latitude"])

primary_fuels = np.array(
    [primary_fuel.replace(" and ", "And").replace(" ", "") for primary_fuel in filtered["primary_fuel"]]
)

# Capacity describes a maximum electric power rate.
# Generation describes the actual electricity output of the plant over a period of time.
capacities = np.array(filtered["capacity_mw"])

countries, continents = wd.get_countries_and_continents(latitudes=latitudes, longitudes=longitudes)

# Power generating assets that are of interest
# assets = [
#     ThermalPowerGeneratingAsset(latitude, longitude, type=primary_fuel, location=continent, capacity=capacity)
#     for latitude, longitude, capacity, primary_fuel, continent in zip(
#         latitudes,
#         longitudes,
#         capacities,
#         primary_fuels,
#         countries,
#     )
# ]

# assets = []
# for latitude, longitude, capacity, primary_fuel, country in zip(
#         latitudes,
#         longitudes,
#         capacities,
#         primary_fuels,
#         countries,
#     ):
#     if country in ['Spain']:
#         assets.append(
#             ThermalPowerGeneratingAsset(latitude, longitude, type=primary_fuel, location=country, capacity=capacity)
#         )

# scenario = "rcp45"
# year = 2050

# hazard_model = calculation.get_default_hazard_model()
# vulnerability_models = calculation.get_default_vulnerability_models()

# results = calculate_impacts(assets, hazard_model, vulnerability_models, scenario=scenario, year=year)
# out = [
#     {
#         "asset": type(result.asset).__name__,
#         "type": getattr(result.asset, "type") if hasattr(result.asset, "type") else None,
#         "capacity": getattr(result.asset, "capacity") if hasattr(result.asset, "capacity") else None,
#         "location": getattr(result.asset, "location") if hasattr(result.asset, "location") else None,
#         "latitude": result.asset.latitude,
#         "longitude": result.asset.longitude,
#         "impact_mean": results[key].impact.mean_impact(),
#         "hazard_type": results[key].impact.hazard_type.__name__,
#     }
#     for result, key in zip(results, results.keys())
# ]


assets = []
for latitude, longitude, capacity, primary_fuel, country in zip(
        latitudes,
        longitudes,
        capacities,
        primary_fuels,
        countries,
    ):
    if country in ['Spain']:
        assets.append(
            ThermalPowerGeneratingAsset(latitude, longitude, type=primary_fuel, location=country, capacity=capacity)
        )













scenario = 'historical' 
vul_models_dict = {
    '1980':[LandslideModel(), SubsidenceModel()]
}

out = []
for year, vulnerability_models in vul_models_dict.items():

    vulnerability_models = {
        ThermalPowerGeneratingAsset:vulnerability_models
    }
    hazard_model = calculation.get_default_hazard_model()
    results = calculate_impacts(assets, hazard_model, vulnerability_models, scenario=scenario, year=int(year))

    for result, key in zip(results, results.keys()):
        out.append  (
            {
                "asset": type(result.asset).__name__,
                "type": getattr(result.asset, "type") if hasattr(result.asset, "type") else None,
                "location": getattr(result.asset, "location") if hasattr(result.asset, "location") else None,
                "latitude": result.asset.latitude,
                "longitude": result.asset.longitude,
                "impact_mean": results[key].impact.mean_impact(),
                "hazard_type": results[key].impact.hazard_type.__name__ if results[key].impact.hazard_type.__name__ !='type' else 'Wind',
                'scenario': scenario,
                'year': int(year),
            }
    )
        
pd.DataFrame.from_dict(out).to_csv(
    os.path.join(cache_folder, "spain_ECB_thermal_power_generation_" + scenario + "_" + str(year) + ".csv")
)










scenario = 'rcp45' 
vul_models_dict = {
    '2040':[WaterstressModel()],
    '2050':[ThermalPowerGenerationCoastalInundationModel(), 
            ThermalPowerGenerationRiverineInundationModel(),
            SevereConvectiveWindstormModel(),
            HighFireModel()],
}

out = []
for year, vulnerability_models in vul_models_dict.items():

    vulnerability_models = {
        ThermalPowerGeneratingAsset:vulnerability_models
    }
    hazard_model = calculation.get_default_hazard_model()
    results = calculate_impacts(assets, hazard_model, vulnerability_models, scenario=scenario, year=int(year))

    for result, key in zip(results, results.keys()):
        out.append  (
            {
                "asset": type(result.asset).__name__,
                "type": getattr(result.asset, "type") if hasattr(result.asset, "type") else None,
                "location": getattr(result.asset, "location") if hasattr(result.asset, "location") else None,
                "latitude": result.asset.latitude,
                "longitude": result.asset.longitude,
                "impact_mean": results[key].impact.mean_impact(),
                "hazard_type": results[key].impact.hazard_type.__name__ if results[key].impact.hazard_type.__name__ !='type' else 'Wind',
                'scenario': scenario,
                'year': int(year),
            }
    )
        
pd.DataFrame.from_dict(out).to_csv(
    os.path.join(cache_folder, "spain_ECB_thermal_power_generation_" + scenario + "_" + str(year) + ".csv")
)













