""" Test asset impact calculations."""

import os
import unittest
from test.base_test import TestWithCredentials
from typing import List

import numpy as np

import physrisk.api.v1.common
import physrisk.data.static.world as wd
from physrisk.kernel import Asset, PowerGeneratingAsset, calculation
from physrisk.kernel.assets import IndustrialActivity, RealEstateAsset, ThermalPowerGeneratingAsset
from physrisk.kernel.hazard_model import HazardEventDataResponse
from physrisk.kernel.impact import calculate_impacts
from physrisk.utils.lazy import lazy_import
from physrisk.vulnerability_models.thermal_power_generation_models import (
    ThermalPowerGenerationCoastalInundationModel,
    ThermalPowerGenerationRiverineInundationModel,
    ThermalPowerGenerationDroughtModel,
    ThermalPowerGenerationAirTemperatureModel,
    ThermalPowerGenerationWaterTemperatureModel,
    ThermalPowerGenerationWaterStressModel,
)

import pandas as pd

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
asset_names = np.array(filtered["name"])

countries, continents = wd.get_countries_and_continents(latitudes=latitudes, longitudes=longitudes)

# Power generating assets that are of interest
assets = [
    ThermalPowerGeneratingAsset(latitude, longitude, type=primary_fuel, location=continent, capacity=capacity)
    for latitude, longitude, capacity, primary_fuel, continent in zip(
        latitudes,
        longitudes,
        capacities,
        primary_fuels,
        countries,
    )
]

for i, asset_name in enumerate(asset_names):
    assets[i].__dict__.update({'asset_name':asset_name})

hazard_indicator_dict = {
    'AirTemperature':'days_tas_above',
    'CoastalInundation': 'flood_depth',
    'RiverineInundation': 'flood_depth',
    'Drought':'months_spei3m_below_minus2',
    'WaterStress': 'water_stress_and_water_supply',
    'WaterTemperature':'weeks_water_temp_above',
}

out = []

vul_models_dict = {

    'ssp126_2020':[ThermalPowerGenerationDroughtModel()],
    'ssp126_2030':[ThermalPowerGenerationAirTemperatureModel(), ThermalPowerGenerationDroughtModel(), ThermalPowerGenerationWaterStressModel(), ThermalPowerGenerationWaterTemperatureModel()],
    'ssp126_2040':[ThermalPowerGenerationAirTemperatureModel(), ThermalPowerGenerationDroughtModel(), ThermalPowerGenerationWaterTemperatureModel()],
    'ssp126_2050':[ThermalPowerGenerationAirTemperatureModel(), ThermalPowerGenerationDroughtModel(), ThermalPowerGenerationWaterStressModel(), ThermalPowerGenerationWaterTemperatureModel()],
    'ssp126_2060':[ThermalPowerGenerationWaterTemperatureModel()],
    'ssp126_2070':[ThermalPowerGenerationWaterTemperatureModel()],
    'ssp126_2075':[ThermalPowerGenerationDroughtModel()],
    'ssp126_2080':[ThermalPowerGenerationWaterStressModel()],
    'ssp126_2090':[ThermalPowerGenerationWaterTemperatureModel()],
    'ssp126_2100':[ThermalPowerGenerationDroughtModel()],

    'ssp245_2030':[ThermalPowerGenerationAirTemperatureModel(), ThermalPowerGenerationWaterTemperatureModel()],
    'ssp245_2040':[ThermalPowerGenerationAirTemperatureModel(), ThermalPowerGenerationWaterTemperatureModel()],
    'ssp245_2050':[ThermalPowerGenerationAirTemperatureModel(), ThermalPowerGenerationWaterTemperatureModel()],
    'ssp245_2060':[ThermalPowerGenerationWaterTemperatureModel()],
    'ssp245_2070':[ThermalPowerGenerationWaterTemperatureModel()],
    'ssp245_2080':[ThermalPowerGenerationWaterTemperatureModel()],
    'ssp245_2090':[ThermalPowerGenerationWaterTemperatureModel()],

    'ssp370_2030':[ThermalPowerGenerationWaterStressModel()],
    'ssp370_2050':[ThermalPowerGenerationWaterStressModel()],
    'ssp370_2080':[ThermalPowerGenerationWaterStressModel()],

    'ssp585_2020':[ThermalPowerGenerationDroughtModel()],
    'ssp585_2030':[ThermalPowerGenerationAirTemperatureModel(), ThermalPowerGenerationDroughtModel(), ThermalPowerGenerationWaterStressModel(), ThermalPowerGenerationWaterTemperatureModel()],
    'ssp585_2040':[ThermalPowerGenerationAirTemperatureModel(), ThermalPowerGenerationDroughtModel()],
    'ssp585_2050':[ThermalPowerGenerationAirTemperatureModel(), ThermalPowerGenerationDroughtModel(), ThermalPowerGenerationWaterStressModel(), ThermalPowerGenerationWaterTemperatureModel()],
    'ssp585_2060':[ThermalPowerGenerationWaterTemperatureModel()],
    'ssp585_2070':[ThermalPowerGenerationWaterTemperatureModel()],
    'ssp585_2075':[ThermalPowerGenerationDroughtModel()],
    'ssp585_2080':[ThermalPowerGenerationWaterStressModel(), ThermalPowerGenerationWaterTemperatureModel()],
    'ssp585_2090':[ThermalPowerGenerationWaterTemperatureModel()],
    'ssp585_2100':[ThermalPowerGenerationDroughtModel()],

    'rcp4p5_2030':[ThermalPowerGenerationCoastalInundationModel(), ThermalPowerGenerationRiverineInundationModel()],
    'rcp4p5_2050':[ThermalPowerGenerationCoastalInundationModel(), ThermalPowerGenerationRiverineInundationModel()],
    'rcp4p5_2080':[ThermalPowerGenerationCoastalInundationModel(), ThermalPowerGenerationRiverineInundationModel()],

    'rcp8p5_2030':[ThermalPowerGenerationCoastalInundationModel(), ThermalPowerGenerationRiverineInundationModel()],
    'rcp8p5_2050':[ThermalPowerGenerationCoastalInundationModel(), ThermalPowerGenerationRiverineInundationModel()],
    'rcp8p5_2080':[ThermalPowerGenerationCoastalInundationModel(), ThermalPowerGenerationRiverineInundationModel()],
}


for scenario_year, vulnerability_models in vul_models_dict.items():

    scenario, year = scenario_year.split('_')

    print(scenario, year)

    vulnerability_models = {
        ThermalPowerGeneratingAsset:vulnerability_models
    }
    hazard_model = calculation.get_default_hazard_model()
    results = calculate_impacts(assets, hazard_model, vulnerability_models, scenario=scenario, year=int(year))

    for result, key in zip(results, results.keys()):

        hazard_type_ = results[key].impact.hazard_type.__name__ if results[key].impact.hazard_type.__name__ !='type' else 'Wind'
        out.append  (
            {
                "asset_name": result.asset.asset_name,
                "asset_subtype": getattr(result.asset, "type") if hasattr(result.asset, "type") else None,
                "latitude": result.asset.latitude,
                "longitude": result.asset.longitude,
                "hazard_type": hazard_type_,
                'indicator_id': hazard_indicator_dict[hazard_type_],
                'display_name': 'display_name_vacio',
                'model':'model_vacio',
                'scenario':scenario,
                'year':int(year),
                'return_periods':{"0":"0"},
                'parameter':0,
                "impact_mean": results[key].impact.mean_impact(),
                "impact_distr_bin_edges":"0;0",
                "impact_distr_p":"0;0",
                "impact_exc_exceed_p":"0;0",
                "impact_exc_values":"0;0",
                "vuln_model_designer":'OS-C-PowerPlants',
            }
    )

# asset_name
# asset_subtype
# latitude
# longitude
# hazard_type
# indicator_id
# display_name
# model
# scenario
# year
# return_periods
# parameter
# impact_mean
# impact_distr_bin_edges
# impact_distr_p
# impact_exc_exceed_p
# impact_exc_values
# vuln_model_designer

df = pd.DataFrame.from_dict(out)

df.to_csv(
    os.path.join(cache_folder, "thermal_power_generation_OSC_all.csv"),
    index=False
)
