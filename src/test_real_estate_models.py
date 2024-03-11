""" Test asset impact calculations."""

import os
import json
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
from physrisk.vulnerability_models.real_estate_models import RealEstateCoastalInundationModel, RealEstateRiverineInundationModel, GenericTropicalCycloneModel, CoolingModel

import pandas as pd

file_ = 'housing_kaggle_spain.json'
f = open(file_, 'r')
houses = json.load(f)
f.close()

asset_df = pd.DataFrame(houses['items'])

longitudes = asset_df.longitude
latitudes = asset_df.latitude
types = asset_df.type
asset_names = asset_df.address
asset_prices = asset_df.price

countries, continents = wd.get_countries_and_continents(latitudes=latitudes, longitudes=longitudes)

# houses_spain = dict()
# houses_spain['items'] = []
# for i_ in range(len(houses['items'])):
#     if countries[i_]  == 'Spain':
#         houses_spain['items'].append(houses['items'][i_])

# with open('housing_kaggle_spain.json', 'w') as fp:
#     json.dump(houses_spain, fp)

assets = [
    RealEstateAsset(latitude, longitude, type=type_, location='Europe') 

    for latitude, longitude, type_ in zip(
        latitudes,
        longitudes,
        types,
        )
]

for i, asset_name in enumerate(asset_names):
    assets[i].__dict__.update({'asset_name':asset_name})

for i, asset_price in enumerate(asset_prices):
    assets[i].__dict__.update({'asset_price':asset_price})



# GENERAR LTV
entrada = np.random.choice(a=[10,20,30], size = len(asset_prices), p=[0.1, 0.2, 0.7])
loan_amounts = (1 - entrada / 100) * asset_prices.to_numpy().astype(float)

for i, loan_amount in enumerate(loan_amounts):
    assets[i].__dict__.update({'asset_loan_amount':loan_amount})



hazard_indicator_dict = {
    'Wind':'max_speed',
    'CoastalInundation': 'flood_depth',
    'RiverineInundation': 'flood_depth',
    'ChronicHeat':'"mean_degree_days_above',
}

# scenario = 'historical'
# vul_models_dict = {
#     '2005':[CoolingModel()],
#     '1980':[RealEstateCoastalInundationModel(), RealEstateRiverineInundationModel()],
#     '2010':[GenericTropicalCycloneModel()],
# }

out = []

vul_models_dict = {
    'ssp119_2050':[GenericTropicalCycloneModel()],

    'ssp126_2030':[CoolingModel()],
    'ssp126_2040':[CoolingModel()],
    'ssp126_2050':[CoolingModel()],

    'ssp245_2030':[CoolingModel()],
    'ssp245_2040':[CoolingModel()],
    'ssp245_2050':[CoolingModel(), GenericTropicalCycloneModel()],

    'ssp585_2030':[CoolingModel()],
    'ssp585_2040':[CoolingModel()],
    'ssp585_2050':[CoolingModel(), GenericTropicalCycloneModel()],
}


for scenario_year, vulnerability_models in vul_models_dict.items():

    scenario, year = scenario_year.split('_')

    vulnerability_models = {
        RealEstateAsset:vulnerability_models
    }
    hazard_model = calculation.get_default_hazard_model()
    results = calculate_impacts(assets, hazard_model, vulnerability_models, scenario=scenario, year=int(year))

    for result, key in zip(results, results.keys()):

        hazard_type_ = results[key].impact.hazard_type.__name__ if results[key].impact.hazard_type.__name__ !='type' else 'Wind'
        out.append  (
            {
                "asset_name": result.asset.asset_name,
                "asset_price": result.asset.asset_price,
                "asset_loan_amount": result.asset.asset_loan_amount,
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
                "impact_distr_bin_edges":";".join(results[key].impact.impact_bins.astype(str)),
                "impact_distr_p":";".join(results[key].impact.prob.astype(str)),
                "impact_exc_exceed_p":"0;0",
                "impact_exc_values":"0;0",
                "vuln_model_designer":'OS-C-RealEstate-LTV',
            }
    )
        
vul_models_dict = {
    'rcp4p5_2030':[RealEstateCoastalInundationModel(), RealEstateRiverineInundationModel()],
    'rcp4p5_2050':[RealEstateCoastalInundationModel(), RealEstateRiverineInundationModel()],
    'rcp4p5_2080':[RealEstateCoastalInundationModel(), RealEstateRiverineInundationModel()],
    'rcp8p5_2030':[RealEstateCoastalInundationModel(), RealEstateRiverineInundationModel()],
    'rcp8p5_2050':[RealEstateCoastalInundationModel(), RealEstateRiverineInundationModel()],
    'rcp8p5_2080':[RealEstateCoastalInundationModel(), RealEstateRiverineInundationModel()],
}

for scenario_year, vulnerability_models in vul_models_dict.items():

    scenario, year = scenario_year.split('_')

    vulnerability_models = {
        RealEstateAsset:vulnerability_models
    }
    hazard_model = calculation.get_default_hazard_model()
    results = calculate_impacts(assets, hazard_model, vulnerability_models, scenario=scenario, year=int(year))

    for result, key in zip(results, results.keys()):
        out.append  (
            {
                "asset_name": result.asset.asset_name,
                "asset_price": result.asset.asset_price,
                "asset_loan_amount": result.asset.asset_loan_amount,
                "asset_subtype": getattr(result.asset, "type") if hasattr(result.asset, "type") else None,
                "latitude": result.asset.latitude,
                "longitude": result.asset.longitude,
                "hazard_type": results[key].impact.hazard_type.__name__ if results[key].impact.hazard_type.__name__ !='type' else 'Wind',
                'indicator_id': hazard_indicator_dict[results[key].impact.hazard_type.__name__],
                'display_name': 'display_name_vacio',
                'model':'model_vacio',
                'scenario':scenario,
                'year':int(year),
                'return_periods':{"0":"0"},
                'parameter':0,
                "impact_mean": results[key].impact.mean_impact(),
                "impact_distr_bin_edges": ";".join(results[key].impact.impact_bins.astype(str)),
                "impact_distr_p": ";".join(results[key].impact.prob.astype(str)),
                "impact_exc_exceed_p":"0;0",
                "impact_exc_values":"0;0",
                "vuln_model_designer":'OS-C-RealEstate-LTV',
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
    os.path.join("real_estate_spain_OSC-LTV_all.csv"),
    index=False
)