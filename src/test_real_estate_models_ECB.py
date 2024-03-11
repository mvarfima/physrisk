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

# for house in houses['items']:

#     house['type'] = 'Buildings/Residential'
#     house['latitude'] = float(house['latitude'])
#     house['longitude'] = float(house['longitude'])

# f = open('housing_kaggle_prueba.json', 'w')
# json.dump(houses, f)
# f.close()

asset_df = pd.DataFrame(houses['items'])

longitudes = asset_df.longitude.values.astype(float)
latitudes = asset_df.latitude.values.astype(float)
types = np.array(['Buildings/Residential']*len(latitudes))

countries, continents = wd.get_countries_and_continents(latitudes=latitudes, longitudes=longitudes)

# houses_spain = dict()
# houses_spain['items'] = []
# for i_ in range(len(houses['items'])):
#     if countries[i_]  == 'Spain':
#         houses_spain['items'].append(houses['items'][i_])

# for house_s in houses_spain['items']:
#     house_s['type'] = 'Buildings/Residential'
#     house_s['latitude'] = float(house_s['latitude'])
#     house_s['longitude'] = float(house_s['longitude'])

# houses_spain_reduced = houses_spain
# houses_spain_reduced['items'] = [house for house in houses_spain['items'][:10]]

# f = open('housing_kaggle_spain_reduced.json', 'w')
# json.dump(houses_spain_reduced, f)
# f.close()

# Power generating assets that are of interest
# assets = []
# for latitude, longitude, type_, continent in zip(
#     latitudes,
#     longitudes,
#     types,
#     continents,
#     ):
#     if type(continent) == str:
#         assets.append(
#             RealEstateAsset(latitude, longitude, type=type_, location=continent) 
#         )

assets = [
    RealEstateAsset(latitude, longitude, type=type_, location='Europe') 

    for latitude, longitude, type_ in zip(
        latitudes,
        longitudes,
        types,
        )
]

scenario = 'historical' # "rcp8p5" 
vul_models_dict = {
    '2005':[CoolingModel()],
    '1980':[RealEstateCoastalInundationModel(), RealEstateRiverineInundationModel()],
    '2010':[GenericTropicalCycloneModel()],
}

out = []
for year, vulnerability_models in vul_models_dict.items():

    vulnerability_models = {
        RealEstateAsset:vulnerability_models
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
        
    # out.append  (
    #     {
    #         "asset": type(result.asset).__name__,
    #         "type": getattr(result.asset, "type") if hasattr(result.asset, "type") else None,
    #         "location": getattr(result.asset, "location") if hasattr(result.asset, "location") else None,
    #         "latitude": result.asset.latitude,
    #         "longitude": result.asset.longitude,
    #         "impact_mean": results[key].impact.mean_impact(),
    #         "hazard_type": results[key].impact.hazard_type.__name__ if results[key].impact.hazard_type.__name__ !='type' else 'Wind',
    #         'scenario': scenario,
    #         'year': int(year),
    #     }
    #     for result, key in zip(results, results.keys())
    # )

pd.DataFrame.from_dict(out).to_csv(
    os.path.join("real_estate_spain_" + scenario + "_" + str(year) + ".csv")
)

scenario = "ssp585" 
year = 2050

results = calculate_impacts(assets, hazard_model, vulnerability_models2, scenario=scenario, year=year)
out1 = [
    {
        "asset": type(result.asset).__name__,
        "type": getattr(result.asset, "type") if hasattr(result.asset, "type") else None,
        "location": getattr(result.asset, "location") if hasattr(result.asset, "location") else None,
        "latitude": result.asset.latitude,
        "longitude": result.asset.longitude,
        "impact_mean": results[key].impact.mean_impact(),
        "hazard_type": results[key].impact.hazard_type.__name__ if results[key].impact.hazard_type.__name__ !='type' else 'Wind',
        'scenario': scenario,
        'year': year,
    }
    for result, key in zip(results, results.keys())
]

pd.DataFrame.from_dict(out1).to_csv(
    os.path.join("real_estate_spain_" + scenario + "_" + str(year) + ".csv")
)







# """ Test asset impact calculations."""

# import unittest
# from test.data.hazard_model_store import TestData, mock_hazard_model_store_inundation

# import numpy as np

# from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
# from physrisk.hazard_models.core_hazards import get_default_source_paths
# from physrisk.kernel.assets import RealEstateAsset
# from physrisk.kernel.hazards import CoastalInundation, RiverineInundation
# from physrisk.kernel.impact import ImpactKey, calculate_impacts
# from physrisk.vulnerability_models.real_estate_models import (
#     RealEstateCoastalInundationModel,
#     RealEstateRiverineInundationModel,
# )


# class TestRealEstateModels():
#     """Tests RealEstateInundationModel."""

#     def test_real_estate_model_details(self):
#         curve = np.array([0.0596, 0.333, 0.505, 0.715, 0.864, 1.003, 1.149, 1.163, 1.163])
#         store = mock_hazard_model_store_inundation(TestData.longitudes, TestData.latitudes, curve)
#         hazard_model = ZarrHazardModel(source_paths=get_default_source_paths(), store=store)

#         # location="Europe", type="Buildings/Residential"
#         assets = [
#             RealEstateAsset(lat, lon, location="Asia", type="Buildings/Industrial")
#             for lon, lat in zip(TestData.longitudes[0:1], TestData.latitudes[0:1])
#         ]

#         scenario = "rcp8p5"
#         year = 2080

#         vulnerability_models = {RealEstateAsset: [RealEstateRiverineInundationModel()]}

#         results = calculate_impacts(assets, hazard_model, vulnerability_models, scenario=scenario, year=year)

#         hazard_bin_edges = results[ImpactKey(assets[0], RiverineInundation)].event.intensity_bin_edges
#         hazard_bin_probs = results[ImpactKey(assets[0], RiverineInundation)].event.prob

#         # check one:
#         # the probability of inundation greater than 0.505m in a year is 1/10.0
#         # the probability of inundation greater than 0.333m in a year is 1/5.0
#         # therefore the probability of an inundation between 0.333 and 0.505 in a year is 1/5.0 - 1/10.0
#         np.testing.assert_almost_equal(hazard_bin_edges[1:3], np.array([0.333, 0.505]))
#         np.testing.assert_almost_equal(hazard_bin_probs[1], 0.1)

#         # check that intensity bin edges for vulnerability matrix are same as for hazard
#         vulnerability_intensity_bin_edges = results[
#             ImpactKey(assets[0], RiverineInundation)
#         ].vulnerability.intensity_bins
#         np.testing.assert_almost_equal(vulnerability_intensity_bin_edges, hazard_bin_edges)

#         # check the impact distribution the matrix is size [len(intensity_bins) - 1, len(impact_bins) - 1]
#         cond_probs = results[ImpactKey(assets[0], RiverineInundation)].vulnerability.prob_matrix[1, :]
#         # check conditional prob for inundation intensity 0.333..0.505
#         mean, std = np.mean(cond_probs), np.std(cond_probs)
#         np.testing.assert_almost_equal(cond_probs.sum(), 1)
#         np.testing.assert_allclose([mean, std], [0.09090909, 0.08184968], rtol=1e-6)

#         # probability that impact occurs between impact bin edge 1 and impact bin edge 2
#         prob_impact = np.dot(
#             hazard_bin_probs, results[ImpactKey(assets[0], RiverineInundation)].vulnerability.prob_matrix[:, 1]
#         )
#         np.testing.assert_almost_equal(prob_impact, 0.19350789547968042)

#         # no check with pre-calculated values for others:
#         np.testing.assert_allclose(
#             results[ImpactKey(assets[0], RiverineInundation)].impact.prob,
#             np.array(
#                 [
#                     0.02815762,
#                     0.1935079,
#                     0.11701139,
#                     0.06043065,
#                     0.03347816,
#                     0.02111368,
#                     0.01504522,
#                     0.01139892,
#                     0.00864469,
#                     0.00626535,
#                     0.00394643,
#                 ]
#             ),
#             rtol=2e-6,
#         )

# aux = TestRealEstateModels()
# aux.test_real_estate_model_details()