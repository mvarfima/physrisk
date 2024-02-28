""" Test asset impact calculations."""

import test.data.hazard_model_store as hms
from test.base_test import TestWithCredentials
from test.data.hazard_model_store import TestData, ZarrStoreMocker
from typing import Sequence

import numpy as np
import pandas as pd
import os
import physrisk.data.static.world as wd


from physrisk import requests
from physrisk.api.v1.impact_req_resp import RiskMeasureKey, RiskMeasuresHelper
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel import calculation
from physrisk.kernel.assets import RealEstateAsset, ThermalPowerGeneratingAsset
from physrisk.kernel.calculation import get_default_vulnerability_models
from physrisk.kernel.hazards import (
    CoastalInundation,
    RiverineInundation,
    ChronicWind,
    Fire,
    WaterStress,
    Landslide,
    Subsidence,
)


from physrisk.vulnerability_models.thermal_power_generation_models_ECB import (
    ThermalPowerGenerationCoastalInundationModel,
    ThermalPowerGenerationRiverineInundationModel,
    SevereConvectiveWindstormModel,
    HighFireModel,
    WaterstressModel,
    LandslideModel,
    SubsidenceModel,
)


from physrisk.kernel.risk import AssetLevelRiskModel, MeasureKey
from physrisk.requests import _create_risk_measures
from physrisk.risk_models.risk_models import ThermalPowerPlantsRiskMeasures


class TestRiskModels(TestWithCredentials):
    def test_risk_indicator_model(self):

        # assets = self._create_assets_real_estate()
        # hazard_model = self._create_hazard_model(scenarios, years)
        assets = self._create_assets_power_plants()
        
        scenario = 'rcp45' 
        vul_models_dict = {
            '2040':[WaterstressModel()],
            '2050':[ThermalPowerGenerationCoastalInundationModel(), 
                    ThermalPowerGenerationRiverineInundationModel(),
                    SevereConvectiveWindstormModel(),
                    HighFireModel()],
        }

        for year, vulnerability_models in vul_models_dict.items():

            vulnerability_models = {
                ThermalPowerGeneratingAsset:vulnerability_models
            }
            hazard_model = calculation.get_default_hazard_model()

            model = AssetLevelRiskModel(
                hazard_model, vulnerability_models, {ThermalPowerGeneratingAsset: ThermalPowerPlantsRiskMeasures()}
            )
            measure_ids_for_asset, definitions = model.populate_measure_definitions(assets)
            _, measures = model.calculate_risk_measures(assets, prosp_scens=[scenario], years=[year])
            risk_measures = _create_risk_measures(measures, measure_ids_for_asset, definitions, assets, [scenario], [year])

        # how to get a score using the MeasureKey
        # measure = measures[MeasureKey(assets[0], scenarios[0], years[0], RiverineInundation)]
        # score = measure.score
        # measure_0 = measure.measure_0
        # np.testing.assert_allclose([measure_0], [0.0])

        # packing up the risk measures, e.g. for JSON transmission:
        # risk_measures = _create_risk_measures(measures, measure_ids_for_asset, definitions, assets, scenarios, years)
        # we still have a key, but no asset:
        # key = RiskMeasureKey(
        #     hazard_type="ChronicWind",
        #     scenario_id=scenarios[0],
        #     year=str(years[0]),
        #     measure_id=risk_measures.score_based_measure_set_defn.measure_set_id,
        # )
        # item = next(m for m in risk_measures.measures_for_assets if m.key == key)
        # score2 = item.scores[0]
        # measure_0_2 = item.measures_0[0]
        # assert score == score2
        # assert measure_0 == measure_0_2

        # helper = RiskMeasuresHelper(risk_measures)
        # asset_scores, measures, definitions = helper.get_measure("RiverineInundation", scenarios[0], years[0])
        # label, description = helper.get_score_details(asset_scores[0], definitions[0])
        # assert asset_scores[0] == 4

    def _create_assets_real_estate(self):
        assets = [
            RealEstateAsset(TestData.latitudes[0], TestData.longitudes[0], location="Europe", type="Buildings/Industrial")
            for i in range(2)
        ]
        return assets
    
    def _create_assets_power_plants(self):
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

        return assets

    def _create_assets_json(self, assets: Sequence[RealEstateAsset]):
        assets_dict = {
            "items": [
                {
                    "asset_class": type(asset).__name__,
                    "type": asset.type,
                    "location": asset.location,
                    "longitude": asset.longitude,
                    "latitude": asset.latitude,
                }
                for asset in assets
            ],
        }
        return assets_dict

    def _create_hazard_model(self, scenarios, years):
        source_paths = get_default_source_paths()

        def sp_riverine(scenario, year):
            return source_paths[RiverineInundation](indicator_id="flood_depth", scenario=scenario, year=year)

        def sp_coastal(scenario, year):
            return source_paths[CoastalInundation](indicator_id="flood_depth", scenario=scenario, year=year)

        def sp_wind(scenario, year):
            return source_paths[ChronicWind](indicator_id="wind25", scenario=scenario, year=year)

        mocker = ZarrStoreMocker()
        return_periods = hms.inundation_return_periods()
        flood_histo_curve = np.array([0.0596, 0.333, 0.505, 0.715, 0.864, 1.003, 1.149, 1.163, 1.163])
        flood_projected_curve = np.array([0.0596, 0.333, 0.605, 0.915, 1.164, 1.503, 1.649, 1.763, 1.963])

        for path in [sp_riverine("historical", 1980), sp_coastal("historical", 1980)]:
            mocker.add_curves_global(path, TestData.longitudes, TestData.latitudes, return_periods, flood_histo_curve)

        for path in [sp_riverine("rcp8p5", 2050), sp_coastal("rcp8p5", 2050)]:
            mocker.add_curves_global(
                path, TestData.longitudes, TestData.latitudes, return_periods, flood_projected_curve
            )

        mocker.add_curves_global(
            sp_wind("historical", -1),
            TestData.longitudes,
            TestData.latitudes,
            TestData.wind_return_periods,
            TestData.wind_intensities_1,
        )
        mocker.add_curves_global(
            sp_wind("rcp8p5", 2050),
            TestData.longitudes,
            TestData.latitudes,
            TestData.wind_return_periods,
            TestData.wind_intensities_2,
        )

        return ZarrHazardModel(source_paths=get_default_source_paths(), store=mocker.store)

    def test_via_requests(self):
        scenarios = ["ssp585"]
        years = [2050]

        assets = self._create_assets()
        # hazard_model = ZarrHazardModel(source_paths=get_default_source_paths())
        hazard_model = self._create_hazard_model(scenarios, years)

        request_dict = {
            "assets": self._create_assets_json(assets),
            "include_asset_level": False,
            "include_measures": True,
            "include_calc_details": False,
            "years": years,
            "scenarios": scenarios,
        }

        request = requests.AssetImpactRequest(**request_dict)
        response = requests._get_asset_impacts(
            request,
            hazard_model,
            vulnerability_models=get_default_vulnerability_models(),
        )
        res = next(
            ma for ma in response.risk_measures.measures_for_assets if ma.key.hazard_type == "RiverineInundation"
        )
        np.testing.assert_allclose(res.measures_0, [0.89306593179, 0.89306593179])
        # json_str = json.dumps(response.model_dump(), cls=NumpyArrayEncoder)

