""" Test asset impact calculations."""
import test.data.hazard_model_store as hms
from test.base_test import TestWithCredentials
from test.data.hazard_model_store import TestData, ZarrStoreMocker
from typing import Sequence

import numpy as np

from physrisk import requests
from physrisk.api.v1.impact_req_resp import RiskMeasureKey, RiskMeasuresHelper
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel.assets import RealEstateAsset, PowerGeneratingAsset
from physrisk.kernel.calculation import get_default_vulnerability_models
from physrisk.kernel.hazards import ChronicHeat, CoastalInundation, RiverineInundation, Wind
from physrisk.kernel.risk import AssetLevelRiskModel, MeasureKey
from physrisk.requests import _create_risk_measures
from physrisk.risk_models.risk_models import RealEstateToyRiskMeasures, PowerGeneratingAssetToyRiskMeasures



from physrisk.kernel import calculation as calc
from physrisk.data.inventory import EmbeddedInventory
from physrisk.data.inventory_reader import InventoryReader
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.data.zarr_reader import ZarrReader
from physrisk.requests import Requester, _create_inventory, create_source_paths



import importlib
import json
from importlib import import_module
from pathlib import WindowsPath
from typing import Any, Dict, List, Optional, Sequence, Type, cast

import numpy as np

import physrisk.data.static.example_portfolios
from physrisk.api.v1.common import Distribution, ExceedanceCurve, VulnerabilityDistrib
from physrisk.api.v1.exposure_req_resp import AssetExposure, AssetExposureRequest, AssetExposureResponse, Exposure
from physrisk.api.v1.hazard_image import HazardImageRequest
from physrisk.data.hazard_data_provider import HazardDataHint
from physrisk.data.inventory import expand
from physrisk.data.inventory_reader import InventoryReader
from physrisk.data.zarr_reader import ZarrReader
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel.exposure import JupterExposureMeasure, calculate_exposures
from physrisk.kernel.hazards import all_hazards
from physrisk.kernel.impact_distrib import EmptyImpactDistrib
from physrisk.kernel.risk import AssetLevelRiskModel, Measure, MeasureKey
from physrisk.kernel.vulnerability_model import VulnerabilityModelBase

from physrisk.api.v1.hazard_data import (
    HazardAvailabilityRequest,
    HazardAvailabilityResponse,
    HazardDataRequest,
    HazardDataResponse,
    HazardDataResponseItem,
    HazardDescriptionRequest,
    HazardDescriptionResponse,
    HazardResource,
    IntensityCurve,
    Scenario,
)
from physrisk.api.v1.impact_req_resp import (
    AcuteHazardCalculationDetails,
    AssetImpactRequest,
    AssetImpactResponse,
    AssetLevelImpact,
    Assets,
    AssetSingleImpact,
    ImpactKey,
    RiskMeasureKey,
    RiskMeasures,
    RiskMeasuresForAssets,
    ScoreBasedRiskMeasureDefinition,
    ScoreBasedRiskMeasureSetDefinition,
)
from physrisk.data.image_creator import ImageCreator
from physrisk.data.inventory import EmbeddedInventory, Inventory
from physrisk.kernel import Asset, Hazard
from physrisk.kernel import calculation as calc
from physrisk.kernel.hazard_model import HazardDataRequest as hmHazardDataRequest
from physrisk.kernel.hazard_model import HazardEventDataResponse as hmHazardEventDataResponse
from physrisk.kernel.hazard_model import HazardModel, HazardParameterDataResponse





from fsspec.implementations.local import LocalFileSystem

class TestRiskModels(TestWithCredentials):
    def test_risk_indicator_model(self):
        assets, request = self._create_assets1()
        # hazard_model = self._create_hazard_model(scenarios, years)

        inventory_reader = InventoryReader( fs = LocalFileSystem() , base_path = r'physrisk/data/static/')
        inventory = _create_inventory(reader=inventory_reader, sources=["hazard"])
        source_paths = create_source_paths(inventory=inventory)
        zarr_store = ZarrReader.create_s3_zarr_store()
        zarr_reader = ZarrReader(store=zarr_store)
        hazard_model = ZarrHazardModel( reader=zarr_reader, source_paths=source_paths, interpolation="floor")

        vulnerability_models = None

        vulnerability_models = (
            calc.get_default_vulnerability_models() if vulnerability_models is None else vulnerability_models
        )

        # we keep API definition of asset separate from internal Asset class; convert by reflection
        # based on asset_class:
        measure_calcs = calc.get_default_risk_measure_calculators()
        risk_model = AssetLevelRiskModel(hazard_model, vulnerability_models, measure_calcs)

        scenarios = [request.scenario] if request.scenarios is None or len(request.scenarios) == 0 else request.scenarios
        scenarios = ['rcp26', 'rcp45', 'rcp85']
        years = [request.year] if request.years is None or len(request.years) == 0 else request.years
        years = [2020, 2030, 2040] + years
        risk_measures = None
        if request.include_measures:
            impacts, measures = risk_model.calculate_risk_measures(assets, scenarios, years)
            measure_ids_for_asset, definitions = risk_model.populate_measure_definitions(assets)
            # create object for API:
            risk_measures = _create_risk_measures(measures, measure_ids_for_asset, definitions, assets, scenarios, years)
        elif request.include_asset_level:
            impacts = risk_model.calculate_impacts(assets, scenarios, years)

        if request.include_asset_level:
            ordered_impacts: Dict[Asset, List[AssetSingleImpact]] = {}
            for k, v in impacts.items():
                if request.include_calc_details:
                    if v.event is not None and v.vulnerability is not None:
                        hazard_exceedance = v.event.to_exceedance_curve()

                        vulnerability_distribution = VulnerabilityDistrib(
                            intensity_bin_edges=v.vulnerability.intensity_bins,
                            impact_bin_edges=v.vulnerability.impact_bins,
                            prob_matrix=v.vulnerability.prob_matrix,
                        )
                        calc_details = AcuteHazardCalculationDetails(
                            hazard_exceedance=ExceedanceCurve(
                                values=hazard_exceedance.values, exceed_probabilities=hazard_exceedance.probs
                            ),
                            hazard_distribution=Distribution(
                                bin_edges=v.event.intensity_bin_edges, probabilities=v.event.prob
                            ),
                            vulnerability_distribution=vulnerability_distribution,
                        )
                else:
                    calc_details = None

                if isinstance(v.impact, EmptyImpactDistrib):
                    continue
                impact_exceedance = v.impact.to_exceedance_curve()
                key = ImpactKey(hazard_type=k.hazard_type.__name__, scenario_id=k.scenario, year=str(k.key_year))
                hazard_impacts = AssetSingleImpact(
                    key=key,
                    impact_type=v.impact.impact_type.name,
                    impact_exceedance=ExceedanceCurve(
                        values=impact_exceedance.values, exceed_probabilities=impact_exceedance.probs
                    ),
                    impact_distribution=Distribution(bin_edges=v.impact.impact_bins, probabilities=v.impact.prob),
                    impact_mean=v.impact.mean_impact(),
                    impact_std_deviation=v.impact.stddev_impact(),
                    calc_details=None if v.event is None else calc_details,
                )
                ordered_impacts.setdefault(k.asset, []).append(hazard_impacts)

            # note that this does rely on ordering of dictionary (post 3.6)
            asset_impacts = [AssetLevelImpact(asset_id="", impacts=a) for a in ordered_impacts.values()]
        else:
            asset_impacts = None


        import pickle
        with open('risk_mv.pickle', 'wb') as handle:
            pickle.dump(ordered_impacts, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def test_risk_indicator_model_working(self):
        scenarios = ["rcp8p5"]
        years = [2050]

        assets = self._create_assets()
        hazard_model = self._create_hazard_model(scenarios, years)

        model = AssetLevelRiskModel(
            hazard_model, get_default_vulnerability_models(), {PowerGeneratingAsset: PowerGeneratingAssetToyRiskMeasures()}
        )
        measure_ids_for_asset, definitions = model.populate_measure_definitions(assets)
        _, measures = model.calculate_risk_measures(assets, prosp_scens=scenarios, years=years)

        # how to get a score using the MeasureKey
        measure = measures[MeasureKey(assets[0], scenarios[0], years[0], RiverineInundation)]
        score = measure.score
        measure_0 = measure.measure_0
        np.testing.assert_allclose([measure_0], [0.89306593179])

        # packing up the risk measures, e.g. for JSON transmission:
        risk_measures = _create_risk_measures(measures, measure_ids_for_asset, definitions, assets, scenarios, years)
        # we still have a key, but no asset:
        key = RiskMeasureKey(
            hazard_type="RiverineInundation",
            scenario_id=scenarios[0],
            year=str(years[0]),
            measure_id=risk_measures.score_based_measure_set_defn.measure_set_id,
        )
        item = next(m for m in risk_measures.measures_for_assets if m.key == key)
        score2 = item.scores[0]
        measure_0_2 = item.measures_0[0]
        assert score == score2
        assert measure_0 == measure_0_2

        helper = RiskMeasuresHelper(risk_measures)
        asset_scores, measures, definitions = helper.get_measure("ChronicHeat", scenarios[0], years[0])
        label, description = helper.get_score_details(asset_scores[0], definitions[0])
        assert asset_scores[0] == 4

    def _create_assets1(self):
        import pickle
        path_pickle = r'C:\Users\mvazquez\ArfimaTools\physrisk_mv\src\request_test.pickle'

        with open(path_pickle, 'rb') as handle:
            request = pickle.load(handle)

        assets = [
            PowerGeneratingAsset(asset.latitude, asset.longitude, type=asset.type, location=asset.location)
            for asset in request.assets.items
        ]

        return assets, request
    
    def _create_assets(self):
        assets = [
            PowerGeneratingAsset(37, -5, location="Europe", type="Gas/Gas")
            for i in range(2)
        ]
        return assets
    
    def _create_assets_working(self):
        assets = [
            RealEstateAsset(TestData.latitudes[0], TestData.longitudes[0], location="Asia", type="Buildings/Industrial")
            for i in range(2)
        ]
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
            return source_paths[Wind](indicator_id="gust_speed_level", scenario=scenario, year=year)

        # def sp_heat(scenario, year):
        #     return source_paths[ChronicHeat](indicator_id="mean_degree_days/above/index", scenario=scenario, year=year)

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
        # mocker.add_curves_global(
        #     sp_heat("historical", -1),
        #     TestData.longitudes,
        #     TestData.latitudes,
        #     TestData.temperature_thresholds,
        #     TestData.degree_days_above_index_1,
        # )
        # mocker.add_curves_global(
        #     sp_heat("rcp8p5", 2050),
        #     TestData.longitudes,
        #     TestData.latitudes,
        #     TestData.temperature_thresholds,
        #     TestData.degree_days_above_index_2,
        # )

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


obj = TestRiskModels()
obj.test_risk_indicator_model()