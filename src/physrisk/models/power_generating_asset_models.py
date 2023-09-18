from typing import Iterable, Union

import pandas as pd
import numpy as np
import os

from ..kernel.assets import Asset, PowerGeneratingAsset
from ..kernel.curve import ExceedanceCurve
from ..kernel.hazard_event_distrib import HazardEventDistrib
from ..kernel.hazard_model import HazardDataRequest, HazardDataResponse, HazardEventDataResponse
from ..kernel.hazards import ChronicHeat, RiverineInundation
from ..kernel.vulnerability_distrib import VulnerabilityDistrib
from ..kernel.vulnerability_model import (
    DeterministicVulnerabilityModel,
    VulnerabilityModelAcuteBase,
    applies_to_assets,
    applies_to_events,
)



if "physical_risk_database" in os.environ:
    base_path_damfun = os.path.join(os.getenv("physical_risk_database"), 'vulnerability', 'damage_functions')
else:
    base_path = os.getcwd()
    base_path = base_path.split('PhysicalRisk')[0]
    base_path_damfun = os.path.join(base_path, 'physical_risk_database', 'vulnerability', 'damage_functions')


@applies_to_events([RiverineInundation])
@applies_to_assets([PowerGeneratingAsset])
class InundationModel(VulnerabilityModelAcuteBase):

    def __init__(self, model="MIROC-ESM-CHEM"):

        data_filename = 'flood_river_huizinga.csv'
        inputfile = os.path.join(base_path_damfun, data_filename)

        asset_type = 'Industrial buildings'
        dm_df = pd.read_csv(inputfile)
        dm_df = dm_df[dm_df.Type == asset_type]

        # default impact curve
        self.__curve_depth = dm_df['Flood depth [m]'].values
        self.__curve_impact = dm_df['Europe_Mean'].values
        self.__model = model
        self.__base_model = "000000000WATCH"
        super().__init__(model, RiverineInundation)
        pass

    def get_data_requests(
        self, asset: Asset, *, scenario: str, year: int
    ) -> Union[HazardDataRequest, Iterable[HazardDataRequest]]:
        """Provide the list of hazard event data requests required in order to calculate
        the VulnerabilityDistrib and HazardEventDistrib for the asset."""

        # histo = HazardDataRequest(
        #     RiverineInundation,
        #     asset.longitude,
        #     asset.latitude,
        #     scenario="historical",
        #     year=1980,
        #     model=self.__base_model,
        # )

        # future = HazardDataRequest(
        #     RiverineInundation, asset.longitude, asset.latitude, scenario=scenario, year=year, model=self.__model
        # )

        # return histo, future
    
        histo = HazardDataRequest(
            RiverineInundation,
            asset.longitude,
            asset.latitude,
            scenario="historical",
            year=1980,
            model=self.__base_model,
        )

        return histo


    def get_distributions(self, asset: Asset, event_data_responses: Iterable[HazardDataResponse]):
        """Return distributions for asset, based on hazard event date:
        VulnerabilityDistrib and HazardEventDistrib."""

        # histo, future = event_data_responses
        # assert isinstance(histo, HazardEventDataResponse)
        # assert isinstance(future, HazardEventDataResponse)

        # protection_return_period = 250.0
        # curve_histo = ExceedanceCurve(1.0 / histo.return_periods, histo.intensities)
        # # the protection depth is the 250-year-return-period inundation depth at the asset location
        # protection_depth = curve_histo.get_value(1.0 / protection_return_period)

        # curve_future = ExceedanceCurve(1.0 / future.return_periods, future.intensities)
        # curve_future = curve_future.add_value_point(protection_depth)

        # depth_bins, probs = curve_future.get_probability_bins()

        # impact_bins = np.interp(depth_bins, self.__curve_depth, self.__curve_impact) / 365.0

        # # keep all bins, but make use of vulnerability matrix to apply protection level
        # # for improved performance we could truncate (and treat identify matrix as a special case)
        # # but this general version allows model uncertainties to be added
        # probs_protected = np.where(depth_bins[1:] <= protection_depth, 0.0, 1.0)

        # vul = VulnerabilityDistrib(RiverineInundation, depth_bins, impact_bins, np.diag(probs_protected))
        # event = HazardEventDistrib(RiverineInundation, depth_bins, probs)

        # return vul, event


        histo = event_data_responses[0]

        curve_histo = ExceedanceCurve(1.0 / histo.return_periods, histo.intensities)
        depth_bins, probs = curve_histo.get_probability_bins()

        impact_bins = np.interp(depth_bins, self.__curve_depth, self.__curve_impact)

        damage_fun_matrix = np.eye(len(depth_bins) - 1)

        vul = VulnerabilityDistrib(RiverineInundation, depth_bins, impact_bins, damage_fun_matrix)
        event = HazardEventDistrib(RiverineInundation, depth_bins, probs)

        return vul, event


@applies_to_events([ChronicHeat])
@applies_to_assets([PowerGeneratingAsset])
class TemperatureModel(DeterministicVulnerabilityModel):
    def __init__(self):
        # does nothing
        pass
