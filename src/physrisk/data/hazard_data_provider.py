import os
from abc import ABC
from pathlib import PurePosixPath
from typing import Dict, List, MutableMapping, Optional

from typing_extensions import Protocol

from physrisk.data.inventory import Inventory
from physrisk.kernel import hazards

from .zarr_reader import ZarrReader


class SourcePath(Protocol):
    """Provides path to hazard event data source. Each source should have its own implementation.
    Args:
        model: model identifier.
        scenario: identifier of scenario, e.g. rcp8p5 (RCP 8.5).
        year: projection year, e.g. 2080.
    """

    def __call__(self, *, model: str, scenario: str, year: int) -> str:
        ...


class HazardDataProvider(ABC):
    def __init__(
        self,
        get_source_path: SourcePath,
        *,
        store: Optional[MutableMapping] = None,
        zarr_reader: Optional[ZarrReader] = None,
        interpolation: Optional[str] = "floor",
    ):
        """Create an EventProvider.

        Args:
            get_source_path: provides the path to the hazard event data source depending on year/scenario/model.
        """
        self._get_source_path = get_source_path
        self._reader = zarr_reader if zarr_reader is not None else ZarrReader(store=store)
        if interpolation not in ["floor", "linear"]:
            raise ValueError("interpolation must be 'floor' or 'linear'")
        self._interpolation = interpolation


class AcuteHazardDataProvider(HazardDataProvider):
    """Provides hazard event intensities for a single Hazard (type of hazard event)."""

    def __init__(
        self,
        get_source_path: SourcePath,
        *,
        store: Optional[MutableMapping] = None,
        zarr_reader: Optional[ZarrReader] = None,
        interpolation: Optional[str] = "floor",
    ):
        super().__init__(get_source_path, store=store, zarr_reader=zarr_reader, interpolation=interpolation)

    def get_intensity_curves(
        self, longitudes: List[float], latitudes: List[float], *, model: str, scenario: str, year: int
    ):
        """Get intensity curve for each latitude and longitude coordinate pair.

        Args:
            longitudes: list of longitudes.
            latitudes: list of latitudes.
            model: model identifier.
            scenario: identifier of scenario, e.g. rcp8p5 (RCP 8.5).
            year: projection year, e.g. 2080.

        Returns:
            curves: numpy array of intensity (no. coordinate pairs, no. return periods).
            return_periods: return periods in years.
        """

        path = self._get_source_path()
        curves, return_periods = self._reader.get_curves(
            path, longitudes, latitudes, self._interpolation
        )  # type: ignore
        return curves, return_periods


class ChronicHazardDataProvider(HazardDataProvider):
    """Provides hazard parameters for a single type of chronic hazard."""

    def __init__(
        self,
        get_source_path: SourcePath,
        *,
        store: Optional[MutableMapping] = None,
        zarr_reader: Optional[ZarrReader] = None,
        interpolation: Optional[str] = "floor",
    ):
        super().__init__(get_source_path, store=store, zarr_reader=zarr_reader, interpolation=interpolation)

    def get_parameters(self, longitudes: List[float], latitudes: List[float], *, model: str, scenario: str, year: int):
        """Get hazard parameters for each latitude and longitude coordinate pair.

        Args:
            longitudes: list of longitudes.
            latitudes: list of latitudes.
            model: model identifier.
            scenario: identifier of scenario, e.g. rcp8p5 (RCP 8.5).
            year: projection year, e.g. 2080.

        Returns:
            parameters: numpy array of parameters
        """

        path = self._get_source_path(model=model, scenario=scenario, year=year)
        parameters, _ = self._reader.get_curves(path, longitudes, latitudes, self._interpolation)
        return parameters[:, 0]


def _data_prefix():
    default_staging_bucket = 'physrisk-hazard-indicators-dev01'
    prefix = 'hazard'
    return os.path.join(default_staging_bucket, prefix, "hazard.zarr").replace('\\','/')


def get_source_path_flood_river_jrc():
    return os.path.join(_data_prefix(), 'flood_river_hist_RP_JRC')

def get_source_path_flood_coastal_jrc():
    return os.path.join(_data_prefix(), 'flood_coastal_hist_RP_JRC')

def get_source_path_wind_ecb():
    return os.path.join(_data_prefix(), 'wind_hist_RP_ECB')

