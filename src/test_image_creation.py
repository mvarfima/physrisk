import os

import numpy as np
import zarr
import zarr.storage

from physrisk.data import colormap_provider
from physrisk.data.image_creator import ImageCreator
from physrisk.data.zarr_reader import ZarrReader

from fsspec.implementations.local import LocalFileSystem

fs = LocalFileSystem()

path = "maps/water_risk/wri/v2/water_demand_ssp126_2030_map/1"
root = zarr.open(store=r"E:/data/hazard/hazard.zarr", mode="r")
store = root.store

z = root[path]
im = z[0,:,:]

creator = ImageCreator(reader=ZarrReader(store))
colormap = colormap_provider.colormap("test")

creator.to_file(os.path.join(os.getcwd(), "test.png"), path)

def get_colors(index: int):
    return colormap[str(index)]

result = converter._to_rgba(im, get_colors)
# Max should be 255, min should be 1. Other values span the 253 elements from 2 to 254.
expected = np.array([[255, 2 + (0.8 - 0.4) * 253 / (1.2 - 0.4)], [2 + (0.5 - 0.4) * 253 / (1.2 - 0.4), 1]])
converter.convert(path, colormap="test")  # check no error running through mocked example.
np.testing.assert_equal(result, expected.astype(np.uint8))


# def test_write_file():
#     # show how to create image from zarr array
#     # useful for testing image generation
#     test_output_dir = "{set me}"
#     test_path = "wildfire/jupiter/v1/wildfire_probability_ssp585_2050_map"
#     store = zarr.DirectoryStore(os.path.join(test_output_dir, "hazard_test", "hazard.zarr"))
#     creator = ImageCreator(ZarrReader(store))
#     creator.to_file(os.path.join(test_output_dir, "test.png"), test_path)

