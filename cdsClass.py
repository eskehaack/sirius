import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

import cdsapi
import earthkit.data as ekd
import xarray as xr

from preprocessor import mean_downscale_3d, bicubic_upscale_3d

class cdsDataset:
    def __init__(
            self, 
            dataset: str = "reanalysis-pan-carra", 
            target: str = "carra2.grib", 
            root_dir: str = "./.data/", 
            variables: list = ["2m_temperature"],
            slices: list | str = "all",
        ) -> None:
        """
        Initializes the cdsDataset class.

        Args:
            dataset (str): The name of the dataset to retrieve.
            target (str): The target file name for the downloaded data.
            root_dir (str): The root directory where the data will be stored.
            variables (list): The list of variables to retrieve.
        """

        self.dataset = dataset
        self.target = target
        self.root_dir = root_dir
        self.variables = variables
        self.slices = slices

        self.data_dir = Path(root_dir) / target

    def _fetch_data(
            self, 
            area: list = [81, 10, 74, 35], # north, west, south, east
            times: list = ["12:00"],
            years: list = ["2025"],
            months: list = ["06"],
            days: list = ["01"]
        ) -> None:
        """
        Fetches the data from the CDS API and saves it to the specified location.
        """
        request = {
            "level_type": "single_levels",
            "variable": self.variables,
            "product_type": "analysis",
            "time": times,
            "year": years,
            "month": months,
            "day": days,
            "data_format": "grib",
            "area": area
        }

        ds = ekd.from_source("cds", self.dataset, request)
        ds.to_target("file", self.data_dir)

        # request = {
        #     "level_type": "single_levels",
        #     "variable": [
        #         "2m_temperature",
        #         "land_sea_mask",
        #         "orography"
        #     ],
        #     "product_type": "analysis",
        #     "time": [
        #         "03:00", "09:00", "15:00",
        #         "21:00"
        #     ],
        #     "year": ["2001"],
        #     "month": [
        #         "01", "02", "03",
        #         "04", "05", "06",
        #         "07", "08", "09",
        #         "10", "11", "12"
        #     ],
        #     "day": [
        #         "01", "02", "03",
        #         "04", "05", "06",
        #         "07", "08", "09",
        #         "10", "11", "12",
        #         "13", "14", "15",
        #         "16", "17", "18",
        #         "19", "20", "21",
        #         "22", "23", "24",
        #         "25", "26", "27",
        #         "28", "29", "30",
        #         "31"
        #     ],
        #     "data_format": "grib"
        # }    

    def read_data(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_dir}")

        return ekd.from_source("file", self.data_dir).to_xarray(add_earthkit_attrs=False)

    def preprocess_to_coarse(
            self, 
            variables: list | str = "t2m", 
            corner: tuple[int,int] = (300, 1100),
            img_size: int = 256,
            downscale_factor: int = 8,
            output_dir: str | Path = Path("./.data/preprocessed/"),
        ):

        if not isinstance(variables, list):
            variables = [variables]

        ds = self.read_data()
        data = ds[variables]

        pp_root = Path(output_dir)
        pp_root.mkdir(parents=True, exist_ok=True)

        for split in ["train", "validation", "test"]:
            split_dir = pp_root / split
            split_dir.mkdir(parents=True, exist_ok=True)

        temporal_dim, x_dim, y_dim = data[variables[0]].shape
        
        for i in tqdm(range(temporal_dim), desc="Preprocessing data"):
            
            # 1. Crop to southern Greenland
            high_res = np.empty((len(variables), img_size, img_size), dtype=np.float32)
            for j, var in enumerate(variables):
                high_res[j] = data[var][i, corner[0]:corner[0]+img_size, corner[1]:corner[1]+img_size].values
            np.save(output_dir / f"target_{i:05d}.npy", high_res)

            # 2. Mean aggregate downscale to [32, 32]
            low_res = mean_downscale_3d(high_res, factor=downscale_factor)
            np.save(output_dir / f"lowres_{i:05d}.npy", low_res)

            # 3. Bicubic upscale back to [256, 256]
            condition = bicubic_upscale_3d(low_res, target_size=high_res.shape[-2:])
            np.save(output_dir / f"condition_{i:05d}.npy", condition)

        print(f"Saved {temporal_dim} samples to {output_dir}")
    
    def slice_data(self, ds, variable: str = "t2m"):
        if self.slices in ["all", "whole", "global"]:
            return ds[variable].values

        x0, x1, y0, y1 = self.slices

        y_slice = slice(y0, y1)
        x_slice = slice(x0, x1)

        sliced_data = ds[variable].isel(y=y_slice, x=x_slice)

        return sliced_data.values
        
    
    def plot_data(
        self,
        variable: str = "t2m",
        outfile: str = "test.jpg",
    ):
        ds = self.read_data()
        data = self.slice_data(ds=ds, variable=variable)

        if variable == "t2m":
            data -= 273.15

        fig = plt.figure(figsize=(9, 8))

        ax = plt.axes()
        mesh = ax.imshow(
            data,
            origin="lower",
            cmap="Greys",
        )

        cbar = plt.colorbar(mesh, ax=ax, shrink=0.75)
        cbar.set_label(f"{variable} [{ds[variable].attrs.get('units', '')}]")

        ax.set_title("2 m Temperature 20010926 12:00 UTC")
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    dataset = cdsDataset(
        target="t2m_topology_coverage_all2001.grib", 
    )
    dataset.preprocess_to_coarse(['t2m', 'lsm', 'orog'])