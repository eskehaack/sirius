import numpy as np
from pathlib import Path
import xarray as xr

import earthkit.data as ekd
import earthkit.plots as ekp

from preprocessor import mean_downscale_3d, bicubic_upscale_3d


class cdsDataset:
    def __init__(
        self,
        dataset: str = "reanalysis-pan-carra",
        target: str = "carra2.grib",
        root_dir: str = "./data/",
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

        root = Path(root_dir)
        root.mkdir(parents=True, exist_ok=True)

        self.data_dir = root / target

    def _fetch_data(
        self,
        area: list = [81, 10, 74, 35],  # north, west, south, east
        times: list = ["12:00"],
        years: list = ["2025"],
        months: list = ["06"],
        days: list = ["01"],
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
            "area": area,
        }

        request = {
            "level_type": "single_levels",
            "variable": [
                "2m_temperature",
                "land_sea_mask",
                "orography"
            ],
            "product_type": "analysis",
            "time": [
                "03:00", "09:00", "15:00",
                "21:00"
            ],
            "year": ["2001"],
            "month": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12"
            ],
            "day": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12",
                "13", "14", "15",
                "16", "17", "18",
                "19", "20", "21",
                "22", "23", "24",
                "25", "26", "27",
                "28", "29", "30",
                "31"
            ],
            "data_format": "grib"
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

        return ekd.from_source("file", self.data_dir).to_xarray(
            add_earthkit_attrs=False
        )

    def preprocess_to_coarse(
        self,
        variables: list | str = "2t",
        corner: tuple[int, int] = (300, 1100),
        img_size: int = 256,
        downscale_factor: int = 8,
        output_dir: str | Path = Path("./.data/preprocessed/"),
        test_split: float = 0.1,
        validation_split: float = 0.1,
    ):

        if not isinstance(variables, list):
            variables = [variables]

        ds = self.read_data()
        data = ds[variables]
        temporal_dim = data.forecast_reference_time.size

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_size = int(temporal_dim * (1 - test_split - validation_split))
        validation_size = int(temporal_dim * validation_split)
        test_size = int(temporal_dim * test_split)
        index_ranges = {
            "train": (0, train_size),
            "validation": (train_size, train_size + validation_size),
            "test": (train_size + validation_size, temporal_dim),
        }

        if sum([train_size, validation_size, test_size]) > temporal_dim:
            raise ValueError("Split sizes exceed the total number of samples")

        for split in index_ranges:
            # 1. Crop to southern Greenland
            high_res = ds.isel(
                forecast_reference_time=slice(*index_ranges[split]),
                x=slice(corner[0], corner[0] + img_size),
                y=slice(corner[1], corner[1] + img_size),
            )[variables]

            if split == "train":
                mean = float(high_res["2t"].mean().values)
                std = float(high_res["2t"].std().values)
                open(output_dir / "mean_std.txt", "x").write(f"{mean},{std}")

            if not mean or not std:
                raise ValueError(
                    "Mean and std must be calculated from the training set before preprocessing other splits."
                )

            high_res = (high_res - mean) / std

            target = high_res["2t"].values
            target = np.expand_dims(target, axis=1)  # shape: (time, 1, height, width)

            high_res_arr: np.ndarray = high_res.to_array().values
            high_res_arr = high_res_arr.transpose(
                1, 0, 2, 3
            )  # shape: (time, channel, height, width)
            # 2. Mean aggregate downscale to [32, 32]
            low_res = mean_downscale_3d(high_res_arr, factor=downscale_factor)

            # 3. Bicubic upscale back to [256, 256]
            condition = bicubic_upscale_3d(low_res, target_size=high_res_arr.shape[-2:])

            # 4. Store condition and target in one Zarr store per split
            condition = np.asarray(condition, dtype=np.float32)
            target = np.asarray(target, dtype=np.float32)

            # Ensure both arrays have shape (time, channel, height, width)
            if condition.ndim != 4 or target.ndim != 4:
                raise ValueError(
                    f"Condition and target must be 4D arrays, but got shapes {condition.shape} and {target.shape}"
                )

            split_dataset = xr.Dataset(
                data_vars={
                    "condition": (
                        (
                            "time",
                            "condition_channel",
                            "condition_x",
                            "condition_y",
                        ),
                        condition,
                    ),
                    "target": (
                        (
                            "time",
                            "target_channel",
                            "target_x",
                            "target_y",
                        ),
                        target,
                    ),
                },
                coords={
                    "time": high_res.forecast_reference_time.values,
                    "condition_channel": variables,
                    "target_channel": ["2t"],
                },
            )

            # Each chunk contains complete images/channels and a subset of time.
            time_chunk_size = min(16, condition.shape[0])

            split_dataset.to_zarr(
                output_dir / f"{split}.zarr",
                mode="w",
                consolidated=True,
                encoding={
                    "condition": {
                        "chunks": (
                            time_chunk_size,
                            condition.shape[1],
                            condition.shape[2],
                            condition.shape[3],
                        ),
                    },
                    "target": {
                        "chunks": (
                            time_chunk_size,
                            target.shape[1],
                            target.shape[2],
                            target.shape[3],
                        ),
                    },
                },
            )

        print(f"Saved {temporal_dim} samples to {output_dir}")

    def plot_data(
        self,
        variable: str = "2t",
        outfile: str = "test.jpg",
    ):
        ds = self.read_data()
        data = ds.sel(variable=variable).isel(forecast_reference_time=0)

        if variable == "t2m":
            data -= 273.15

        chart = ekp.Map(
            data,
            title=f"{variable} at {data.forecast_reference_time.values}",
            cmap="coolwarm",
        )
        chart.save(outfile)


if __name__ == "__main__":
    dataset = cdsDataset(
        target="2001_all_temp_lsm_orog.grib",
    )
    dataset._fetch_data()
    dataset.preprocess_to_coarse(
        variables=["2t", "lsm", "orog"],
        corner=(300, 1100),
        img_size=256,
        downscale_factor=8,
        output_dir="./data/preprocessed/",
        test_split=0.0,
        validation_split=0.1,
    )
