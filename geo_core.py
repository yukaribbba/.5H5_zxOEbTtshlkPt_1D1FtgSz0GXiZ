#!/usr/bin/python
import bz2
import glob
import math
import os
import queue
import re
import shutil
import time as time_calc
import urllib.error
import urllib.request
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from functools import partial
from urllib.parse import unquote

import boto3
import dask.array as da
import eumdac
import hdf5plugin
import numpy as np
import requests
import trollimage.xrimage
import xarray as xr
from botocore import UNSIGNED
from botocore.client import Config
from osgeo import gdal
from pyproj import Proj

import satpy
import satpy.enhancements
from mosaic_core import Mosaic
from satpy import Scene, config, find_files_and_readers, utils
from satpy.composites import LOG, MASKING_COMPOSITOR_METHODS, GenericCompositor
from satpy.enhancements import exclude_alpha, on_dask_array
from satpy.readers import generic_image
from satpy.writers import geotiff

gdal_options = ("tfw",
                "rpb",
                "rpctxt",
                "interleave",
                "tiled",
                "blockxsize",
                "blockysize",
                "nbits",
                "compress",
                "num_threads",
                "predictor",
                "discard_lsb",
                "sparse_ok",
                "jpeg_quality",
                "jpegtablesmode",
                "zlevel",
                "photometric",
                "alpha",
                "profile",
                "bigtiff",
                "pixeltype",
                "copy_src_overviews",
                "in_memory",
                # COG driver options (different from GTiff above)
                "blocksize",
                "resampling",
                "quality",
                "level",
                "overview_resampling",
                "warp_resampling",
                "overview_compress",
                "overview_quality",
                "overview_predictor",
                "tiling_scheme",
                "zoom_level_strategy",
                "target_srs",
                "res",
                "extent",
                "aligned_levels",
                "add_alpha",
                "in_memory"
                )


class GeoProcessor:
    def __init__(self, config_data: dict, sat_index: int, proj_index: int, product_index_list: list, sos=False):
        super().__init__()
        hdf5plugin.get_config()
        self.config_data = config_data
        self.sat_index = sat_index
        self.proj_index = proj_index
        self.product_index_list = product_index_list
        self.sos = sos

        self.global_satellites = self.config_data["global_satellites"]
        self.conda_path = self.config_data["basic_path"]["conda_path"]
        self.base_path = self.config_data["basic_path"]["base_path"]
        self.satpy_config_path = self.config_data["basic_path"]["satpy_config_path"]
        self.background_mask_path = self.config_data["basic_path"]["background_mask_path"]

        satellites_dict = self.config_data["satellites"]
        self.sat = list(satellites_dict.keys())[sat_index]
        self.reader_name = self.config_data["satellites"][self.sat]["reader_name"]

        proj_dict = self.config_data["projection"][self.sat]
        proj_name = list(proj_dict.keys())[self.proj_index]
        self.target_proj = proj_dict[proj_name]["proj"]
        self.target_width = int(proj_dict[proj_name]["width"])
        self.target_height = int(proj_dict[proj_name]["height"])
        self.target_xy = proj_dict[proj_name]["extent"]
        self.area = proj_dict[proj_name]["area_id"]

        self.proj_indicator = self.target_proj.rsplit(" ")[0].strip("+proj=")

        self.area_min_x = float(self.target_xy.split()[0])
        self.area_min_y = float(self.target_xy.split()[1])
        self.area_max_x = float(self.target_xy.split()[2])
        self.area_max_y = float(self.target_xy.split()[3])
        self.width_extent = self.area_max_x - self.area_min_x
        self.height_extent = self.area_max_y - self.area_min_y
        self.resolution_x = self.width_extent / self.target_width
        self.resolution_y = self.height_extent / self.target_height
        if "latlon" in self.target_proj or "longlat" in self.target_proj:
            self.resolution_x = self.resolution_x * 111319.5
            self.resolution_y = self.resolution_y * 111319.5
        self.resolution = round(self.resolution_x)
        self.area_id = f"{self.proj_indicator}_{self.area}_{str(self.resolution)}"

        self.original_geos_proj = satellites_dict[self.sat]["original_geos_proj"]

        self.band_list = []
        self.band_image_list = []
        self.exp_bands_count_list = []
        self.background_local_file_list = []
        self.background_satpy_file_list = []
        self.landmask_local_file_list = []
        self.landmask_satpy_file_list = []
        self.globalmask_local_file_list = []
        self.globalmask_satpy_file_list = []
        self.product_list = []

        satellite = "G16" if self.sat in ["G16", "G17", "G18", "G19"] else self.sat
        product_dict = self.config_data["composites"][satellite]

        for product_idx in product_index_list:
            product = list(product_dict.keys())[product_idx]
            self.product_list.append(product)

            composites = self.config_data["composites"][satellite][product]
            self.band_list.extend(composites["band_list"])
            self.band_image_list.extend(composites["band_image_list"])
            self.exp_bands_count_list.append(composites["expected_bands_count"])

            for key, target_list, base_path in [
                ("background_local_file", self.background_local_file_list, self.background_mask_path),
                ("background_satpy_file", self.background_satpy_file_list, self.base_path),
                ("landmask_local_file", self.landmask_local_file_list, self.background_mask_path),
                ("landmask_satpy_file", self.landmask_satpy_file_list, self.base_path),
                ("globalmask_local_file", self.globalmask_local_file_list, self.background_mask_path),
                ("globalmask_satpy_file", self.globalmask_satpy_file_list, self.base_path),
            ]:
                value = composites.get(key)
                target_list.append(f"{base_path}/{value}" if value else None)

        self.new_product_list = []
        self.new_exp_bands_count_list = []

        special_sos_composites_dict = self.config_data.get("sos_special_composites", {}) if self.sos else {}
        for idx, product in enumerate(self.product_list):
            if self.sos and product in special_sos_composites_dict:
                composite_info = special_sos_composites_dict[product]
                self.new_product_list.extend(composite_info["actual_composite"])
                self.new_exp_bands_count_list.extend(composite_info["expected_bands_count"])
            else:
                self.new_product_list.append(product)
                self.new_exp_bands_count_list.append(self.exp_bands_count_list[idx])

        seen = set()
        pop_list = [idx for idx, item in enumerate(self.new_product_list) if item in seen or seen.add(item)]
        pop_list.reverse()

        for pop in pop_list:
            self.new_product_list.pop(pop)
            self.new_exp_bands_count_list.pop(pop)

        self.band_list = list({}.fromkeys(self.band_list).keys())
        self.band_image_list = list({}.fromkeys(self.band_image_list).keys())

        gdal.SetCacheMax(4096 * 1024 * 1024)
        gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")

    def update_sat(self, new_sat):
        self.sat = new_sat

        satellites_dict = self.config_data["satellites"]
        self.reader_name = self.config_data["satellites"][self.sat]["reader_name"]

        proj_dict = self.config_data["projection"][self.sat]
        proj_name = list(proj_dict.keys())[self.proj_index]
        self.target_proj = proj_dict[proj_name]["proj"]
        self.target_width = int(proj_dict[proj_name]["width"])
        self.target_height = int(proj_dict[proj_name]["height"])
        self.target_xy = proj_dict[proj_name]["extent"]
        self.area = proj_dict[proj_name]["area_id"]

        self.proj_indicator = self.target_proj.rsplit(" ")[0].strip("+proj=")

        self.area_min_x = float(self.target_xy.split()[0])
        self.area_min_y = float(self.target_xy.split()[1])
        self.area_max_x = float(self.target_xy.split()[2])
        self.area_max_y = float(self.target_xy.split()[3])
        self.width_extent = self.area_max_x - self.area_min_x
        self.height_extent = self.area_max_y - self.area_min_y
        self.resolution_x = self.width_extent / self.target_width
        self.resolution_y = self.height_extent / self.target_height
        self.resolution = round(self.resolution_x)
        self.area_id = f"{self.proj_indicator}_{self.area}_{str(self.resolution)}"

        self.original_geos_proj = satellites_dict[self.sat]["original_geos_proj"]

        self.band_list = []
        self.band_image_list = []
        self.exp_bands_count_list = []
        self.background_local_file_list = []
        self.background_satpy_file_list = []
        self.landmask_local_file_list = []
        self.landmask_satpy_file_list = []
        self.globalmask_local_file_list = []
        self.globalmask_satpy_file_list = []
        self.product_list = []

        satellite = "G16" if self.sat in ["G16", "G17", "G18", "G19"] else self.sat
        product_dict = self.config_data["composites"][satellite]

        for product_idx in self.product_index_list:
            product = list(product_dict.keys())[product_idx]
            self.product_list.append(product)

            composites = self.config_data["composites"][satellite][product]
            self.band_list.extend(composites["band_list"])
            self.band_image_list.extend(composites["band_image_list"])
            self.exp_bands_count_list.append(composites["expected_bands_count"])

            for key, target_list, base_path in [
                ("background_local_file", self.background_local_file_list, self.background_mask_path),
                ("background_satpy_file", self.background_satpy_file_list, self.base_path),
                ("landmask_local_file", self.landmask_local_file_list, self.background_mask_path),
                ("landmask_satpy_file", self.landmask_satpy_file_list, self.base_path),
                ("globalmask_local_file", self.globalmask_local_file_list, self.background_mask_path),
                ("globalmask_satpy_file", self.globalmask_satpy_file_list, self.base_path),
            ]:
                value = composites.get(key)
                target_list.append(f"{base_path}/{value}" if value else None)

        self.new_product_list = []
        self.new_exp_bands_count_list = []

        special_sos_composites_dict = self.config_data.get("sos_special_composites", {}) if self.sos else {}
        for idx, product in enumerate(self.product_list):
            if self.sos and product in special_sos_composites_dict:
                composite_info = special_sos_composites_dict[product]
                self.new_product_list.extend(composite_info["actual_composite"])
                self.new_exp_bands_count_list.extend(composite_info["expected_bands_count"])
            else:
                self.new_product_list.append(product)
                self.new_exp_bands_count_list.append(self.exp_bands_count_list[idx])

        seen = set()
        pop_list = [idx for idx, item in enumerate(self.new_product_list) if item in seen or seen.add(item)]
        pop_list.reverse()

        for pop in pop_list:
            self.new_product_list.pop(pop)
            self.new_exp_bands_count_list.pop(pop)

        self.band_list = list({}.fromkeys(self.band_list).keys())
        self.band_image_list = list({}.fromkeys(self.band_image_list).keys())

    def valid_area(self) -> tuple[bool, list[str]]:
        area_errors = [
            (lambda: len(self.target_xy.split()) != 4, '\n"非经纬度xy坐标"填写有误'),
            (lambda: self.target_width <= 0 or self.target_height <= 0, '\n"宽/高"填写有误'),
            (lambda: self.resolution_x != self.resolution_y, '\n宽/高方向分辨率不相等\n检查 "宽/高" 和 "非经纬度xy坐标"')
        ]

        errors: list[str] = []
        area_pass = False

        for condition, error_message in area_errors:
            if condition(self):
                errors.append(error_message)

        if not errors:
            area_pass = True

        return area_pass, errors

    def arrange_sat_files(self) -> tuple[list[str], dict[str, dict[str, str]], list[str]]:
        arrange_funcs = {
            "GK2A": lambda: arrange_gk2a_files(self.base_path, self.band_list),
            "HS": lambda: arrange_himawari_files(self.base_path, self.band_list),
            "FY4A": lambda: arrange_fy4_files(self.sat, self.base_path, self.band_list),
            "FY4B": lambda: arrange_fy4_files(self.sat, self.base_path, self.band_list),
            "FY4C": lambda: arrange_fy4_files(self.sat, self.base_path, self.band_list),
            "G16": lambda: arrange_goes_files(self.sat, self.base_path, self.band_list),
            "G17": lambda: arrange_goes_files(self.sat, self.base_path, self.band_list),
            "G18": lambda: arrange_goes_files(self.sat, self.base_path, self.band_list),
            "G19": lambda: arrange_goes_files(self.sat, self.base_path, self.band_list),
            "MTG-I1": lambda:arrange_fci_files(self.sat, self.base_path, self.band_list)
        }

        if self.sat not in arrange_funcs:
            return [], {}, []

        func = arrange_funcs[self.sat]
        missing_file, process_dict, actual_file, errors = func(self)

        return missing_file, process_dict, errors

    def get_geos_crop_boundary(self) -> tuple[float | int, float | int, float | int, float | int]:
        void_file = f"/vsimem/void_{self.sat}_{self.area_id}.tif"
        crop_ref_file = f"/vsimem/void_{self.sat}_{self.area_id}_crop_ref.tif"

        projection = self.target_proj
        geotransform = (self.area_min_x, self.resolution_x, 0.0, self.area_max_y, 0.0, -self.resolution_y)
        with gdal.GetDriverByName("GTiff").Create(void_file, self.target_width, self.target_height,
                                                  1, gdal.GDT_Byte) as void_ds:
            void_ds.SetProjection(projection)
            void_ds.SetGeoTransform(geotransform)
            void_ds.FlushCache()

        warp_options = gdal.WarpOptions(format="GTiff", dstSRS=self.original_geos_proj, xRes=10000, yRes=10000,
                                        multithread=True,
                                        errorThreshold=0, warpMemoryLimit=21474836480,
                                        resampleAlg=gdal.GRA_NearestNeighbour,
                                        warpOptions=["NUM_THREADS=ALL_CPUS"],
                                        callback=gdal.TermProgress_nocb)

        with (gdal.Warp(crop_ref_file, void_file, options=warp_options) as crop_ds):
            crop_geotransform = crop_ds.GetGeoTransform()

            if self.target_proj != self.original_geos_proj:
                min_x = crop_geotransform[0]
                max_y = crop_geotransform[3]
                max_x = min_x + crop_geotransform[1] * crop_ds.RasterXSize
                min_y = max_y + crop_geotransform[5] * crop_ds.RasterYSize
                self_tolerance = 0.25 if "latlon" in self.original_geos_proj or "longlat" in self.original_geos_proj \
                                 else 1000
                res_tolerance = 0.25 if "latlon" in self.target_proj or "longlat" in self.target_proj else 1000
                crop_min_x = round(min_x / self_tolerance) * self_tolerance - 5 * round(
                    self.resolution / res_tolerance) * self_tolerance
                crop_min_y = round(min_y / self_tolerance) * self_tolerance - 5 * round(
                    self.resolution / res_tolerance) * self_tolerance
                crop_max_x = round(max_x / self_tolerance) * self_tolerance + 5 * round(
                    self.resolution / res_tolerance) * self_tolerance
                crop_max_y = round(max_y / self_tolerance) * self_tolerance + 5 * round(
                    self.resolution / res_tolerance) * self_tolerance
                crop_boundary = (crop_min_x, crop_min_y, crop_max_x, crop_max_y)

            else:
                if not self.target_width == self.target_height:
                    crop_min_x = self.area_min_x
                    crop_min_y = self.area_min_y
                    crop_max_x = self.area_max_x
                    crop_max_y = self.area_max_y
                    crop_boundary = (crop_min_x, crop_min_y, crop_max_x, crop_max_y)
                else:
                    crop_boundary = ()

        gdal.Unlink(void_file)
        gdal.Unlink(crop_ref_file)

        return crop_boundary

    @staticmethod
    def background_file_handling(local_file_list: list[str], satpy_file_list: list[str],
                                 image_geometry: tuple[str,
                                                       tuple[int | float,
                                                             int | float,
                                                             int | float,
                                                             int | float,
                                                             int | float,
                                                             int | float],
                                                       int, int,
                                                       int | float,
                                                       int | float,
                                                       int | float,
                                                       int | float]) -> None:

        proj, geotransform, width, height, area_min_x, area_min_y, area_max_x, area_max_y = image_geometry

        def do_warp():
            gdalwarp_by_extent(proj, gdal.GRA_NearestNeighbour, local_file, satpy_file,
                               target_width=width, target_height=height,
                               area_min_x=area_min_x, area_min_y=area_min_y,
                               area_max_x=area_max_x, area_max_y=area_max_y)

        for local_file, satpy_file in zip(local_file_list, satpy_file_list):
            if not (local_file and satpy_file):
                continue

            if not os.path.exists(satpy_file):
                do_warp()

            else:
                with gdal.Open(satpy_file) as satpy_ds:
                    satpy_proj = satpy_ds.GetProjection()
                    satpy_tran = satpy_ds.GetGeoTransform()

                if satpy_tran != geotransform or Proj(satpy_proj) != Proj(proj):
                    gdal.Unlink(satpy_file)
                    do_warp()

    def fix_area_for_non_fldk_and_geos(self, folder: str) -> tuple[int, int,
                                                                   int | float,
                                                                   int | float,
                                                                   int | float,
                                                                   int | float,]:
        files = find_files_and_readers(base_dir=folder, reader=self.reader_name)
        scn0 = Scene(filenames=files)
        scn0.load([self.band_image_list[0]])
        actual_minx, actual_miny, actual_maxx, actual_maxy = scn0[self.band_image_list[0]].attrs["area"].area_extent

        actual_width = scn0[self.band_image_list[0]].attrs["area"].width
        actual_height = scn0[self.band_image_list[0]].attrs["area"].height
        actual_resolution_x = (actual_maxx - actual_minx) / actual_width
        acutal_resolution_y = (actual_maxy - actual_miny) / actual_height

        res_x_ratio = actual_resolution_x / self.resolution_x
        res_y_ratio = acutal_resolution_y / self.resolution_y

        target_width = round(actual_width * res_x_ratio)
        target_height = round(actual_height * res_y_ratio)
        area_min_x, area_min_y, area_max_x, area_max_y = actual_minx, actual_miny, actual_maxx, actual_maxy

        del scn0
        return target_width, target_height, area_min_x, area_min_y, area_max_x, area_max_y

    def satpy_processor(self, folder: str, sat_name: str, scan_area: str,
                        crop_boundary: tuple) -> dict[str, dict[str, str | datetime]]:
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=UserWarning)
        warnings.simplefilter(action="ignore", category=RuntimeWarning)

        trollimage.xrimage.XRImage.rio_save = rio_save
        geotiff.GeoTIFFWriter.GDAL_OPTIONS = gdal_options
        satpy.readers.generic_image._mask_image_data = _mask_image_data
        satpy.composites.MaskingCompositor = MyOwnMaskingCompositor
        satpy.composites.CloudCompositor = MyOwnCloudCompositor
        satpy.composites.HighCloudCompositor.__bases__ = (MyOwnCloudCompositor,)
        satpy.composites.LowCloudCompositor.__bases__ = (MyOwnCloudCompositor,)
        satpy.enhancements._jma_true_color_reproduction = _jma_true_color_reproduction

        config.set(config_path=[self.satpy_config_path])
        utils.debug_off()

        area_min_x = self.area_min_x
        area_min_y = self.area_min_y
        area_max_x = self.area_max_x
        area_max_y = self.area_max_y
        target_width = self.target_width
        target_height = self.target_height
        geotransform = (area_min_x, self.resolution_x, 0.0, self.area_max_y, 0.0, -self.resolution_y)

        tmp_folder = f"{folder}/work_tmp"
        if not os.path.exists(tmp_folder) or (os.path.exists(tmp_folder) and not os.path.isdir(tmp_folder)):
            os.mkdir(tmp_folder)

        if scan_area != "FLDK" and "geos" in self.target_proj:
            (target_width, target_height,
             area_min_x, area_min_y, area_max_x, area_max_y) = self.fix_area_for_non_fldk_and_geos(folder)

        is_global_latlon = (("latlon" in self.target_proj or "longlat" in self.target_proj)
                            and area_min_x == -180 and area_min_y == -90
                            and area_max_x == 180 and area_max_y == 90)

        self.background_file_handling(self.background_local_file_list, self.background_satpy_file_list,
                                      (self.target_proj, geotransform, target_width, target_height,
                                      area_min_x, area_min_y, area_max_x, area_max_y))
        self.background_file_handling(self.landmask_local_file_list, self.landmask_satpy_file_list,
                                      (self.target_proj, geotransform, target_width, target_height,
                                      area_min_x, area_min_y, area_max_x, area_max_y))
        self.background_file_handling(self.globalmask_local_file_list, self.globalmask_satpy_file_list,
                                      (self.target_proj, geotransform, target_width, target_height,
                                      area_min_x, area_min_y, area_max_x, area_max_y))

        start_load = time_calc.perf_counter()
        files = find_files_and_readers(base_dir=folder, reader=self.reader_name)
        scn = Scene(filenames=files)
        scn.load(self.band_image_list)

        if scan_area == "FLDK":
            if self.target_proj == self.original_geos_proj and target_width == target_height:
                scn2 = scn
            else:
                scn2 = scn if is_global_latlon else scn.crop(xy_bbox=crop_boundary)
        else:
            scn2 = scn

        end_load = time_calc.perf_counter()
        print(f"加载卫星数据集耗时: {(end_load - start_load): .3f}")

        res_list = []
        image: dict[str, dict[str, str]] = {}
        start_save = time_calc.perf_counter()
        for band_image in self.band_image_list:
            output = (f"{sat_name}_{scn[band_image].attrs["sensor"]}_{band_image}_"
                      f"{scn.start_time.strftime("%Y%m%d%H%M%S")}_{scan_area}")
            output_filename = f"{output}.tif"
            attrs = {"proj_id": self.area_id.split("_")[0],
                     "area_id": self.area_id.split("_")[1],
                     "platform_name": scn[band_image].attrs["platform_name"].lower(),
                     "sensor": scn[band_image].attrs["sensor"],
                     "start_time": scn[band_image].attrs["start_time"].strftime("%Y%m%d%H%M%S.%fZ"),
                     "end_time": scn[band_image].attrs["end_time"].strftime("%Y%m%d%H%M%S.%fZ"),
                     "orb_params": scn[band_image].attrs["orbital_parameters"]}
            res = scn2.save_dataset(band_image, filename=output_filename, base_dir=tmp_folder, writer="geotiff",
                                   blockxsize=512, blockysize=512,
                                   enhance=False, dtype=np.float32, fill_value=np.nan, compute=False, tags=attrs,
                                   in_memory=True)

            res_list.append(res)

            # image[output] = {"output_tif": f"{tmp_folder}/{output}.tif",
            #                  "warp_tif": f"{tmp_folder}/{output}_{self.area_id}.tif",
            #                  }
            # image[output] = {"output_tif": f"/vsimem/{output}.tif",
            #                  "warp_tif": f"{tmp_folder}/{output}_{self.area_id}.tif",
            #                  }
            image[output] = {"output_tif": f"/vsimem/{output}.tif",
                             "warp_tif": f"/vsimem/{output}_{self.area_id}.tif",
                             }

        compute_writer_results(res_list)
        end_save = time_calc.perf_counter()
        print(f"存储为中介数据耗时: {(end_save - start_save): .3f}")

        del scn, scn2, res, res_list

        start_warp = time_calc.perf_counter()
        for output_info in image.values():
            gdalwarp_by_extent(self.target_proj, gdal.GRA_Lanczos, output_info["output_tif"], output_info["warp_tif"],
                               target_width=target_width, target_height=target_height,
                               area_min_x=area_min_x, area_min_y=area_min_y,
                               area_max_x=area_max_x, area_max_y=area_max_y,
                               creationoptions=["TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"])
            gdal.Unlink(output_info["output_tif"])
        end_warp = time_calc.perf_counter()
        print(f"中介数据重投影耗时: {(end_warp - start_warp): .3f}")

        with gdal.Open(image[list(image.keys())[0]]["warp_tif"]) as ds:
            mask_data = ds.GetRasterBand(1).ReadAsArray(0, 0, target_width, target_height)
            mask_data = np.where(np.isnan(mask_data), 0, 255)

        start_final_load = time_calc.perf_counter()
        files = [output_info["warp_tif"] for output_info in image.values()]
        scn = Scene(filenames=files, reader=f"geo_float_image_{self.reader_name}")
        scn.load(self.new_product_list)
        end_final_load = time_calc.perf_counter()
        print(f"加载重投影后的中介数据耗时: {(end_final_load - start_final_load): .3f}")

        res_list2 = []
        product_dict: dict[str, dict[str, str | datetime]] = {}
        start_final_product = time_calc.perf_counter()
        for idx, product in enumerate(self.new_product_list):
            if "geos" in self.target_proj:
                lons, lats = scn[scn.keys()[0]["name"]].attrs["area"].get_lonlats()
                for output_naming in scn.keys():
                    scn[output_naming["name"]].data = np.where(np.isinf(lons), np.nan, scn[output_naming["name"]].data)
                mask_data = np.where(np.isinf(lons), 0, mask_data)

            exp_bands_count = self.new_exp_bands_count_list[idx]
            fill_value = None if exp_bands_count in [2, 4] or "geos" in self.target_proj else 0

            final = f"{sat_name}_{product}_{self.area_id}_{scn.start_time.strftime("%Y%m%d_%H%M%S")}_{scan_area}"
            final_filename = f"{final}.tif"
            final_file = f"{self.base_path}/{final_filename}"
            boundary_file = f"{self.base_path}/zzz_boundary_{final}.tif"

            res2 = scn.save_dataset(product, filename=final_filename, base_dir=self.base_path, writer="geotiff",
                                    blockxsize=512, blockysize=512,
                                    fill_value=fill_value, compute=False, compress="deflate", zlevel=6, predictor=2)

            res_list2.append(res2)

            if os.path.exists(boundary_file):
                gdal.Unlink(boundary_file)

            with gdal.GetDriverByName("GTiff").Create(boundary_file, target_width, target_height, 1, gdal.GDT_Byte,
                                    {"COMPRESS": "DEFLATE", "ZLEVEL": "6", "PREDICTOR": "2"}) as mask_ds:
                mask_ds.SetProjection(self.target_proj)
                mask_ds.SetGeoTransform(geotransform)
                mask_ds.GetRasterBand(1).WriteArray(mask_data)
                mask_ds.GetRasterBand(1).SetNoDataValue(255)
                mask_ds.FlushCache()

            product_dict[product] = {"file_name": final_file,
                                     "boundary_name": boundary_file,
                                     "sat_name": sat_name,
                                     "scan_area": scan_area,
                                     "start_date": scn.start_time}
        compute_writer_results(res_list2)
        end_final_product = time_calc.perf_counter()
        print(f"存为最终产品耗时: {(end_final_product - start_final_product): .3f}")

        del scn, res2, res_list2
        return product_dict

    def run_satpy_processor_core(self, sat: str, arrange_time: float | int,
                                 process_dict: dict[str, dict[str, str]]) -> tuple[dict[str, dict[str, str]],
                                                                                   dict[str, str],
                                                                                   dict[str,
                                                                                        dict[str, list[dict] | int]]]:
        product_dict_list: list[dict[str, dict[str, str | datetime]]] = []
        mosaic_piece: dict[str, dict[str, list[dict] | int]] = {}

        time_info: dict[str, dict[str, str]] = {}
        err_info: dict[str, str] = {}

        print(f"开始处理{sat}卫星数据")
        start_sat = time_calc.perf_counter()
        crop_boundary = self.get_geos_crop_boundary()

        process_folders = list(process_dict.keys())
        count = len(process_folders)
        done_count = 0

        for process_folder, folder_info in process_dict.items():
            start = time_calc.perf_counter()

            sat_name = folder_info["sat_name"]
            scan_area = folder_info["scan_area"]
            try:
                product_dict = self.satpy_processor(process_folder, sat_name, scan_area, crop_boundary)

                product_dict_list.append(product_dict)

            except Exception as e:
                print(e)
                err_info[process_folder] = f"{sat}卫星下的{process_folder}文件夹报错, 报错信息为: {e}"

            end = time_calc.perf_counter()
            done_count = done_count + 1
            print(f"{sat}卫星单个节点数据耗时: {(end - start): .3f}")
            print("******************************\n"
                  "\n"
                  f"-----时间节点完成: {done_count} / {count}------\n"
                  "\n"
                  "******************************\n")

        end_sat = time_calc.perf_counter()
        single_sat_timing = f"{self.sat}卫星处理总耗时: {get_elapsed_time(start_sat, arrange_time + end_sat)}"
        print(single_sat_timing, "\n")


        time_info[sat] = {"start_folder": f"{self.sat}卫星处理开始于: {process_folders[0]}",
                          "end_folder": f"{self.sat}卫星处理结束于: {process_folders[-1]}",
                          "total_time": single_sat_timing} if count > 0 else \
                         {"start_folder": f"{self.sat}卫星并无数据",
                          "end_folder": f"{self.sat}卫星并无数据",
                          "total_time": single_sat_timing}

        for product, exp_bands_count in zip(self.new_product_list, self.new_exp_bands_count_list):
            mosaic_piece[product] = {"images": [], "exp_bands_count": exp_bands_count}
            for product_dict in product_dict_list:
                if product in product_dict:
                    mosaic_piece[product]["images"].append(product_dict[product])

        return time_info, err_info, mosaic_piece

    def run_satpy_processor(self) -> tuple[dict[str,
                                           dict[str, list[str] | str]],
                                           dict[str, dict[str, str]],
                                           float,
                                           dict[str, list[str]],
                                           dict[str, list[str]],
                                           dict[str, str]]:
        print("\033[H\033[J")

        area_pass, errors = self.valid_area()

        sat_list = [self.sat] if not self.sos else self.global_satellites

        satpy_process_time_dict: dict[str, dict[str, str]] = {}
        satpy_error_dict: dict[str, str] = {}
        mosaic_dict: dict[str, dict[str, list[str] | int]] = {}

        arrange_error_dict: dict[str, list[str]] = {}
        satpy_missing_dict: dict[str, list[str]] = {}

        start_bj_time = f"卫星数据处理开始于: {(datetime.now(UTC) + timedelta(hours=8)).strftime(
            "%Y年%m月%d日%H时%M分%S秒")}"
        start_0 = time_calc.perf_counter()

        for sat in sat_list:
            self.update_sat(sat)
            print(f"开始整理{sat}卫星数据")
            start_arrange = time_calc.perf_counter()
            missing_file, process_dict, errors = self.arrange_sat_files()
            end_arrange = time_calc.perf_counter()
            print(f"整理卫星数据耗时: {(end_arrange - start_arrange): .3f}\n")

            for err in errors:
                print(err)

            arrange_error_dict[sat] = errors

            if area_pass and not missing_file:
                time_info, err_info, mosaic_piece = self.run_satpy_processor_core(sat,
                                                                                  end_arrange - start_arrange,
                                                                                  process_dict)
                satpy_process_time_dict.update(time_info)
                satpy_error_dict.update(err_info)
                mosaic_dict.update(mosaic_piece)

            else:
                for error in errors:
                    print(error)

                if missing_file:
                    satpy_missing_dict[sat] = [missing for missing in missing_file]
                    print("\n以下数据集有缺失:")
                    for file in missing_file:
                        print(file)

        end_0 = time_calc.perf_counter()
        end_bj_time = f"卫星数据处理结束于: {(datetime.now(UTC) + timedelta(hours=8)).strftime(
            "%Y年%m月%d日%H时%M分%S秒")}"

        all_sats_time = end_0 - start_0
        all_sats_timing = f"所有卫星处理总耗时: {get_elapsed_time(start_0, end_0)}"

        if self.sos:
            print(all_sats_timing)

        satpy_process_time_dict["satpy_summary"] = {"total_time": all_sats_timing,
                                                    "start_bj_time": start_bj_time,
                                                    "end_bj_time": end_bj_time}

        for err in satpy_error_dict.values():
            print(err)

        return (mosaic_dict, satpy_process_time_dict, all_sats_time, arrange_error_dict,
                satpy_missing_dict, satpy_error_dict)

    def write_black_image(self, mosaic_output: str, exp_bands_count: int, metadata: dict):
        with gdal.GetDriverByName("GTiff").Create(
                mosaic_output, self.target_width, self.target_height, exp_bands_count, gdal.GDT_Byte,
                {"COMPRESS": "DEFLATE", "ZLEVEL": "6", "PREDICTOR": "2"}) as ds:
            ds.SetProjection(self.target_proj)
            ds.SetGeoTransform((self.area_min_x, self.resolution_x, 0.0,
                                self.area_max_y, 0.0, -self.resolution_y))
            ds.SetMetadata(metadata)
            ds.FlushCache()

    def global_mosaic_single_node(self, date: datetime,
                                  mosaic_dict: dict[str, dict[str, list[dict] | str | int]]) -> tuple[dict[str,
                                                                                                      list[str]],
                                                                                                      dict[str, str]]:
        def gk2a_is_tolerated(dt: datetime):
            return (
                    (dt.hour == 15 and dt.minute == 20 and datetime(dt.year, 4, 20) <= dt <= datetime(dt.year,
                                                                                                      8, 21)) or
                    (dt.hour == 6 and dt.minute == 0 and datetime(dt.year, 8, 22) <= dt <= datetime(dt.year,
                                                                                                    12, 20)) or
                    (dt.hour == 0 and dt.minute == 40 and (
                            datetime(dt.year, 12, 21) <= dt <= datetime(dt.year, 12, 31) or
                            datetime(dt.year, 1, 1) <= dt <= datetime(dt.year, 4, 19)))
            )
        sat_region_config = {
            "GK2A": ("asia_west_pac1_pass", "raster_asia_west_pac1_image", ["GK2A"], "FLDK", gk2a_is_tolerated),
            "HS": ("asia_west_pac2_pass", "raster_asia_west_pac2_image", ["H08", "H09"], "FLDK",
                   lambda d: d.hour in [2, 14] and d.minute == 40),
            "GOES-WEST": ("goes_west_pass", "raster_goes_west_image", ["G17", "G18"], "FLDK", None),
            "GOES-EAST": ("goes_east_pass", "raster_goes_east_image", ["G16"], "FLDK", None),
            "MTG-FD": ("europe_pass", "raster_europe_image", ["MTG-I1"], "FLDK", None),
        }

        failure_info: dict[str, list[str]] = {}
        error_info: dict[str, str] = {}
        clean_info: dict[str, str] = {}

        date_str = date.strftime("%Y%m%d_%H%M%S")

        res_list = []
        mosaics = []

        failure_key = date.strftime("%Y年%m月%d日%H时%M分")
        failure_info[failure_key] = []

        tmp_folder = f"{self.base_path}/global_{date_str}_tmp"
        if not os.path.exists(tmp_folder) or (os.path.exists(tmp_folder) and not os.path.isdir(tmp_folder)):
            os.mkdir(tmp_folder)
        clean_info[tmp_folder] = "folder"

        if len(mosaic_dict.keys()) == 0:
            failure_info[failure_key].append("所有产品全部无法生成")

        for product, info in mosaic_dict.items():
            image_dict_list = info["images"]
            exp_bands_count = info["exp_bands_count"]

            pass_dict = {key: False for key, *_ in sat_region_config.values()}
            raster_images = {key: {} for key, *_ in sat_region_config.values()}

            raster_list = []
            mask_list = []

            for image_dict in image_dict_list:
                sat_name = image_dict["sat_name"]
                scan_area = image_dict["scan_area"]
                start_date = image_dict["start_date"]

                if abs(date - start_date) > timedelta(seconds=80) or scan_area != "FLDK":
                    continue

                for region_pass, region_img, sats, _, _ in sat_region_config.values():
                    if sat_name in sats:
                        pass_dict[region_pass] = True
                        raster_images[region_img]["file"] = image_dict["file_name"]
                        raster_images[region_img]["boundary"] = image_dict["boundary_name"]

            for sat_key, (region_pass, region_img, _, _, time_check) in sat_region_config.items():
                if pass_dict[region_pass]:
                    raster_list.append(raster_images[region_img]["file"])
                    mask_list.append(raster_images[region_img]["boundary"])
                else:
                    if time_check is None or not time_check(date):
                        failure_info[failure_key].append(f"{sat_key}: {product}")

            metadata = {"start_time": date.strftime("%Y%m%d%H%M%S.%fZ"),
                        "end_time": (date + timedelta(minutes=10)).strftime("%Y%m%d%H%M%S.%fZ"),
                        "proj_id": self.area_id.split("_")[0],
                        "area_id": self.area_id.split("_")[1], }

            # mosaic_output = f"{tmp_folder}/mosaic_{product}_{date_str}.tif"
            mosaic_output = f"/vsimem/mosaic_{product}_{date_str}.tif"

            if raster_list:
                mosaic_operator = Mosaic(raster_list, 0, mosaic_output, mask_list)
                mosaic_error = mosaic_operator.large_feather_mosaic(need_final_bound=False, tags=metadata,
                                                                    format_kwargs={"TILED": "YES", "BLOCKXSIZE": "512",
                                                                                   "BLOCKYSIZE": "512",
                                                                                   "COMPRESS": "DEFLATE",
                                                                                   "ZLEVEL": "6"})

                if mosaic_error:
                    self.write_black_image(mosaic_output, exp_bands_count, metadata)
                    error_info[failure_key] = mosaic_error

                for img in raster_list:
                    clean_info[img] = "file"

                for mask in mask_list:
                    clean_info[mask] = "file"

            else:
                self.write_black_image(mosaic_output, exp_bands_count, metadata)

            mosaics.append(mosaic_output)

        if len(mosaic_dict.keys()) > 0:
            scn = Scene(filenames=mosaics, reader="geo_int_global_image", reader_kwargs={"mode": "int"})
            scn.load(self.product_list)

            for product, bands_count in zip(self.product_list, self.exp_bands_count_list):
                fill_value = None if bands_count in [2, 4] else 0

                final = f"{product}_{self.area_id}_{scn.start_time.strftime("%Y%m%d_%H%M%S")}.tif"
                res = scn.save_dataset(product, filename=final, base_dir=self.base_path, writer="geotiff",
                                       blockxsize=512, blockysize=512, compress="deflate", zlevel=6, predictor=2,
                                       fill_value=fill_value, compute=False)
                res_list.append(res)

            compute_writer_results(res_list)

        return failure_info, clean_info

    @staticmethod
    def handling_global_mosaic_results(start_end: tuple[str, str],
                                       report_dicts: tuple[dict[str, list[str]],
                                                           dict[str, str],
                                                           dict[str, dict[str, str]],
                                                           dict[str, list[str]],
                                                           dict[str, str],
                                                           dict[str, list[str]]]) -> tuple[dict[str, str],
                                                                                           dict[str, str]]:
        start_dt_str, end_dt_str = start_end
        (failure_dict, mosaic_process_time_dict,
         satpy_process_time_dict, satpy_missing_dict, satpy_error_dict, arrange_error_dict) = report_dicts

        process_msg_list: list[str] = []
        warning_msg = ", 但有错误发生" if len(failure_dict.keys()) > 0 else ""
        process_topic = f"{start_dt_str}至{end_dt_str}的全球影像生成完毕{warning_msg}"
        process_msg_list.append(mosaic_process_time_dict["all_time"])
        process_msg_list.append("\n")

        for time_info in satpy_process_time_dict.values():
            for timing in time_info.values():
                process_msg_list.append(timing)
            process_msg_list.append("\n")
        process_msg_list.append("\n")

        for key, timing in mosaic_process_time_dict:
            if key != "all_time":
                process_msg_list.append(timing)
        process_msg_list.append("\n")
        process_msg = "\n".join(process_msg_list)

        process_email_dict = {"topic": process_topic, "message": process_msg}

        warning_email_dict: dict[str, str] = {}
        if len(failure_dict.keys()) > 0:
            warning_topic = f"警告!!! {start_dt_str}至{end_dt_str}的全球影像生成出现问题"

            warning_msg_list = ["下面列出可能缺失产品的时间点: \n"]
            for time_node, product in failure_dict:
                warning = f"{time_node}, 缺失如下产品: {", ".join(product)}"
                warning_msg_list.append(warning)
            warning_msg_list.append("\n")

            for sat, missing_list in satpy_missing_dict:
                warning_msg_list.append(f"{sat}卫星的下列数据有缺失: \n")
                warning_msg_list.extend(missing_list)
            warning_msg_list.append("\n")

            if len(satpy_error_dict.keys()) > 0:
                warning_msg_list.append("satpy处理过程中报错: \n")
                for err in satpy_error_dict.values():
                    warning_msg_list.append(err)
                warning_msg_list.append("\n")

            for sat, err_list in arrange_error_dict:
                if err_list:
                    warning_msg_list.append(f"{sat}卫星数据整理时发生错误: \n")
                    warning_msg_list.extend(err_list)
                    warning_msg_list.append("\n")
            warning_msg_list.append("\n")
            warning_msg = "\n".join(warning_msg_list)

            warning_email_dict = {"topic": warning_topic, "message": warning_msg}

        return process_email_dict, warning_email_dict

    def global_mosaic(self, start_dt, end_dt):
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=UserWarning)
        warnings.simplefilter(action="ignore", category=RuntimeWarning)

        trollimage.xrimage.XRImage.rio_save = rio_save
        satpy.readers.generic_image._mask_image_data = _mask_image_data
        satpy.composites.MaskingCompositor = MyOwnMaskingCompositor
        satpy.composites.CloudCompositor = MyOwnCloudCompositor
        satpy.composites.HighCloudCompositor.__bases__ = (MyOwnCloudCompositor,)
        satpy.composites.LowCloudCompositor.__bases__ = (MyOwnCloudCompositor,)
        satpy.enhancements._jma_true_color_reproduction = _jma_true_color_reproduction

        (mosaic_dict, satpy_process_time_dict, all_sats_time, arrange_error_dict,
         satpy_missing_dict, satpy_error_dict) = self.run_satpy_processor()

        start_dt = datetime.strptime(start_dt, "%Y%m%d%H%M")
        end_dt = datetime.strptime(end_dt, "%Y%m%d%H%M")

        date_minus = int((end_dt - start_dt).total_seconds())
        count = date_minus // 600

        config.set(config_path=[self.satpy_config_path])
        utils.debug_off()

        start_bj_time = f"融合处理开始于: {(datetime.now(UTC) + timedelta(hours=8)).strftime(
            "%Y年%m月%d日%H时%M分%S秒")}"
        start_0 = time_calc.perf_counter()

        failure_dict: dict[str, list[str]] = {}
        clean_dict: dict[str, str] = {}

        done_count = 0
        for date in date_range(start_dt, end_dt, timedelta(minutes=10)):
            start_time_node = time_calc.perf_counter()

            failure_info, clean_info = self.global_mosaic_single_node(date, mosaic_dict)
            failure_dict.update(failure_info)
            clean_dict.update(clean_info)

            done_count = done_count + 1

            end_time_node = time_calc.perf_counter()
            single_node_timing = f"单个时间节点融合耗时: {get_elapsed_time(start_time_node, end_time_node)}"
            print(single_node_timing)
            print("\n"
                  "******************************\n"
                  "\n"
                  f"---单个时间节点融合完成: {done_count} / {count}---\n"
                  "\n"
                  "******************************\n")

        end_0 = time_calc.perf_counter()
        end_bj_time = f"融合处理结束于: {(datetime.now(UTC) + timedelta(hours=8)).strftime(
            "%Y年%m月%d日%H时%M分%S秒")}"

        all_nodes_timing = f"所有时间节点融合总耗时: {get_elapsed_time(start_0, end_0)}"
        print(all_nodes_timing)

        all_timing = f"处理全流程总耗时: {get_elapsed_time(start_0, end_0 + all_sats_time)}"
        print(all_timing)

        start_dt_str = start_dt.strftime("%Y年%m月%d日%H时%M分")
        end_dt_str = end_dt.strftime("%Y年%m月%d日%H时%M分")
        dt_range = f"起始时间节点: {start_dt_str}   结束时间节点: {end_dt_str}"
        mosaic_process_time_dict = {"all_time": all_timing,
                                    "total_time": all_nodes_timing,
                                    "start_bj_time": start_bj_time,
                                    "end_bj_time": end_bj_time,
                                    "dt_range": dt_range}

        process_email_dict, warning_email_dict = self.handling_global_mosaic_results(
            (start_dt_str, end_dt_str), (failure_dict, mosaic_process_time_dict, satpy_process_time_dict,
                                         satpy_missing_dict, satpy_error_dict, arrange_error_dict))

        return clean_dict, process_email_dict, warning_email_dict

    def run_clean(self, clean_dict: dict = None, all_clean: bool = False):
        sat_list = [self.sat] if not self.sos else self.global_satellites
        for sat in sat_list:
            self.update_sat(sat)
            print(f"开始善后{sat}卫星数据")
            missing_file, process_dict, errors = self.arrange_sat_files()
            if all_clean:
                for folder in process_dict:
                    try:
                        shutil.rmtree(folder, ignore_errors=True)
                    except Exception as e:
                        print(e)
                        pass
            print(f"{sat}卫星数据善后完毕")

        if clean_dict is not None:
            print("开始善后其他临时文件")
            for key in clean_dict.keys():
                if clean_dict[key] == "folder":
                    try:
                        shutil.rmtree(key, ignore_errors=True)
                    except Exception as e:
                        print(e)
                        pass

                elif clean_dict[key] == "file":
                    try:
                        os.remove(key)
                    except Exception as e:
                        print(e)
                        pass
            print("其他临时文件善后完毕")


class GeoCalculator:
    def __init__(self, config_data, sat_index, calc_latlon, calc_proj, calc_name, calc_area, calc_area_id,
                 need_square_correction=False, square_correction_lim=400):
        super().__init__()
        self.config_data = config_data
        self.sat_index = sat_index
        satellites_dict = config_data["satellites"]
        self.calc_sat = list(satellites_dict.keys())[self.sat_index]
        self.calc_latlon = calc_latlon
        self.calc_proj = calc_proj
        self.calc_name = calc_name
        self.calc_area = calc_area
        self.calc_area_id = calc_area_id
        self.need_square_correction = need_square_correction
        self.square_correction_lim = square_correction_lim

    def projection_calc(self):
        def calculate_diff(value):
            if value % 2 != 0:
                min_diff = (value - 1) / 2
                max_diff = (value + 1) / 2
            else:
                min_diff = max_diff = value / 2

            return min_diff, max_diff

        target_res = 4000 / 111 * 0.001 if "latlon" in self.calc_proj or "longlat" in self.calc_proj else 4000

        min_lon, min_lat, max_lon, max_lat = map(float, self.calc_latlon.split())
        min_lon_digits, min_lat_digits, max_lon_digits, max_lat_digits = map(decimal_digits, self.calc_latlon.split())

        max_digits = max(min_lon_digits, min_lat_digits, max_lon_digits, max_lat_digits)

        width_latlon = round((max_lon - min_lon) * math.pow(10, max_digits))
        height_latlon = round((max_lat - min_lat) * math.pow(10, max_digits))
        res_latlon = 1 / math.pow(10, max_digits)
        tran_latlon = (min_lon, res_latlon, 0.0, max_lat, 0.0, -res_latlon)

        with gdal.GetDriverByName("VRT").Create("/vsimem/sample.vrt", width_latlon, height_latlon,
                                                1, gdal.GDT_Byte) as ds:
            ds.SetProjection("EPSG:4326")
            ds.SetGeoTransform(tran_latlon)

            warp_options = gdal.WarpOptions(format="GTiff", dstSRS=self.calc_proj, xRes=target_res, yRes=target_res,
                                            multithread=True,
                                            errorThreshold=0, warpMemoryLimit=21474836480,
                                            resampleAlg=gdal.GRA_NearestNeighbour,
                                            warpOptions=["NUM_THREADS=ALL_CPUS"],
                                            callback=gdal.TermProgress_nocb)

            ds2 = gdal.Warp("/vsimem/samplewarp.tif", ds, options=warp_options)

            gt = ds2.GetGeoTransform()
            min_x = gt[0]
            max_y = gt[3]
            max_x = min_x + gt[1] * ds2.RasterXSize
            min_y = max_y + gt[5] * ds2.RasterYSize

        tolerance = 0.25 if "latlon" in self.calc_proj or "longlat" in self.calc_proj else 1000
        unit = "d" if "latlon" in self.calc_proj or "longlat" in self.calc_proj else "m"

        temp_min_x = round(min_x / tolerance) * tolerance - 1 * round(target_res / tolerance) * tolerance
        temp_min_y = round(min_y / tolerance) * tolerance - 1 * round(target_res / tolerance) * tolerance
        temp_max_x = round(max_x / tolerance) * tolerance + 1 * round(target_res / tolerance) * tolerance
        temp_max_y = round(max_y / tolerance) * tolerance + 1 * round(target_res / tolerance) * tolerance

        temp_width = round((temp_max_x - temp_min_x) / target_res)
        temp_height = round((temp_max_y - temp_min_y) / target_res)

        resolutions = [500, 1000, 2000, 4000]
        multi = [8, 4, 2, 1]

        lon0_match = re.search(r"\+lon_0=(\S+)", self.calc_proj)
        fi_lon0 = round(float(lon0_match.group(1))) if lon0_match else 0

        proj_indicator = re.search(r"\+proj=(\S+)", self.calc_proj).group(1)

        utc_date = datetime.now(UTC).strftime("%Y%m%d%H%M%S")

        for res in resolutions:
            factor = multi[resolutions.index(res)]

            if not self.need_square_correction:
                width = temp_width * factor
                height = temp_height * factor
                min_x_diff, max_x_diff = (0, 0)
                min_y_diff, max_y_diff = (0, 0)
            else:
                width = height = max(self.square_correction_lim, max(temp_width, temp_height)) * factor
                width_diff = width - temp_width
                height_diff = height - temp_height

                min_x_diff, max_x_diff = calculate_diff(width_diff)
                min_y_diff, max_y_diff = calculate_diff(height_diff)

            fi_min_x = round(temp_min_x - min_x_diff * res)
            fi_min_y = round(temp_min_y - min_y_diff * res)
            fi_max_x = round(temp_max_x + max_x_diff * res)
            fi_max_y = round(temp_max_y + max_y_diff * res)

            new_projection = {
                "name": self.calc_name,
                "area": self.calc_area,
                "center_lon": fi_lon0,
                "area_id": self.calc_area_id,
                "proj": self.calc_proj,
                "width": width,
                "height": height,
                "extent": f"{fi_min_x} {fi_min_y} {fi_max_x} {fi_max_y}",
                "res": res,
                "date": int(utc_date),
            }

            self.config_data["projection"][self.calc_sat][(f"{proj_indicator}_{self.calc_area_id}_"
                                                           f"{fi_lon0}_{res}{unit}_{utc_date}")] = new_projection

        return self.config_data


class GeoDownloader:
    def __init__(self, config_data, sat_index, product_index_list, start_date, start_time, end_date, end_time,
                 sos=False):
        super().__init__()
        self.config_data = config_data
        self.product_index_list = product_index_list
        self.sos = sos

        self.global_satellites = config_data["global_satellites"]
        self.conda_path = config_data["basic_path"]["conda_path"]
        self.base_path = config_data["basic_path"]["base_path"]
        self.satellites_dict = config_data["satellites"]
        self.goes_area = config_data["goes_area"]
        self.sat = list(self.satellites_dict.keys())[sat_index]

        self.start_date = start_date
        self.start_time = start_time
        self.end_date = end_date
        self.end_time = end_time

        self.band_list = []

        self.satellite = "G16" if self.sat in ["G16", "G17", "G18", "G19"] else self.sat
        product_dict = config_data["composites"][self.satellite]

        for product_idx in product_index_list:
            product = list(product_dict.keys())[product_idx]
            composites = config_data["composites"][self.satellite][product]
            for band in composites["band_list"]:
                self.band_list.append(band)

        self.band_list = list({}.fromkeys(self.band_list).keys())

    def update_sat(self, new_sat):
        self.sat = new_sat
        self.band_list = []

        self.satellite = "G16" if self.sat in ["G16", "G17", "G18", "G19"] else self.sat
        product_dict = self.config_data["composites"][self.satellite]

        for product_idx in self.product_index_list:
            product = list(product_dict.keys())[product_idx]
            composites = self.config_data["composites"][self.satellite][product]
            for band in composites["band_list"]:
                self.band_list.append(band)

        self.band_list = list({}.fromkeys(self.band_list).keys())

    def arrange_sat_files(self):
        if self.sat == "GK2A":
            missing_file, process_dict, actual_file, errors = arrange_gk2a_files(self.base_path, self.band_list,
                                                                                 only_need_existed=True,
                                                                                 move_to_folder=False)

        elif self.sat == "HS":
            missing_file, process_dict, actual_file, errors = arrange_himawari_files(self.base_path, self.band_list,
                                                                                     only_need_existed=True,
                                                                                     move_to_folder=False)

        elif self.sat in ["FY4A", "FY4B", "FY4C"]:
            missing_file, process_dict, actual_file, errors = arrange_fy4_files(self.sat, self.base_path,
                                                                                self.band_list,
                                                                                move_to_folder=False)

        elif self.sat in ["G16", "G17", "G18", "G19"]:
            missing_file, process_dict, actual_file, errors = arrange_goes_files(self.sat, self.base_path,
                                                                                 self.band_list,
                                                                                 only_need_existed=True,
                                                                                 move_to_folder=False)

        elif self.sat in ["MTG-I1", "MTG-I2"]:
            missing_file, process_dict, actual_file, errors = arrange_fci_files(self.sat, self.base_path,
                                                                                self.band_list,
                                                                                only_need_existed=True,
                                                                                move_to_folder=False)

        else:
            missing_file = []
            process_dict = {}
            actual_file = []
            errors = []

        return missing_file, process_dict, actual_file, errors

    def correct_date(self):
        def found_nearest_minute(date, minute_threshold):
            remainder = date.minute % minute_threshold

            if minute_threshold % 2 == 0:
                if remainder >= minute_threshold / 2:
                    date = date + timedelta(minutes=minute_threshold - remainder)
                else:
                    date = date - timedelta(minutes=remainder)

            else:
                if remainder >= (minute_threshold + 1) / 2:
                    date = date + timedelta(minutes=minute_threshold - remainder)
                else:
                    date = date - timedelta(minutes=remainder)

            return date

        h8_start_dt = "201511010000"
        h8_end_dt = "202212130450"
        h9_start_dt = "202212131000"
        gk2a_start_dt = "201908010000"
        g16_start_dt = "201702280000"
        g17_start_dt = "201808280000"
        g18_start_dt = "202207280000"
        g16_mode_switch_dt = "201904021600"

        og_date1 = datetime.strptime(self.start_date + self.start_time, "%Y%m%d%H%M")
        og_date2 = datetime.strptime(self.end_date + self.end_time, "%Y%m%d%H%M")

        if og_date1.minute % 10 != 0:
            og_date1 = found_nearest_minute(og_date1, 10)

        if og_date2.minute % 10 != 0:
            og_date2 = found_nearest_minute(og_date2, 10)

        if self.sat in ["GK2A", "HS", "G16", "G17", "G18"]:
            lst = ["GK2A", "HS", "G16", "G17", "G18"]
            start_dt_lst = [gk2a_start_dt, h8_start_dt, g16_start_dt, g17_start_dt, g18_start_dt]
            idx = lst.index(self.sat)
            start_dt = start_dt_lst[idx]

            if og_date1 < datetime.strptime(start_dt, "%Y%m%d%H%M"):
                og_date1 = datetime.strptime(start_dt, "%Y%m%d%H%M")

            if og_date2 < datetime.strptime(start_dt, "%Y%m%d%H%M"):
                og_date2 = datetime.strptime(start_dt, "%Y%m%d%H%M") + timedelta(minutes=10)

        date1_str = og_date1.strftime("%Y%m%d%H%M")
        date2_str = og_date2.strftime("%Y%m%d%H%M")

        return date1_str, date2_str

    def download_sat_files(self, start_dt, end_dt, need_download=False):
        sat_list = [self.sat] if not self.sos else self.global_satellites

        download_time_dict = {}
        download_error_dict = {}
        link_length_dict = {}

        start_bj_time = f"卫星数据下载开始于: {(datetime.now(UTC) + timedelta(hours=8)).strftime(
            "%Y年%m月%d日%H时%M分%S秒")}"
        start_0 = time_calc.perf_counter()

        for sat in sat_list:
            self.update_sat(sat)
            print(f"开始整理{sat}卫星数据")
            start_arrange = time_calc.perf_counter()
            missing_file, process_dict, actual_file, errors = self.arrange_sat_files()
            end_arrange = time_calc.perf_counter()
            print(f"整理卫星数据耗时: {(end_arrange - start_arrange): .3f}"
                  "\n")

            notication = f"开始下载{sat}卫星数据" if need_download else f"开始生成{sat}卫星数据链接"
            print(notication)
            start_sat = time_calc.perf_counter()

            if self.sat == "HS":
                dl_list = himawari_aws_link(start_dt, end_dt, self.band_list, avoid_file=actual_file)

            elif self.sat == "GK2A":
                dl_list = gk2a_aws_link(start_dt, end_dt, self.band_list, avoid_file=actual_file)

            elif self.sat in ["G16", "G17", "G18", "G19"]:
                areas = ["FD"] if len(self.goes_area) == 0 else self.goes_area
                dl_list = goes_aws_link(self.sat, start_dt, end_dt, self.band_list, areas, avoid_file=actual_file)

            elif self.sat in ["MTG-I1", "MTG-I2", "MTG-I3", "MTG-I4"]:
                dl_list = mtg_datastore_link(self.sat, start_dt, end_dt, self.band_list, avoid_file=actual_file)

            else:
                dl_list = []

            dl_list = list(dict.fromkeys(dl_list))
            merged_txt = f"unicorn_{self.sat}_{start_dt[:8]}_{start_dt[8:12]}_{end_dt[:8]}_{end_dt[8:12]}.txt"

            if len(dl_list) > 0:
                if os.path.exists(merged_txt):
                    file = open(merged_txt, "r")
                    lines = file.readlines()
                    file.close()
                    os.remove(merged_txt)
                    for line in lines:
                        line = line.replace("\n", "")
                        if os.path.basename(line) not in actual_file:
                            dl_list.append(line)

                dl_list = list(dict.fromkeys(dl_list))
                dl_list.sort()

                if need_download:
                    mtg = True if self.sat in ["MTG-I1", "MTG-I2", "MTG-I3", "MTG-I4"] else False
                    errors = url_requests_downloader(dl_list, self.base_path, threads=20, mtg=mtg)
                    if len(errors) > 0:
                        download_error_dict[sat] = str(errors)

                else:
                    with open(merged_txt, "w+") as file:
                        for link in dl_list:
                            file.write(f"{link}\n")

            link_length_dict[sat] = len(dl_list)

            end_sat = time_calc.perf_counter()
            single_sat_time = end_sat - start_sat
            single_sat_hour = int(single_sat_time // 3600)
            single_sat_minute = int((single_sat_time - single_sat_hour * 3600) // 60)
            single_sat_second = round(single_sat_time - single_sat_hour * 3600 - single_sat_minute * 60)

            single_sat_timing = f"{self.sat}卫星下载总耗时: {single_sat_hour}小时{single_sat_minute}分{single_sat_second}秒"
            if need_download:
                print(single_sat_timing)
            download_time_dict[sat] = single_sat_time

        end_0 = time_calc.perf_counter()
        end_bj_time = f"卫星数据下载结束于: {(datetime.now(UTC) + timedelta(hours=8)).strftime(
            "%Y年%m月%d日%H时%M分%S秒")}"

        all_sats_time = end_0 - start_0
        all_sats_hour = int(all_sats_time // 3600)
        all_sats_minute = int((all_sats_time - all_sats_hour * 3600) // 60)
        all_sats_second = round(all_sats_time - all_sats_hour * 3600 - all_sats_minute * 60)

        all_sats_timing = f"所有卫星下载总耗时: {all_sats_hour}小时{all_sats_minute}分{all_sats_second}秒"
        if self.sos:
            print(all_sats_timing)

        download_time_dict["download_summary"] = {"total_time": all_sats_timing,
                                                  "start_bj_time": start_bj_time,
                                                  "end_bj_time": end_bj_time}

        download_msg_list = []
        warning_msg = ", 但有错误发生" if len(download_error_dict.keys()) > 0 else ""
        download_topic = (f"{datetime.strptime(start_dt, "%Y%m%d%H%M").strftime("%Y年%m月%d日%H时%M分")}至"
                          f"{datetime.strptime(end_dt, "%Y%m%d%H%M").strftime("%Y年%m月%d日%H时%M分")}的卫星数据下载完毕"
                          f"{warning_msg}")
        download_msg_list.append(all_sats_timing)
        download_msg_list.append("\n")
        download_msg = "\n".join(download_msg_list)

        download_email_dict = {"topic": download_topic,
                               "message": download_msg}

        warning_email_dict = {}
        if len(download_error_dict.keys()) > 0:
            warning_topic = (f"警告!!! "
                             f"{(datetime.strptime(start_dt, "%Y%m%d%H%M")).strftime("%Y年%m月%d日%H时%M分")}至"
                             f"{(datetime.strptime(end_dt, "%Y%m%d%H%M")).strftime("%Y年%m月%d日%H时%M分")}的卫星数据下载出现问题")

            warning_msg_list = []

            for key in download_error_dict:
                warning_msg_list.append(f"{key}卫星的数据下载时报错: \n")
                for item in download_error_dict[key]:
                    warning_msg_list.append(item)
                warning_msg_list.append("\n")
            warning_msg_list.append("\n")
            warning_msg = "\n".join(warning_msg_list)

            warning_email_dict = {"topic": warning_topic,
                                  "message": warning_msg}

        return link_length_dict, download_email_dict, warning_email_dict


def get_elapsed_time(start, end):
    elapsed = end - start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}小时{int(m)}分{int(round(s))}秒"


def date_range(start_date: datetime, end_date: datetime, delta: timedelta):
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += delta


def valid_url(link_list):
    dl_list = []

    def is_valid_url(link):
        try:
            urllib.request.urlopen(link)
            dl_list.append(link)

        except urllib.error.HTTPError:
            pass

    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(is_valid_url, link_list)

    return dl_list


def url_requests_downloader(link_list, base_path, threads=10, mtg=False):
    errors = queue.Queue()

    def download(link):
        filename = os.path.basename(unquote(link)).split("?")[0] + ".zip" if mtg else os.path.basename(link)
        filepath = os.path.join(base_path, filename)

        if not os.path.exists(filepath):
            try:
                with requests.get(link, timeout=(5, 30)) as response:
                    response.raise_for_status()
                    with open(filepath, "wb") as f:
                        f.write(response.content)
            except Exception as e:
                print(e)
                errors.put(e)


    with ThreadPoolExecutor(max_workers=threads if not mtg else 1) as executor:
        executor.map(download, link_list)

    error_list = []
    while not errors.empty():
        error_list.append(errors.get())

    return error_list


def decimal_digits(decimal_str):
    location = decimal_str.find(".")
    if location == -1:
        return 0
    else:
        return len(decimal_str) - location - 1


def gdalwarp_by_extent(target_proj, resampler, infile, outfile, callback=None, source_proj=None,
                       target_width=None, target_height=None,
                       area_min_x=None, area_min_y=None, area_max_x=None, area_max_y=None, creationoptions=None):
    warp_options = gdal.WarpOptions(format="GTiff", srcSRS=source_proj, dstSRS=target_proj,
                                    width=target_width, height=target_height,
                                    outputBounds=(area_min_x, area_min_y, area_max_x, area_max_y),
                                    multithread=True,
                                    errorThreshold=0, warpMemoryLimit=21474836480,
                                    resampleAlg=resampler,
                                    warpOptions=["NUM_THREADS=ALL_CPUS"],
                                    creationOptions=creationoptions,
                                    callback=callback)

    gdal.Warp(outfile, infile, options=warp_options)


def move_file(data, dst, errors):
    if not os.path.exists(dst) or (os.path.exists(dst) and not os.path.isdir(dst)):
        os.mkdir(dst)

    try:
        shutil.move(data, dst)
    except Exception as e:
        errors.append(f"{data}文件移动时报错, 文件可能被占用, 报错信息为: {e}")


def rename_file_ext(path, file_with_path, old_ext, new_ext, errors):
    filename = os.path.splitext(os.path.basename(file_with_path))[0]

    try:
        os.rename(f"{path}/{filename}.{old_ext}", f"{path}/{filename}.{new_ext}")
    except Exception as e:
        errors.append(f"{filename}文件重命名时报错, 文件可能被占用, 报错信息为: {e}")


def arrange_himawari_files(path, band_list, only_need_existed=False, move_to_folder=True):
    data_files = []
    tmp_folders = []

    missing_file = []
    actual_bz2_file = []

    file_move_dict = {}
    dst_dict = {}
    process_dict = {}

    errors = []

    sat_names = ["H08", "H09"]
    scan_areas = ["FLDK", "R301", "R302", "R303", "R304", "JP01", "JP02", "JP03", "JP04"]
    segments = [10, 1, 1, 1, 1, 1, 1, 1, 1]

    all_bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10",
                 "B11", "B12", "B13", "B14", "B15", "B16"]

    all_suffix = {"B01": "R10", "B02": "R10", "B03": "R05", "B04": "R10", "B05": "R20", "B06": "R20", "B07": "R20",
                  "B08": "R20", "B09": "R20", "B10": "R20", "B11": "R20", "B12": "R20", "B13": "R20", "B14": "R20",
                  "B15": "R20", "B16": "R20"}

    def extract_bz2(bz2_file):
        newfilepath = bz2_file[:-4]
        with bz2.BZ2File(bz2_file, 'rb') as zip_file, open(newfilepath, 'wb') as new_file:
            new_file.write(zip_file.read())
        os.remove(bz2_file)

    bz2_list = glob.glob(f"{path}/HS_H*_*_*_B*_*_R*_S*.DAT.bz2")
    with ThreadPoolExecutor(max_workers=20) as executor:
        executor.map(extract_bz2, bz2_list)

    for band in band_list:
        all_bands.remove(band)

    for satellite in sat_names:
        for scan_area in scan_areas:
            folders = glob.glob(f"{path}/{satellite}_*_*_{scan_area}")
            for tmp_folder in folders:
                if os.path.isdir(tmp_folder):
                    tmp_folders.append(tmp_folder)
                    all_items = glob.glob(f"{tmp_folder}/*")
                    for item in all_items:
                        if os.path.isfile(item):
                            if "desktop.ini" not in item:
                                data_files.append(item)

    partial_move_files = partial(shutil.move, dst=path)
    with ThreadPoolExecutor(max_workers=20) as executor_move_files:
        executor_move_files.map(partial_move_files, data_files)

    partial_del_folders = partial(shutil.rmtree, ignore_errors=True)
    with ThreadPoolExecutor(max_workers=20) as executor_del_folders:
        executor_del_folders.map(partial_del_folders, tmp_folders)

    if only_need_existed:
        for scan_area in scan_areas:
            wait = glob.glob(f"{path}/HS_H*_*_*_B*_{scan_area}_R*_S*.DAT")
            for file in wait:
                bz2_filename = os.path.basename(file) + ".bz2"
                actual_bz2_file.append(bz2_filename)

        return missing_file, process_dict, actual_bz2_file, errors

    for scan_area in scan_areas:
        sense_dt_list = []

        wait = glob.glob(f"{path}/HS_H*_*_*_B*_{scan_area}_R*_S*.DAT")
        for file in wait:
            filename = os.path.basename(file).split("_")
            sat_name = filename[1]
            sense_date = filename[2]
            sense_time = filename[3]
            sense_dt_list.append(f"{sat_name}_{sense_date}_{sense_time}")
            file_move_dict[file] = f"{sat_name}_{sense_date}_{sense_time}_{scan_area}"

            bz2_filename = os.path.basename(file) + ".bz2"
            actual_bz2_file.append(bz2_filename)

        sense_dt_list = list({}.fromkeys(sense_dt_list).keys())

        for dt in sense_dt_list:
            sat_name, sense_date, sense_time = dt.split("_")

            for band in band_list:
                for i in range(1, segments[scan_areas.index(scan_area)] + 1):
                    seg = f"{i:02d}10"
                    band_file = (f"HS_{sat_name}_{sense_date}_{sense_time}_{band}_{scan_area}_"
                                 f"{all_suffix[band]}_S{seg}.DAT")
                    if not os.path.exists(f"{path}/{band_file}"):
                        missing_file.append(band_file)

    if len(missing_file) == 0 and move_to_folder:
        for file in file_move_dict.keys():
            dst_dict[file] = f"{path}/{file_move_dict[file]}"

        for file in dst_dict.keys():
            dst = dst_dict[file]
            move_file(file, dst, errors)

        for folder in file_move_dict.values():
            folder_sat_name = folder.split("_")[0]
            folder_scan_area = folder.split("_")[3]
            process_dict[f"{path}/{folder}"] = {"sat_name": folder_sat_name, "scan_area": folder_scan_area}

    return missing_file, process_dict, actual_bz2_file, errors


def arrange_gk2a_files(path, band_list, only_need_existed=False, move_to_folder=True):
    data_files = []
    tmp_folders = []

    missing_file = []
    actual_file = []

    file_move_dict = {}
    dst_dict = {}
    process_dict = {}

    errors = []

    scan_areas = ["FLDK", "ELA"]
    scan_areas_in_file = ["fd", "ela"]
    all_bands = ["vi004", "vi005", "vi006", "vi008", "nr013", "nr016", "sw038", "wv063", "wv069", "wv073",
                 "ir087", "ir096", "ir105", "ir112", "ir123", "ir133"]

    all_suffix = {"vi004": "010", "vi005": "010", "vi006": "005", "vi008": "010", "nr013": "020", "nr016": "020",
                  "sw038": "020", "wv063": "020", "wv069": "020", "wv073": "020", "ir087": "020", "ir096": "020",
                  "ir105": "020", "ir112": "020", "ir123": "020", "ir133": "020"}

    gk2a_hdf = glob.glob(f"{path}/gk2a_ami_le1b_*_*ge_*.hdf")
    for hdf in gk2a_hdf:
        rename_file_ext(path, hdf, "hdf", "nc", errors)

    for band in band_list:
        band = band.lower()
        all_bands.remove(band)

    for scan_area in scan_areas:
        folders = glob.glob(f"{path}/GK2A_*_*_{scan_area}")
        for tmp_folder in folders:
            if os.path.isdir(tmp_folder):
                tmp_folders.append(tmp_folder)
                all_items = glob.glob(f"{tmp_folder}/*")
                for item in all_items:
                    if os.path.isfile(item):
                        if "desktop.ini" not in item:
                            data_files.append(item)

    partial_move_files = partial(shutil.move, dst=path)
    with ThreadPoolExecutor(max_workers=20) as executor_move_files:
        executor_move_files.map(partial_move_files, data_files)

    partial_del_folders = partial(shutil.rmtree, ignore_errors=True)
    with ThreadPoolExecutor(max_workers=20) as executor_del_folders:
        executor_del_folders.map(partial_del_folders, tmp_folders)

    if only_need_existed:
        for scan_area in scan_areas_in_file:
            wait = glob.glob(f"{path}/gk2a_ami_le1b_*_{scan_area}*ge_*.nc")
            for file in wait:
                nc_file = os.path.basename(file)
                actual_file.append(nc_file)

        return missing_file, process_dict, actual_file, errors

    for scan_area in scan_areas_in_file:
        sense_dt_list = []

        wait = glob.glob(f"{path}/gk2a_ami_le1b_*_{scan_area}*ge_*.nc")
        for file in wait:
            filename = os.path.basename(file).split("_")
            sense_dt = filename[5].replace(".nc", "")
            sense_dt_list.append(sense_dt)
            file_move_dict[file] = (f"GK2A_{filename[5][0:8]}_{filename[5][8:12]}_"
                                    f"{scan_areas[scan_areas_in_file.index(scan_area)]}")

            nc_file = os.path.basename(file)
            actual_file.append(nc_file)

        sense_dt_list = list({}.fromkeys(sense_dt_list).keys())

        for dt in sense_dt_list:
            for band in band_list:
                band = band.lower()
                band_file = f"gk2a_ami_le1b_{band}_{scan_area}{all_suffix[band]}ge_{dt}.nc"
                if not os.path.exists(f"{path}/{band_file}"):
                    missing_file.append(band_file)

    if len(missing_file) == 0 and move_to_folder:
        for file in file_move_dict.keys():
            dst_dict[file] = f"{path}/{file_move_dict[file]}"

        for file in dst_dict.keys():
            dst = dst_dict[file]
            move_file(file, dst, errors)

        for folder in file_move_dict.values():
            folder_scan_area = folder.split("_")[3]
            process_dict[f"{path}/{folder}"] = {"sat_name": "GK2A", "scan_area": folder_scan_area}

    return missing_file, process_dict, actual_file, errors


def arrange_goes_files(sat, path, band_list, only_need_existed=False, move_to_folder=True):
    data_files = []
    tmp_folders = []

    missing_file = []
    actual_file = []

    file_move_dict = {}
    dst_dict = {}
    process_dict = {}

    errors = []

    sat_name = sat
    scan_areas = ["FLDK", "CONUS", "MESO1", "MESO2"]
    scan_areas_in_file = ["F", "C", "M1", "M2"]

    time_deltas = [timedelta(seconds=30), timedelta(seconds=10), timedelta(seconds=3), timedelta(seconds=3)]

    for scan_area in scan_areas:
        folders = glob.glob(f"{path}/{sat_name}_m*_s*_e*_{scan_area}")
        for tmp_folder in folders:
            if os.path.isdir(tmp_folder):
                tmp_folders.append(tmp_folder)
                all_items = glob.glob(f"{tmp_folder}/*")
                for item in all_items:
                    if os.path.isfile(item):
                        if "desktop.ini" not in item:
                            data_files.append(item)

    partial_move_files = partial(shutil.move, dst=path)
    with ThreadPoolExecutor(max_workers=20) as executor_move_files:
        executor_move_files.map(partial_move_files, data_files)

    partial_del_folders = partial(shutil.rmtree, ignore_errors=True)
    with ThreadPoolExecutor(max_workers=20) as executor_del_folders:
        executor_del_folders.map(partial_del_folders, tmp_folders)

    if only_need_existed:
        for scan_area in scan_areas_in_file:
            wait = glob.glob(f"{path}/OR_ABI-L1b-Rad{scan_area}*-M*C*_{sat_name}_s*_e*_c*.nc")
            for file in wait:
                nc_file = os.path.basename(file)
                actual_file.append(nc_file)

        return missing_file, process_dict, actual_file, errors

    for scan_area in scan_areas_in_file:
        sense_dt_list = []
        missing_list = []

        wait = glob.glob(f"{path}/OR_ABI-L1b-Rad{scan_area}*-M*C*_{sat_name}_s*_e*_c*.nc")
        for file in wait:
            filename = os.path.basename(file).split("_")
            scan_mode = filename[1].split("-")[3][1]
            start_dt = filename[3].replace("s", "")
            end_dt = filename[4].replace("e", "")
            create_dt = filename[5].replace(".nc", "").replace("c", "")
            sense_dt_list.append(f"{scan_mode}_{start_dt}_{end_dt}_{create_dt}")
            file_move_dict[file] = f"{sat_name}_m{scan_mode}_s{start_dt}_e{end_dt}_{scan_areas[scan_areas_in_file.index(scan_area)]}"

            nc_file = os.path.basename(file)
            actual_file.append(nc_file)

        sense_dt_list = list({}.fromkeys(sense_dt_list).keys())

        for dt in sense_dt_list:
            mode, start, end, create = dt.split("_")

            for band in band_list:
                band_file = f"OR_ABI-L1b-Rad{scan_area}-M{mode}{band}_{sat_name}_s{start}_e{end}_c{create}.nc"
                if not os.path.exists(f"{path}/{band_file}"):
                    missing_list.append(band_file)

        def check_missing_file(miss):
            pop_item = None
            missing_fname = os.path.splitext(os.path.split(miss)[1])[0]
            missing_fname = missing_fname.split("_")
            missing_chanel = missing_fname[1].split("-")[3][2:5]
            missing_scan_mode = missing_fname[1].split("-")[3][1]
            missing_start = datetime.strptime(missing_fname[3].replace("s", ""), "%Y%j%H%M%S%f")
            missing_end = datetime.strptime(missing_fname[4].replace("e", ""), "%Y%j%H%M%S%f")

            time_del = time_deltas[scan_areas_in_file.index(scan_area)]

            for actual in actual_file:
                actual_filename = os.path.splitext(os.path.split(actual)[1])[0]
                actual_filename = actual_filename.split("_")
                actual_chanel = actual_filename[1].split("-")[3][2:5]
                actual_scan_mode = actual_filename[1].split("-")[3][1]
                actual_start = datetime.strptime(actual_filename[3].replace("s", ""), "%Y%j%H%M%S%f")
                actual_end = datetime.strptime(actual_filename[4].replace("e", ""), "%Y%j%H%M%S%f")

                time_pass = False
                if abs(actual_start - missing_start) <= time_del:
                    if abs(actual_end - missing_end) <= time_del or actual_scan_mode == missing_scan_mode:
                        time_pass = True

                if missing_chanel == actual_chanel and time_pass:
                    pop_item = miss
                    break
            return pop_item

        with ThreadPoolExecutor(max_workers=100) as executor:
            results = executor.map(check_missing_file, missing_list)

        pop_list = [missing_list.index(result) for result in results if result is not None]
        pop_list = list({}.fromkeys(pop_list))
        pop_list.reverse()

        for pop in pop_list:
            missing_list.pop(pop)

        for missing in missing_list:
            missing_file.append(missing)

    if len(missing_file) == 0 and move_to_folder:
        new_file_move_dict = file_move_dict.copy()

        for file in file_move_dict.keys():
            satellite_name, mode, start, end, area = file_move_dict[file].split("_")
            start = start.replace("s", "")
            end = end.replace("e", "")

            time_delta = time_deltas[scan_areas.index(area)]

            for other_file in file_move_dict.keys():
                other_satellite_name, other_mode, other_start, other_end, other_area = file_move_dict[other_file].split("_")
                other_start = other_start.replace("s", "")
                other_end = other_end.replace("e", "")

                if area == other_area and abs(datetime.strptime(start, "%Y%j%H%M%S%f") -
                                              datetime.strptime(other_start, "%Y%j%H%M%S%f")) <= time_delta:
                    if abs(datetime.strptime(end, "%Y%j%H%M%S%f") - datetime.strptime(other_end, "%Y%j%H%M%S%f")) or mode == other_mode:
                        new_file_move_dict[other_file] = file_move_dict[file]

        for file in new_file_move_dict.keys():
            dst_dict[file] = f"{path}/{new_file_move_dict[file]}"

        for file in dst_dict.keys():
            dst = dst_dict[file]
            move_file(file, dst, errors)

        for folder in new_file_move_dict.values():
            folder_scan_area = folder.split("_")[4]
            process_dict[f"{path}/{folder}"] = {"sat_name": sat_name, "scan_area": folder_scan_area}

    for missing in missing_file:
        idx6 = missing_file.index(missing)

        missing_filename = os.path.splitext(os.path.split(missing)[1])[0]
        missing_filename = missing_filename.split("_")

        missing_area = missing_filename[1].split("-")[2].replace("Rad", "")
        missing_band = missing_filename[1].split("-")[3][2:5]
        missing_start_dt = datetime.strptime(missing_filename[3].replace("s", ""), "%Y%j%H%M%S%f")

        missing_file.pop(idx6)
        missing_file.insert(idx6, f"Area: {missing_area} Band: {missing_band} Start date: {missing_start_dt}")

    missing_file = list({}.fromkeys(missing_file).keys())
    missing_file.sort()

    return missing_file, process_dict, actual_file, errors


def arrange_fci_files(sat, path, band_list, only_need_existed=False, move_to_folder=True):
    data_files = []
    tmp_folders = []

    missing_file = []
    actual_file = []

    file_move_dict = {}
    dst_dict = {}
    process_dict = {}

    errors = []

    sat_name = sat.replace("G", "").replace("-", "")
    scan_areas = ["FLDK", "EU"]
    scan_areas_in_file = ["FD", "Q4"]

    def extract_zip(zip_file):
        extract_dir = os.path.dirname(zip_file)
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            for data in zip_ref.namelist():
                if str(data).endswith(".nc"):
                    zip_ref.extract(data, extract_dir)
        os.remove(zip_file)

    zip_list = glob.glob(f"{path}/W_XX-EUMETSAT-Darmstadt,IMG+SAT,*+FCI-1C-RRAD-*.zip")
    with ThreadPoolExecutor(max_workers=20) as executor:
        executor.map(extract_zip, zip_list)

    all_bands = ["FDHSI", "HRFI"]
    for band in band_list:
        all_bands.remove(band)

    for scan_area in scan_areas:
        folders = glob.glob(f"{path}/{sat}_*_cycle*_{scan_area}")
        for tmp_folder in folders:
            if os.path.isdir(tmp_folder):
                tmp_folders.append(tmp_folder)
                all_items = glob.glob(f"{tmp_folder}/*")
                for item in all_items:
                    if os.path.isfile(item):
                        if "desktop.ini" not in item:
                            data_files.append(item)

    partial_move_files = partial(shutil.move, dst=path)
    with ThreadPoolExecutor(max_workers=20) as executor_move_files:
        executor_move_files.map(partial_move_files, data_files)

    partial_del_folders = partial(shutil.rmtree, ignore_errors=True)
    with ThreadPoolExecutor(max_workers=20) as executor_del_folders:
        executor_del_folders.map(partial_del_folders, tmp_folders)

    if only_need_existed:
        for scan_area in scan_areas_in_file:
            wait = glob.glob(
                f"{path}/W_XX-EUMETSAT-Darmstadt,IMG+SAT,{sat_name}+FCI-1C-RRAD-*-{scan_area}--CHK-*--*-NC4E_C_EUMT_*_"
                f"IDPFI_OPE_*_*_N_*_*_*_*.nc")

            for file in wait:
                nc_file = os.path.basename(file)
                actual_file.append(nc_file)

        return missing_file, process_dict, actual_file, errors

    for scan_area in scan_areas_in_file:
        sense_dt_list = []
        missing_list = []
        actual_list = []
        wait = glob.glob(
            f"{path}/W_XX-EUMETSAT-Darmstadt,IMG+SAT,{sat_name}+FCI-1C-RRAD-*-{scan_area}--CHK-*--*-NC4E_C_EUMT_*_"
            f"IDPFI_OPE_*_*_N_*_*_*_*.nc")

        for file in wait:
            filename = os.path.basename(file)
            filename = filename.split("_")
            component = filename[1].split("-")[9]
            purpose = filename[1].split("-")[11]
            start_time = filename[7]
            end_time = filename[8]
            process_time = filename[4]
            special_compression = filename[10]
            disposition_mode = filename[11]
            cycle = filename[12]
            sense_dt_list.append(f"{component}_{purpose}_{start_time}_{end_time}_{process_time}_{special_compression}_"
                                 f"{disposition_mode}_{cycle}")
            file_move_dict[
                file] = f"{sat}_{start_time[0:8]}_cycle{cycle}_{scan_areas[scan_areas_in_file.index(scan_area)]}"

            nc_file = os.path.basename(file)
            actual_file.append(nc_file)

        # sense_dt_list = list({}.fromkeys(sense_dt_list).keys())
        #
        # for dt in sense_dt_list:
        #     component, purpose, start_time, end_time, process_time, special_compression, disposition_mode, cycle = dt.split(
        #         "_")
        #
        #     for band in band_list:
        #         for seg in range(1, 42):
        #             seg = str(seg).rjust(4, "0")
        #             band_file = (
        #                 f"W_XX-EUMETSAT-Darmstadt,IMG+SAT,{sat_name}+FCI-1C-RRAD-{band}-{scan_area}--CHK-{component}--{purpose}-"
        #                 f"NC4E_C_EUMT_{process_time}_IDPFI_OPE_{start_time}_{end_time}_N_{special_compression}_"
        #                 f"{disposition_mode}_{cycle}_{seg}.nc")
        #
        #             if not os.path.exists(f"{path}/{band_file}"):
        #                 missing_list.append(band_file)
        #             else:
        #                 actual_list.append(band_file)
        #
        # def check_missing_file(missing, actual_list):
        #     pop_idx = None
        #     missing_filename = os.path.splitext(os.path.split(missing)[1])[0]
        #     missing_filename = missing_filename.split("_")
        #     missing_start_date = missing_filename[7][0:8]
        #     missing_end_date = missing_filename[8][0:8]
        #     missing_cycle = missing_filename[12]
        #     missing_seg = missing_filename[13]
        #
        #     for actual in actual_list:
        #         actual_filename = os.path.splitext(os.path.split(actual)[1])[0]
        #         actual_filename = actual_filename.split("_")
        #         actual_start_date = actual_filename[7][0:8]
        #         actual_end_date = actual_filename[8][0:8]
        #         actual_cycle = actual_filename[12]
        #         actual_seg = actual_filename[13]
        #
        #         if ((missing_start_date == actual_start_date or missing_end_date == actual_end_date) and
        #                 missing_cycle == actual_cycle and missing_seg == actual_seg):
        #             pop_idx = missing
        #             break
        #     return pop_idx
        #
        # with ThreadPoolExecutor(max_workers=100) as executor:
        #     results = executor.map(lambda m: check_missing_file(m, actual_list), missing_list)
        #
        # pop_list = [missing_list.index(result) for result in results if result is not None]
        # pop_list = list({}.fromkeys(pop_list))
        # pop_list.reverse()
        #
        # for pop in pop_list:
        #     missing_list.pop(pop)
        #
        # for missing in missing_list:
        #     missing_file.append(missing)

    if len(missing_file) == 0 and move_to_folder:
        for file in file_move_dict.keys():
            dst_dict[file] = f"{path}/{file_move_dict[file]}"

        for file in dst_dict.keys():
            dst = dst_dict[file]
            move_file(file, dst, errors)

        for folder in file_move_dict.values():
            process_dict[f"{path}/{folder}"] = {"sat_name": sat, "scan_area": "FLDK"}

    for missing in missing_file:
        idx6 = missing_file.index(missing)

        missing_filename = os.path.splitext(os.path.split(missing)[1])[0]
        missing_filename = missing_filename.split("_")

        missing_band = missing_filename[1].split(",")[2].split("-")[3]
        missing_date = missing_filename[7][0:8]
        missing_cycle = missing_filename[12]
        missing_seg = missing_filename[13]

        missing_file.pop(idx6)
        missing_file.insert(idx6,
                            f"Band: {missing_band} Date: {missing_date} Cycle: {missing_cycle} Segment: {missing_seg}")

    missing_file = list({}.fromkeys(missing_file).keys())

    return missing_file, process_dict, actual_file, errors


def arrange_fy4_files(sat, path, band_list, move_to_folder=True):
    data_files = []
    tmp_folders = []

    missing_file = []
    actual_file = []

    file_move_dict = {}
    dst_dict = {}
    process_dict = {}

    errors = []

    sat_name = sat

    scan_areas = ["FLDK", "REGC"]
    scan_area_shorts = ["DISK", "REGC"]
    all_bands = ["0500M", "1000M", "2000M", "4000M"]

    fy4_hdf = glob.glob(f"{path}/FY4*-_AGRI--_N_*_*_L1-_FDI-_MULT_NOM_*_*_*00M_*.HDF")
    partial_rename_ext = partial(rename_file_ext, path=path, old_ext="HDF", new_ext="hdf", errors=errors)
    with ThreadPoolExecutor(max_workers=20) as executor_partial_rename_ext:
        executor_partial_rename_ext.map(partial_rename_ext, fy4_hdf)

    for band in band_list:
        all_bands.remove(band)

    for scan_area in scan_areas:
        folders = glob.glob(f"{path}/{sat_name}_sdt*_edt*_{scan_area}")
        for tmp_folder in folders:
            if os.path.isdir(tmp_folder):
                tmp_folders.append(tmp_folder)
                all_items = glob.glob(f"{tmp_folder}/*")
                for item in all_items:
                    if os.path.isfile(item):
                        if "desktop.ini" not in item:
                            data_files.append(item)

    partial_move_files = partial(shutil.move, dst=path)
    with ThreadPoolExecutor(max_workers=20) as executor_move_files:
        executor_move_files.map(partial_move_files, data_files)

    partial_del_folders = partial(shutil.rmtree, ignore_errors=True)
    with ThreadPoolExecutor(max_workers=20) as executor_del_folders:
        executor_del_folders.map(partial_del_folders, tmp_folders)

    for scan_area in scan_area_shorts:
        sense_dt_list = []

        wait = glob.glob(f"{path}/{sat_name}-_AGRI--_N_{scan_area}_*_L1-_FDI-_MULT_NOM_*_*_*00M_V0001.hdf")
        for file in wait:
            filename = os.path.splitext(os.path.split(file)[1])[0]
            filename = filename.split("_")
            lon_0 = filename[4]
            start_dt = filename[9]
            end_dt = filename[10]
            sense_dt_list.append(f"{sat_name}_{scan_area}_{lon_0}_{start_dt}_{end_dt}")
            file_move_dict[file] = (f"{sat_name}_sdt{start_dt}_edt{end_dt}_"
                                    f"{scan_areas[scan_area_shorts.index(scan_area)]}")

        sense_dt_list = list({}.fromkeys(sense_dt_list).keys())

        for dt in sense_dt_list:
            satellite_name, area, lon0, start, end = dt.split("_")

            for band in band_list:
                band_file = (f"{satellite_name}-_AGRI--_N_{scan_area}_{lon0}_L1-_FDI-_MULT_NOM_{start}_{end}_{band}_"
                             f"V0001.hdf")
                if not os.path.exists(f"{path}/{band_file}"):
                    missing_file.append(band_file)
                else:
                    actual_file.append(band_file)

            for other_band in all_bands:
                band_file = (f"{sat_name}-_AGRI--_N_{scan_area}_{lon0}_L1-_FDI-_MULT_NOM_{start}_{end}_{other_band}_"
                             f"V0001.hdf")
                if os.path.exists(f"{path}/{band_file}"):
                    actual_file.append(band_file)

    if len(missing_file) == 0 and move_to_folder:
        for file in file_move_dict.keys():
            dst_dict[file] = f"{path}/{file_move_dict[file]}"

        for file in dst_dict.keys():
            dst = dst_dict[file]
            move_file(file, dst, errors)

        for folder in file_move_dict.values():
            folder_scan_area = folder.split("_")[3]
            process_dict[f"{path}/{folder}"] = {"sat_name": sat_name, "scan_area": folder_scan_area}

    return missing_file, process_dict, actual_file, errors


def gk2a_aws_link(start_date_str, end_date_str, band_list, avoid_file: list = None):
    all_suffix = {"vi004": "010", "vi005": "010", "vi006": "005", "vi008": "010",
                  "nr013": "020", "nr016": "020",
                  "sw038": "020", "wv063": "020", "wv069": "020", "wv073": "020",
                  "ir087": "020", "ir096": "020",
                  "ir105": "020", "ir112": "020", "ir123": "020", "ir133": "020"}
    dl_list = []
    tmp_list = []
    gk2a_nasa_mode = "https://data.nas.nasa.gov/geonex/geonexdata/GK2A/AMI/L1B/FD"

    aws_start_dt = "202302160000"

    start_date = datetime.strptime(start_date_str, "%Y%m%d%H%M")
    end_date = datetime.strptime(end_date_str, "%Y%m%d%H%M") - timedelta(minutes=10)

    minus = int((end_date - start_date).total_seconds())

    for i in range(0, minus + 600, 600):
        date = start_date + timedelta(seconds=i)
        date_str = date.strftime("%Y%m%d%H%M")
        year, month, day, hour, minute = date_str[:4], date_str[4:6], date_str[6:8], date_str[8:10], date_str[10:12]

        new_date = f"{year}{month}{day}"
        new_time = f"{hour}{minute}"

        bucket_name = "noaa-gk2a-pds"
        aws_mode = f"https://{bucket_name}.s3.amazonaws.com"
        client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        response = client.list_objects_v2(Bucket=bucket_name,
                                          Prefix=f"AMI/L1B/FD/{year}{month}/{day}/{hour}")

        for band in band_list:
            band = band.lower()
            if date < datetime.strptime(aws_start_dt, "%Y%m%d%H%M"):
                file = f"gk2a_ami_le1b_{band}_fd{all_suffix[band]}ge_{new_date}{new_time}.nc"
                link = f"{gk2a_nasa_mode}/{year}{month}/{day}/{hour}/{file}"
                tmp_list.append(link)
            else:
                for content in response.get("Contents", []):
                    filename = content["Key"].replace(f"AMI/L1B/FD/{year}{month}/{day}/{hour}/", "")
                    filename_split = filename.split("_")
                    file_band = filename_split[3]
                    file_dt = filename_split[5][:12]

                    link = f"{aws_mode}/AMI/L1B/FD/{year}{month}/{day}/{hour}/{filename}"

                    if file_band == band and file_dt == f"{year}{month}{day}{hour}{minute}":
                        dl_list.append(link)

    valid_list = valid_url(tmp_list)
    dl_list.extend(valid_list)

    if avoid_file is not None:
        final_dl_list = [link for link in dl_list if os.path.basename(link) not in avoid_file]
    else:
        final_dl_list = dl_list

    final_dl_list.sort()

    return final_dl_list


def himawari_aws_link(start_date_str, end_date_str, band_list, avoid_file: list = None):
    all_suffix = {"B01": "R10", "B02": "R10", "B03": "R05", "B04": "R10", "B05": "R20",
                  "B06": "R20", "B07": "R20",
                  "B08": "R20", "B09": "R20", "B10": "R20", "B11": "R20", "B12": "R20",
                  "B13": "R20", "B14": "R20",
                  "B15": "R20", "B16": "R20"}

    dl_list = []
    tmp_list = []

    h8_start_dt = "201511010000"
    h8_end_dt = "202212130450"
    h9_start_dt = "202212131000"
    aws_start_dt = "201912091810"

    h8_nasa_mode = "https://data.nas.nasa.gov/geonex/geonexdata/HIMAWARI8/JMA-L1B/AHI/Hsfd"

    start_date = datetime.strptime(start_date_str, "%Y%m%d%H%M")
    end_date = datetime.strptime(end_date_str, "%Y%m%d%H%M") - timedelta(minutes=10)

    minus = int((end_date - start_date).total_seconds())

    for i in range(0, minus + 600, 600):
        date = start_date + timedelta(seconds=i)
        date_str = date.strftime("%Y%m%d%H%M")
        year, month, day, hour, minute = date_str[:4], date_str[4:6], date_str[6:8], date_str[8:10], date_str[10:12]

        new_date = f"{year}{month}{day}"
        new_time = f"{hour}{minute}"

        if datetime.strptime(h8_end_dt, "%Y%m%d%H%M") < date < datetime.strptime(h9_start_dt, "%Y%m%d%H%M"):
            continue

        sat_name = "H08" if date <= datetime.strptime(h8_end_dt, "%Y%m%d%H%M") else "H09"

        bucket_name = "noaa-himawari8" if sat_name == "H08" else "noaa-himawari9"
        aws_mode = f"https://{bucket_name}.s3.amazonaws.com"
        client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        response = client.list_objects_v2(Bucket=bucket_name,
                                          Prefix=f"AHI-L1b-FLDK/{year}/{month}/{day}/{hour}{minute}")

        for band in band_list:
            if date < datetime.strptime(aws_start_dt, "%Y%m%d%H%M"):
                for num in range(1, 11):
                    seg = f"{num:02d}10"
                    file = f"HS_{sat_name}_{new_date}_{new_time}_{band}_{all_suffix[band]}_FLDK_{seg}.DAT.bz2"

                    link = f"{h8_nasa_mode}/{year}/{month}/{day}/{new_date}{hour}/{minute}/{band}/{file}"

                    tmp_list.append(link)

            else:
                for num in range(1, 11):
                    seg = f"{num:02d}10"
                    for content in response.get("Contents", []):
                        filename = content["Key"].replace(f"AHI-L1b-FLDK/{year}/{month}/{day}/{hour}{minute}/", "")
                        filename_split = filename.split("_")
                        file_band = filename_split[4]
                        file_dt = filename_split[2] + filename_split[3]
                        file_seg = filename_split[7][:5]

                        link = f"{aws_mode}/AHI-L1b-FLDK/{year}/{month}/{day}/{hour}{minute}/{filename}"

                        if file_band == band and file_dt == f"{year}{month}{day}{hour}{minute}" \
                                and file_seg == f"S{seg}":
                            dl_list.append(link)

    valid_list = valid_url(tmp_list)
    dl_list.extend(valid_list)

    if avoid_file is not None:
        final_dl_list = [link for link in dl_list if os.path.basename(link) not in avoid_file]
    else:
        final_dl_list = dl_list

    final_dl_list.sort()

    return final_dl_list


def mtg_datastore_link(sat, start_date_str, end_date_str, band_list, avoid_file: list = None):
    lst = ["MTG-I1", "MTG-I2", "MTG-I3", "MTG-I4"]
    platforms = ["MTI1", "MTI2", "MTI3", "MTI4"]

    bands = ["FDHSI", "HRFI"]
    datastore_ids = ["EO:EUM:DAT:0662", "EO:EUM:DAT:0665"]

    dl_list = []

    start_date = datetime.strptime(start_date_str, "%Y%m%d%H%M")
    end_date = datetime.strptime(end_date_str, "%Y%m%d%H%M") - timedelta(seconds=1)

    consumer_key = "Ij21HQ3icd8eY352nKymL5P1xUga"
    consumer_secret = "_nU7zzczMJ3128fhTDPcVqipBcoa"

    credentials = (consumer_key, consumer_secret)
    token = eumdac.AccessToken(credentials, validity=86400)
    datastore = eumdac.DataStore(token)

    for band in band_list:
        selected_collection = datastore.get_collection(datastore_ids[bands.index(band)])
        products = selected_collection.search(sat=platforms[lst.index(sat)],
                                              dtstart=start_date, dtend=end_date, coverage="FD")
        for product in products:
            dl_list.append(product.url)

    if avoid_file is not None:
        avoid_components = []
        final_dl_list = []
        for file in avoid_file:
            filename = os.path.basename(file)
            filename = filename.split("_")
            band = filename[1].split("-")[5]
            coverage = filename[1].split("-")[6]
            start_time = filename[7]
            cycle = filename[12]
            avoid_components.append(f"{band}_{coverage}_{start_time}_{cycle}")

        for link in dl_list:
            zip_filename = os.path.basename(link).split("?")[0]
            zip_filename = zip_filename.split("_")
            zip_band = zip_filename[1].split("-")[5]
            zip_coverage = zip_filename[1].split("-")[6]
            zip_start_time = zip_filename[7]
            zip_cycle = zip_filename[12]

            if f"{zip_band}_{zip_coverage}_{zip_start_time}_{zip_cycle}" not in avoid_components:
                final_dl_list.append(link)
    else:
        final_dl_list = dl_list

    final_dl_list.sort()

    return final_dl_list


def goes_aws_link(sat, start_date_str, end_date_str, band_list, area_list, avoid_file: list = None):
    lst = ["G16", "G17", "G18", "G19"]
    platforms = ["goes16", "goes17", "goes18", "goes19"]

    dl_list = []

    mode_list = ["FD", "CONUS", "MESO"]
    s3_mode_list = ["ABI-L1b-RadF", "ABI-L1b-RadC", "ABI-L1b-RadM"]
    time_delta_list = [timedelta(seconds=600), timedelta(seconds=120), timedelta(seconds=30)]
    bucket_name = f"noaa-{platforms[lst.index(sat)]}"

    start_date = datetime.strptime(start_date_str, "%Y%m%d%H%M")
    end_date = datetime.strptime(end_date_str, "%Y%m%d%H%M")

    minus = int((end_date - start_date).total_seconds())

    for i in range(0, minus + 600, 3600):
        date = start_date + timedelta(seconds=i)
        date_str = date.strftime("%Y%m%d%H%M")
        day_num = date.strftime("%j")
        year = date_str[:4]
        hour = date_str[8:10]

        for area in area_list:
            idx = 0 if area not in mode_list else mode_list.index(area)
            s3_mode = s3_mode_list[idx]
            time_delta = time_delta_list[idx]
            aws_mode = f"https://{bucket_name}.s3.amazonaws.com/{s3_mode}"

            client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
            response = client.list_objects_v2(Bucket=bucket_name,
                                              Prefix=f"{s3_mode}/{year}/{day_num}/{hour}")

            for band in band_list:
                for content in response.get("Contents", []):
                    filename = content["Key"]
                    filename = filename.replace(f"{s3_mode}/{year}/{day_num}/{hour}/", "")
                    filename_split = filename.split("_")
                    file_band = filename_split[1].split("-")[3][-3:]
                    scan_mode = filename_split[1].split("-")[3][1]
                    file_start_dt = datetime.strptime(filename_split[3].replace("s", ""), "%Y%j%H%M%S%f")
                    file_end_dt = datetime.strptime(filename_split[4].replace("e", ""), "%Y%j%H%M%S%f")

                    link = f"{aws_mode}/{year}/{day_num}/{hour}/{filename}"

                    if file_band == band and scan_mode == "6":
                        if start_date < end_date:
                            if start_date <= file_start_dt <= end_date and start_date <= file_end_dt <= end_date:
                                dl_list.append(link)

                            elif file_start_dt <= start_date <= file_end_dt <= end_date:
                                if abs(file_start_dt - start_date) <= time_delta:
                                    dl_list.append(link)

                            elif start_date <= file_start_dt <= end_date <= file_end_dt:
                                if abs(file_end_dt - end_date) <= time_delta:
                                    dl_list.append(link)

                        elif start_date == end_date:
                            if file_start_dt <= start_date <= file_end_dt:
                                if abs(file_start_dt - start_date) <= time_delta or (
                                        abs(file_end_dt - start_date) <= time_delta):
                                    dl_list.append(link)

                            elif start_date <= file_start_dt:
                                if abs(file_start_dt - start_date) <= time_delta:
                                    dl_list.append(link)

    if avoid_file is not None:
        final_dl_tmp_list = [link for link in dl_list if os.path.basename(link) not in avoid_file]
    else:
        final_dl_tmp_list = dl_list

    final_dl_list = []
    seen_files = []

    for link in final_dl_tmp_list:
        filename = os.path.basename(link).split("_")
        channel = filename[1]
        start_dt = datetime.strptime(filename[3].replace("s", ""), "%Y%j%H%M%S%f")
        end_dt = datetime.strptime(filename[4].replace("e", ""), "%Y%j%H%M%S%f")

        is_unique = True
        for seen_channel, seen_start, seen_end in seen_files:
            if channel == seen_channel and abs((start_dt - seen_start).total_seconds()) < 1 and abs(
                    (end_dt - seen_end).total_seconds()) < 1:
                is_unique = False
                break

        if is_unique:
            final_dl_list.append(link)
            seen_files.append((channel, start_dt, end_dt))

    final_dl_list.sort()

    return final_dl_list


class MyOwnCloudCompositor(GenericCompositor):
    def __init__(self, name, transition_min=258.15, transition_max=298.15, transition_gamma=3.0,
                 invert_alpha=False, rgba=False, **kwargs):
        super().__init__(name, **kwargs)
        self.transition_min = transition_min
        self.transition_max = transition_max
        self.transition_gamma = transition_gamma
        self.invert_alpha = invert_alpha
        self.rgba = rgba

    def __call__(self, projectables, **kwargs):
        """Generate the composite."""
        data = projectables[0]

        tr_min = self.transition_min
        tr_max = self.transition_max
        gamma = self.transition_gamma

        slope = 1 / (tr_min - tr_max)
        offset = 1 - slope * tr_min

        alpha = data.where(data > tr_min, 1.)
        alpha = alpha.where(data <= tr_max, 0.)
        alpha = alpha.where((data <= tr_min) | (data > tr_max), slope * data + offset)

        if self.invert_alpha:
            alpha.data = 1.0 - alpha.data

        alpha **= gamma
        if self.rgba:
            res = super().__call__((data, data, data, alpha), **kwargs)
        else:
            res = super().__call__((data, alpha), **kwargs)
        return res



class MyOwnMaskingCompositor(GenericCompositor):
    def __init__(self, name, transparency=None, conditions=None, alpha=False, **kwargs):
        super().__init__(name, **kwargs)
        if transparency:
            LOG.warning("Using 'transparency' is deprecated in "
                        "MaskingCompositor, use 'conditions' instead.")
            self.conditions = []
            for key, transp in transparency.items():
                self.conditions.append({"method": "equal",
                                        "value": key,
                                        "transparency": transp})
            LOG.info("Converted 'transparency' to 'conditions': %s",
                     str(self.conditions))
        else:
            self.conditions = conditions
        if self.conditions is None:
            raise ValueError("Masking conditions not defined.")
        self.alpha = alpha

    def __call__(self, projectables, *args, **kwargs):
        """Call the compositor."""
        if len(projectables) != 2:
            raise ValueError("Expected 2 datasets, got %d" % (len(projectables),))
        projectables = self.match_data_arrays(projectables)
        data_in = projectables[0]
        mask_in = projectables[1]

        alpha_attrs = data_in.attrs.copy()
        alpha = self._get_alpha_bands(data_in, mask_in, alpha_attrs)
        new_data_in = []

        if self.alpha:
            if "bands" in data_in.dims:
                for b in data_in["bands"]:
                    band_data = data_in.sel(bands=b).data
                    new_data_in.append(xr.DataArray(data=band_data, attrs=alpha_attrs, dims=data_in[0].dims,
                                                    coords=data_in[0].coords))


            else:
                band_data = data_in.data
                new_data_in.append(xr.DataArray(data=band_data, attrs=alpha_attrs, dims=data_in[0].dims,
                                                coords=data_in[0].coords))

            new_data_in.append(alpha)

        else:
            if "bands" in data_in.dims:
                for b in data_in["bands"]:
                    band_data = data_in.sel(bands=b).data * alpha.data
                    new_data_in.append(xr.DataArray(data=band_data, attrs=alpha_attrs, dims=data_in[0].dims,
                                                    coords=data_in[0].coords))

            else:
                band_data = data_in.data * alpha.data
                new_data_in.append(xr.DataArray(data=band_data, attrs=alpha_attrs, dims=data_in[0].dims,
                                                coords=data_in[0].coords))

        res = super().__call__(new_data_in, **kwargs)
        return res

    @staticmethod
    def _get_mask(method, value, mask_data):
        if method not in MASKING_COMPOSITOR_METHODS:
            raise AttributeError("Unsupported Numpy method %s, use one of %s",
                                 method, str(MASKING_COMPOSITOR_METHODS))

        func = getattr(np, method)

        if value is None:
            return func(mask_data)
        return func(mask_data, value)

    @staticmethod
    def _set_data_nans(data, mask, attrs):
        for i, dat in enumerate(data):
            data[i] = xr.where(mask, np.nan, dat)
            data[i].attrs = attrs

        return data

    def _get_alpha_bands(self, data, mask_in, alpha_attrs):
        try:
            mask_in = mask_in.isel(bands=0)
        except ValueError:
            pass
        mask_data = mask_in.data
        alpha = da.ones((data[0].sizes["y"],
                         data[0].sizes["x"]),
                        chunks=data[0].chunks)

        for condition in self.conditions:
            method = condition["method"]
            value = condition.get("value", None)
            transparency = condition["transparency"]
            mask = self._get_mask(method, value, mask_data)

            if transparency == 100.0:
                data = self._set_data_nans(data, mask, alpha_attrs)
            alpha_val = 1. - transparency / 100.
            alpha = da.where(mask, alpha_val, alpha)

        return xr.DataArray(data=alpha, attrs=alpha_attrs,
                            dims=data[0].dims, coords=data[0].coords)


def _mask_image_data(data, info):
    info.get("a")
    return data


@exclude_alpha
@on_dask_array
def _jma_true_color_reproduction(img_data, platform=None):
    ccm_dict = {"fy-4a": np.array([[1.1629, 0.1539, -0.2175],
                                   [-0.0252, 0.8725, 0.1300],
                                   [-0.0204, -0.1100, 1.0633]]),
                "fy-4b": np.array([[1.1619, 0.1542, -0.2168],
                                   [-0.0271, 0.8749, 0.1295],
                                   [-0.0202, -0.1103, 1.0634]]),
                "himawari-8": np.array([[1.1629, 0.1539, -0.2175],
                                        [-0.0252, 0.8725, 0.1300],
                                        [-0.0204, -0.1100, 1.0633]]),
                "himawari-9": np.array([[1.1619, 0.1542, -0.2168],
                                        [-0.0271, 0.8749, 0.1295],
                                        [-0.0202, -0.1103, 1.0634]]),
                "goes-16": np.array([[1.1425, 0.1819, -0.2250],
                                     [-0.0951, 0.9363, 0.1360],
                                     [-0.0113, -0.1179, 1.0621]]),
                "goes-17": np.array([[1.1437, 0.1818, -0.2262],
                                     [-0.0952, 0.9354, 0.1371],
                                     [-0.0113, -0.1178, 1.0620]]),
                "goes-18": np.array([[1.1629, 0.1539, -0.2175],
                                     [-0.0252, 0.8725, 0.1300],
                                     [-0.0204, -0.1100, 1.0633]]),
                "mtg-i1": np.array([[0.9007, 0.2086, -0.0100],
                                    [-0.0475, 1.0662, -0.0414],
                                    [-0.0123, -0.1342, 1.0794]]),
                "geo-kompsat-2a": np.array([[1.1661, 0.1489, -0.2157],
                                            [-0.0255, 0.8745, 0.1282],
                                            [-0.0205, -0.1103, 1.0637]]),
                }

    if platform is None:
        raise ValueError("Missing platform name.")

    try:
        ccm = ccm_dict[platform.lower()]
    except KeyError:
        raise KeyError(f"No conversion matrix found for platform {platform}")

    output = da.dot(img_data.T, ccm.T)
    return output.T


def rio_save(self, filename, fformat=None, fill_value=None,
             dtype=np.uint8, compute=True, tags=None,
             keep_palette=False, cmap=None, overviews=None,
             overviews_minsize=256, overviews_resampling=None,
             include_scale_offset_tags=False,
             scale_offset_tags=None,
             colormap_tag=None,
             driver=None,
             in_memory=False,
             **format_kwargs):
    from osgeo import gdal
    gdal.SetCacheMax(4096 * 1024 * 1024)
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")

    fformat = fformat or os.path.splitext(filename)[1][1:]
    drivers = {"jpg": "JPEG",
               "png": "PNG",
               "tif": "GTiff",
               "tiff": "GTiff",
               "jp2": "JP2OpenJPEG"}
    data_types = {np.uint8: gdal.GDT_Byte,
                  np.uint16: gdal.GDT_UInt16,
                  np.int16: gdal.GDT_Int16,
                  np.uint32: gdal.GDT_UInt32,
                  np.int32: gdal.GDT_Int32,
                  np.float32: gdal.GDT_Float32,
                  np.float64: gdal.GDT_Float64,
                  np.complex64: gdal.GDT_CFloat32,
                  np.complex128: gdal.GDT_CFloat64}
    color_interp = {"R": gdal.GCI_RedBand,
                    "G": gdal.GCI_GreenBand,
                    "B": gdal.GCI_BlueBand,
                    "A": gdal.GCI_AlphaBand,
                    "C": gdal.GCI_CyanBand,
                    "M": gdal.GCI_MagentaBand,
                    "Y": gdal.GCI_YellowBand,
                    "K": gdal.GCI_BlackBand,
                    "H": gdal.GCI_HueBand,
                    "S": gdal.GCI_SaturationBand,
                    "L": gdal.GCI_LightnessBand}
    if dtype not in data_types.keys():
        raise ValueError("Invalid data types")

    driver = driver or drivers.get(fformat, fformat)
    if driver == "COG" and overviews == []:
        overviews = None
    if include_scale_offset_tags:
        warnings.warn(
            "include_scale_offset_tags is deprecated, please use "
            "scale_offset_tags to indicate tag labels",
            DeprecationWarning, stacklevel=2)
        scale_offset_tags = scale_offset_tags or ("scale", "offset")

    if tags is None:
        tags = {}

    data, mode = self.finalize(fill_value, dtype=dtype,
                               keep_palette=keep_palette)
    data = data.transpose("bands", "y", "x")
    if mode in ["L", "LA"]:
        color_interp = {"L": gdal.GCI_GrayIndex,
                        "A": gdal.GCI_AlphaBand}

    crs = None
    gcps = None
    transform = None
    if driver in ["COG", "GTiff", "JP2OpenJPEG"]:
        if not np.issubdtype(data.dtype, np.floating):
            format_kwargs.setdefault("compress", "DEFLATE")
        photometric_map = {
            "RGB": "RGB",
            "RGBA": "RGB",
            "CMYK": "CMYK",
            "CMYKA": "CMYK",
            "YCBCR": "YCBCR",
            "YCBCRA": "YCBCR",
        }
        if mode.upper() in photometric_map:
            format_kwargs.setdefault("photometric",
                                     photometric_map[mode.upper()])

        from trollimage._xrimage_rasterio import get_data_arr_crs_transform_gcps
        crs, transform, gcps = get_data_arr_crs_transform_gcps(data)

        stime = data.attrs.get("start_time")
        if stime:
            stime_str = stime.strftime("%Y:%m:%d %H:%M:%S")
            tags.setdefault("TIFFTAG_DATETIME", stime_str)
    if driver == "JPEG" and "A" in mode:
        raise ValueError("JPEG does not support alpha")

    gcps_gdal = None
    if gcps:
        gcps_gdal = [gdal.GCP(gcp.x, gcp.y, gcp.z, gcp.col, gcp.row) for gcp in gcps]

    enhancement_colormap = self._get_colormap_from_enhancement_history(data)
    if colormap_tag and enhancement_colormap is not None:
        tags[colormap_tag] = enhancement_colormap.to_csv()
    if scale_offset_tags:
        self._add_scale_offset_to_tags(scale_offset_tags, data, tags)

    if driver == "COG":
        format_kwargs = self._gtiff_to_cog_kwargs(format_kwargs)

    new_format_kwargs = {}
    for key, value in format_kwargs.items():
        if value:
            new_key = str(key).upper()
            new_value = str(value).upper()
            if new_key not in {"SPARSE_OK", "WEBP_LOSSLESS"}:
                if new_value == "TRUE":
                    new_value = "YES"
                elif new_value == "FALSE":
                    new_value = "NO"
            new_format_kwargs[new_key] = new_value


    if os.path.exists(filename):
        gdal.Unlink(filename)

    if compute:
        if in_memory:
            filename = f"/vsimem/{os.path.basename(filename)}"
        with gdal.GetDriverByName(driver).Create(filename, data.sizes["x"], data.sizes["y"], data.sizes["bands"],
                                                 data_types[dtype], new_format_kwargs) as ds:
            ds.SetMetadata(tags)

            if crs:
                wkt = crs.to_wkt("WKT2_2018")
                ds.SetProjection(wkt)
            if transform:
                geotrans = transform.to_gdal()
                ds.SetGeoTransform(geotrans)
            if gcps_gdal:
                ds.SetGCPs(gcps_gdal, "")

            total_data = data.data.compute()
            for i in range(1, data.sizes["bands"] + 1):
                ds.GetRasterBand(i).SetColorInterpretation(color_interp[mode[i - 1]])
                if fill_value:
                    if fill_value == np.nan and dtype not in [np.float32, np.float64]:
                        raise ValueError("np.nan only for float32/64 image")
                    ds.GetRasterBand(i).SetNoDataValue(fill_value)

                band_data = total_data[i - 1]
                ds.GetRasterBand(i).WriteArray(band_data)

            ds.FlushCache()

        del band_data, data, total_data
        return 0

    gdal_dict = {"in_memory": in_memory,
                 "dirver": driver,
                 "filename": filename,
                 "xarray": data,
                 "dtype": data_types[dtype],
                 "other_kwargs": new_format_kwargs,
                 "tags": tags,
                 "proj": crs,
                 "transform": transform,
                 "gcps_gdal": gcps_gdal,
                 "fill_value": fill_value,
                 "colors": [color_interp[mode[i - 1]] for i in range(1, data.sizes["bands"] + 1)]}

    to_store = (data.data, gdal_dict)
    return to_store


def compute_writer_results(results):
    def gdal_dict_to_geotiff(tuple_in):
        gdal.SetCacheMax(4096 * 1024 * 1024)
        gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")

        dask_arr, gdal_dict = tuple_in
        in_memory, driver, filename, data, dtype, other_kwargs, tags, proj, transform, gcps_gdal, fill_value, colors = gdal_dict.values()
        total_data = data.data.compute()

        if in_memory:
            filename = f"/vsimem/{os.path.basename(filename)}"
        with gdal.GetDriverByName(driver).Create(filename, data.sizes["x"], data.sizes["y"], data.sizes["bands"],
                                                 dtype, other_kwargs) as ds:
            ds.SetMetadata(tags)

            if proj:
                wkt = proj.to_wkt("WKT2_2018")
                ds.SetProjection(wkt)
            if transform:
                geotrans = transform.to_gdal()
                ds.SetGeoTransform(geotrans)
            if gcps_gdal:
                ds.SetGCPs(gcps_gdal, "")

            for i in range(1, data.sizes["bands"] + 1):
                ds.GetRasterBand(i).SetColorInterpretation(colors[i - 1])
                if fill_value:
                    if fill_value == np.nan and dtype not in [np.float32, np.float64]:
                        raise ValueError("np.nan only for float32/64 image")
                    ds.GetRasterBand(i).SetNoDataValue(fill_value)

                band_data = total_data[i - 1]
                ds.GetRasterBand(i).WriteArray(band_data)

            ds.FlushCache()

        del band_data, data, total_data

    if not results:
        return

    if isinstance(results[0], list):
        results = results[0]

    # for res in results:
    #     gdal_dict_to_geotiff(res)

    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(gdal_dict_to_geotiff, results)
