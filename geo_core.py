#!/usr/bin/python
import bz2
import glob
import html.parser
import http.cookiejar
import math
import os
import queue
import re
import shutil
import time as time_calc
import urllib.error
import urllib.parse
import urllib.request
import warnings
import xml.etree.ElementTree
import zipfile
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
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
from mosaic_core import Mosaic
from osgeo import gdal
from pyproj import Proj

import satpy
import satpy.enhancements
from satpy import Scene, config, find_files_and_readers, utils
from satpy.composites import LOG, MASKING_COMPOSITOR_METHODS, GenericCompositor
from satpy.enhancements import exclude_alpha, on_dask_array
from satpy.readers import generic_image
from satpy.writers import geotiff

GDAL_OPTIONS = ("tfw",
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
                )

DATETIME_FMT = "%Y%m%d%H%M"
DATETIME_FMT_CN = "%Y年%m月%d日%H时%M分"
DATETIME_S_FMT_CN = "%Y年%m月%d日%H时%M分%S秒"

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
        self.sat = list(satellites_dict)[sat_index]
        self.reader_name = self.config_data["satellites"][self.sat]["reader_name"]

        proj_dict = self.config_data["projection"][self.sat]
        proj_name = list(proj_dict)[self.proj_index]
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
        self.resolution = round(self.resolution_x) if self.resolution_x >= 1 else self.resolution_x
        decimal_degree_meter_factor = 111319.5 if "latlon" in self.target_proj or "longlat" in self.target_proj else 1
        self.area_id = f"{self.proj_indicator}_{self.area}_{round(self.resolution * decimal_degree_meter_factor)}"

        self.original_geos_proj = satellites_dict[self.sat]["original_geos_proj"]

        self.band_list = []
        self.band_image_list = []
        self.exp_bands_count_list = []
        self.background_local_file_list = []
        self.background_satpy_file_list = []
        self.day_background_local_file_list = []
        self.day_background_satpy_file_list = []
        self.night_background_local_file_list = []
        self.night_background_satpy_file_list = []
        self.landmask_local_file_list = []
        self.landmask_satpy_file_list = []
        self.globalmask_local_file_list = []
        self.globalmask_satpy_file_list = []
        self.product_list = []

        satellite = {"GOES-WEST": "GOES-WEST",
                     "GOES-EAST": "GOES-WEST",
                     "MTG-FD": "MTG-FD",
                     "MTG-EU": "MTG-FD"}.get(self.sat, self.sat)
        product_dict = self.config_data["composites"][satellite]

        for product_idx in product_index_list:
            product = list(product_dict)[product_idx]
            self.product_list.append(product)

            composites = self.config_data["composites"][satellite][product]
            self.band_list.extend(composites["band_list"])
            self.band_image_list.extend(composites["band_image_list"])
            self.exp_bands_count_list.append(composites["expected_bands_count"])

            for key, target_list, base_path in [
                ("background_local_file", self.background_local_file_list, self.background_mask_path),
                ("background_satpy_file", self.background_satpy_file_list, self.base_path),
                ("day_background_local_file", self.day_background_local_file_list, self.background_mask_path),
                ("day_background_satpy_file", self.day_background_satpy_file_list, self.base_path),
                ("night_background_local_file", self.night_background_local_file_list, self.background_mask_path),
                ("night_background_satpy_file", self.night_background_satpy_file_list, self.base_path),
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

        self.band_list = list({}.fromkeys(self.band_list))
        self.band_image_list = list({}.fromkeys(self.band_image_list))

        gdal.UseExceptions()
        gdal.SetCacheMax(4096)
        gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")

    def update_sat(self, new_sat) -> None:
        self.sat = new_sat

        satellites_dict = self.config_data["satellites"]
        self.reader_name = self.config_data["satellites"][self.sat]["reader_name"]

        proj_dict = self.config_data["projection"][self.sat]
        proj_name = list(proj_dict)[self.proj_index]
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
        self.resolution = round(self.resolution_x) if self.resolution_x >= 1 else self.resolution_x
        decimal_degree_meter_factor = 111319.5 if "latlon" in self.target_proj or "longlat" in self.target_proj else 1
        self.area_id = f"{self.proj_indicator}_{self.area}_{round(self.resolution * decimal_degree_meter_factor)}"

        self.original_geos_proj = satellites_dict[self.sat]["original_geos_proj"]

        self.band_list = []
        self.band_image_list = []
        self.exp_bands_count_list = []
        self.background_local_file_list = []
        self.background_satpy_file_list = []
        self.day_background_local_file_list = []
        self.day_background_satpy_file_list = []
        self.night_background_local_file_list = []
        self.night_background_satpy_file_list = []
        self.landmask_local_file_list = []
        self.landmask_satpy_file_list = []
        self.globalmask_local_file_list = []
        self.globalmask_satpy_file_list = []
        self.product_list = []

        satellite = {"GOES-WEST": "GOES-WEST",
                     "GOES-EAST": "GOES-WEST",
                     "MTG-FD": "MTG-FD",
                     "MTG-EU": "MTG-FD"}.get(self.sat, self.sat)
        product_dict = self.config_data["composites"][satellite]

        for product_idx in self.product_index_list:
            product = list(product_dict)[product_idx]
            self.product_list.append(product)

            composites = self.config_data["composites"][satellite][product]
            self.band_list.extend(composites["band_list"])
            self.band_image_list.extend(composites["band_image_list"])
            self.exp_bands_count_list.append(composites["expected_bands_count"])

            for key, target_list, base_path in [
                ("background_local_file", self.background_local_file_list, self.background_mask_path),
                ("background_satpy_file", self.background_satpy_file_list, self.base_path),
                ("day_background_local_file", self.day_background_local_file_list, self.background_mask_path),
                ("day_background_satpy_file", self.day_background_satpy_file_list, self.base_path),
                ("night_background_local_file", self.night_background_local_file_list, self.background_mask_path),
                ("night_background_satpy_file", self.night_background_satpy_file_list, self.base_path),
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

        self.band_list = list({}.fromkeys(self.band_list))
        self.band_image_list = list({}.fromkeys(self.band_image_list))

    def valid_area(self) -> tuple[bool, list[str]]:
        area_errors = [
            (lambda: len(self.target_xy.split()) != 4, '\n"非经纬度xy坐标"填写有误'),
            (lambda: self.target_width <= 0 or self.target_height <= 0, '\n"宽/高"填写有误'),
            (lambda: self.resolution_x != self.resolution_y, '\n宽/高方向分辨率不相等\n检查 "宽/高" 和 "非经纬度xy坐标"')
        ]

        errors: list[str] = []
        area_pass = False

        for condition, error_message in area_errors:
            if condition():
                errors.append(error_message)

        if not errors:
            area_pass = True

        return area_pass, errors

    def arrange_sat_files(
            self,
            all_clean=False,
            post_process_clean=False
    ) -> tuple[list[str], dict[str, dict[str, str]], list[str], list[str]]:
        arrange_funcs = {
            "GK2A": lambda: arrange_gk2a_files(self.base_path, self.band_list, only_need_existed=all_clean),
            "HS": lambda: arrange_himawari_files(self.base_path, self.band_list, only_need_existed=all_clean),
            "FY4A": lambda: arrange_fy4_files(self.sat, self.base_path, self.band_list, only_need_existed=all_clean),
            "FY4B": lambda: arrange_fy4_files(self.sat, self.base_path, self.band_list, only_need_existed=all_clean),
            "FY4C": lambda: arrange_fy4_files(self.sat, self.base_path, self.band_list, only_need_existed=all_clean),
            "GOES-WEST": lambda: arrange_goes_files(self.sat, self.base_path, self.band_list,
                                                    only_need_existed=all_clean),
            "GOES-EAST": lambda: arrange_goes_files(self.sat, self.base_path, self.band_list,
                                                    only_need_existed=all_clean),
            "MTG-FD": lambda:arrange_fci_files(self.sat, self.base_path, self.band_list,
                                               only_need_existed=True if post_process_clean else all_clean),
            "MTG-EU": lambda: arrange_fci_files(self.sat, self.base_path, self.band_list,
                                                only_need_existed=True if post_process_clean else all_clean)
        }

        if self.sat not in arrange_funcs:
            return [], {}, [], []

        func = arrange_funcs[self.sat]
        missing_file, process_dict, actual_file, errors = func()

        return missing_file, process_dict, actual_file, errors

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
                                        callback=None)

        with gdal.Warp(crop_ref_file, void_file, options=warp_options) as crop_ds:
            crop_geotransform = crop_ds.GetGeoTransform()
            crop_ds_xsize = crop_ds.RasterXSize
            crop_ds_ysize = crop_ds.RasterYSize

        if self.target_proj != self.original_geos_proj:
            min_x = crop_geotransform[0]
            max_y = crop_geotransform[3]
            max_x = min_x + crop_geotransform[1] * crop_ds_xsize
            min_y = max_y + crop_geotransform[5] * crop_ds_ysize
            self_tolerance = 0.25 if "latlon" in self.target_proj or "longlat" in self.target_proj else 1000
            res_tolerance = 0.25 if "latlon" in self.target_proj or "longlat" in self.target_proj else 1000

            pad = 5 * round(self.resolution / res_tolerance) * self_tolerance
            round_coord = lambda v: round(v / self_tolerance) * self_tolerance
            crop_min_x = round_coord(min_x) - pad
            crop_min_y = round_coord(min_y) - pad
            crop_max_x = round_coord(max_x) + pad
            crop_max_y = round_coord(max_y) + pad
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
    def background_file_handling(
            local_file_list: list[str],
            satpy_file_list: list[str],
            image_geometry: tuple[str,
                                  tuple[int | float, int | float, int | float, int | float, int | float, int | float],
                                  int, int, int | float, int | float, int | float, int | float]
    ) -> None:

        proj, geotransform, width, height, area_min_x, area_min_y, area_max_x, area_max_y = image_geometry

        def do_warp():
            gdalwarp_by_extent(proj, gdal.GRA_NearestNeighbour, local_file, satpy_file,
                               target_width=width, target_height=height,
                               area_min_x=area_min_x, area_min_y=area_min_y,
                               area_max_x=area_max_x, area_max_y=area_max_y,
                               format_kwargs={"TILED": "YES", "BLOCKXSIZE": "512", "BLOCKYSIZE": "512",
                                              "COMPRESS": "DEFLATE", "ZLEVEL": "6"})

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

        trollimage.xrimage.XRImage.save = save
        trollimage.xrimage.XRImage.rio_save = rio_save
        geotiff.GeoTIFFWriter.GDAL_OPTIONS = GDAL_OPTIONS
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

        for local_file_list, satpy_file_list in [
            (self.background_local_file_list, self.background_satpy_file_list),
            (self.day_background_local_file_list, self.day_background_satpy_file_list),
            (self.night_background_local_file_list, self.night_background_satpy_file_list),
            (self.landmask_local_file_list, self.landmask_satpy_file_list),
            (self.globalmask_local_file_list, self.globalmask_satpy_file_list)]:
            self.background_file_handling(local_file_list, satpy_file_list,
                                          (self.target_proj, geotransform, target_width, target_height,
                                           area_min_x, area_min_y, area_max_x, area_max_y)
                                          )

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
                                   compress="deflate", zlevel=6, predictor=3, in_memory=True)

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
                               format_kwargs={"TILED": "YES", "BLOCKXSIZE": "512", "BLOCKYSIZE": "512",
                                              "COMPRESS": "DEFLATE", "ZLEVEL": "6"})
            gdal.Unlink(output_info["output_tif"])
        end_warp = time_calc.perf_counter()
        print(f"中介数据重投影耗时: {(end_warp - start_warp): .3f}")

        with gdal.Open(image[list(image)[0]]["warp_tif"]) as ds:
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

    def run_satpy_processor_single_sat(
            self
    ) -> tuple[list[str], list[str], list[str], list[str], dict[str, dict[str, list[dict] | int]]]:
        product_dict_list: list[dict[str, dict[str, str | datetime]]] = []
        mosaic_piece: dict[str, dict[str, list[dict] | int]] = {}
        time_info: list[str] = []
        err_info: list[str] = []

        print(f"开始整理{self.sat}卫星数据")
        start_arrange = time_calc.perf_counter()
        missing_file, process_dict, _, arr_errors = self.arrange_sat_files()
        end_arrange = time_calc.perf_counter()
        arrange_time = end_arrange - start_arrange
        print(f"整理卫星数据耗时: {arrange_time: .3f}")

        arrange_err_info = arr_errors
        if arr_errors:
            print("数据整理时发生错误:")
            for err in arr_errors:
                print(err)

        missing_info = missing_file
        if missing_file:
            print("以下数据集有缺失:")
            for file in missing_file:
                print(file)

            return time_info, arrange_err_info, missing_info, err_info, mosaic_piece

        print(f"开始处理{self.sat}卫星数据")
        start_sat = time_calc.perf_counter()
        crop_boundary = self.get_geos_crop_boundary()

        process_folders = list(process_dict)
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
                err_info.append(f"{self.sat}卫星下的{process_folder}文件夹报错, 报错信息为: {e}")

            end = time_calc.perf_counter()
            done_count = done_count + 1
            print(f"{self.sat}卫星单个节点数据耗时: {(end - start): .3f}")
            print("******************************\n"
                  "\n"
                  f"-----时间节点完成: {done_count} / {count}------\n"
                  "\n"
                  "******************************")

        end_sat = time_calc.perf_counter()
        single_sat_timing = f"{self.sat}卫星处理总耗时: {get_elapsed_time(start_sat, arrange_time + end_sat)}\n"
        print(single_sat_timing)

        time_info = [f"{self.sat}卫星处理开始于: {process_folders[0]}",
                     f"{self.sat}卫星处理结束于: {process_folders[-1]}",
                     single_sat_timing] if count > 0 else [f"{self.sat}卫星并无数据", single_sat_timing]

        for product, exp_bands_count in zip(self.new_product_list, self.new_exp_bands_count_list):
            mosaic_piece[product] = {"images": [], "exp_bands_count": exp_bands_count}
            for product_dict in product_dict_list:
                if product in product_dict:
                    mosaic_piece[product]["images"].append(product_dict[product])

        return time_info, arrange_err_info, missing_info, err_info, mosaic_piece

    def run_satpy_processor(
            self
    ) -> tuple[dict[str, dict[str, list[dict] | int]], list[str], float, dict[str, list[str]], dict[str, list[str]],
               dict[str, list[str]], dict[str, str], dict[str, str]]:
        print("\033[H\033[J")

        satpy_process_time_list: list[str] = []
        satpy_error_dict: dict[str, list[str]] = {}
        arrange_error_dict: dict[str, list[str]] = {}
        satpy_missing_dict: dict[str, list[str]] = {}
        mosaic_dict: dict[str, dict[str, list[dict] | int]] = {}

        process_email_dict: dict[str, str] = {}
        warning_email_dict: dict[str, str] = {}

        area_pass, area_errors = self.valid_area()
        if not area_pass:
            for error in area_errors:
                print(error)
                return (mosaic_dict, satpy_process_time_list, 0, arrange_error_dict,
                        satpy_missing_dict, satpy_error_dict, process_email_dict, warning_email_dict)

        start_bj_time = f"卫星数据处理开始于: {(datetime.now(UTC) + timedelta(hours=8)).strftime(
            DATETIME_S_FMT_CN)}"
        start_0 = time_calc.perf_counter()

        result = defaultdict(dict)
        sat_list = [self.sat] if not self.sos else self.global_satellites
        for sat in sat_list:
            self.update_sat(sat)

            time_info, arrange_err_info, missing_info, err_info, mosaic_piece = self.run_satpy_processor_single_sat()
            satpy_process_time_list.extend(time_info)
            if arrange_err_info:
                arrange_error_dict[sat] = arrange_err_info
            if missing_info:
                satpy_missing_dict[sat] = missing_info
            if err_info:
                satpy_error_dict[sat] = err_info

            for product, product_infos in mosaic_piece.items():
                if "images" in product_infos:
                    result[product].setdefault("images", []).extend(product_infos["images"])
                if "exp_bands_count" in product_infos and "exp_bands_count" not in result[product]:
                    result[product]["exp_bands_count"] = product_infos["exp_bands_count"]

        mosaic_dict = dict(result)

        end_0 = time_calc.perf_counter()
        end_bj_time = f"卫星数据处理结束于: {(datetime.now(UTC) + timedelta(hours=8)).strftime(
            DATETIME_S_FMT_CN)}"

        all_sats_time = end_0 - start_0
        all_sats_timing = f"所有卫星处理总耗时: {get_elapsed_time(start_0, end_0)}"

        if self.sos:
            print(all_sats_timing)

        satpy_process_time_list.extend([all_sats_timing, start_bj_time, end_bj_time])

        for err_list in satpy_error_dict.values():
            for err in err_list:
                print(err)

        process_email_dict, warning_email_dict = self.post_process_message_handling(
            (satpy_process_time_list, satpy_missing_dict, satpy_error_dict, arrange_error_dict))

        return (mosaic_dict, satpy_process_time_list, all_sats_time, arrange_error_dict,
                satpy_missing_dict, satpy_error_dict, process_email_dict, warning_email_dict)

    def write_black_image(self, mosaic_output: str, exp_bands_count: int, metadata: dict):
        with gdal.GetDriverByName("GTiff").Create(
                mosaic_output, self.target_width, self.target_height, exp_bands_count, gdal.GDT_Byte,
                {"COMPRESS": "DEFLATE", "ZLEVEL": "6", "PREDICTOR": "2"}) as ds:
            ds.SetProjection(self.target_proj)
            ds.SetGeoTransform((self.area_min_x, self.resolution_x, 0.0,
                                self.area_max_y, 0.0, -self.resolution_y))
            ds.SetMetadata(metadata)
            ds.FlushCache()

    def global_mosaic_single_node_product(
            self,
            date: datetime,
            mosaic_dict_info: dict[str, list[dict] | str | int],
            product: str
    ) -> tuple[list[str], list[str], list[str], list[str]]:
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
            "GOES-EAST": ("goes_east_pass", "raster_goes_east_image", ["G16", "G19"], "FLDK", None),
            "MTG-FD": ("europe_pass", "raster_europe_image", ["MTG-I1"], "FLDK", None),
        }

        date_str = date.strftime("%Y%m%d_%H%M%S")
        image_dict_list = mosaic_dict_info["images"]
        exp_bands_count = mosaic_dict_info["exp_bands_count"]

        pass_dict = {key: False for key, *_ in sat_region_config.values()}
        raster_images = {key: {} for _, key, *_ in sat_region_config.values()}

        raster_list: list[str] = []
        mask_list: list[str] = []

        failure_list: list[str] = []
        mosaic_err_list: list[str] = []
        clean_list: list[str] = []

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
                    failure_list.append(f"{sat_key}: {product}")

        metadata = {"start_time": date.strftime("%Y%m%d%H%M%S.%fZ"),
                    "end_time": (date + timedelta(minutes=10)).strftime("%Y%m%d%H%M%S.%fZ"),
                    "proj_id": self.area_id.split("_")[0],
                    "area_id": self.area_id.split("_")[1], }

        # tmp_folder = f"{self.base_path}/global_{date_str}_tmp"
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
                mosaic_err_list.append(mosaic_error)

        else:
            self.write_black_image(mosaic_output, exp_bands_count, metadata)

        clean_list.extend(raster_list)
        clean_list.extend(mask_list)
        mosaic_list = [mosaic_output]
        return mosaic_list, failure_list, mosaic_err_list, clean_list

    def global_mosaic_single_node(
            self,
            date: datetime,
            mosaic_dict: dict[str, dict[str, list[dict] | str | int]]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
        failure_info: dict[str, list[str]] = {}
        mosaic_error_info: dict[str, list[str]] = {}
        clean_info: dict[str, list[str]] = {}

        date_str = date.strftime("%Y%m%d_%H%M%S")

        res_list = []
        mosaics: list[str] = []

        failure_key = date.strftime(DATETIME_FMT_CN)
        failure_info[failure_key]: list[str] = []
        mosaic_error_info[failure_key]: list[str] = []

        tmp_folder = f"{self.base_path}/global_{date_str}_tmp"
        if not os.path.exists(tmp_folder) or (os.path.exists(tmp_folder) and not os.path.isdir(tmp_folder)):
            os.mkdir(tmp_folder)
        clean_info["folder"] = [tmp_folder]
        clean_info["file"]: list[str] = []

        if not mosaic_dict:
            failure_info[failure_key].append("所有产品全部无法生成")

        for product, info in mosaic_dict.items():
            (mosaic_list, failure_list,
             mosaic_err_list, clean_list) = self.global_mosaic_single_node_product(date, info, product)
            mosaics.extend(mosaic_list)
            failure_info[failure_key].extend(failure_list)
            mosaic_error_info[failure_key].extend(mosaic_err_list)
            clean_info["file"].extend(clean_list)

        if mosaic_dict:
            scn = Scene(filenames=mosaics, reader="geo_int_global_image", reader_kwargs={"mode": "int"})
            scn.load(self.product_list)

            for product, bands_count in zip(self.product_list, self.exp_bands_count_list):
                fill_value = None if bands_count in [2, 4] else 0

                final = f"GEOs_{product}_{self.area_id}_{scn.start_time.strftime("%Y%m%d_%H%M%S")}.tif"
                res = scn.save_dataset(product, filename=final, base_dir=self.base_path, writer="geotiff",
                                       blockxsize=512, blockysize=512, compress="jpeg", jpeg_quality=80,
                                       fill_value=fill_value, compute=False)
                res_list.append(res)

            compute_writer_results(res_list)

        return failure_info, mosaic_error_info, clean_info

    def global_mosaic(self, start_dt, end_dt):
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=UserWarning)
        warnings.simplefilter(action="ignore", category=RuntimeWarning)

        trollimage.xrimage.XRImage.save = save
        trollimage.xrimage.XRImage.rio_save = rio_save
        satpy.readers.generic_image._mask_image_data = _mask_image_data
        satpy.composites.MaskingCompositor = MyOwnMaskingCompositor
        satpy.composites.CloudCompositor = MyOwnCloudCompositor
        satpy.composites.HighCloudCompositor.__bases__ = (MyOwnCloudCompositor,)
        satpy.composites.LowCloudCompositor.__bases__ = (MyOwnCloudCompositor,)
        satpy.enhancements._jma_true_color_reproduction = _jma_true_color_reproduction

        (mosaic_dict_list, satpy_process_time_list, all_sats_time, arrange_error_dict,
         satpy_missing_dict, satpy_error_dict, _, _) = self.run_satpy_processor()

        start_dt = datetime.strptime(start_dt, DATETIME_FMT)
        end_dt = datetime.strptime(end_dt, DATETIME_FMT)

        count = len(list(date_range(start_dt, end_dt - timedelta(minutes=10), timedelta(minutes=10))))

        config.set(config_path=[self.satpy_config_path])
        utils.debug_off()

        start_bj_time = f"融合处理开始于: {(datetime.now(UTC) + timedelta(hours=8)).strftime(
            DATETIME_S_FMT_CN)}"
        start_0 = time_calc.perf_counter()

        failure_dict: dict[str, list[str]] = {}
        mosaic_error_dict: dict[str, list[str]] = {}
        clean_dict: dict[str, list[str]] = {}

        done_count = 0
        for date in date_range(start_dt, end_dt - timedelta(minutes=10), timedelta(minutes=10)):
            start_time_node = time_calc.perf_counter()

            failure_info, mosaic_error_info, clean_info = self.global_mosaic_single_node(date, mosaic_dict_list)
            failure_key = date.strftime(DATETIME_FMT_CN)
            if failure_info[failure_key]:
                failure_dict.update(failure_info)
            if mosaic_error_info[failure_key]:
                mosaic_error_dict.update(mosaic_error_info)
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
            DATETIME_S_FMT_CN)}"

        all_nodes_timing = f"所有时间节点融合总耗时: {get_elapsed_time(start_0, end_0)}"
        print(all_nodes_timing)

        all_timing = f"处理全流程总耗时: {get_elapsed_time(start_0, end_0 + all_sats_time)}"
        print(all_timing)

        start_dt_str = start_dt.strftime(DATETIME_FMT_CN)
        end_dt_str = end_dt.strftime(DATETIME_FMT_CN)
        dt_range = f"起始时间节点: {start_dt_str}\n结束时间节点: {end_dt_str}"
        mosaic_process_time_list = [all_timing, all_nodes_timing, start_bj_time, end_bj_time, dt_range]

        process_email_dict, warning_email_dict = self.post_process_message_handling(
            (satpy_process_time_list, satpy_missing_dict, satpy_error_dict, arrange_error_dict),
            (mosaic_process_time_list, (start_dt_str, end_dt_str), failure_dict, mosaic_error_dict))

        return clean_dict, process_email_dict, warning_email_dict

    def post_process_message_handling(
            self,
            satpy_part: tuple[list[str], dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]],
            mosaic_part: tuple[list[str], tuple[str, str], dict[str, list[str]], dict[str, list[str]]] | None = None
    ) -> tuple[dict[str, str], dict[str, str]]:
        satpy_process_time_list, satpy_missing_dict, satpy_error_dict, arrange_error_dict = satpy_part
        (mosaic_process_time_list, start_end,
         failure_dict, mosaic_error_dict) = mosaic_part if mosaic_part else ([], (), {}, {})
        start_dt_str, end_dt_str = start_end if start_end else ("", "")

        process_msg_list: list[str] = []
        warning_msg = ", 但有错误发生" if failure_dict else ""

        image_in_topic = f"{start_dt_str}至{end_dt_str}的全球静止卫星融合影像" if mosaic_part else f"{self.sat}的影像"
        process_topic = f"{image_in_topic}生成完毕{warning_msg}"
        process_msg_list.extend(satpy_process_time_list)
        process_msg_list.append("\n")
        if mosaic_part:
            process_msg_list.extend(mosaic_process_time_list)
            process_msg_list.append("\n")

        process_msg = "\n".join(process_msg_list)
        process_email_dict = {"topic": process_topic, "message": process_msg}

        warning_topic = f"警告!!! {image_in_topic}生成出现问题"
        warning_msg_list = []
        if mosaic_part:
            warning_msg_list.append("下面列出可能缺失产品的时间点: \n")
            for time_node, product_list in failure_dict.items():
                warning = f"{time_node}, 缺失如下产品: {", ".join(product_list)}"
                warning_msg_list.append(warning)
            warning_msg_list.append("\n")

            for time_node, err_list in mosaic_error_dict.items():
                warning_msg_list.append(f"{time_node}, 羽化融合发生错误: \n")
                warning_msg_list.extend(err_list)
            warning_msg_list.append("\n")

        for sat, missing_list in satpy_missing_dict.items():
            warning_msg_list.append(f"{sat}卫星的下列数据有缺失: \n")
            warning_msg_list.extend(missing_list)
            warning_msg_list.append("\n")

        for sat, err_list in arrange_error_dict.items():
            warning_msg_list.append(f"{sat}卫星数据整理时发生错误: \n")
            warning_msg_list.extend(err_list)
            warning_msg_list.append("\n")

        for sat, err_list in satpy_error_dict.items():
            warning_msg_list.append("satpy处理过程中报错: \n")
            warning_msg_list.extend(err_list)
            warning_msg_list.append("\n")

        warning_msg = "\n".join(warning_msg_list)

        warning_email_dict = {
            "topic": warning_topic,
            "message": warning_msg
        } if failure_dict or satpy_missing_dict or satpy_error_dict or arrange_error_dict else {}

        return process_email_dict, warning_email_dict

    def run_clean(self, clean_dict: dict[str, list[str]] | None = None, all_clean=False) -> None:
        sat_list = [self.sat] if not self.sos else self.global_satellites
        for sat in sat_list:
            self.update_sat(sat)
            print(f"开始善后{sat}卫星数据")
            _, _, actual_file, _ = self.arrange_sat_files(all_clean=all_clean, post_process_clean=not all_clean)

            if not all_clean:
                print(f"{sat}卫星数据善后完毕\n")
                continue

            for file in actual_file:
                try:
                    os.remove(file)
                except Exception as e:
                    print(e)

            print(f"{sat}卫星数据善后完毕\n")

        if not clean_dict:
            return

        print("开始善后其他临时文件")
        for item_type, items in clean_dict.items():
            if not isinstance(items, list):
                continue
            for item in items:
                try:
                    if item_type == "folder":
                        shutil.rmtree(item, ignore_errors=True)
                    elif item_type == "file":
                        os.remove(item)
                except Exception as e:
                    print(e)

        print("其他临时文件善后完毕\n")



class GeoCalculator:
    def __init__(self, config_data, sat_index, calc_latlon, calc_proj, calc_name, calc_area, calc_area_id,
                 need_square_correction=False, square_correction_lim=400):
        super().__init__()
        self.config_data = config_data
        self.sat_index = sat_index
        satellites_dict = config_data["satellites"]
        self.calc_sat = list(satellites_dict)[self.sat_index]
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

            ds.FlushCache()
            ds2.FlushCache()

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
    def __init__(self, config_data: dict, sat_index: int, product_index_list: list[int],
                 start_date_str: str, start_time_str: str, end_date_str: str, end_time_str: str, sos=False):
        super().__init__()
        self.config_data = config_data
        self.product_index_list = product_index_list
        self.sos = sos

        self.global_satellites = config_data["global_satellites"]
        self.conda_path = config_data["basic_path"]["conda_path"]
        self.base_path = config_data["basic_path"]["base_path"]
        self.satellites_dict = config_data["satellites"]
        self.goes_area = config_data["goes_area"]
        self.himawari_area = config_data["himawari-area"]
        self.sat = list(self.satellites_dict)[sat_index]

        self.og_start_dt = datetime.strptime(start_date_str + start_time_str, DATETIME_FMT)
        self.og_end_dt = datetime.strptime(end_date_str + end_time_str, DATETIME_FMT)
        self.start_dt_str, self.end_dt_str = self.correct_date()
        self.start_dt = datetime.strptime(self.start_dt_str, DATETIME_FMT)
        self.end_dt = datetime.strptime(self.end_dt_str, DATETIME_FMT)

        self.band_list = []

        satellite = {"GOES-WEST": "GOES-WEST",
                     "GOES-EAST": "GOES-WEST",
                     "MTG-FD": "MTG-FD",
                     "MTG-EU": "MTG-FD"}.get(self.sat, self.sat)
        product_dict = config_data["composites"][satellite]

        for product_idx in product_index_list:
            product = list(product_dict)[product_idx]
            composites = config_data["composites"][satellite][product]
            self.band_list.extend(composites["band_list"])

        self.band_list = list({}.fromkeys(self.band_list))

    def update_sat(self, new_sat: str):
        self.sat = new_sat
        self.band_list = []

        satellite = {"GOES-WEST": "GOES-WEST",
                     "GOES-EAST": "GOES-WEST",
                     "MTG-FD": "MTG-FD",
                     "MTG-EU": "MTG-FD"}.get(self.sat, self.sat)
        product_dict = self.config_data["composites"][satellite]

        for product_idx in self.product_index_list:
            product = list(product_dict)[product_idx]
            composites = self.config_data["composites"][satellite][product]
            self.band_list.extend(composites["band_list"])

        self.band_list = list({}.fromkeys(self.band_list))

    def arrange_sat_files(self) -> tuple[list[str], list[str]]:
        arrange_funcs = {
            "GK2A": lambda: arrange_gk2a_files(self.base_path, self.band_list, only_need_existed=True),
            "HS": lambda: arrange_himawari_files(self.base_path, self.band_list, only_need_existed=True),
            "FY4A": lambda: arrange_fy4_files(self.sat, self.base_path, self.band_list, only_need_existed=True),
            "FY4B": lambda: arrange_fy4_files(self.sat, self.base_path, self.band_list, only_need_existed=True),
            "FY4C": lambda: arrange_fy4_files(self.sat, self.base_path, self.band_list, only_need_existed=True),
            "GOES-WEST": lambda: arrange_goes_files(self.sat, self.base_path, self.band_list, only_need_existed=True),
            "GOES-EAST": lambda: arrange_goes_files(self.sat, self.base_path, self.band_list, only_need_existed=True),
            "MTG-FD": lambda: arrange_fci_files(self.sat, self.base_path, self.band_list, only_need_existed=True),
            "MTG-EU": lambda: arrange_fci_files(self.sat, self.base_path, self.band_list, only_need_existed=True)
        }

        if self.sat not in arrange_funcs:
            return [], []

        func = arrange_funcs[self.sat]
        _, _, actual_file, errors = func()

        return actual_file, errors

    def correct_date(self) -> tuple[str, str]:
        def found_nearest_minute(date, minute_threshold):
            remainder = date.minute % minute_threshold
            remainder_threshold = minute_threshold / 2 if minute_threshold % 2 == 0 else (minute_threshold + 1) / 2
            date = date + timedelta(minutes=minute_threshold - remainder) \
                if remainder >= remainder_threshold else date - timedelta(minutes=remainder)

            return date

        start_dt_sat = {"GK2A": datetime(2019, 7, 25, 0, 0),
                        "HS": datetime(2015, 7, 1, 0, 0),
                        "GOES-WEST": datetime(2018, 11, 15, 0, 0),
                        "GOES-EAST": datetime(2017, 12, 15, 0, 0),
                        "MTG-FD": datetime(2024, 9, 25, 0, 0),
                        "MTG-EU": datetime(2026, 1, 1, 0, 0)}

        og_date1 = found_nearest_minute(self.og_start_dt, 10) if self.og_start_dt.minute % 10 != 0 else self.og_start_dt
        og_date2 = found_nearest_minute(self.og_end_dt, 10) if self.og_end_dt.minute % 10 != 0 else self.og_end_dt

        if self.sat in start_dt_sat:
            start_dt = start_dt_sat[self.sat]
            og_date1 = max(og_date1, start_dt)
            og_date2 = max(og_date2, start_dt + timedelta(minutes=10))

        date1_str = og_date1.strftime(DATETIME_FMT)
        date2_str = og_date2.strftime(DATETIME_FMT)

        return date1_str, date2_str

    def download_sat_files_single_sat(
            self,
            need_download: bool,
            gk2a_download_mode="api",
            hs_download_mode="api"
    ) -> tuple[list[str], dict[str, dict[str, list[str]]], int, list[str], list[str], list[str]]:
        link_funcs = {
            "HS": lambda avoid_file: self.gk2a_himawari_api_link(avoid_file),
            "GK2A": lambda avoid_file: self.gk2a_himawari_api_link(avoid_file),
            "GOES-WEST": lambda avoid_file: self.goes_aws_link(avoid_file),
            "GOES-EAST": lambda avoid_file: self.goes_aws_link(avoid_file),
            "MTG-FD": lambda avoid_file: self.mtg_datastore_link(avoid_file),
            "MTG-EU": lambda avoid_file: self.mtg_datastore_link(avoid_file),
            "FY4A": lambda: ({}, [], []),
            "FY4B": lambda: ({}, [], []),
            "FY4C": lambda: ({}, [], [])
        }
        if gk2a_download_mode == "aws":
            link_funcs.update({"GK2A": lambda avoid_file: self.himawari_gk2a_aws_link(avoid_file)})
        if hs_download_mode == "aws":
            link_funcs.update({"HS": lambda avoid_file: self.himawari_gk2a_aws_link(avoid_file)})

        print(f"开始整理{self.sat}卫星数据")
        start_arrange = time_calc.perf_counter()
        actual_file, arr_errors = self.arrange_sat_files()
        end_arrange = time_calc.perf_counter()
        print(f"整理卫星数据耗时: {(end_arrange - start_arrange): .3f}")

        arrange_err_info = arr_errors
        if arr_errors:
            print("数据整理时发生错误:")
            for err in arr_errors:
                print(err)

        dl_start_notify = f"开始下载{self.sat}卫星数据" if need_download else f"开始生成{self.sat}卫星数据链接"
        print(dl_start_notify)
        start_sat = time_calc.perf_counter()

        link_func = link_funcs[self.sat]
        dl_dict, dl_list, link_errors = link_func(actual_file)
        time_node_info = dl_dict
        link_err_info = link_errors

        merged_txt = f"unicorn_{self.sat}_{self.start_dt_str[:8]}_{self.start_dt_str[8:12]}_{self.end_dt_str[:8]}_{self.end_dt_str[8:12]}.txt"
        if os.path.exists(merged_txt):
            with open(merged_txt, "r") as file:
                lines = file.readlines()
            os.remove(merged_txt)

            ext = ".bz2" if self.sat == "HS" else ""
            avoid_filename = set([os.path.basename(file) + ext for file in actual_file] if actual_file else [])
            if self.sat == "GK2A" and gk2a_download_mode == "api":
                condition = lambda line: extract_filename_gk2a_api_link(line.strip()) not in avoid_filename
            else:
                condition = lambda line: os.path.basename(line.strip()) not in avoid_filename
            dl_list.extend([line.strip() for line in lines if condition(line)])

        dl_list = list(dict.fromkeys(dl_list))
        dl_list.sort()

        download_err_info: list[str] = []
        if need_download:
            dl_errors = url_requests_downloader(dl_list, self.base_path, threads=20)
            download_err_info = dl_errors

        elif not need_download and dl_list:
            with open(merged_txt, "w+") as file:
                lines = [link + "\n" for link in dl_list]
                file.writelines(lines)

        link_length = len(dl_list)

        end_sat = time_calc.perf_counter()

        operation_str = "下载" if need_download else "生成链接"
        single_sat_timing = f"{self.sat}卫星{operation_str}总耗时: {get_elapsed_time(start_sat, end_sat)}\n"
        print(single_sat_timing)
        download_time_info = [single_sat_timing]

        return download_time_info, time_node_info, link_length, arrange_err_info, link_err_info, download_err_info

    def download_sat_files(self, need_download=False):
        sat_list = [self.sat] if not self.sos else self.global_satellites

        arrange_error_dict: dict[str, list[str]] = {}
        link_error_dict: dict[str, list[str]] = {}
        time_node_dict: dict[str, dict[str, dict[str, list[str]]]] = {}
        link_length_dict: dict[str, int] = {}
        download_error_dict: dict[str, list[str]] = {}
        download_time_list: list[str] = []

        start_bj_time = f"卫星数据下载开始于: {(datetime.now(UTC) + timedelta(hours=8)).strftime(DATETIME_S_FMT_CN)}"
        start_0 = time_calc.perf_counter()

        for sat in sat_list:
            self.update_sat(sat)

            (download_time_info, time_node_info, link_length,
             arrange_err_info, link_err_info, download_err_info) = self.download_sat_files_single_sat(need_download)

            download_time_list.extend(download_time_info)
            link_length_dict[sat] = link_length
            if time_node_info:
                time_node_dict[sat] = time_node_info
            if arrange_err_info:
                arrange_error_dict[sat] = arrange_err_info
            if link_err_info:
                link_error_dict[sat] = link_err_info
            if download_err_info:
                download_error_dict[sat] = download_err_info

        end_0 = time_calc.perf_counter()
        end_bj_time = f"卫星数据下载结束于: {(datetime.now(UTC) + timedelta(hours=8)).strftime(
            DATETIME_S_FMT_CN)}"

        operation_str = "下载" if need_download else "生成链接"
        all_sats_timing = f"所有卫星{operation_str}总耗时: {get_elapsed_time(start_0, end_0)}"
        print(all_sats_timing)
        download_time_list.extend([all_sats_timing, start_bj_time, end_bj_time])

        download_email_dict, warning_email_dict = self.post_download_message_handling((download_time_list,
                                                                                       arrange_error_dict,
                                                                                       link_error_dict,
                                                                                       download_error_dict))
        return link_length_dict, download_email_dict, warning_email_dict

    def post_download_message_handling(
            self,
            download_part: tuple[list[str], dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]):
        download_time_list, arrange_error_dict, link_error_dict, download_error_dict = download_part
        download_msg_list: list[str] = []
        warning_msg = ", 但有错误发生" if download_error_dict else ""
        download_topic = (f"{self.start_dt.strftime(DATETIME_FMT_CN)}至{self.end_dt.strftime(DATETIME_FMT_CN)}"
                          f"的卫星数据下载完毕{warning_msg}")
        download_msg_list.extend(download_time_list)
        download_msg_list.append("\n")
        download_msg = "\n".join(download_msg_list)
        download_email_dict = {"topic": download_topic,
                               "message": download_msg}

        warning_topic = (
            f"警告!!! {self.start_dt.strftime(DATETIME_FMT_CN)}至{self.end_dt.strftime(DATETIME_FMT_CN)}"
            f"的卫星数据下载出现问题")
        warning_msg_list: list[str] = []
        for sat, err_list in arrange_error_dict.items():
            warning_msg_list.append(f"在整理{sat}卫星的数据以获取已有文件时报错: \n")
            warning_msg_list.extend(err_list)
            warning_msg_list.append("\n")

        for sat, err_list in link_error_dict.items():
            warning_msg_list.append(f"在生成{sat}卫星的链接时报错: \n")
            warning_msg_list.extend(err_list)
            warning_msg_list.append("\n")

        for sat, err_list in download_error_dict.items():
            warning_msg_list.append(f"{sat}卫星的数据下载时报错: \n")
            warning_msg_list.extend(err_list)
            warning_msg_list.append("\n")

        warning_msg_list.append("\n")
        warning_msg = "\n".join(warning_msg_list)

        warning_email_dict = {
            "topic": warning_topic,
            "message": warning_msg} if download_error_dict or link_error_dict or arrange_error_dict else {}

        return download_email_dict, warning_email_dict

    def gk2a_aws_single_hour(
            self,
            time_node: datetime
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], list[str]]:
        gk2a_nasa_mode = "https://data.nas.nasa.gov/geonex/geonexdata/GK2A/AMI/L1B/FD"
        aws_start_dt = datetime(2023, 2, 16, 0, 0)
        all_suffix = {"vi004": "010", "vi005": "010", "vi006": "005", "vi008": "010",
                      "nr013": "020", "nr016": "020",
                      "sw038": "020", "wv063": "020", "wv069": "020", "wv073": "020",
                      "ir087": "020", "ir096": "020",
                      "ir105": "020", "ir112": "020", "ir123": "020", "ir133": "020"}

        dl_dict: dict[str, list[str]] = {}
        tmp_dict: dict[str, list[str]] = {}
        errors: list[str] = []

        date_str = time_node.strftime(DATETIME_FMT)
        year, month, day, hour = date_str[:4], date_str[4:6], date_str[6:8], date_str[8:10]

        hour_start = time_node.replace(minute=0)
        hour_end = hour_start + timedelta(minutes=50)
        if time_node < aws_start_dt:
            for tmp_dt in date_range(hour_start, hour_end, timedelta(minutes=10)):
                if tmp_dt < self.start_dt or tmp_dt > self.end_dt - timedelta(minutes=10):
                    continue

                tmp_dt_str = tmp_dt.strftime(DATETIME_FMT)
                tmp_minute = tmp_dt_str[10:12]
                tmp_dict[tmp_dt_str]: list[str] = []
                for band in self.band_list:
                    band = band.lower()
                    file = f"gk2a_ami_le1b_{band}_fd{all_suffix[band]}ge_{year}{month}{day}{hour}{tmp_minute}.nc"
                    link = f"{gk2a_nasa_mode}/{year}{month}/{day}/{hour}/{file}"
                    tmp_dict[tmp_dt_str].append(link)

            return dl_dict, tmp_dict, errors

        bucket_name = "noaa-gk2a-pds"
        aws_mode = f"https://{bucket_name}.s3.amazonaws.com"
        client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        try:
            response = client.list_objects_v2(Bucket=bucket_name, Prefix=f"AMI/L1B/FD/{year}{month}/{day}/{hour}")
        except Exception as e:
            errors.append(f"在连接GK2A S3存储桶时发生错误, 时间节点{year}-{month}-{day}-{hour}, 报错信息: {e}")
            return dl_dict, tmp_dict, errors

        for tmp_dt in date_range(hour_start, hour_end, timedelta(minutes=10)):
            if tmp_dt < self.start_dt or tmp_dt > self.end_dt - timedelta(minutes=10):
                continue

            tmp_dt_str = tmp_dt.strftime(DATETIME_FMT)
            tmp_minute = tmp_dt_str[10:12]
            dl_dict[tmp_dt_str]: list[str] = []
            for band in self.band_list:
                band = band.lower()
                for content in response.get("Contents", []):
                    filename = os.path.basename(content["Key"])
                    filename_split = filename.split("_")
                    file_band = filename_split[3]
                    file_dt = filename_split[5][:12]

                    if file_band == band and file_dt == f"{year}{month}{day}{hour}{tmp_minute}":
                        link = f"{aws_mode}/AMI/L1B/FD/{year}{month}/{day}/{hour}/{filename}"
                        dl_dict[tmp_dt_str].append(link)

        return dl_dict, tmp_dict, errors

    def himawari_aws_single_node(
            self,
            time_node: datetime
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], list[str]]:
        all_suffix = {"B01": "R10", "B02": "R10", "B03": "R05", "B04": "R10", "B05": "R20",
                      "B06": "R20", "B07": "R20",
                      "B08": "R20", "B09": "R20", "B10": "R20", "B11": "R20", "B12": "R20",
                      "B13": "R20", "B14": "R20",
                      "B15": "R20", "B16": "R20"}
        h8_end_dt = datetime(2022, 12, 13, 4, 50)
        h9_start_dt = datetime(2022, 12, 13, 10, 0)
        aws_start_dt = datetime(2019, 12, 9, 18, 10)
        h8_nasa_mode = "https://data.nas.nasa.gov/geonex/geonexdata/HIMAWARI8/JMA-L1B/AHI/Hsfd"

        dl_dict: dict[str, list[str]] = {}
        tmp_dict: dict[str, list[str]] = {}
        errors: list[str] = []

        sat_name = "H08" if time_node <= h8_end_dt else "H09"

        date_str = time_node.strftime(DATETIME_FMT)
        year, month, day, hour, minute = date_str[:4], date_str[4:6], date_str[6:8], date_str[8:10], date_str[10:12]
        dl_dict[date_str]: list[str] = []

        new_date = f"{year}{month}{day}"
        new_time = f"{hour}{minute}"

        if h8_end_dt < time_node < h9_start_dt:
            return dl_dict, tmp_dict, errors

        if time_node < aws_start_dt:
            for band in self.band_list:
                for num in range(1, 11):
                    seg = f"S{num:02d}10"
                    file = f"HS_{sat_name}_{new_date}_{new_time}_{band}_{all_suffix[band]}_FLDK_{seg}.DAT.bz2"
                    link = f"{h8_nasa_mode}/{year}/{month}/{day}/{new_date}{hour}/{minute}/{band}/{file}"
                    tmp_dict[date_str].append(link)

            return dl_dict, tmp_dict, errors

        bucket_name = "noaa-himawari8" if sat_name == "H08" else "noaa-himawari9"
        aws_mode = f"https://{bucket_name}.s3.amazonaws.com"
        client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        try:
            response = client.list_objects_v2(Bucket=bucket_name,
                                              Prefix=f"AHI-L1b-FLDK/{year}/{month}/{day}/{hour}{minute}")
        except Exception as e:
            errors.append(f"在连接{sat_name} S3存储桶时发生错误, 时间节点{year}-{month}-{day}-{hour}-{minute}, 报错信息: {e}")
            return dl_dict, tmp_dict, errors

        for band in self.band_list:
            for num in range(1, 11):
                seg = f"S{num:02d}10"
                for content in response.get("Contents", []):
                    filename = os.path.basename(content["Key"])
                    filename_split = filename.split("_")
                    file_band = filename_split[4]
                    file_dt = filename_split[2] + filename_split[3]
                    file_seg = filename_split[7][:5]
                    link = f"{aws_mode}/AHI-L1b-FLDK/{year}/{month}/{day}/{hour}{minute}/{filename}"

                    if file_band == band and file_dt == f"{year}{month}{day}{hour}{minute}" and file_seg == seg:
                        dl_dict[date_str].append(link)

        return dl_dict, tmp_dict, errors

    def himawari_gk2a_aws_link(
            self,
            avoid_file: list[str] | None = None
    ) -> tuple[dict[str, dict[str, list[str]]], list[str], list[str]]:
        ext = ".bz2" if self.sat == "HS" else ""
        avoid_filename = set([os.path.basename(file) + ext for file in avoid_file] if avoid_file else [])

        dl_dict: dict[str, dict[str, list[str]]] = {}
        tmp_dict: dict[str, dict[str, list[str]]] = {}
        errors: list[str] = []

        dl_dict["FD"]: dict[str, list[str]] = {}
        tmp_dict["FD"]: dict[str, list[str]] = {}

        start_hour = self.start_dt.replace(minute=0)
        end_hour = (self.end_dt + timedelta(hours=1)).replace(minute=0) if self.end_dt.minute != 0 else self.end_dt

        dt_range_start = self.start_dt if self.sat == "HS" else start_hour
        dt_range_end = self.end_dt - timedelta(minutes=10) if self.sat == "HS" else end_hour
        time_intervene = timedelta(minutes=10) if self.sat == "HS" else timedelta(hours=1)
        for date in date_range(dt_range_start, dt_range_end, time_intervene):
            if self.sat == "HS":
                dl_dict_node, tmp_dict_node, errors_node = self.himawari_aws_single_node(date)
            else:
                dl_dict_node, tmp_dict_node, errors_node = self.gk2a_aws_single_hour(date)
            dl_dict["FD"].update(dl_dict_node)
            tmp_dict["FD"].update(tmp_dict_node)
            errors.extend(errors_node)

        tmp_list = extract_link_from_dl_dict(tmp_dict)
        valid_list = valid_url(tmp_list)
        new_tmp_dict = filter_dl_dict(tmp_dict, lambda link: link in set(valid_list))

        merged_dl_dict = merge_dl_dicts(new_tmp_dict, dl_dict)
        merged_dl_list = extract_link_from_dl_dict(merged_dl_dict)
        print(merged_dl_list)
        final_dl_dict = filter_dl_dict(merged_dl_dict, lambda link: os.path.basename(link) not in avoid_filename)
        final_dl_list = extract_link_from_dl_dict(final_dl_dict)
        final_dl_list.sort()

        return final_dl_dict, final_dl_list, errors

    def gk2a_api_single_area_hour(
            self,
            time_node: datetime,
            area,
    ) -> tuple[dict[str, list[str]], list[str]]:
        url = "http://api.nmsc.kma.go.kr:9080/api/GK2A/LE1B"
        api_key = "NMSC610a32d21941491cb2591a59ad97009b"

        dl_dict: dict[str, list[str]] = {}
        errors: list[str] = []

        date_str = time_node.strftime(DATETIME_FMT)
        year, month, day, hour = date_str[:4], date_str[4:6], date_str[6:8], date_str[8:10]

        hour_start = time_node.replace(minute=0)
        hour_end = hour_start + timedelta(minutes=50)
        hour_start_str = hour_start.strftime(DATETIME_FMT)
        hour_end_str = hour_end.strftime(DATETIME_FMT)

        api_list = []
        for band in self.band_list:
            api_url = f"{url}/{band}/{area}/dataList?sDate={hour_start_str}&eDate={hour_end_str}&format=xml&key={api_key}"
            try:
                response = requests.get(api_url)
                root = xml.etree.ElementTree.fromstring(response.text)
                api_list.extend([(band, date.text) for date in root.findall("item")])
            except Exception as e:
                errors.append(f"在连接GK2A OpenAPI时发生错误, "
                              f"区域{area}, 波段{band}, 时间节点{year}-{month}-{day}-{hour}, 报错信息: {e}")
                continue

        for tmp_dt in date_range(hour_start, hour_end, timedelta(minutes=10)):
            if tmp_dt < self.start_dt or tmp_dt > self.end_dt - timedelta(minutes=10):
                continue

            tmp_dt_str = tmp_dt.strftime(DATETIME_FMT)
            dl_dict[tmp_dt_str]: list[str] = []
            for band in self.band_list:
                for res_band, res_date_str in api_list:
                    link = f"{url}/{res_band}/{area}/data?date={res_date_str}&key={api_key}"

                    if res_band == band and res_date_str == tmp_dt_str:
                        dl_dict[tmp_dt_str].append(link)

        return dl_dict, errors

    def himawari_api_single_area_hour(
            self,
            time_node: datetime,
            area,
    ) -> tuple[dict[str, list[str]], list[str]]:
        host = "himawari.diasjp.net"
        url = "http://" + host + "/expert/bin/search.cgi"
        username = "lpeacemaker711@gmail.com"
        password = "lulu1989"
        segments = 10 if area == "FLDK" else 1

        dl_dict: dict[str, list[str]] = {}
        errors: list[str] = []

        date_str = time_node.strftime(DATETIME_FMT)
        year, month, day, hour = date_str[:4], date_str[4:6], date_str[6:8], date_str[8:10]

        hour_start_for_dias = time_node.replace(minute=0) + timedelta(minutes=10)
        hour_end_for_dias = hour_start_for_dias + timedelta(minutes=40)

        hour_start_for_dias_str = hour_start_for_dias.strftime("%Y-%m-%dT%H:%M")
        hour_end_for_dias_str = hour_end_for_dias.strftime("%Y-%m-%dT%H:%M")

        hour_start = time_node.replace(minute=0)
        hour_end = hour_start + timedelta(minutes=50)

        param = ["format=xml", f"start={hour_start_for_dias_str}", f"end={hour_end_for_dias_str}",
                 f"area={area}", "type=HS", "band=" + ",".join(self.band_list)]

        access = DIASAccess(username, password)
        try:
            response = access.open(url, "&".join(param).encode("utf-8"))
            root = xml.etree.ElementTree.fromstring(response.read())
            response.close()
            dias_list = [item.attrib["id"] for item in root.findall("item")]
        except Exception as e:
            errors.append(f"在连接dias-jp时发生错误, 区域{area}, 时间节点{year}-{month}-{day}-{hour}, 报错信息: {e}")
            return dl_dict, errors

        for tmp_dt in date_range(hour_start, hour_end, timedelta(minutes=10)):
            if tmp_dt < self.start_dt or tmp_dt > self.end_dt - timedelta(minutes=10):
                continue

            tmp_dt_str = tmp_dt.strftime(DATETIME_FMT)
            dl_dict[tmp_dt_str]: list[str] = []
            for band in self.band_list:
                for num in range(1, segments + 1):
                    seg = f"S{num:02d}10"
                    for file in dias_list:
                        filename = os.path.basename(file)
                        filename_split = filename.split("_")
                        file_band = filename_split[4]
                        file_dt = filename_split[2] + filename_split[3]
                        file_seg = filename_split[7][:5]
                        link = f"http://{host}/expert/original/bin/original-download.cgi?file={file}"
                        if file_band == band and file_dt == tmp_dt_str and file_seg == seg:
                            dl_dict[tmp_dt_str].append(link)

        return dl_dict, errors

    def gk2a_himawari_api_link(
            self,
            avoid_file: list[str] | None = None
    ) -> tuple[dict[str, dict[str, list[str]]], list[str], list[str]]:
        ext = ".bz2" if self.sat == "HS" else ""
        avoid_filename = set([os.path.basename(file) + ext for file in avoid_file] if avoid_file else [])

        dl_dict: dict[str, dict[str, list[str]]] = {}
        errors: list[str] = []

        start_hour = self.start_dt.replace(minute=0)
        end_hour = (self.end_dt + timedelta(hours=1)).replace(minute=0) if self.end_dt.minute != 0 else self.end_dt

        areas = self.himawari_area if self.sat == "HS" else ["FD"]
        for area in areas:
            dl_dict[area]: dict[str, list[str]] = {}
            for date in date_range(start_hour, end_hour, timedelta(hours=1)):
                if self.sat == "HS":
                    dl_dict_hour, errors_hour = self.himawari_api_single_area_hour(date, area)
                else:
                    dl_dict_hour, errors_hour = self.gk2a_api_single_area_hour(date, area)
                dl_dict[area].update(dl_dict_hour)
                errors.extend(errors_hour)

        if self.sat == "HS":
            condition = lambda link: os.path.basename(link) not in avoid_filename
        else:
            condition = lambda link: extract_filename_gk2a_api_link(link) not in avoid_filename
        final_dl_dict = filter_dl_dict(dl_dict, condition)
        final_dl_list = extract_link_from_dl_dict(final_dl_dict)
        final_dl_list.sort()

        return final_dl_dict, final_dl_list, errors

    def goes_aws_single_area_hour(
            self,
            time_node: datetime,
            area: str
    ) -> tuple[dict[str, list[str]], list[str]]:
        scan_areas = {"FD": "ABI-L1b-RadF", "CONUS": "ABI-L1b-RadC", "MESO": "ABI-L1b-RadM"}
        time_deltas = {"FD": timedelta(seconds=420), "CONUS": timedelta(seconds=84), "MESO": timedelta(seconds=42)}

        time_delta = time_deltas[area]
        s3_mode = scan_areas[area]

        dl_dict: dict[str, list[str]] = {}
        errors: list[str] = []

        if self.sat == "GOES-EAST":
            sat_name = "goes16" if time_node < datetime(2025, 4, 2, 0, 0) else "goes19"
        else:
            sat_name = "goes17" if time_node < datetime(2022, 7, 28, 0, 0) else "goes18"
        scan_mode = "3" if time_node <= datetime(2019, 4, 2, 16, 0) else "6"
        bucket_name = f"noaa-{sat_name}"
        aws_mode = f"https://{bucket_name}.s3.amazonaws.com/{s3_mode}"

        date_str = time_node.strftime(DATETIME_FMT)
        day_num = time_node.strftime("%j")
        year = date_str[:4]
        hour = date_str[8:10]

        client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        try:
            response = client.list_objects_v2(Bucket=bucket_name,
                                              Prefix=f"{s3_mode}/{year}/{day_num}/{hour}")
        except Exception as e:
            errors.append(f"在连接{sat_name} S3存储桶时发生错误, 区域{area}, 时间节点{year}-{day_num}-{hour}, 报错信息: {e}")
            return dl_dict, errors

        hour_start = time_node.replace(minute=0)
        hour_end = hour_start + timedelta(minutes=50)
        for tmp_dt in date_range(hour_start, hour_end, timedelta(minutes=10)):
            if tmp_dt < self.start_dt or tmp_dt > self.end_dt - timedelta(minutes=10):
                continue

            tmp_dt_str = tmp_dt.strftime(DATETIME_FMT)
            dl_dict[tmp_dt_str]: list[str] = []
            for band in self.band_list:
                for content in response.get("Contents", []):
                    filename = os.path.basename(content["Key"])
                    filename_split = filename.split("_")
                    file_band = filename_split[1].split("-")[3][-3:]
                    file_scan_mode = filename_split[1].split("-")[3][1]
                    file_start_dt = datetime.strptime(filename_split[3].replace("s", ""), "%Y%j%H%M%S%f")
                    file_end_dt = datetime.strptime(filename_split[4].replace("e", ""), "%Y%j%H%M%S%f")

                    latest_start = max(tmp_dt, file_start_dt)
                    earliest_end = min(tmp_dt + timedelta(minutes=10), file_end_dt)

                    if earliest_end - latest_start >= time_delta and file_band == band and file_scan_mode == scan_mode:
                        link = f"{aws_mode}/{year}/{day_num}/{hour}/{filename}"
                        dl_dict[tmp_dt_str].append(link)

        return dl_dict, errors

    def goes_aws_link(
            self,
            avoid_file: list[str] | None = None
    ) -> tuple[dict[str, dict[str, list[str]]], list[str], list[str]]:
        avoid_filename = set([os.path.basename(file) for file in avoid_file] if avoid_file else [])

        dl_dict: dict[str, dict[str, list[str]]] = {}
        errors: list[str] = []

        start_hour = self.start_dt.replace(minute=0)
        end_hour = (self.end_dt + timedelta(hours=1)).replace(minute=0) if self.end_dt.minute != 0 else self.end_dt

        for area in self.goes_area:
            dl_dict[area]: dict[str, list[str]] = {}
            for date in date_range(start_hour, end_hour, timedelta(hours=1)):
                dl_dict_hour, errors_hour = self.goes_aws_single_area_hour(date, area)
                dl_dict[area].update(dl_dict_hour)
                errors.extend(errors_hour)

        final_dl_tmp_dict = filter_dl_dict(dl_dict, lambda url: os.path.basename(url) not in avoid_filename)
        final_dl_tmp_list = extract_link_from_dl_dict(final_dl_tmp_dict)

        inspected_list: list[str] = []
        seen_files: list[tuple[str, datetime, datetime]] = []

        for link in final_dl_tmp_list:
            filename = os.path.basename(link).split("_")
            channel = filename[1]
            start_dt = datetime.strptime(filename[3].replace("s", ""), "%Y%j%H%M%S%f")
            end_dt = datetime.strptime(filename[4].replace("e", ""), "%Y%j%H%M%S%f")

            is_unique = True
            for seen_channel, seen_start, seen_end in seen_files:
                if channel == seen_channel and abs((start_dt - seen_start).total_seconds()) <= 1 and abs(
                        (end_dt - seen_end).total_seconds()) <= 1:
                    is_unique = False
                    break

            if is_unique:
                inspected_list.append(link)
                seen_files.append((channel, start_dt, end_dt))

        final_dl_dict = filter_dl_dict(final_dl_tmp_dict, lambda url: url in inspected_list)
        final_dl_list = extract_link_from_dl_dict(final_dl_dict)
        final_dl_list.sort()

        return final_dl_dict, final_dl_list, errors

    def mtg_datastore_single_hour(
            self,
            time_node: datetime,
    ) -> tuple[dict[str, list[str]], list[str]]:
        area = "FD" if self.sat == "MTG-FD" else "Q4"
        time_delta = timedelta(seconds=420) if self.sat == "MTG-FD" else timedelta(seconds=105)
        datastore_bands = {"FDHSI": "EO:EUM:DAT:0662", "HRFI": "EO:EUM:DAT:0665"}

        consumer_key = "Ij21HQ3icd8eY352nKymL5P1xUga"
        consumer_secret = "_nU7zzczMJ3128fhTDPcVqipBcoa"
        credentials = (consumer_key, consumer_secret)
        token = eumdac.AccessToken(credentials, validity=86400)
        datastore = eumdac.DataStore(token)

        dl_dict: dict[str, list[str]] = {}
        errors: list[str] = []

        if self.sat == "MTG-FD":
            sat_name = "MTI1" if time_node < datetime(2027, 1, 1, 0, 0) else "MTI3"
        else:
            sat_name = "MTI2" if time_node < datetime(2028, 1, 1, 0, 0) else "MTI4"

        hour_start = time_node.replace(minute=0)
        hour_end = hour_start + timedelta(hours=1)
        hour_list: list[str] = []
        for band in self.band_list:
            datastore_id = datastore_bands[band]
            selected_collection = datastore.get_collection(datastore_id)
            try:
                products = selected_collection.search(sat=sat_name,
                                                      dtstart=hour_start, dtend=hour_end - timedelta(seconds=1),
                                                      coverage=area)
                hour_list.extend([product.url for product in products])
            except Exception as e:
                errors.append(f"在连接eumetsat datastore时发生错误, 卫星{sat_name}, 波段{band}, "
                              f"起止时间{hour_start.strftime(DATETIME_FMT)}-{hour_end.strftime(DATETIME_FMT)}, "
                              f"报错信息: {e}")

        for tmp_dt in date_range(hour_start, hour_end, timedelta(minutes=10)):
            if tmp_dt < self.start_dt or tmp_dt > self.end_dt - timedelta(minutes=10):
                continue

            tmp_dt_str = tmp_dt.strftime(DATETIME_FMT)
            dl_dict[tmp_dt_str]: list[str] = []

            for link in hour_list:
                filename_split = os.path.basename(link).split("_")
                file_start_dt = datetime.strptime(filename_split[7].replace("s", ""), "%Y%m%d%H%M%S")
                file_end_dt = datetime.strptime(filename_split[8].replace("e", ""), "%Y%m%d%H%M%S")

                latest_start = max(tmp_dt, file_start_dt)
                earliest_end = min(tmp_dt + timedelta(minutes=10), file_end_dt)

                if earliest_end - latest_start >= time_delta:
                    dl_dict[tmp_dt_str].append(link)
        return dl_dict, errors

    def mtg_datastore_link(
            self,
            avoid_file: list[str] | None = None
    ) -> tuple[dict[str, dict[str, list[str]]], list[str], list[str]]:
        avoid_filename = set([os.path.basename(file) for file in avoid_file] if avoid_file else [])

        area = "FD" if self.sat == "MTG-FD" else "Q4"

        dl_dict: dict[str, dict[str, list[str]]] = {}
        errors: list[str] = []

        start_hour = self.start_dt.replace(minute=0)
        end_hour = (self.end_dt + timedelta(hours=1)).replace(minute=0) if self.end_dt.minute != 0 else self.end_dt

        dl_dict[area]: dict[str, list[str]] = {}
        for date in date_range(start_hour, end_hour, timedelta(hours=1)):
            dl_dict_hour, errors_hour = self.mtg_datastore_single_hour(date)
            dl_dict[area].update(dl_dict_hour)
            errors.extend(errors_hour)

        final_dl_dict = filter_dl_dict(dl_dict, lambda link: os.path.basename(link) not in avoid_filename)
        final_dl_list = extract_link_from_dl_dict(final_dl_dict)
        final_dl_list.sort()

        return final_dl_dict, final_dl_list, errors


def get_elapsed_time(start: int | float, end: int | float) -> str:
    elapsed = end - start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}小时{int(m)}分{int(round(s))}秒"


def date_range(start_date: datetime, end_date: datetime, delta: timedelta):
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += delta


def merge_dl_dicts(d1: dict[str, dict[str, list[str]]],
                   d2: dict[str, dict[str, list[str]]]) -> dict[str, dict[str, list[str]]]:
    merged = defaultdict(lambda: defaultdict(list))

    for outer_key, inner_dict in d1.items():
        for inner_key, url_list in inner_dict.items():
            merged[outer_key][inner_key].extend(url_list)

    for outer_key, inner_dict in d2.items():
        for inner_key, url_list in inner_dict.items():
            merged[outer_key][inner_key].extend(url_list)

    result: dict[str, dict[str, list[str]]] = {}
    for outer_key, inner_dict in merged.items():
        result[outer_key] = {}
        for inner_key, url_list in inner_dict.items():
            result[outer_key][inner_key] = list(dict.fromkeys(url_list))

    return result


def extract_link_from_dl_dict(dl_dict: dict[str, dict[str, list[str]]]) -> list[str]:
    return [link for area_dl_dict in dl_dict.values() for dl_list in area_dl_dict.values() for link in dl_list]


def filter_dl_dict(dl_dict: dict[str, dict[str, list[str]]],
                   condition: Callable[[str], bool]) -> dict[str, dict[str, list[str]]]:
    return {area: {date_str: [link for link in dl_list if condition(link)]
                   for date_str, dl_list in area_dl_dict.items()}
            for area, area_dl_dict in dl_dict.items()}


def valid_url(link_list: list[str]) -> list[str]:
    dl_queue = queue.Queue()

    def is_valid_url(link: str):
        try:
            urllib.request.urlopen(link)
            dl_queue.put(link)
        except urllib.error.HTTPError:
            pass

    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(is_valid_url, link_list)

    dl_list = []
    while not dl_queue.empty():
        dl_list.append(dl_queue.get())

    return list(dl_queue.queue)


def extract_filename_gk2a_api_link(link: str):
    res_suffix = {"vi004": "010", "vi005": "010", "vi006": "005", "vi008": "010", "nr013": "020", "nr016": "020",
                  "sw038": "020", "wv063": "020", "wv069": "020", "wv073": "020", "ir087": "020", "ir096": "020",
                  "ir105": "020", "ir112": "020", "ir123": "020", "ir133": "020"}

    link_split = link.split("/")
    band = link_split[6]
    area = link_split[7]
    date = link_split[8].split("=")[1][:12]
    filename = f"gk2a_ami_le1b_{band.lower()}_{area.lower()}{res_suffix[band.lower()]}ge_{date}.nc"

    return filename


def url_requests_downloader(link_list: list[str], base_path: str, threads=10) -> list[str]:
    errors = queue.Queue()
    def download(link: str):
        if "EUMETSAT-Darmstadt" in link:
            filename = os.path.basename(unquote(link)).split("?")[0] + ".zip"
        elif "api.nmsc" in link:
            filename = extract_filename_gk2a_api_link(link)
        else:
            filename = os.path.basename(link)

        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            return

        try:
            with requests.get(link, timeout=(5, 30)) as response:
                response.raise_for_status()
                with open(filepath, "wb") as f:
                    f.write(response.content)
        except Exception as e:
            print(e)
            errors.put(f"{link}下载时发生错误, 报错信息: {e}")

    def diasjp_download(link):
        host = "himawari.diasjp.net"
        url = "http://" + host + "/expert/original/bin/original-download.cgi"
        username = "lpeacemaker711@gmail.com"
        password = "lulu1989"

        file_path_server = link.split("=")[1]
        filename = f"{os.path.basename(file_path_server)}.zip"
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            return

        access = DIASAccess(username, password)
        param = ["archiver=zip", f"file={file_path_server}"]
        try:
            response = access.open(url, "&".join(param).encode("utf-8"))
        except Exception as e:
            errors.put(f"{link}下载时发生错误, 报错信息: {e}")
            return

        with open(filepath, "wb") as f:
            while True:
                buf = response.read(32768)
                if not buf:
                    break
                f.write(buf)
        response.close()

    mtg_links = [link for link in link_list if "EUMETSAT-Darmstadt" in link]
    himawari_api_links = [link for link in link_list if "diasjp" in link]
    gk2a_api_links = [link for link in link_list if "api.nmsc" in link]
    other_links = [link for link in link_list if "EUMETSAT-Darmstadt" not in link and "diasjp" not in link]

    error_list: list[str] = []
    for link_list in [other_links, himawari_api_links, gk2a_api_links, mtg_links]:
        dl_threads = 1 if link_list in [mtg_links, gk2a_api_links] else threads

        if not link_list == himawari_api_links:
            with ThreadPoolExecutor(max_workers=dl_threads) as executor:
                executor.map(download, link_list)
            while not errors.empty():
                error_list.append(errors.get())
            continue

        with ThreadPoolExecutor(max_workers=dl_threads) as executor:
            executor.map(diasjp_download, link_list)
        while not errors.empty():
            error_list.append(errors.get())

    return error_list


def decimal_digits(decimal_str: str) -> int:
    location = decimal_str.find(".")
    digits = 0 if location == -1 else len(decimal_str) - location - 1
    return digits


def gdalwarp_by_extent(target_proj: str, resampler, infile: str, outfile: str,
                       target_width: int, target_height: int,
                       area_min_x: int | float, area_min_y: int | float,
                       area_max_x: int | float, area_max_y: int | float,
                       format_kwargs: dict | None = None, callback=None, source_proj: str | None = None):
    with gdal.Open(infile) as ds:
        base_dtype = ds.GetRasterBand(1).DataType
    predictor = "2" if base_dtype in [1, 2, 3, 4, 5, 14] else "3"

    format_kwargs.setdefault("BIGTIFF", "IF_SAFER")
    format_kwargs.setdefault("PREDICTOR", predictor)
    creationoptions = [f"{pair[0]}={pair[1]}" for pair in format_kwargs.items()]

    bigtiff_format_kwargs = format_kwargs.copy()
    bigtiff_format_kwargs.update({"BIGTIFF": "YES"})
    bigtiff_creationoptions = [f"{pair[0]}={pair[1]}" for pair in format_kwargs.items()]

    warp_options_dict = {"format": "GTiff",
                         "srcSRS": source_proj,
                         "dstSRS": target_proj,
                         "width": target_width,
                         "height": target_height,
                         "outputBounds": (area_min_x, area_min_y, area_max_x, area_max_y),
                         "multithread": True,
                         "errorThreshold": 0,
                         "warpMemoryLimit": 21474836480,
                         "resampleAlg": resampler,
                         "warpOptions": ["NUM_THREADS=ALL_CPUS"],
                         "creationOptions": creationoptions,
                         "callback": callback}
    bigtiff_warp_options_dict = warp_options_dict.copy()
    bigtiff_warp_options_dict.update({"creationOptions": bigtiff_creationoptions})

    warp_options = gdal.WarpOptions(**warp_options_dict)
    bigtiff_warp_options = gdal.WarpOptions(**bigtiff_warp_options_dict)

    try:
        gdal.Warp(outfile, infile, options=warp_options)
    except RuntimeError:
        try:
            gdal.Warp(outfile, infile, options=bigtiff_warp_options)
        except Exception as e:
            print(e)
    except Exception as e:
        print(e)


def rename_file_ext(path: str, file_with_path: str, old_ext: str, new_ext: str) -> list[str]:
    errors: list[str] = []

    filename_without_ext = os.path.splitext(os.path.basename(file_with_path))[0]
    try:
        os.rename(f"{path}/{filename_without_ext}.{old_ext}", f"{path}/{filename_without_ext}.{new_ext}")
    except Exception as e:
        errors.append(f"将{filename_without_ext}文件重命名时报错, 文件可能被占用, 报错信息为: {e}")

    return errors


def move_data_files_out_folder(path: str, sat_names: list[str],
                               scan_areas: list[str], folder_mid_part_scheme: str) -> list[str]:
    data_files: list[str] = []
    tmp_folders: list[str] = []
    errors: list[str] = []

    for satellite in sat_names:
        for scan_area in scan_areas:
            area_tmp_folder = [
                tmp_folder for tmp_folder in glob.glob(f"{path}/{satellite}_{folder_mid_part_scheme}_{scan_area}") if
                 os.path.isdir(tmp_folder)]
            tmp_folders.extend(area_tmp_folder)

            data_files.extend(
                [item for tmp_folder in area_tmp_folder for item in glob.glob(f"{tmp_folder}/*") if
                 os.path.isfile(item) and "desktop.ini" not in item]
            )

    for data in data_files:
        try:
            if "EUMETSAT-Darmstadt" in data and data.endswith(".nc"):
                os.remove(data)
            else:
                shutil.move(data, path)
        except Exception as e:
            errors.append(f"将{data}文件从处理文件夹移出时报错, 文件可能被占用, 报错信息为: {e}")

    for tmp_folder in tmp_folders:
        shutil.rmtree(tmp_folder, ignore_errors=True)

    return errors

def move_data_files_into_folder(folder_sat_name_idx: int, folder_scan_area_idx: int,
                                file_move_dict: dict[str, str]) -> tuple[dict[str, dict[str, str]], list[str]]:
    errors: list[str] = []
    process_dict: dict[str, dict[str, str]] = {}

    for file, folder in file_move_dict.items():
        if not os.path.exists(folder) or (os.path.exists(folder) and not os.path.isdir(folder)):
            os.mkdir(folder)
        try:
            shutil.move(file, folder)
        except Exception as e:
            errors.append(f"将{file}文件移动到处理文件夹时报错, 文件可能被占用, 报错信息为: {e}")

    for folder in file_move_dict.values():
        folder_sat_name = os.path.basename(folder).split("_")[folder_sat_name_idx]
        folder_scan_area = os.path.basename(folder).split("_")[folder_scan_area_idx]
        process_dict[folder] = {"sat_name": folder_sat_name, "scan_area": folder_scan_area}

    return process_dict, errors


def extract_bz2(bz2_file: str):
    newfilepath = bz2_file[:-4]
    with bz2.BZ2File(bz2_file, "rb") as zip_file, open(newfilepath, "wb") as new_file:
        new_file.write(zip_file.read())
    os.remove(bz2_file)


def extract_diasjp_zip(zip_file: str):
    output_dir = os.path.dirname(zip_file)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        for item in zip_ref.infolist():
            if item.filename.endswith(".bz2") and not item.is_dir():
                filename = os.path.basename(item.filename)
                output_path = os.path.join(output_dir, filename)
                with zip_ref.open(item) as source, open(output_path, "wb") as target:
                    target.write(source.read())
    os.remove(zip_file)


def arrange_himawari_files(
        path: str,
        band_list: list[str],
        only_need_existed=False
) -> tuple[list[str], dict[str, dict[str, str]], list[str], list[str]]:
    missing_file: list[str] = []
    actual_file: list[str] = []

    file_move_dict: dict[str, str] = {}
    process_dict: dict[str, dict[str, str]] = {}

    errors: list[str] = []

    sat_names = ["H08", "H09"]
    scan_areas = ["FLDK", "R301", "R302", "R303", "R304", "JP01", "JP02", "JP03", "JP04"]
    res_suffix = {"B01": "R10", "B02": "R10", "B03": "R05", "B04": "R10", "B05": "R20", "B06": "R20", "B07": "R20",
                  "B08": "R20", "B09": "R20", "B10": "R20", "B11": "R20", "B12": "R20", "B13": "R20", "B14": "R20",
                  "B15": "R20", "B16": "R20"}

    diasjp_zip_list = glob.glob(f"{path}/HS_H*_*_*_B*_*_R*_S*.DAT.bz2.zip")
    with ThreadPoolExecutor(max_workers=20) as executor:
        executor.map(extract_diasjp_zip, diasjp_zip_list)

    bz2_list = glob.glob(f"{path}/HS_H*_*_*_B*_*_R*_S*.DAT.bz2")
    with ThreadPoolExecutor(max_workers=20) as executor:
        executor.map(extract_bz2, bz2_list)

    err = move_data_files_out_folder(path, sat_names, scan_areas, "*_*")
    errors.extend(err)

    if only_need_existed:
        actual_file = [file for scan_area in scan_areas
                            for file in glob.glob(f"{path}/HS_H*_*_*_B*_{scan_area}_R*_S*.DAT")]

        return missing_file, process_dict, actual_file, errors

    for scan_area in scan_areas:
        segments = 10 if scan_area == "FLDK" else 1
        sense_dt_list: list[str] = []

        wait = glob.glob(f"{path}/HS_H*_*_*_B*_{scan_area}_R*_S*.DAT")
        for file in wait:
            filename = os.path.basename(file).split("_")
            sat_name, sense_date, sense_time = filename[1], filename[2], filename[3]
            sense_dt_list.append(f"{sat_name}_{sense_date}_{sense_time}")
            file_move_dict[file] = f"{path}/{sat_name}_{sense_date}_{sense_time}_{scan_area}"
            actual_file.append(file)

        sense_dt_list = list({}.fromkeys(sense_dt_list))

        for dt in sense_dt_list:
            sat_name, sense_date, sense_time = dt.split("_")

            for band in band_list:
                for num in range(1, segments + 1):
                    seg = f"S{num:02d}10"
                    band_file = (f"HS_{sat_name}_{sense_date}_{sense_time}_{band}_{scan_area}_"
                                 f"{res_suffix[band]}_{seg}.DAT")
                    if not os.path.exists(f"{path}/{band_file}"):
                        missing_file.append(band_file)

    if not missing_file:
        process_dict, err = move_data_files_into_folder(0, 3, file_move_dict)
        errors.extend(err)

    return missing_file, process_dict, actual_file, errors


def arrange_gk2a_files(
        path: str,
        band_list: list[str],
        only_need_existed=False,
) -> tuple[list[str], dict[str, dict[str, str]], list[str], list[str]]:
    missing_file: list[str] = []
    actual_file: list[str] = []

    file_move_dict: dict[str, str] = {}
    process_dict: dict[str, dict[str, str]] = {}

    errors: list[str] = []

    scan_areas = {"FLDK": "fd", "ELA": "ela"}
    res_suffix = {"vi004": "010", "vi005": "010", "vi006": "005", "vi008": "010", "nr013": "020", "nr016": "020",
                  "sw038": "020", "wv063": "020", "wv069": "020", "wv073": "020", "ir087": "020", "ir096": "020",
                  "ir105": "020", "ir112": "020", "ir123": "020", "ir133": "020"}

    gk2a_hdf = glob.glob(f"{path}/gk2a_ami_le1b_*_*ge_*.hdf")
    for hdf in gk2a_hdf:
        err = rename_file_ext(path, hdf, "hdf", "nc")
        errors.extend(err)

    err = move_data_files_out_folder(path, ["GK2A"], list(scan_areas), "*_*")
    errors.extend(err)

    if only_need_existed:
        actual_file = [file for scan_area in scan_areas.values()
                            for file in glob.glob(f"{path}/gk2a_ami_le1b_*_{scan_area}*ge_*.nc")]

        return missing_file, process_dict, actual_file, errors

    for scan_area_in_folder, scan_area_in_file in scan_areas.items():
        sense_dt_list: list[str] = []

        wait = glob.glob(f"{path}/gk2a_ami_le1b_*_{scan_area_in_file}*ge_*.nc")
        for file in wait:
            filename = os.path.basename(file).split("_")
            sense_dt = filename[5].replace(".nc", "")
            sense_dt_list.append(sense_dt)
            file_move_dict[file] = f"{path}/GK2A_{filename[5][0:8]}_{filename[5][8:12]}_{scan_area_in_folder}"
            actual_file.append(file)

        sense_dt_list = list({}.fromkeys(sense_dt_list))

        for dt in sense_dt_list:
            for band in band_list:
                band = band.lower()
                band_file = f"gk2a_ami_le1b_{band}_{scan_area_in_file}{res_suffix[band]}ge_{dt}.nc"
                if not os.path.exists(f"{path}/{band_file}"):
                    missing_file.append(band_file)

    if not missing_file:
        process_dict, err = move_data_files_into_folder(0, 3, file_move_dict)
        errors.extend(err)

    return missing_file, process_dict, actual_file, errors


def remove_missing_duplicates(missing_list: list[str], time_delta: timedelta, actual_file: list[str]):
    def check_missing(missing: str):
        pop_item = None
        missing_filename = os.path.splitext(os.path.basename(missing))[0].split("_")
        missing_band = missing_filename[1].split("-")[3][2:5]
        missing_scan_mode = missing_filename[1].split("-")[3][1]
        missing_start_dt = datetime.strptime(missing_filename[3].replace("s", ""), "%Y%j%H%M%S%f")
        missing_end_dt = datetime.strptime(missing_filename[4].replace("e", ""), "%Y%j%H%M%S%f")

        for actual in actual_file:
            actual_filename = os.path.splitext(os.path.basename(actual))[0].split("_")
            actual_band = actual_filename[1].split("-")[3][2:5]
            actual_scan_mode = actual_filename[1].split("-")[3][1]
            actual_start_dt = datetime.strptime(actual_filename[3].replace("s", ""), "%Y%j%H%M%S%f")
            actual_end_dt = datetime.strptime(actual_filename[4].replace("e", ""), "%Y%j%H%M%S%f")

            time_pass = False
            if abs(actual_start_dt - missing_start_dt) <= time_delta:
                if abs(actual_end_dt - missing_end_dt) <= time_delta or actual_scan_mode == missing_scan_mode:
                    time_pass = True

            if missing_band == actual_band and time_pass:
                pop_item = missing
                break

        return pop_item

    with ThreadPoolExecutor(max_workers=100) as executor:
        results = executor.map(check_missing, missing_list)

    pop_list = [missing_list.index(result) for result in results if result is not None]
    pop_list = list({}.fromkeys(pop_list))
    pop_list.reverse()

    for pop in pop_list:
        missing_list.pop(pop)


def find_similar_folders(folders: list[str], file_move_dict: dict[str, str]) -> dict[str, str]:
    def find_similar_time_node(folder: str) -> dict[str, str]:
        time_deltas = {"FLDK": timedelta(seconds=30),
                       "CONUS": timedelta(seconds=10),
                       "MESO1": timedelta(seconds=3),
                       "MESO2": timedelta(seconds=3)}

        single_file_move_dict: dict[str, str] = {}

        _, mode, start, end, area = os.path.basename(folder).split("_")
        start = start.replace("s", "")
        end = end.replace("e", "")
        time_delta = time_deltas[area]

        for alt_file, alt_folder in file_move_dict.items():
            _, alt_mode, alt_start, alt_end, alt_area = os.path.basename(alt_folder).split("_")
            alt_start = alt_start.replace("s", "")
            alt_end = alt_end.replace("e", "")

            if area == alt_area and abs(datetime.strptime(start, "%Y%j%H%M%S%f") -
                                        datetime.strptime(alt_start, "%Y%j%H%M%S%f")) <= time_delta:
                if abs(datetime.strptime(end, "%Y%j%H%M%S%f") - datetime.strptime(alt_end,
                                                                                  "%Y%j%H%M%S%f")) or mode == alt_mode:
                    single_file_move_dict[alt_file] = folder

        return single_file_move_dict

    new_file_move_dict: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=100) as executor:
        results = executor.map(find_similar_time_node, folders)

    for result in results:
        new_file_move_dict.update(result)

    return new_file_move_dict


def arrange_goes_files(
        sat: str,
        path: str,
        band_list: list[str],
        only_need_existed=False
) -> tuple[list[str], dict[str, dict[str, str]], list[str], list[str]]:
    missing_file: list[str] = []
    actual_file: list[str] = []

    file_move_dict: dict[str, str] = {}
    process_dict: dict[str, dict[str, str]] = {}

    errors: list[str] = []

    sat_names = ["G16", "G19"] if sat == "GOES-EAST" else ["G17", "G18"]
    scan_areas = {"FLDK": "F", "CONUS": "C", "MESO1": "M1", "MESO2": "M2"}
    time_deltas = {"FLDK":timedelta(seconds=30),
                   "CONUS": timedelta(seconds=10),
                   "MESO1": timedelta(seconds=3),
                   "MESO2": timedelta(seconds=3)}

    err = move_data_files_out_folder(path, sat_names, list(scan_areas), "m*_s*_e*")
    errors.extend(err)

    if only_need_existed:
        actual_file = [file for sat_name in sat_names
                            for scan_area in scan_areas.values()
                            for file in glob.glob(f"{path}/OR_ABI-L1b-Rad{scan_area}*-M*C*_{sat_name}_s*_e*_c*.nc")]

        return missing_file, process_dict, actual_file, errors

    for sat_name in sat_names:
        for scan_area_folder, scan_area_file in scan_areas.items():
            time_delta = time_deltas[scan_area_folder]
            sense_dt_list: list[str] = []
            missing_list: list[str] = []

            wait = glob.glob(f"{path}/OR_ABI-L1b-Rad{scan_area_file}*-M*C*_{sat_name}_s*_e*_c*.nc")
            for file in wait:
                filename = os.path.basename(file).split("_")
                scan_mode = filename[1].split("-")[3][1]
                start_dt = filename[3].replace("s", "")
                end_dt = filename[4].replace("e", "")
                create_dt = filename[5].replace(".nc", "").replace("c", "")
                sense_dt_list.append(f"{scan_mode}_{start_dt}_{end_dt}_{create_dt}")
                scan_area_folder = scan_area_folder
                file_move_dict[file] = f"{path}/{sat_name}_m{scan_mode}_s{start_dt}_e{end_dt}_{scan_area_folder}"
                actual_file.append(file)

            sense_dt_list = list({}.fromkeys(sense_dt_list))

            for dt in sense_dt_list:
                mode, start, end, create = dt.split("_")
                for band in band_list:
                    band_file = f"OR_ABI-L1b-Rad{scan_area_file}-M{mode}{band}_{sat_name}_s{start}_e{end}_c{create}.nc"
                    if not os.path.exists(f"{path}/{band_file}"):
                        missing_list.append(band_file)

            remove_missing_duplicates(missing_list, time_delta, actual_file)
            missing_file.extend(missing_list)

    if not missing_file:
        new_file_move_dict = find_similar_folders(list(file_move_dict.values()), file_move_dict)

        process_dict, err = move_data_files_into_folder(0, 4, new_file_move_dict)
        errors.extend(err)

    for idx, missing in enumerate(missing_file):
        missing_filename = os.path.splitext(os.path.split(missing)[1])[0]
        missing_filename = missing_filename.split("_")

        missing_area = missing_filename[1].split("-")[2].replace("Rad", "")
        missing_band = missing_filename[1].split("-")[3][2:5]
        missing_start_dt = datetime.strptime(missing_filename[3].replace("s", ""), "%Y%j%H%M%S%f")

        missing_file.pop(idx)
        missing_file.insert(idx, f"Area: {missing_area} Band: {missing_band} Start date: {missing_start_dt}")

    missing_file = list({}.fromkeys(missing_file))
    missing_file.sort()

    return missing_file, process_dict, actual_file, errors


def extract_fci_zip(file_move_dict: dict[str, str]) -> dict[str, dict[str, str]]:
    process_dict: dict[str, dict[str, str]] = {}

    for zip_file, folder in file_move_dict.items():
        if not os.path.exists(folder) or (os.path.exists(folder) and not os.path.isdir(folder)):
            os.mkdir(folder)

        folder_sat_name = os.path.basename(folder).split("_")[0]
        folder_scan_area = os.path.basename(folder).split("_")[4]
        process_dict[folder] = {"sat_name": folder_sat_name, "scan_area": folder_scan_area}

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            for data in zip_ref.namelist():
                if str(data).endswith(".nc"):
                    zip_ref.extract(data, folder)

    return process_dict


def arrange_fci_files(
        sat: str,
        path: str,
        band_list: list[str],
        only_need_existed=False
) -> tuple[list[str], dict[str, dict[str, str]], list[str], list[str]]:
    missing_file: list[str] = []
    actual_file: list[str] = []

    file_move_dict: dict[str, str] = {}
    process_dict: dict[str, dict[str, str]] = {}

    errors: list[str] = []

    sat_names = {"MTG-I1": "MTI1", "MTG-I3": "MTI3"} if sat == "MTG-FD" else {"MTG-I2": "MTI2", "MTG-I4": "MTI4"}
    scan_areas = {"FLDK": "FD", "EU": "Q4"}
    commission_operations = ["C", "O"]

    err = move_data_files_out_folder(path, list(sat_names), list(scan_areas), "*_cycle*")
    errors.extend(err)

    if only_need_existed:
        actual_file = [file for sat_name in sat_names.values()
                            for scan_area in scan_areas.values()
                            for status in commission_operations
                            for file in glob.glob(
                                            f"{path}/W_XX-EUMETSAT-Darmstadt,IMG+SAT,{sat_name}+FCI-1C-RRAD-FDHSI-"
                                            f"{scan_area}--x-x---x_C_EUMT_*_IDPFI_OPE_*_*_N__{status}_*_0000.zip")]

        return missing_file, process_dict, actual_file, errors

    for sat_name_in_folder, sat_name_in_file in sat_names.items():
        for scan_area_in_folder, scan_area_in_file in scan_areas.items():
            for status in commission_operations:
                sense_dt_list: list[str] = []
                wait = glob.glob(f"{path}/W_XX-EUMETSAT-Darmstadt,IMG+SAT,{sat_name_in_file}+FCI-1C-RRAD-FDHSI-"
                                 f"{scan_area_in_file}--x-x---x_C_EUMT_*_IDPFI_OPE_*_*_N__{status}_*_0000.zip")

                for file in wait:
                    filename = os.path.basename(file).split("_")
                    create_dt = filename[4]
                    start_dt = filename[7]
                    end_dt = filename[8]
                    cycle = filename[12]
                    sense_dt_list.append(f"{create_dt}_{start_dt}_{end_dt}_{cycle}")
                    file_move_dict[
                        file] = f"{path}/{sat_name_in_folder}_{status}_{start_dt[0:8]}_cycle{cycle}_{scan_area_in_folder}"
                    actual_file.append(file)

                for dt in sense_dt_list:
                    create_dt, start_dt, end_dt, cycle = dt.split("_")
                    for band in band_list:
                        band_file = (f"W_XX-EUMETSAT-Darmstadt,IMG+SAT,{sat_name_in_file}+FCI-1C-RRAD-{band}-"
                                     f"{scan_area_in_file}--x-x---x_C_EUMT_{create_dt}_IDPFI_OPE_{start_dt}_{end_dt}_"
                                     f"N__{status}_{cycle}_0000.zip")
                        if not os.path.exists(f"{path}/{band_file}"):
                            missing_file.append(band_file)

    if not missing_file:
        process_dict = extract_fci_zip(file_move_dict)

    return missing_file, process_dict, actual_file, errors


def arrange_fy4_files(
        sat: str,
        path: str,
        band_list: list[str],
        only_need_existed=False
) -> tuple[list[str], dict[str, dict[str, str]], list[str], list[str]]:
    missing_file: list[str] = []
    actual_file: list[str] = []

    file_move_dict: dict[str, str] = {}
    process_dict: dict[str, dict[str, str]] = {}

    errors: list[str] = []

    sat_name = sat
    scan_areas = {"FLDK": "DISK", "REGC": "REGC"}

    fy4_hdf = glob.glob(f"{path}/FY4*-_AGRI--_N_*_*_L1-_FDI-_MULT_NOM_*_*_*00M_*.HDF")
    for hdf in fy4_hdf:
        err = rename_file_ext(path, hdf, "HDF", "hdf")
        errors.extend(err)

    err = move_data_files_out_folder(path, [sat_name], list(scan_areas), "sdt*_edt*")
    errors.extend(err)

    if only_need_existed:
        actual_file = [file for scan_area in scan_areas.values()
                            for file in glob.glob(f"{path}/{sat_name}-_AGRI--_N_{scan_area}_*_L1-"
                                                  f"_FDI-_MULT_NOM_*_*_*00M_V0001.hdf")]

        return missing_file, process_dict, actual_file, errors

    for scan_area_folder, scan_area_file in scan_areas.items():
        sense_dt_list = []
        wait = glob.glob(f"{path}/{sat_name}-_AGRI--_N_{scan_area_file}_*_L1-_FDI-_MULT_NOM_*_*_*00M_V0001.hdf")
        for file in wait:
            filename = os.path.splitext(os.path.split(file)[1])[0]
            filename = filename.split("_")
            lon_0 = filename[4]
            start_dt = filename[9]
            end_dt = filename[10]
            sense_dt_list.append(f"{sat_name}_{lon_0}_{start_dt}_{end_dt}")
            file_move_dict[file] = f"{path}/{sat_name}_sdt{start_dt}_edt{end_dt}_{scan_area_folder}"
            actual_file.append(file)

        sense_dt_list = list({}.fromkeys(sense_dt_list))

        for dt in sense_dt_list:
            sat_name, lon0, start_dt, end_dt = dt.split("_")

            for band in band_list:
                band_file = (f"{sat_name}-_AGRI--_N_{scan_area_file}_{lon0}_L1-_FDI-_MULT_NOM_"
                             f"{start_dt}_{end_dt}_{band}_V0001.hdf")
                if not os.path.exists(f"{path}/{band_file}"):
                    missing_file.append(band_file)

    if not missing_file:
        process_dict, err = move_data_files_into_folder(0, 3, file_move_dict)
        errors.extend(err)

    return missing_file, process_dict, actual_file, errors


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

        res = super().__call__((data, data, data, alpha), **kwargs) if self.rgba else (
            super().__call__((data, alpha), **kwargs))
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

        is_multi_band = "bands" in data_in.dims
        bands = data_in["bands"] if is_multi_band else [None]

        for b in bands:
            band_data = data_in.sel(bands=b).data if is_multi_band else data_in.data
            if not self.alpha or (self.alpha and len(bands) in [2, 4]):
                band_data = band_data * alpha.data

            new_data_in.append(xr.DataArray(
                data=band_data,
                attrs=alpha_attrs,
                dims=data_in[0].dims,
                coords=data_in[0].coords
            ))

        if self.alpha and len(bands) not in [2, 4]:
            new_data_in.append(xr.DataArray(
                data=alpha.data * 255,
                attrs=alpha_attrs,
                dims=data_in[0].dims,
                coords=data_in[0].coords
            ))

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
                "goes-19": np.array([[0.9481, 0.3706, -0.2194],
                                     [-0.0150, 0.8605, 0.1317],
                                     [-0.0174, -0.1009, 1.0512]]),
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


def save(self, filename, fformat=None, fill_value=None, compute=True,driver=None, **format_kwargs):
    import logging
    logger = logging.getLogger(__name__)

    kwformat = format_kwargs.pop("format", None)
    fformat = fformat or kwformat or os.path.splitext(filename)[1][1:]
    if fformat in ("tif", "tiff", "jp2"):
        try:
            return self.rio_save(filename, fformat=fformat, driver=driver, fill_value=fill_value, compute=compute,
                                 **format_kwargs)
        except ImportError:
            logger.warning("Missing 'rasterio' dependency to save GeoTIFF image. Will try using PIL...")
    return self.pil_save(filename, fformat, fill_value, compute=compute, **format_kwargs)


def gdal_write(xarray_with_gdal_kwargs):
    (data, in_memory, driver, filename, gdal_dtype, format_kwargs, tags, proj, transform, gcps_gdal,
     fill_value, colors, overviews) = xarray_with_gdal_kwargs

    if in_memory:
        filename = f"/vsimem/{os.path.basename(filename)}"

    total_data = data.data.compute()
    with gdal.GetDriverByName(driver).Create(filename, data.sizes["x"], data.sizes["y"], data.sizes["bands"],
                                             gdal_dtype, format_kwargs) as ds:
        ds.SetMetadata(tags)

        if proj:
            ds.SetProjection(proj.to_wkt("WKT2_2018"))
        if transform:
            ds.SetGeoTransform(transform.to_gdal())
        if gcps_gdal:
            ds.SetGCPs(gcps_gdal, "")

        for i in range(1, data.sizes["bands"] + 1):
            ds.GetRasterBand(i).WriteArray(total_data[i - 1])

            if fill_value:
                ds.GetRasterBand(i).SetNoDataValue(fill_value)

            ds.GetRasterBand(i).SetColorInterpretation(colors[i - 1])

        if overviews:
            ds.BuildOverviews("LANCZOS", overviews)

        ds.FlushCache()

    del data, total_data

def xarray_and_gdal_kwargs_to_geotiff(xarray_with_gdal_kwargs):
    gdal.UseExceptions()
    gdal.SetCacheMax(4096)
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")

    (data, in_memory, driver, filename, gdal_dtype, format_kwargs, tags, proj, transform, gcps_gdal,
     fill_value, colors, overviews) = xarray_with_gdal_kwargs

    format_kwargs.update({"BIGTIFF": "IF_SAFER"})
    if colors[-1] == gdal.GCI_AlphaBand:
        format_kwargs.update({"ALPHA": "YES"})
    bigtiff_format_kwargs = format_kwargs.copy()
    bigtiff_format_kwargs.update({"BIGTIFF": "YES"})

    try:
        gdal_write(xarray_with_gdal_kwargs)
    except RuntimeError:
        if os.path.exists(filename):
            gdal.Unlink(filename)
        try:
            xarray_with_bigtiff_gdal_kwargs = (data, in_memory, driver, filename, gdal_dtype, bigtiff_format_kwargs,
                                               tags, proj, transform, gcps_gdal, fill_value, colors, overviews)
            gdal_write(xarray_with_bigtiff_gdal_kwargs)
        except Exception as e:
            print(e)
    except Exception as e:
        print(e)


def rio_save(self, filename: str, fformat: str | None=None, fill_value: float | None=None, dtype=np.uint8,
             compute=True, tags: dict | None=None, overviews: list[int] | None=None, driver: str | None=None,
             in_memory=False, **format_kwargs) -> tuple:
    gdal.UseExceptions()
    gdal.SetCacheMax(4096)
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")

    for key in list(format_kwargs):
        if key not in GDAL_OPTIONS or key == "in_memory":
            format_kwargs.pop(key, None)

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
    if dtype not in data_types:
        raise ValueError("Invalid data types")
    if fill_value and np.isnan(fill_value) and dtype not in [np.float32, np.float64]:
        raise ValueError("np.nan only for float32/64 image")

    driver = driver or drivers.get(fformat, fformat)
    if driver == "COG" and overviews == []:
        overviews = None

    if tags is None:
        tags = {}

    data, mode = self.finalize(fill_value, dtype=dtype, keep_palette=False)
    data = data.transpose("bands", "y", "x")
    if mode in ["L", "LA"]:
        color_interp = {"L": gdal.GCI_GrayIndex,
                        "A": gdal.GCI_AlphaBand}
    elif mode in ["YCBCR", "YCBCRA"]:
        color_interp = {"Y": gdal.GCI_YCbCr_YBand,
                        "CB": gdal.GCI_YCbCr_CbBand,
                        "CR": gdal.GCI_YCbCr_CrBand}
    if driver == "JPEG" and "A" in mode:
        raise ValueError("JPEG does not support alpha")

    crs = None
    gcps = None
    transform = None
    if driver in ["COG", "GTiff", "JP2OpenJPEG"]:
        format_kwargs.setdefault("compress", "DEFLATE")
        if format_kwargs.get("compress", "") == "JPEG" and dtype != np.uint8:
            raise ValueError("JPEG compress only for Byte images")

        photometric_map = {
            "RGB": "RGB",
            "RGBA": "RGB",
            "CMYK": "CMYK",
            "CMYKA": "CMYK",
            "YCBCR": "YCBCR",
            "YCBCRA": "YCBCR",
        }
        if mode.upper() in photometric_map:
            format_kwargs.setdefault("photometric", photometric_map[mode.upper()])

        from trollimage._xrimage_rasterio import get_data_arr_crs_transform_gcps
        crs, transform, gcps = get_data_arr_crs_transform_gcps(data)

        stime = data.attrs.get("start_time")
        if stime:
            stime_str = stime.strftime("%Y:%m:%d %H:%M:%S")
            tags.setdefault("TIFFTAG_DATETIME", stime_str)

    gcps_gdal = [gdal.GCP(gcp.x, gcp.y, gcp.z, gcp.col, gcp.row) for gcp in gcps] if gcps else None

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

    colors = [color_interp[mode[i - 1]] for i in range(1, data.sizes["bands"] + 1)] \
        if mode not in ["YCBCR", "YCBCRA"] else list(color_interp.values())
    xarray_with_gdal_kwargs = (data, in_memory, driver, filename, data_types[dtype], new_format_kwargs,
                               tags, crs, transform, gcps_gdal,
                               fill_value, colors, overviews)

    if compute:
        xarray_and_gdal_kwargs_to_geotiff(xarray_with_gdal_kwargs)
        return ()

    return xarray_with_gdal_kwargs


def compute_writer_results(results):
    if not results:
        return

    if isinstance(results[0], list):
        results = results[0]

    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(xarray_and_gdal_kwargs_to_geotiff, results)

class CASLoginParser(html.parser.HTMLParser):
    def __init__(self):
        html.parser.HTMLParser.__init__(self)
        self.action = None
        self.data = {}

    def handle_starttag(self, tagname, attribute):
        if tagname.lower() == "form":
            attribute = dict(attribute)
            if "action" in attribute:
                self.action = attribute["action"]
        elif tagname.lower() == "input":
            attribute = dict(attribute)
            if "name" in attribute and "value" in attribute:
                self.data[attribute["name"]] = attribute["value"]

class DIASAccess:
    def __init__(self, username, password):
        self.__cas_url = "https://auth.diasjp.net/cas/login?"
        self.__username = username
        self.__password = password
        cj = http.cookiejar.CookieJar()
        self.__opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    def open(self, url, data=None):
        response = self.__opener.open(url, data)
        response_url = response.geturl()

        if response_url != url and response_url.startswith(self.__cas_url):
            response = self.__login_cas(response)
            if data is not None:
                response.close()
                response = self.__opener.open(url, data)

        return response

    def __login_cas(self, response):
        parser = CASLoginParser()
        parser.feed(str(response.read()))
        parser.close()

        if parser.action is None:
            raise LoginError("Not login page")

        action_url = urllib.parse.urljoin(response.geturl(), parser.action)
        data = parser.data
        data["username"] = self.__username
        data["password"] = self.__password

        response.close()
        response = self.__opener.open(action_url,
                                      urllib.parse.urlencode(data).encode("utf-8"))

        if response.geturl() == action_url:
            raise LoginError("Authorization fail")

        return response

class LoginError(Exception):
    def __init__(self, e):
        Exception.__init__(self, e)
