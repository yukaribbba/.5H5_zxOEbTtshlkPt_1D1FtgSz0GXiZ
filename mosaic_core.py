import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Union

import numpy as np
import SimpleITK as sitk
from osgeo import gdal
from osgeo_utils import gdal_merge

GDAL_SITK_DTYPE = {1: sitk.sitkUInt8, 2: sitk.sitkUInt16, 3: sitk.sitkInt16, 4: sitk.sitkUInt32,
                   5: sitk.sitkInt32, 6: sitk.sitkFloat32, 7: sitk.sitkFloat64, 8: sitk.sitkComplexFloat32,
                   9: sitk.sitkComplexFloat64, 10: sitk.sitkComplexFloat32, 11: sitk.sitkComplexFloat64,
                   12: sitk.sitkUInt64, 13: sitk.sitkInt64, 14: sitk.sitkInt8}
GDAL_DTYPE_LIMIT = {1: np.iinfo(np.uint8).max, 2: np.iinfo(np.uint16).max, 3: np.iinfo(np.int16).max,
                    4: np.iinfo(np.uint32).max, 5: np.iinfo(np.int32).max, 6: np.finfo(np.float32).max,
                    7: np.finfo(np.float64).max, 8: np.finfo(np.float32).max, 9: np.finfo(np.float32).max,
                    10: np.finfo(np.float32).max, 11: np.finfo(np.float64).max, 12: np.iinfo(np.uint64).max,
                    13: np.iinfo(np.int64).max, 14: np.iinfo(np.int8).max}
ACCEPTED_DTYPE = [1, 2, 3, 4, 5, 6, 7, 14]
ACCEPTED_DTYPE_STR = ", ".join([gdal.GetDataTypeName(dtype) for dtype in ACCEPTED_DTYPE])
DRIVER = gdal.GetDriverByName("GTiff")
CREATE_OPTIONS = {"TILED": "YES", "BLOCKXSIZE": "512", "BLOCKYSIZE": "512"}
BANDS_MODE = {1: "L",
              2: "LA",
              3: "RGB",
              4: "RGBA"}
COLOR_INTERP = {"R": gdal.GCI_RedBand,
                "G": gdal.GCI_GreenBand,
                "B": gdal.GCI_BlueBand,
                "A": gdal.GCI_AlphaBand,
                "L": gdal.GCI_GrayIndex}


@dataclass
class ImageData:
    bands: dict[int, np.ndarray]
    distance: np.ndarray
    binary: np.ndarray


def get_real_geotiffs(input_images: list | None):
    bands_count: list[int] = []
    ready_image_files: list[str] = []

    if input_images is None:
        return bands_count, ready_image_files

    for image in input_images:
        if image and os.path.exists(image):
            with gdal.Open(image) as ds:
                if ds.GetGeoTransform() and ds.GetProjection():
                    bands = ds.RasterCount
                    bands_count.append(bands)
                    ready_image_files.append(image)

    return bands_count, ready_image_files


class Mosaic:
    def __init__(self, input_images: list, nodata: Union[int, float], output: str,
                 input_boundaries: list | None = None):
        super().__init__()
        gdal.UseExceptions()

        bands_count, ready_image_files = get_real_geotiffs(input_images)
        ready_boundary_files = get_real_geotiffs(input_boundaries)[1]

        self.base_bands = max(bands_count)
        if self.base_bands > 4:
            raise ValueError(f"At least one of the files has {self.base_bands} bands. Only 4 or less can be supported.")

        self.mode = BANDS_MODE[self.base_bands]
        self.input_images = ready_image_files
        self.input_boundaries = input_boundaries if (
                input_boundaries and len(ready_boundary_files) == len(ready_image_files)) else None
        self.nodata = nodata
        self.output = output
        if os.path.exists(self.output):
            gdal.Unlink(self.output)

        path = os.getcwd() if len(os.path.split(self.output)[0]) == 0 else os.path.split(self.output)[0]
        filename = os.path.basename(self.output).split(".")[0]
        filename = filename.split(".")[0]
        self.output_boundary = f"{path}/zzz_boundary_{filename}.tif"
        if os.path.exists(self.output_boundary):
            gdal.Unlink(self.output_boundary)

        with gdal.Open(ready_image_files[0]) as first_ds:
            self.base_proj = first_ds.GetProjection()
            self.base_res_x = first_ds.GetGeoTransform()[1]
            self.base_res_y = first_ds.GetGeoTransform()[5]
            self.base_dtype = first_ds.GetRasterBand(1).DataType

        if self.base_dtype not in ACCEPTED_DTYPE:
            raise ValueError(f"{gdal.GetDataTypeName(self.base_dtype)} is not supported. "
                             f"Supported data type: {ACCEPTED_DTYPE_STR}")

        gdal.SetCacheMax(4096)
        gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")

    def adjust_dtype_add_bands(self, image: str):
        path = os.path.split(image)[0]
        if len(path) == 0:
            path = os.getcwd()
        filename = os.path.basename(image)
        filename = filename.split(".")[0]
        output_image = f"{path}/{filename}_add_bands.tif"

        if os.path.exists(output_image):
            gdal.Unlink(output_image)

        with gdal.Open(image) as ds:
            bands = ds.RasterCount
            width = ds.RasterXSize
            height = ds.RasterYSize
            dtype = ds.GetRasterBand(1).DataType
            proj = ds.GetProjection()
            tran = ds.GetGeoTransform()

        if bands == self.base_bands and dtype == self.base_dtype:
            return image

        with DRIVER.Create(output_image, width, height, self.base_bands, self.base_dtype,
                           CREATE_OPTIONS) as t_ds:
            t_ds.SetProjection(proj)
            t_ds.SetGeoTransform(tran)

            with gdal.Open(image) as ds:
                for i in range(1, self.base_bands + 1):
                    data = (ds.GetRasterBand(i).ReadAsArray(0, 0, width, height)
                            if i <= bands
                            else (np.full((height, width), GDAL_DTYPE_LIMIT[self.base_dtype]) if i in [2, 4]
                                  else ds.GetRasterBand(bands).ReadAsArray(0, 0, width, height)))

                    t_band = t_ds.GetRasterBand(i)
                    t_band.WriteArray(data)

            t_ds.FlushCache()

        return output_image

    def adjust_proj_res(self, image, boundary_resample=False):
        path = os.path.split(image)[0]
        if len(path) == 0:
            path = os.getcwd()
        filename = os.path.basename(image).split(".")[0]
        output_image = f"{path}/{filename}_warped.tif"

        if os.path.exists(output_image):
            gdal.Unlink(output_image)

        with gdal.Open(image) as ds:
            proj = ds.GetProjection()
            res_x = ds.GetGeoTransform()[1]
            res_y = ds.GetGeoTransform()[5]

        all_the_same = (proj == self.base_proj and res_x == self.base_res_x and res_y == self.base_res_y)

        if all_the_same:
            return image

        print(f"{image}需要重投影")
        print(f"基准x分辨率为{self.base_res_x}, 文件x分辨率为{res_x}")
        print(f"基准y分辨率为{self.base_res_y}, 文件y分辨率为{res_y}")
        resampler = gdal.GRA_NearestNeighbour if boundary_resample else gdal.GRA_Lanczos

        warp_options = gdal.WarpOptions(dstSRS=self.base_proj,
                                        xRes=self.base_res_x,
                                        yRes=self.base_res_y,
                                        multithread=True,
                                        errorThreshold=0, warpMemoryLimit=21474836480,
                                        resampleAlg=resampler,
                                        warpOptions=["NUM_THREADS=ALL_CPUS"],
                                        callback=gdal.TermProgress_nocb)
        gdal.Warp(output_image, image, options=warp_options)

        return output_image

    def normalize_images(self):
        ready_images: list[str] = []
        for idx, image in enumerate(self.input_images):
            add_bands_image = self.adjust_dtype_add_bands(image)
            ready_image = self.adjust_proj_res(add_bands_image)

            try:
                ready_boundary = self.adjust_proj_res(self.input_boundaries[idx])
                self.input_boundaries[idx] = ready_boundary
            except (IndexError, TypeError):
                boundary = self.search_boundary(image)
                if boundary:
                    self.adjust_proj_res(boundary, boundary_resample=True)

            ready_images.append(ready_image)

        return ready_images

    @staticmethod
    def search_boundary(input_image: str):
        path = os.path.split(input_image)[0]
        if len(path) == 0:
            path = os.getcwd()
        filename = os.path.basename(input_image).split(".")[0]

        bin_file = None
        possible_binary_list = [f"{path}/zzz_boundary_{filename}.tif",
                                f"{path}/_boundary_mask_{filename}.tif",
                                f"{path}/_mask_{filename}.tif"]
        for pos in possible_binary_list:
            if os.path.exists(pos):
                with gdal.Open(pos) as ds:
                    if ds.GetGeoTransform() and ds.GetProjection():
                        bin_file = pos
                        break

        return bin_file

    @staticmethod
    def run_command(cmd: list):
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
            for line in proc.stdout:
                print(line, end="", flush=True)
            for line in proc.stderr:
                print(line, end="", file=sys.stderr, flush=True)
            proc.wait()
            if proc.returncode != 0:
                stderr = proc.stderr.read() if proc.stderr else None
                raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=stderr)

    def try_bigtiff(self, argv1: list, argv1_bigtiff: list, output_image: str):
        error = None
        try:
            self.run_command(argv1)
        except subprocess.CalledProcessError:
            if os.path.exists(output_image):
                gdal.Unlink(output_image)

            print("文件可能超过4GB, 因此需启用BigTIFF后重试")

            try:
                self.run_command(argv1_bigtiff)
            except subprocess.CalledProcessError as e:
                error = e.stderr
                if os.path.exists(output_image):
                    gdal.Unlink(output_image)

        return error

    def simple_mosaic(self, format_kwargs: dict | None = None):
        if format_kwargs is None:
            format_kwargs = {}

        predictor = "2" if self.base_dtype in [1, 2, 3, 4, 5, 14] else "3"

        format_kwargs.setdefault("BIGTIFF", "IF_SAFER")
        format_kwargs.setdefault("GDAL_CACHEMAX", "4096")
        format_kwargs.setdefault("PREDICTOR", predictor)

        options = [item
            for pair in format_kwargs.items()
            for item in (["--config", f"{pair[0]}={pair[1]}"] if pair[0] in ["GDAL_NUM_THREADS", "GDAL_CACHEMAX"]
                         else ["-co", f"{pair[0]}={pair[1]}"])]

        ready_images = self.normalize_images()

        argv1 = ["python", "-m", "osgeo_utils.gdal_merge", "-n", "nan" if np.isnan(self.nodata) else str(self.nodata),
                 "-o", self.output]
        argv1.extend(options)
        argv1.extend(ready_images)

        bigtiff_format_kwargs = format_kwargs.copy()
        bigtiff_format_kwargs.update({"BIGTIFF": "YES"})

        bigtiff_options = [
            item
            for pair in bigtiff_format_kwargs.items()
            for item in (["--config", f"{pair[0]}={pair[1]}"] if pair[0] in ["GDAL_NUM_THREADS", "GDAL_CACHEMAX"]
                         else ["-co", f"{pair[0]}={pair[1]}"])]

        argv1_bigtiff = ["python", "-m", "osgeo_utils.gdal_merge", "-n", str(self.nodata), "-o", self.output]
        argv1_bigtiff.extend(bigtiff_options)
        argv1_bigtiff.extend(ready_images)

        error = self.try_bigtiff(argv1, argv1_bigtiff, self.output)

        argv2 = ["python", "-m", "osgeo_utils.gdal_merge", "-o", self.output_boundary, "-n", str(0),
                 "-co", "COMPRESS=DEFLATE", "-co", "ZLEVEL=6", "-co", "PREDICTOR=2"]
        argv2_bigtiff = ["python", "-m", "osgeo_utils.gdal_merge", "-o", self.output_boundary, "-n", str(0),
                         "-co", "BIGTIFF=YES", "-co", "COMPRESS=DEFLATE", "-co", "ZLEVEL=6", "-co", "PREDICTOR=2"]

        boundary_list = self.input_boundaries if self.input_boundaries else [
            boundary for image in ready_images if (boundary := self.search_boundary(image))]
        boundary_need = self.input_boundaries or len(boundary_list) == len(ready_images)

        argv2.extend(boundary_list)
        argv2_bigtiff.extend(boundary_list)

        if error is None and boundary_need:
            error = self.try_bigtiff(argv2, argv2_bigtiff, self.output_boundary)

        return self.output_boundary, error

    def get_binary_image(self, image_idx: int, image: str):
        if self.input_boundaries:
            bin_file = self.input_boundaries[image_idx]
            with gdal.Open(bin_file) as bin_ds:
                bin_data = bin_ds.GetRasterBand(1).ReadAsArray(0, 0, bin_ds.RasterXSize, bin_ds.RasterYSize)
            bin_data = 255 - bin_data
            binary_image = sitk.GetImageFromArray(bin_data)

        else:
            bin_file = self.search_boundary(image)

            if bin_file:
                with gdal.Open(bin_file) as bin_ds:
                    bin_data = bin_ds.GetRasterBand(1).ReadAsArray(0, 0, bin_ds.RasterXSize, bin_ds.RasterYSize)
                bin_data = 255 - bin_data
                binary_image = sitk.GetImageFromArray(bin_data)
            else:
                sitk_image = sitk.ReadImage(image, GDAL_SITK_DTYPE[self.base_dtype])

                binary_filter = sitk.BinaryThresholdImageFilter()
                binary_filter.SetLowerThreshold(self.nodata)
                binary_filter.SetUpperThreshold(self.nodata)
                binary_filter.SetInsideValue(255)
                binary_filter.SetOutsideValue(0)

                binary_image = binary_filter.Execute(sitk_image)

        return binary_image

    def write_final_image(self, image_geometry: tuple, format_kwargs: dict, tags: dict, mosaic_arrays: dict):
        xsize, ysize, geotransform = image_geometry
        sum_dis_array = sum(mosaic_arrays[img].distance for img in mosaic_arrays)

        with DRIVER.Create(self.output, xsize, ysize, self.base_bands, self.base_dtype, format_kwargs) as final_ds:
            final_ds.SetProjection(self.base_proj)
            final_ds.SetGeoTransform(geotransform)
            final_ds.SetMetadata(tags)

            for band in range(1, self.base_bands + 1):
                sum_band_arrays = [mosaic_arrays[img].bands[band] * mosaic_arrays[img].distance
                                   for img in mosaic_arrays]

                image_array = sum(sum_band_arrays) / sum_dis_array
                final_ds.GetRasterBand(band).WriteArray(image_array)
                final_ds.GetRasterBand(band).SetNoDataValue(self.nodata)
                final_ds.GetRasterBand(band).SetColorInterpretation(COLOR_INTERP[self.mode[band - 1]])

            final_ds.FlushCache()

    def write_final_boundary(self, image_geometry: tuple, format_kwargs: dict, mosaic_arrays: dict):
        xsize, ysize, geotransform = image_geometry
        sum_bin_array = sum(mosaic_arrays[img].binary for img in mosaic_arrays)

        with DRIVER.Create(self.output_boundary, xsize, ysize, 1, gdal.GDT_Byte, format_kwargs) as final_boundary_ds:
            final_boundary_ds.SetProjection(self.base_proj)
            final_boundary_ds.SetGeoTransform(geotransform)
            final_boundary_ds.GetRasterBand(1).WriteArray(sum_bin_array)
            final_boundary_ds.FlushCache()

    def large_feather_mosaic(self, need_final_bound=True, tags: dict | None = None, format_kwargs: dict | None = None):
        if tags is None:
            tags = {}

        if format_kwargs is None:
            format_kwargs = {}

        predictor = "2" if self.base_dtype in [1, 2, 3, 4, 5, 14] else "3"

        format_kwargs.setdefault("BIGTIFF", "IF_SAFER")
        format_kwargs.setdefault("PREDICTOR", predictor)

        if self.base_dtype in [1, 2] and self.base_bands in [3, 4]:
            format_kwargs.update({"PHOTOMETRIC": "RGB"})

        bigtiff_format_kwargs = format_kwargs.copy()
        bigtiff_format_kwargs.update({"BIGTIFF": "YES"})

        bin_format_kwargs = {"COMPRESS": "DEFLATE", "ZLEVEL": 6, "PREDICTOR": "2", "BIGTIFF": "IF_SAFER"}
        bin_bigtiff_format_kwargs = bin_format_kwargs.copy()
        bin_bigtiff_format_kwargs.update({"BIGTIFF": "YES"})

        ready_images = self.normalize_images()
        file_infos = gdal_merge.names_to_fileinfos(ready_images)

        ulx = file_infos[0].ulx
        uly = file_infos[0].uly
        lrx = file_infos[0].lrx
        lry = file_infos[0].lry

        for fi in file_infos:
            ulx = min(ulx, fi.ulx)
            uly = max(uly, fi.uly)
            lrx = max(lrx, fi.lrx)
            lry = min(lry, fi.lry)

        geotransform = [ulx, self.base_res_x, 0, uly, 0, self.base_res_y]

        xsize = int((lrx - ulx) / geotransform[1] + 0.5)
        ysize = int((lry - uly) / geotransform[5] + 0.5)

        fi_processed = 0
        mosaic_arrays: dict[str, ImageData] = {}

        for idx, image in enumerate(ready_images):
            filename = os.path.basename(image).split(".")[0]

            with gdal.Open(image) as ds:
                proj = ds.GetProjection()
                tran = ds.GetGeoTransform()

            temp_image = f"/vsimem/tmp_{filename}.tif"
            with DRIVER.Create(temp_image, xsize, ysize, self.base_bands, self.base_dtype,
                               CREATE_OPTIONS) as temp_image_ds:
                temp_image_ds.SetProjection(self.base_proj)
                temp_image_ds.SetGeoTransform(geotransform)
                single_image_info = gdal_merge.names_to_fileinfos([image])

                bands_dict: dict[int, np.ndarray] = {}
                for band in range(1, self.base_bands + 1):
                    single_image_info[0].copy_into(temp_image_ds, band, band, self.nodata)
                    data = temp_image_ds.GetRasterBand(band).ReadAsArray(0, 0, xsize, ysize)
                    bands_dict[band] = data
                temp_image_ds.FlushCache()
            gdal.Unlink(temp_image)

            binary_image = self.get_binary_image(idx, image)

            distance_map = sitk.DanielssonDistanceMap(binary_image, inputIsBinary=True, useImageSpacing=True)
            distance_map_data = sitk.GetArrayFromImage(distance_map)
            distance_map_data = np.where(distance_map_data <= 0, 0, distance_map_data)

            distance = f"/vsimem/dis_{filename}.tif"
            with DRIVER.Create(distance, xsize, ysize, 1, gdal.GDT_Float32, CREATE_OPTIONS) as distance_ds:
                distance_ds.GetRasterBand(1).WriteArray(distance_map_data)
                distance_ds.SetProjection(proj)
                distance_ds.SetGeoTransform(tran)
                distance_ds.FlushCache()

            temp_distance = f"/vsimem/tmp_dis_{filename}.tif"
            with DRIVER.Create(temp_distance, xsize, ysize, 1, gdal.GDT_Float32, CREATE_OPTIONS) as temp_distance_ds:
                temp_distance_ds.SetProjection(self.base_proj)
                temp_distance_ds.SetGeoTransform(geotransform)
                single_distance_info = gdal_merge.names_to_fileinfos([distance])

                single_distance_info[0].copy_into(temp_distance_ds, 1, 1, None)
                distance_data = temp_distance_ds.GetRasterBand(1).ReadAsArray(0, 0, xsize, ysize)

                temp_distance_ds.FlushCache()
            gdal.Unlink(temp_distance)
            gdal.Unlink(distance)

            binary = f"/vsimem/bin_{filename}.tif"
            with DRIVER.Create(binary, xsize, ysize, 1, gdal.GDT_Byte) as binary_ds:
                binary_ds.GetRasterBand(1).WriteArray(255 - sitk.GetArrayFromImage(binary_image))
                binary_ds.SetProjection(proj)
                binary_ds.SetGeoTransform(tran)
                binary_ds.FlushCache()

            temp_binary = f"/vsimem/tmp_bin_{filename}.tif"
            with DRIVER.Create(temp_binary, xsize, ysize, 1, gdal.GDT_Byte) as temp_binary_ds:
                temp_binary_ds.SetProjection(self.base_proj)
                temp_binary_ds.SetGeoTransform(geotransform)
                single_binary_info = gdal_merge.names_to_fileinfos([binary])

                single_binary_info[0].copy_into(temp_binary_ds, 1, 1, None)
                binary_data = temp_binary_ds.GetRasterBand(1).ReadAsArray(0, 0, xsize, ysize)

                temp_binary_ds.FlushCache()
            gdal.Unlink(temp_binary)
            gdal.Unlink(binary)

            image_data = ImageData(bands_dict, distance_data, binary_data)
            mosaic_arrays[image] = image_data

            fi_processed = fi_processed + 1
            gdal.TermProgress_nocb(fi_processed / float(len(file_infos)))

        try_bigtiff = False
        image_done = False
        error = None

        try:
            self.write_final_image((xsize, ysize, geotransform), format_kwargs, tags, mosaic_arrays)
            image_done = True
        except RuntimeError:
            try_bigtiff = True
        except Exception as e:
            error = e

        if try_bigtiff:
            if os.path.exists(self.output):
                gdal.Unlink(self.output)

            print("文件可能超过4GB, 因此需启用BigTIFF后重试")

            try:
                self.write_final_image((xsize, ysize, geotransform), bigtiff_format_kwargs, tags, mosaic_arrays)
                image_done = True
            except Exception as e:
                error = e

        if need_final_bound and image_done:
            boundary_try_bigtiff = False
            try:
                self.write_final_boundary((xsize, ysize, geotransform), bin_format_kwargs, mosaic_arrays)
            except RuntimeError:
                boundary_try_bigtiff = True
            except Exception as e:
                error = e

            if boundary_try_bigtiff:
                if os.path.exists(self.output_boundary):
                    gdal.Unlink(self.output_boundary)

                print("文件可能超过4GB, 因此需启用BigTIFF后重试")

                try:
                    self.write_final_boundary((xsize, ysize, geotransform), bin_bigtiff_format_kwargs, mosaic_arrays)
                except Exception as e:
                    error = e

        return error
