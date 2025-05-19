import os
import warnings
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
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=UserWarning)
        warnings.simplefilter(action="ignore", category=RuntimeWarning)
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

    def get_final_geometry_from_images(self, file_infos: list[gdal_merge.file_info]):
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

        return xsize, ysize, geotransform

    def normalize_format_kwargs(self, format_kwargs):
        predictor = "2" if self.base_dtype in [1, 2, 3, 4, 5, 14] else "3"
        format_kwargs.setdefault("BIGTIFF", "IF_SAFER")
        format_kwargs.setdefault("PREDICTOR", predictor)
        if self.base_dtype in [1, 2] and self.base_bands in [3, 4]:
            format_kwargs.update({"PHOTOMETRIC": "RGB"})
        if self.base_dtype in [1, 2] and self.base_bands in [2, 4]:
            format_kwargs.update({"ALPHA": "YES"})
        bigtiff_format_kwargs = format_kwargs.copy()
        bigtiff_format_kwargs.update({"BIGTIFF": "YES"})

        bin_format_kwargs = {"COMPRESS": "DEFLATE", "ZLEVEL": "6", "PREDICTOR": "2", "BIGTIFF": "IF_SAFER"}
        bin_bigtiff_format_kwargs = bin_format_kwargs.copy()
        bin_bigtiff_format_kwargs.update({"BIGTIFF": "YES"})

        return format_kwargs, bigtiff_format_kwargs, bin_format_kwargs, bin_bigtiff_format_kwargs

    def simple_mosaic_write_image(self, file_infos: list[gdal_merge.file_info], output: str,
                                  bands, format_kwargs: dict, nodata: float | int):
        xsize, ysize, geotransform = self.get_final_geometry_from_images(file_infos)
        with DRIVER.Create(output, xsize, ysize, bands, self.base_dtype, format_kwargs) as final_ds:
            final_ds.SetProjection(self.base_proj)
            final_ds.SetGeoTransform(geotransform)

            fi_processed = 0
            for fi in file_infos:
                for band in range(1, bands + 1):
                    fi.copy_into(final_ds, band, band, nodata)
                fi_processed += 1
                gdal.TermProgress_nocb(float(fi_processed / len(file_infos)))

        gdal.TermProgress_nocb(1.0)

    def simple_mosaic(self, format_kwargs: dict | None = None):
        if format_kwargs is None:
            format_kwargs = {}

        (format_kwargs, bigtiff_format_kwargs,
         bin_format_kwargs, bin_bigtiff_format_kwargs) = self.normalize_format_kwargs(format_kwargs)

        ready_images = self.normalize_images()
        file_infos = gdal_merge.names_to_fileinfos(ready_images)
        wrapped_final_image = lambda: self.simple_mosaic_write_image(file_infos, self.output,
                                                                     self.base_bands, format_kwargs, self.nodata)
        wrapped_final_image_bigtiff = lambda: self.simple_mosaic_write_image(file_infos, self.output,
                                                                             self.base_bands, bigtiff_format_kwargs,
                                                                             self.nodata)
        error = self.try_big_tiff(wrapped_final_image, wrapped_final_image_bigtiff, self.output)

        boundary_list = self.input_boundaries if self.input_boundaries else [boundary for image in ready_images if
                                                                             (boundary := self.search_boundary(image))]
        boundary_need = self.input_boundaries or len(boundary_list) == len(ready_images)

        if not error and boundary_need:
            bin_file_infos = gdal_merge.names_to_fileinfos(boundary_list)
            wrapped_final_boundary = lambda: self.simple_mosaic_write_image(bin_file_infos, self.output_boundary, 1,
                                                                            bin_format_kwargs, 0)
            wrapped_final_boundary_bigtiff = lambda: self.simple_mosaic_write_image(bin_file_infos,
                                                                                    self.output_boundary, 1,
                                                                                    bin_bigtiff_format_kwargs, 0)
            error = self.try_big_tiff(wrapped_final_boundary, wrapped_final_boundary_bigtiff, self.output_boundary)

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

    @staticmethod
    def write_tmp_dis_bin(tmp_file: str, image_geometry: tuple, dtype, data: np.ndarray):
        proj, tran = image_geometry

        with DRIVER.Create(tmp_file, data.shape[1], data.shape[0], 1, dtype, CREATE_OPTIONS) as single_ds:
            single_ds.SetProjection(proj)
            single_ds.SetGeoTransform(tran)
            single_ds.GetRasterBand(1).WriteArray(data)
            single_ds.FlushCache()

    def single_merged_into_scene_and_get_data(self, files: tuple, image_geometry: tuple,
                                              dtype, del_input_file=True):
        input_single_file, merged_file = files
        xsize, ysize, geotransform = image_geometry

        with gdal.Open(input_single_file) as ds:
            bands = ds.RasterCount

        with DRIVER.Create(merged_file, xsize, ysize, bands, dtype, CREATE_OPTIONS) as merged_ds:
            merged_ds.SetProjection(self.base_proj)
            merged_ds.SetGeoTransform(geotransform)
            single_image_info = gdal_merge.names_to_fileinfos([input_single_file])

            extract_data: list[np.ndarray] = []
            for band in range(1, bands + 1):
                single_image_info[0].copy_into(merged_ds, band, band, self.nodata)
                data = merged_ds.GetRasterBand(band).ReadAsArray(0, 0, xsize, ysize)
                extract_data.append(data)

            merged_ds.FlushCache()

        gdal.Unlink(merged_file)
        if del_input_file:
            gdal.Unlink(input_single_file)

        return extract_data

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

    @staticmethod
    def try_big_tiff(func, func_bigtiff, output):
        error = None

        try:
            func()
        except RuntimeError:
            if os.path.exists(output):
                gdal.Unlink(output)
            print("文件可能超过4GB, 因此需启用BigTIFF后重试")
            try:
                func_bigtiff()
            except Exception as e:
                if os.path.exists(output):
                    gdal.Unlink(output)
                error = e
        except Exception as e:
            if os.path.exists(output):
                gdal.Unlink(output)
            error = e

        return error

    def large_feather_mosaic(self, need_final_bound=True, tags: dict | None = None, format_kwargs: dict | None = None):
        if tags is None:
            tags = {}

        if format_kwargs is None:
            format_kwargs = {}

        (format_kwargs, bigtiff_format_kwargs,
         bin_format_kwargs, bin_bigtiff_format_kwargs) = self.normalize_format_kwargs(format_kwargs)

        ready_images = self.normalize_images()
        file_infos = gdal_merge.names_to_fileinfos(ready_images)
        image_geometry = self.get_final_geometry_from_images(file_infos)

        fi_processed = 0
        mosaic_arrays: dict[str, ImageData] = {}

        for idx, image in enumerate(ready_images):
            filename = os.path.basename(image).split(".")[0]

            with gdal.Open(image) as ds:
                proj = ds.GetProjection()
                tran = ds.GetGeoTransform()

            tmp_image = f"/vsimem/tmp_{filename}.tif"
            merged_image_data = self.single_merged_into_scene_and_get_data((image, tmp_image),
                                                                           image_geometry,
                                                                           self.base_dtype, False)
            bands_dict = {idx + 1: band_data for idx, band_data in enumerate(merged_image_data)}

            binary_image = self.get_binary_image(idx, image)
            distance_map = sitk.DanielssonDistanceMap(binary_image, inputIsBinary=True, useImageSpacing=True)
            distance_map_data = sitk.GetArrayFromImage(distance_map)
            distance_map_data = np.where(distance_map_data <= 0, 0, distance_map_data)

            tmp_distance = f"/vsimem/tmp_dis_{filename}.tif"
            merged_tmp_distance = f"/vsimem/tmp_merged_dis_{filename}.tif"
            self.write_tmp_dis_bin(tmp_distance, (proj, tran), gdal.GDT_Float32, distance_map_data)
            merged_distance_data = self.single_merged_into_scene_and_get_data((tmp_distance, merged_tmp_distance),
                                                                              image_geometry,
                                                                              gdal.GDT_Float32)

            tmp_binary = f"/vsimem/tmp_bin_{filename}.tif"
            merged_tmp_binary = f"/vsimem/tmp_merged_bin_{filename}.tif"
            self.write_tmp_dis_bin(tmp_binary, (proj, tran), gdal.GDT_Byte, 255 - sitk.GetArrayFromImage(binary_image))
            merged_binary_data = self.single_merged_into_scene_and_get_data((tmp_binary, merged_tmp_binary),
                                                                            image_geometry,
                                                                            gdal.GDT_Byte)


            image_data = ImageData(bands_dict, merged_distance_data[0], merged_binary_data[0])
            mosaic_arrays[image] = image_data

            fi_processed += 1
            gdal.TermProgress_nocb(fi_processed / len(ready_images) * 0.7)

        wrapped_final_image = lambda: self.write_final_image(image_geometry, format_kwargs, tags, mosaic_arrays)
        wrapped_final_image_bigtiff = lambda : self.write_final_image(image_geometry, bigtiff_format_kwargs, tags,
                                                                      mosaic_arrays)
        error = self.try_big_tiff(wrapped_final_image, wrapped_final_image_bigtiff, self.output)
        gdal.TermProgress_nocb(1.0)

        if need_final_bound and not error:
            wrapped_final_boundary = lambda: self.write_final_boundary(image_geometry, bin_format_kwargs,
                                                                       mosaic_arrays)
            wrapped_final_boundary_bigtiff = lambda: self.write_final_boundary(image_geometry,
                                                                               bin_bigtiff_format_kwargs,
                                                                               mosaic_arrays)
            error = self.try_big_tiff(wrapped_final_boundary, wrapped_final_boundary_bigtiff, self.output_boundary)

        return error
