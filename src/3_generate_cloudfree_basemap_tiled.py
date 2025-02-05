import os
import numpy as np
import rioxarray
import xarray as xr
from osgeo import gdal, osr
from typing import List, Dict, Tuple
import multiprocessing
from functools import partial
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import box
from rasterio import transform as rio_transform

# Explicitly enable exceptions for GDAL
gdal.UseExceptions()


def extract_geometries(
    input_directories: List[str],
    target_epsg: str = "EPSG:4326",
    num_processes: int = None,
) -> List[Dict]:
    """
    Extract geometry information from all GeoTIFF files in the input directories.

    Args:
    input_directories (List[str]): Paths to the directories containing GeoTIFF files for each band.
    target_epsg (str): EPSG code for the target CRS to transform GeoTIFFs.
    num_processes (int): Number of processes to use. If None, uses the number of CPU cores.

    Returns:
    List[Dict]: A list of dictionaries, each containing geometry information for a GeoTIFF file.
    """
    if num_processes is None:
        num_processes = max(10, multiprocessing.cpu_count() // 10)

    all_files = []
    for directory in input_directories:
        files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith((".tif", ".tiff"))
        ]
        all_files.extend(files)

    with multiprocessing.Pool(processes=num_processes) as pool:
        process_func = partial(process_file, target_epsg=target_epsg)
        geometries = list(
            tqdm(
                pool.imap(process_func, all_files),
                total=len(all_files),
                desc="Extracting geometries",
            )
        )

    # Filter out None values (failed processings)
    return [g for g in geometries if g is not None]


def process_file(filepath: str, target_epsg: str) -> Dict:
    """
    Process a single GeoTIFF file: reproject if necessary and extract geometry information.

    Args:
    filepath (str): Path to the GeoTIFF file.
    target_epsg (str): EPSG code for the target CRS.

    Returns:
    Dict: Geometry information for the processed file.
    """
    try:
        # Reproject the raster if necessary
        was_reprojected = reproject_raster(filepath, target_epsg)

        dataset = gdal.Open(filepath)
        if dataset is None:
            print(f"Warning: Could not open {filepath}. Skipping.")
            return None

        # Get geotransform
        geotransform = dataset.GetGeoTransform()

        # Calculate bounding box
        minx = geotransform[0]
        maxy = geotransform[3]
        maxx = minx + geotransform[1] * dataset.RasterXSize
        miny = maxy + geotransform[5] * dataset.RasterYSize

        # Get pixel size
        x_res = geotransform[1]
        y_res = abs(geotransform[5])

        # Get CRS
        crs = dataset.GetProjection()
        srs = osr.SpatialReference(wkt=crs)
        epsg = srs.GetAuthorityCode(None)

        geometry = {
            "filepath": filepath,
            "bbox": (minx, miny, maxx, maxy),
            "pixel_size": (x_res, y_res),
            "crs": crs,
            "epsg": epsg,
            "dimensions": (dataset.RasterXSize, dataset.RasterYSize),
            "was_reprojected": was_reprojected,
        }

        if was_reprojected:
            print(f"Reprojected: {filepath}")

        return geometry

    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None

    finally:
        # Close the dataset
        dataset = None


def reproject_raster(input_path: str, target_epsg: str) -> bool:
    """
    Reproject a raster to the target EPSG and replace the original file.

    Args:
    input_path (str): Path to the input GeoTIFF file.
    target_epsg (str): EPSG code for the target CRS.

    Returns:
    bool: True if reprojection was performed, False if not needed.
    """
    src_ds = gdal.Open(input_path)
    src_srs = osr.SpatialReference(wkt=src_ds.GetProjection())
    src_epsg = src_srs.GetAuthorityCode(None)

    if src_epsg == target_epsg.split(":")[1]:
        src_ds = None
        return False

    # Create a temporary file for the reprojected raster
    temp_file = input_path + ".temp.tif"

    # Set the target SRS
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(int(target_epsg.split(":")[1]))

    # Reproject the raster
    gdal.Warp(temp_file, src_ds, dstSRS=target_srs)

    src_ds = None  # Close the dataset

    # Replace the original file with the reprojected one
    os.remove(input_path)
    os.rename(temp_file, input_path)

    return True


def create_cloud_mask(
    scl_file: str,
    tile_bbox: Tuple[float, float, float, float],
    target_epsg: str,
    tile_size: Tuple[int, int],
) -> xr.DataArray:
    """
    Create a cloud mask from the SCL file.

    Args:
    scl_file (str): Path to the SCL file.
    tile_bbox (Tuple[float, float, float, float]): Bounding box of the tile.
    target_epsg (str): Target EPSG code.
    tile_size (Tuple[int, int]): Size of the tile in pixels.

    Returns:
    xr.DataArray: Cloud mask (True for clear pixels, False for cloudy pixels).
    """
    with rioxarray.open_rasterio(scl_file) as scl:
        if scl.rio.crs.to_string() != target_epsg:
            scl = scl.rio.reproject(target_epsg)

        scl = scl.rio.clip_box(*tile_bbox)

        if scl.rio.shape != tile_size:
            transform = rio_transform.from_bounds(
                *tile_bbox, tile_size[0], tile_size[1]
            )
            scl = scl.rio.reproject(target_epsg, shape=tile_size, transform=transform)

        cloud_mask = (scl != 3) & (scl != 8) & (scl != 9)
        return cloud_mask.squeeze()  # Remove single-dimensional entries


def process_tile(
    tile_bbox: Tuple[float, float, float, float],
    intersecting_files: List[Dict],
    target_epsg: str,
    tile_size: Tuple[int, int],
    scl_directory: str,
) -> xr.DataArray:
    """
    Process a single tile by calculating the mean of intersecting rasters using rioxarray for 3 bands,
    and apply cloud masking.

    Args:
    tile_bbox (Tuple[float, float, float, float]): Bounding box of the tile (minx, miny, maxx, maxy).
    intersecting_files (List[Dict]): List of files intersecting with the tile.
    target_epsg (str): Target EPSG code.
    tile_size (Tuple[int, int]): Size of the tile in pixels.
    scl_directory (str): Directory containing SCL files.

    Returns:
    xr.DataArray: Processed tile data with 3 bands, or None if no valid data.
    """
    tile_data = []
    reference_data = None
    # print(f"Number of intersecting files: {len(intersecting_files)}")
    for file in intersecting_files:
        try:
            with rioxarray.open_rasterio(file["filepath"]) as src:
                # Check if the source data is valid
                if src.isnull().all():
                    print(
                        f"Warning: File {file['filepath']} contains all null values. Skipping."
                    )
                    continue

                # Reproject to target CRS if needed
                if src.rio.crs.to_string() != target_epsg:
                    src = src.rio.reproject(target_epsg)

                # Clip to tile extent
                clipped = src.rio.clip_box(*tile_bbox)

                # Check if clipping resulted in any data
                if clipped.isnull().all():
                    print(
                        f"Warning: Clipping {file['filepath']} resulted in all null values. Skipping."
                    )
                    continue

                # Ensure the clipped data has the correct dimensions
                if clipped.rio.shape != tile_size:
                    transform = rio_transform.from_bounds(
                        *tile_bbox, tile_size[0], tile_size[1]
                    )
                    clipped = clipped.rio.reproject(
                        target_epsg, shape=tile_size, transform=transform
                    )
                # Verify the shape after reprojection
                if clipped.rio.shape != tile_size:
                    print(
                        f"Error: Reprojection of {file['filepath']} resulted in incorrect shape. Expected {tile_size}, got {clipped.rio.shape}. Skipping."
                    )
                    continue

                # Set or match to reference data
                if reference_data is None:
                    reference_data = clipped
                else:
                    clipped = clipped.rio.reproject_match(reference_data)

                # Create and apply cloud mask
                scl_file = os.path.join(
                    scl_directory,
                    os.path.basename(file["filepath"]).replace("RGB_", ""),
                )
                if os.path.exists(scl_file):
                    cloud_mask = create_cloud_mask(
                        scl_file, tile_bbox, target_epsg, tile_size
                    )
                    # Dirty fix for a coordinate bug between cloud_mask and image_array
                    # Unfixable with rio.reproject_match
                    cloud_mask["x"] = clipped.x.values
                    cloud_mask["y"] = clipped.y.values
                    clipped = clipped.where(cloud_mask, other=np.nan)
                else:
                    print(
                        f"Warning: SCL file not found for {file['filepath']}. Skipping cloud masking for this file."
                    )

                tile_data.append(clipped)
                # print(f"Processed file {file['filepath']} #{count}")
        except Exception as e:
            print(f"Error processing file {file['filepath']}: {str(e)}")

    if not tile_data:
        print(f"No valid data for tile with bbox {tile_bbox}")
        return None

    # Stack all tile data along a new dimension
    stacked_data = xr.concat(tile_data, dim="source")

    # All zero values are considered nodata
    stacked_data = stacked_data.where(stacked_data != 0, other=np.nan)
    # Calculate mean across all valid data for each band separately
    mean_tile = stacked_data.mean(dim="source", skipna=True)

    # Sanity checks
    if mean_tile.isnull().all():
        print(f"Warning: Resulting tile contains all null values for bbox {tile_bbox}")
        return None

    valid_percentage = (~mean_tile.isnull()).sum().item() / (mean_tile.size)
    if valid_percentage < 0.2:  # Less than 20% valid data
        print(
            f"Warning: Resulting tile contains too few valid pixels ({valid_percentage:.2%}) for bbox {tile_bbox}"
        )
        return None

    return mean_tile, valid_percentage


def process_and_save_tile(args):
    """
    Process a single tile and save it to a file.

    Args:
    args (tuple): A tuple containing (tile_bbox, tile_index, geometries, geoseries, target_epsg, tile_size, pixel_size, output_folder, scl_directory)

    Returns:
    str: A message indicating the tile has been processed and saved, or skipped if no valid data.
    """
    (
        tile_bbox,
        tile_index,
        geometries,
        geoseries,
        target_epsg,
        tile_size,
        pixel_size,
        output_folder,
        scl_directory,
    ) = args

    # Create a box for the tile
    tile_box = box(*tile_bbox)
    tile_gs = gpd.GeoSeries([tile_box], crs=target_epsg)

    # Find intersecting geometries
    intersecting_indices = geoseries.intersects(tile_gs.iloc[0])
    intersecting_files = [
        geometries[i] for i, intersects in enumerate(intersecting_indices) if intersects
    ]

    if not intersecting_files:
        return f"Skipped tile {tile_index + 1} (no intersecting files)"

    return_value = process_tile(
        tile_bbox, intersecting_files, target_epsg, tile_size, scl_directory
    )

    if return_value is None:
        mean_tile = None

    else:
        mean_tile, valid_percentage = return_value

    if mean_tile is None:
        print(f"Skipped tile {tile_index + 1} (no valid data)")
        return "Skipped"

    # Create output file for this tile
    output_file = os.path.join(output_folder, f"tile_{tile_index:04d}.tif")
    mean_tile.rio.to_raster(output_file)

    print(
        f"Processed and saved tile {tile_index + 1} valid data {valid_percentage} (intersecting files: {len(intersecting_files)})"
    )
    return "Processed"


def create_geoseries(geometries: List[Dict], target_epsg: str) -> gpd.GeoSeries:
    """
    Create a GeoSeries from the list of geometries.

    Args:
    geometries (List[Dict]): List of geometry dictionaries.
    target_epsg (str): Target EPSG code.

    Returns:
    gpd.GeoSeries: GeoSeries of geometry bounds.
    """
    bounds = [box(*geom["bbox"]) for geom in geometries]
    gs = gpd.GeoSeries(bounds, crs=geometries[0]["epsg"])
    return gs.to_crs(target_epsg)


def calculate_mean_raster(
    geometries: List[Dict],
    overall_bbox: Tuple[float, float, float, float],
    tile_size: Tuple[int, int],
    pixel_size: Tuple[float, float],
    target_epsg: str,
    output_folder: str,
    scl_directory: str,
    num_processes: int = None,
):
    """
    Calculate mean raster by processing tiles in parallel and saving individual tile files.

    Args:
    geometries (List[Dict]): List of geometry dictionaries for input files.
    overall_bbox (Tuple[float, float, float, float]): Overall bounding box.
    tile_size (Tuple[int, int]): Size of each tile in pixels.
    pixel_size (Tuple[float, float]): Pixel size in target CRS units.
    target_epsg (str): Target EPSG code.
    output_folder (str): Folder to save output tile files.
    scl_directory (str): Directory containing SCL files.
    num_processes (int): Number of parallel processes to use.
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    geoseries = create_geoseries(geometries, target_epsg)
    tile_size_geo = (tile_size[0] * pixel_size[0], tile_size[1] * pixel_size[1])
    tiles = generate_tiles(overall_bbox, tile_size_geo)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Prepare arguments for process_and_save_tile
    args_list = [
        (
            tile_bbox,
            tile_index,
            geometries,
            geoseries,
            target_epsg,
            tile_size,
            pixel_size[0],
            output_folder,
            scl_directory,
        )
        for tile_index, tile_bbox in enumerate(tiles)
    ]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_and_save_tile, args_list),
                total=len(tiles),
                desc="Processing and saving tiles",
            )
        )

    # Print summary of results
    processed_count = sum(1 for r in results if r.startswith("Processed"))
    skipped_count = sum(1 for r in results if r.startswith("Skipped"))
    print(f"\nProcessed {processed_count} tiles, skipped {skipped_count} tiles.")


def generate_tiles(
    overall_bbox: Tuple[float, float, float, float], tile_size_geo: Tuple[float, float]
) -> List[Tuple[float, float, float, float]]:
    """
    Generate a list of tile bounding boxes covering the overall extent.

    Args:
    overall_bbox (Tuple[float, float, float, float]): Overall bounding box (minx, miny, maxx, maxy).
    tile_size_geo (Tuple[float, float]): Tile size in geographical units.

    Returns:
    List[Tuple[float, float, float, float]]: List of tile bounding boxes.
    """
    tiles = []
    x = overall_bbox[0]
    while x < overall_bbox[2]:
        y = overall_bbox[1]
        while y < overall_bbox[3]:
            tile_bbox = (
                x,
                y,
                min(x + tile_size_geo[0], overall_bbox[2]),
                min(y + tile_size_geo[1], overall_bbox[3]),
            )
            tiles.append(tile_bbox)
            y += tile_size_geo[1]
        x += tile_size_geo[0]
    return tiles


def create_vrt(output_folder: str, vrt_filename: str):
    """
    Create a VRT file from all the GeoTIFF tiles in the output folder.

    Args:
    output_folder (str): Path to the folder containing the tile files.
    vrt_filename (str): Name of the VRT file to be created.
    """
    vrt_path = os.path.join(output_folder, vrt_filename)
    tile_files = [
        os.path.join(output_folder, f)
        for f in os.listdir(output_folder)
        if f.endswith(".tif")
    ]

    vrt_options = gdal.BuildVRTOptions(resampleAlg="nearest", addAlpha=False)
    vrt_dataset = gdal.BuildVRT(vrt_path, tile_files, options=vrt_options)
    vrt_dataset = None  # This will close the dataset and flush it to disk

    print(f"Created VRT file: {vrt_path}")
    return vrt_path


def main():
    input_directories = [
        "/datadisk3/botswana/data/2022/05/RGB",
    ]
    output_folder = "/datadisk3/botswana/data/2022/05/MEAN_RGB_TILED"
    scl_directory = "/datadisk3/botswana/data/2022/05/SCL"
    target_epsg = "EPSG:32735"  # UTM Zone 35S
    num_processes = (
        10  # Set to a specific number if you don't want to use all CPU cores
    )
    tile_size = (5000, 5000)  # Size of each tile in pixels
    vrt_filename = "mean_rgb_mosaic.vrt"

    print(f"Extracting geometries from {', '.join(input_directories)}...")
    geometries = extract_geometries(input_directories, target_epsg, num_processes)

    print(f"Extracted geometry information for {len(geometries)} files.")

    # Calculate overall bounding box and pixel size
    min_x = min(g["bbox"][0] for g in geometries)
    min_y = min(g["bbox"][1] for g in geometries)
    max_x = max(g["bbox"][2] for g in geometries)
    max_y = max(g["bbox"][3] for g in geometries)
    overall_bbox = (min_x, min_y, max_x, max_y)

    # Use the smallest pixel size found
    pixel_size = min((g["pixel_size"] for g in geometries), key=lambda x: x[0])

    print("Calculating mean raster tiles with cloud masking...")
    calculate_mean_raster(
        geometries,
        overall_bbox,
        tile_size,
        pixel_size,
        target_epsg,
        output_folder,
        scl_directory,
        num_processes,
    )

    print("Creating VRT file...")
    create_vrt(output_folder, vrt_filename)

    print(
        f"Mean RGB raster calculation with cloud masking complete. Output tiles saved in: {output_folder}"
    )
    print(f"VRT file created: {os.path.join(output_folder, vrt_filename)}")


if __name__ == "__main__":
    main()
