import os
import rioxarray
import xarray as xr
from typing import List
import numpy as np
from multiprocessing import Pool, cpu_count
import shutil
from tqdm import tqdm

def get_free_space(path="."):
    """Get the free space on the disk containing the given path."""
    total, used, free = shutil.disk_usage(path)
    return free / (2**30)  # Convert bytes to gigabytes

def process_single_file(args):
    """Process a single file for RGB stacking."""
    file, input_directories, output_directory = args
    
    # Check if output file already exists
    output_file = os.path.join(output_directory, f"RGB_{file}")
    if os.path.exists(output_file):
        return f"Skipped (already exists): {output_file}"

    rgb_bands = []
    for dir in input_directories:
        file_path = os.path.join(dir, file)
        if not os.path.exists(file_path):
            return f"Skipped (missing input): {file_path}"
        try:
            with rioxarray.open_rasterio(file_path) as src:
                rgb_bands.append(src.squeeze())  # Remove single-dimensional entries
        except Exception as e:
            return f"Error opening {file_path}: {str(e)}"
    
    if len(rgb_bands) != 3:
        return f"Skipped (insufficient bands): {file}"

    try:
        rgb_stack = xr.concat(rgb_bands, dim='band')
        rgb_stack['band'] = ['red', 'green', 'blue']
        rgb_stack.rio.to_raster(output_file, driver='GTiff')
        return f"Processed: {output_file}"
    except Exception as e:
        return f"Error processing {file}: {str(e)}"

def stack_rgb_geotiffs_parallel(input_directories: List[str], output_directory: str, max_workers: int = None):
    """
    Stack single-band GeoTIFFs from separate directories into 3-band RGB GeoTIFFs using parallel processing.

    Args:
    input_directories (List[str]): List of three directories containing single-band GeoTIFFs for R, G, B respectively.
    output_directory (str): Directory to save the output 3-band RGB GeoTIFFs.
    max_workers (int): Maximum number of worker processes. Defaults to number of CPU cores.

    Returns:
    None
    """
    if len(input_directories) != 3:
        raise ValueError("Exactly three input directories (R, G, B) must be provided.")

    os.makedirs(output_directory, exist_ok=True)

    # Check free space
    free_space = get_free_space(output_directory)
    if free_space < 10:  # Adjust this threshold as needed
        print(f"Warning: Only {free_space:.2f} GB free on disk. This may not be enough.")
        # Optionally, you could return early or raise an exception here

    files = [f for f in os.listdir(input_directories[0]) if f.endswith('.tif')]

    if max_workers is None:
        max_workers = 20

    # Prepare arguments for multiprocessing
    args = [(file, input_directories, output_directory) for file in files]

    # Use multiprocessing to process files in parallel with tqdm progress bar
    with Pool(processes=max_workers) as pool:
        results = list(tqdm(pool.imap(process_single_file, args), total=len(files), desc="Processing files"))

    # Count and print results
    processed = sum(1 for r in results if r.startswith("Processed:"))
    skipped_existing = sum(1 for r in results if r.startswith("Skipped (already exists):"))
    skipped_missing = sum(1 for r in results if r.startswith("Skipped (missing input):"))
    skipped_bands = sum(1 for r in results if r.startswith("Skipped (insufficient bands):"))
    errors = sum(1 for r in results if r.startswith("Error"))

    print(f"Processing complete:")
    print(f"  Successfully processed: {processed}")
    print(f"  Skipped (already exist): {skipped_existing}")
    print(f"  Skipped (missing input): {skipped_missing}")
    print(f"  Skipped (insufficient bands): {skipped_bands}")
    print(f"  Errors: {errors}")

    # Print error messages
    for result in results:
        if result.startswith("Error"):
            print(result)

if __name__ == '__main__':
    input_directories = [
        "/datadisk3/botswana/data/2022/05/B04",  # Red band
        "/datadisk3/botswana/data/2022/05/B03",  # Green band
        "/datadisk3/botswana/data/2022/05/B02"   # Blue band
    ]
    output_directory = "/datadisk3/botswana/data/2022/05/RGB"

    stack_rgb_geotiffs_parallel(input_directories, output_directory)