import os
import time
import requests
from shapely.geometry import shape
from pystac_client import Client as STACClient
from planetary_computer import sign_inplace, sas
from tqdm import tqdm
import concurrent.futures

# GeoJSON structure for Botswana
botswana_geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "Botswana Bounding Box"},
            "geometry": {
                "coordinates": [
                    [
                        [19.860850204666235, -27.2369783323106],
                        [29.074299775286875, -27.28492079006547],
                        [29, -17.5],
                        [20, -17.5],
                        [19.860850204666235, -27.2369783323106],
                    ]
                ],
                "type": "Polygon",
            },
        }
    ],
}

# Extract the bounding box from the GeoJSON
botswana_shape = shape(botswana_geojson["features"][0]["geometry"])
botswana_bbox = botswana_shape.bounds  # (minx, miny, maxx, maxy)

def create_directory_structure(base_dir, year, month):
    """Create directory structure for a given year and month."""
    month_dir = os.path.join(base_dir, f"{year}", f"{month:02d}")
    for band in ["B03", "B02", "B04", "SCL"]:
        os.makedirs(os.path.join(month_dir, band), exist_ok=True)
    return month_dir

def download_asset(item, asset_key, output_dir):
    asset = item.assets[asset_key]
    signed_url = sas.sign(asset.href)
    output_path = os.path.join(output_dir, f"{item.id}.tif")

    # Check if file already exists
    if os.path.exists(output_path):
        return output_path

    # Download file
    response = requests.get(signed_url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as file, tqdm(
        desc=f"{asset_key} - {item.id}",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        leave=False,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

    return output_path

def process_item(item, month_dir, progress_bars):
    asset_paths = {}
    for key in ["B02", "B03", "B04", "SCL"]:
        asset_dir = os.path.join(month_dir, key)
        asset_paths[key] = download_asset(item, key, asset_dir)
        progress_bars[key].update(1)
    return asset_paths

def process_month(year, month, base_dir):
    print(f"\nProcessing data for {year}-{month:02d}")
    start_time = time.time()

    month_dir = create_directory_structure(base_dir, year, month)

    # Set up date range for the month
    start = f"{year}-{month:02d}-01"
    end = f"{year}-{month+1:02d}-01" if month < 12 else f"{year+1}-01-01"

    # Query Planetary Computer STAC API
    catalog = STACClient.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=sign_inplace
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=botswana_bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": 80}},
    )

    items = list(search.items())
    if not items:
        print(f"No data available for {year}-{month:02d}")
        return None

    print(f"Found {len(items)} items.")

       # Set up progress bars
    progress_bars = {
        "B02": tqdm(total=len(items), desc="Downloading B02", leave=True),
        "B03": tqdm(total=len(items), desc="Downloading B03", leave=True),
        "B04": tqdm(total=len(items), desc="Downloading B04", leave=True),
        "SCL": tqdm(total=len(items), desc="Downloading SCL", leave=True),
    }

        # Process items and download assets
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(process_item, item, month_dir, progress_bars)
            for item in items
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    # Close progress bars
    for bar in progress_bars.values():
        bar.close()

    print(f"Completed processing for {year}-{month:02d} in {time.time() - start_time:.2f} seconds.")

if name == "__main__":
		base_dir = "data"
		year = 2022
		month = 5
		process_month(year, month, base_dir)