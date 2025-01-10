# solafune_country_basemaps
Code for generating country scale cloudfree basemaps from Sentinel-2 imagery

This is the accompanying code for the blogpost from Solafune here: **ADD LINK ONCE ARTICLE IS LIVE**

Three scripts are in this repo:

*1_download_sentinel_bands*: This script downloads the red, green, blue and cloud mask bands from the Sentinel-2 repo on Planetary Computer.
*2_stack_bands*: This script stacks the red, green and blue bands into a single RGB tif file.
*3_generate_cloudfree_basemap_tiled*: This script generates cloudfree basemap tiles and vrt that unifies them.

The python version used is 3.10.14