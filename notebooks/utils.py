import os
import re
import glob
import ee
import json
import geemap
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

###############################################################################
#                                                                             #
#                1. GLOBAL CONFIGURATION & CONSTANTS                          #
#                                                                             #
###############################################################################

# 1. Directory Definitions
DEFAULT_INPUT_DIR = "/content/glance_data/masked"
DEFAULT_OUTPUT_DIR = "/content/glance_data/output"

# 2. Data Definitions
NODATA_VALUE = 255
GLANCE_COLLECTION_ID = "projects/GLANCE/DATASETS/V001"
GLANCE_CLASS_BAND = "LC"

# 3. Class Metadata
GLANCE_METADATA = {
    1: {'name': 'Water',      'color': '#0000FF'},
    2: {'name': 'Ice/Snow',   'color': '#AAAAFF'},
    3: {'name': 'Developed',  'color': '#FF0000'},
    4: {'name': 'Barren',     'color': '#964B00'},
    5: {'name': 'Trees',      'color': '#006400'},
    6: {'name': 'Shrub',      'color': '#FFBB22'},
    7: {'name': 'Herbaceous', 'color': '#FFFF4C'}
}

###############################################################################
#                                                                             #
#                  2. HELPER FUNCTIONS (FILE & SYSTEM)                        #
#                                                                             #
###############################################################################

def get_year_from_filename(filepath):
    """
    Extracts the 4-digit year from the filename using a regular expression.

    Parameters
    ----------
    filepath : str
        The full path or filename of the raster image (e.g., '/path/to/glance_2005.tif').

    Returns
    -------
    int
        The 4-digit year extracted from the filename. Returns 0 if no pattern matches.
    """
    filename = os.path.basename(filepath)
    match = re.search(r"(\d{4})\.tif$", filename)
    if match:
        return int(match.group(1))
    return 0

###############################################################################
#                                                                             #
#                  3. VISUALIZATION FUNCTIONS (EARTH ENGINE)                  #
#                                                                             #
###############################################################################

# ... (sem alterações nesta seção)

###############################################################################
#                                                                             #
#                  4. ANALYSIS FUNCTIONS (LOCAL RASTER)                       #
#                                                                             #
###############################################################################

# ... (sem alterações nesta seção)

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import rasterio
from pyproj import Transformer, Geod
from matplotlib_scalebar.scalebar import ScaleBar

# 1. Provide a stub for north_arrow if not defined
def north_arrow(
    ax,
    location="upper right",
    shadow=False,
    rotation=None,
    scale=0.3,
) -> None:
    """
    Placeholder for north_arrow function.
    """
    pass

from pyproj import Geod

def compute_display_pixel_size_km(
    lon_l: float,
    lon_r: float,
    lat_mid: float,
    width_pixels: int,
) -> float:
    """
    Calculate the real size of a pixel in kilometers for the scale bar.
    """
    geod = Geod(ellps="WGS84")
    lon_mid = (lon_l + lon_r) / 2.0
    _, _, dist1 = geod.inv(lon_l, lat_mid, lon_mid, lat_mid)
    _, _, dist2 = geod.inv(lon_mid, lat_mid, lon_r, lat_mid)
    width_m = dist1 + dist2
    dx_km = (width_m / width_pixels) / 1000.0
    return dx_km

def plot_number_of_changes_map(
    output_dir: str,
    nodata_val: int,
    raster_filename: str = "Number_of_Changes_Raster",
    scale_factor: float = 0.05,
) -> None:
    """
    Plot the Number of Changes raster map with cartographic elements.

    Esta função lê vários tiles de GEE usando um VRT e faz downsample para evitar OOM.
    """
    # 1. Locate all raster tiles exported by GEE
    raster_files = glob.glob(
        os.path.join(
            output_dir,
            f"{raster_filename}*.tif",
        )
    )
    if not raster_files:
        raise FileNotFoundError(f"Raster tiles not found for prefix: {raster_filename}.")

    # 2. Create a temporary Virtual Raster (VRT) to merge tiles
    vrt_path = os.path.join(output_dir, "merged_changes.vrt")
    files_str = " ".join([f'"{f}"' for f in raster_files])
    os.system(f"gdalbuildvrt {vrt_path} {files_str}")

    # 4. Read the raster and basic metadata with downsampling
    with rasterio.open(vrt_path) as src:
        bounds = src.bounds
        width_pixels = src.width
        lon_l = bounds.left
        lon_r = bounds.right
        lat_mid = (bounds.bottom + bounds.top) / 2.0

        data = src.read(1)
        data = np.ma.masked_equal(
            data,
            nodata_val,
        )

        left, bottom, right, top = src.bounds
        src_crs = src.crs
        # Adjust the affine transform for the new resolution
        transform = src.transform * src.transform.scale(
            (
                src.width / data.shape[1]
            ),
            (
                src.height / data.shape[0]
            ),
        )

    # 3. Calculate the pixel size for the scale bar
    pixel_size_km = compute_display_pixel_size_km(
        lon_l=lon_l,
        lon_r=lon_r,
        lat_mid=lat_mid,
        width_pixels=width_pixels,
    )

    # ... (restante da função mantém igual, sem alterações)

# ... (restante do arquivo sem alteração)
