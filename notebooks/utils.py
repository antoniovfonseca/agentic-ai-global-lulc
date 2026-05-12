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

def get_glance_map(year):
    """
    Generates an interactive map with GLANCE land cover data for a given year.

    This function assumes that the Earth Engine API (ee) has already been
    authenticated and initialized in the main execution environment.

    Parameters
    ----------
    year : int
        The year to visualize (e.g., 2010). 
        GLANCE data is typically available from 2001 to 2019.

    Returns
    -------
    geemap.Map
        An interactive map object centered on the data with the land cover
        layer added. Returns None if data cannot be loaded.
    """

    # 1. Dynamic Visualization Parameters construction
    # Extract IDs to sort correctly
    class_ids = sorted(GLANCE_METADATA.keys())
    
    # Earth Engine expects hex colors without '#', so we strip it
    palette = [GLANCE_METADATA[i]['color'].lstrip('#') for i in class_ids]
    
    vis_params = {
        'min': class_ids[0],
        'max': class_ids[-1],
        'palette': palette
    }

    # 2. Initialize Map
    m = geemap.Map()
    m.setCenter(0, 20, 2)

    # 3. Filter Data
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    try:
        # Use global constant for Collection ID
        collection = ee.ImageCollection(GLANCE_COLLECTION_ID)
        image = collection.filterDate(start_date, end_date).mosaic()
        
        # Use global constant for Band Name
        glance_layer = image.select(GLANCE_CLASS_BAND)

        # 4. Add Layer to Map
        m.addLayer(glance_layer, vis_params, f"GLANCE LC {year}")
        
        # 5. Dynamic Legend construction
        # Creates a dictionary 'Name': 'Color' for the legend
        legend_dict = {
            meta['name']: meta['color'] 
            for meta in GLANCE_METADATA.values()
        }
        
        m.add_legend(title="GLANCE Land Cover", legend_dict=legend_dict)
        
        return m

    except Exception as e:
        print(f"Error loading GLANCE data for {year}: {e}")
        return None

def view_local_rasters(input_dir=DEFAULT_INPUT_DIR):
    """
    Creates an interactive map displaying all .tif files found in the specified directory.
    This is useful for visually validating the result of the masking process.

    Parameters
    ----------
    input_dir : str, optional
        The directory path containing the masked .tif files.
        Defaults to DEFAULT_INPUT_DIR.

    Returns
    -------
    geemap.Map
        An interactive map object with the raster layers added.
        Returns None if no .tif files are found.
    """
    # 1. Initialize Map
    m = geemap.Map()
    
    # 2. Search for files
    search_pattern = os.path.join(input_dir, "*.tif")
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print(f"No .tif files found in {input_dir}")
        return None

    # 3. Prepare Colors (Palette)
    # Extract colors from metadata to match the charts.
    # We pass the list of hex codes (e.g., ['#0000FF', ...])
    class_ids = sorted(GLANCE_METADATA.keys())
    palette = [GLANCE_METADATA[i]['color'] for i in class_ids]
    
    # 4. Add each file as a Layer
    print(f"Loading {len(files)} local raster layers...")
    
    for filepath in files:
        filename = os.path.basename(filepath)
        year_match = re.search(r"(\d{4})", filename)
        layer_name = f"Masked {year_match.group(1)}" if year_match else filename
        
        try:
            m.add_raster(
                filepath, 
                layer_name=layer_name, 
                palette=palette,  # Tries to apply the specific class colors
                nodata=NODATA_VALUE
            )
        except Exception as e:
            print(f"Could not load {filename}: {e}")

    # 5. Add Legend
    legend_dict = {
        meta['name']: meta['color'] 
        for meta in GLANCE_METADATA.values()
    }
    m.add_legend(title="Classes", legend_dict=legend_dict)

    return m

###############################################################################
#                                                                             #
#                  4. ANALYSIS FUNCTIONS (LOCAL RASTER)                       #
#                                                                             #
###############################################################################

def export_global_pixel_counts_tasks(
    year_list: list,
    drive_folder: str,
    scale: int = 30,
    max_pixels: float = 1e13
) -> list:
    """
    Triggers Earth Engine tasks to calculate the frequency histogram (pixel counts)
    of the global GLANCE categorical images for a list of years and exports them 
    to Google Drive as CSV files.

    Parameters
    ----------
    year_list : list of int
        A list of 4-digit years to process (e.g., [2001, 2002, 2003]).
    drive_folder : str
        The name of the Google Drive folder where the CSVs will be saved.
    scale : int, optional
        The scale in meters for the GEE reduction. Default is 30.
    max_pixels : float, optional
        The maximum number of pixels to process in GEE. Default is 1e13.

    Returns
    -------
    list of ee.batch.Task
        A list containing all the triggered Earth Engine tasks.
    """
    # 1. Define global bounding box geometry
    global_geom = ee.Geometry.Polygon(
        [[[-180, -90], [180, -90], [180, 90], [-180, 90]]],
        None,
        False
    )

    # 2. Access the GLANCE collection using global constants defined in utils.py
    glance_collection = ee.ImageCollection(GLANCE_COLLECTION_ID).select(GLANCE_CLASS_BAND)

    # 3. Initialize an empty list to store the tasks
    tasks_list = []

    # 4. Iterate over the provided list of years
    for year in year_list:
        # 5. Filter the collection for the specific year
        image_year = glance_collection.filter(
            ee.Filter.calendarRange(year, year, 'year')
        ).mosaic()

        # 6. Calculate the frequency histogram
        # Note: tileScale=16 is used to avoid memory limit errors in global reductions
        histogram = image_year.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=global_geom,
            scale=scale,
            maxPixels=max_pixels,
            tileScale=16
        )

        # 7. Extract the dictionary and convert it to a FeatureCollection
        counts_dict = ee.Dictionary(histogram.get(GLANCE_CLASS_BAND))
        feature_collection = ee.FeatureCollection([
            ee.Feature(None, counts_dict)
        ])

        # 8. Define the export task parameters
        export_name = f"Pixel_Counts_LULC_{year}"
        task = ee.batch.Export.table.toDrive(
            collection=feature_collection,
            description=export_name,
            folder=drive_folder,
            fileFormat='CSV'
        )

        # 9. Start the task and append it to the list
        task.start()
        print(f"Task started: {export_name} (Scale: {scale}m)")
        tasks_list.append(task)

    # 10. Return the list of triggered tasks
    return tasks_list

def plot_pixel_counts_bar_chart(
    input_dir: str,
    class_labels_dict: dict,
    output_dir: str,
) -> None:
    """
    Reads yearly GEE pixel count CSVs, aggregates them, and generates a 
    stacked bar chart of pixel counts per class over time.

    Parameters
    ----------
    input_dir : str
        Directory path containing the Earth Engine CSVs (e.g., Pixel_Counts_LULC_2001.csv).
    class_labels_dict : dict
        Dictionary mapping class IDs to metadata (must contain "name" and "color").
    output_dir : str
        Directory path where the output plot will be saved.
    """
    import os
    import glob
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    # 1. Read and aggregate GEE CSVs
    csv_pattern = os.path.join(input_dir, "*.csv")
    csv_files = glob.glob(csv_pattern)

    yearly_data = {}
    for file in csv_files:
        basename = os.path.basename(file)
        # Extract 4-digit year from the filename
        match = re.search(r"(\d{4})", basename)
        if not match:
            continue
        
        year = int(match.group(1))
        df_csv = pd.read_csv(file)
        
        row_dict = {}
        for col in df_csv.columns:
            try:
                # Convert string column names (e.g., '1', '2') to integer IDs
                class_id = int(col)
                if class_id in class_labels_dict:
                    class_name = class_labels_dict[class_id]["name"]
                    row_dict[class_name] = df_csv[col].sum()
            except ValueError:
                # Ignore non-integer columns (like 'system:index')
                pass
        
        if row_dict:
            yearly_data[year] = row_dict

    if not yearly_data:
        print(f"No valid GEE CSV data found in {input_dir}")
        return

    # Create pivot table from aggregated data
    pivot_pixels = pd.DataFrame.from_dict(yearly_data, orient='index').fillna(0)
    pivot_pixels.sort_index(inplace=True)
    pivot_pixels.index.name = "Year"

    years_array = pivot_pixels.index.values

    # 2. Determine Y-axis scaling factor and label
    max_val = pivot_pixels.to_numpy().max()

    if max_val >= 1_000_000_000_000:
        scale_factor = 1_000_000_000_000
        y_label = "Class Area (trillion pixels)"
    elif max_val >= 1_000_000_000:
        scale_factor = 1_000_000_000
        y_label = "Class Area (billion pixels)"
    elif max_val >= 1_000_000:
        scale_factor = 1_000_000
        y_label = "Class Area (million pixels)"
    elif max_val >= 1_000:
        scale_factor = 1_000
        y_label = "Class Area (thousand pixels)"
    elif max_val >= 100:
        scale_factor = 100
        y_label = "Class Area (hundred pixels)"
    else:
        scale_factor = 1
        y_label = "Class Area (pixels)"

    pivot_scaled = pivot_pixels / scale_factor

    # 3. Prepare color map and sorting logic
    class_ids_plot = sorted(
        class_labels_dict.keys(),
    )

    color_map = {
        class_labels_dict[class_id]["name"]: class_labels_dict[class_id]["color"]
        for class_id in class_ids_plot
    }

    # Calculate Net Change
    first_year = years_array[0]
    last_year = years_array[-1]
    net_change_per_class = (
        pivot_scaled.loc[last_year]
        - pivot_scaled.loc[first_year]
    )

    # Map names back to IDs for tie-breaking
    name_to_id_map = {
        v["name"]: k
        for k, v in class_labels_dict.items()
    }

    df_sorting = net_change_per_class.to_frame(
        name="net_change",
    )
    df_sorting["class_id"] = df_sorting.index.map(
        name_to_id_map,
    )

    # Sort: Net Change (Desc) then Class ID (Desc)
    classes_for_stack = list(
        df_sorting.sort_values(
            by=[
                "net_change",
                "class_id",
            ],
            ascending=[
                False,
                False,
            ],
        ).index,
    )

    # Legend order: Reversed stack order
    classes_for_legend = list(
        reversed(classes_for_stack),
    )

    # 4. Generate the Stacked Bar Chart
    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        ),
    )

    x = np.arange(
        len(years_array),
    )
    width = 0.9
    base = np.zeros(
        len(years_array),
        dtype=float,
    )
    patches_by_class: dict[str, plt.Artist] = {}

    for cls in classes_for_stack:
        if cls not in pivot_scaled.columns:
            continue

        values_cls = pivot_scaled[cls].reindex(
            years_array,
            fill_value=0.0,
        ).values

        bars = ax.bar(
            x,
            values_cls,
            bottom=base,
            width=width,
            label=cls,
            color=color_map.get(cls, "gray"),
        )
        patches_by_class[cls] = bars[0]
        base += values_cls

    # 5. Configure Axes
    ax.set_xticks(
        x,
    )
    ax.set_xticklabels(
        years_array,
    )

    # Adaptive rotation for X-axis labels
    n_labels = len(years_array)

    # 0 degrees if <= 6 labels, else 90 degrees
    if n_labels <= 6:
        rotation = 0
        ha = "center"
    else:
        rotation = 90
        ha = "center"

    plt.setp(
        ax.get_xticklabels(),
        rotation=rotation,
        ha=ha,
    )

    ax.tick_params(
        axis="both",
        labelsize=12,
    )
    ax.set_ylabel(
        y_label,
        fontsize=18,
    )
    ax.set_xlabel(
        "Time points",
        fontsize=18,
    )
    ax.set_title(
        "Class Area at Time Points",
        fontsize=20,
    )

    # Y-axis limit and formatting
    y_max_scaled = base.max() * 1.1 if base.max() > 0 else 1.0
    ax.set_ylim(
        0,
        y_max_scaled,
    )
    ax.yaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=8,
            integer=True,
        ),
    )

    # 6. Add Legend
    handles = [
        patches_by_class[cls]
        for cls in classes_for_legend
        if cls in patches_by_class
    ]
    labels = [
        cls
        for cls in classes_for_legend
        if cls in patches_by_class
    ]

    leg = ax.legend(
        handles,
        labels,
        title="Class",
        title_fontsize=16,
        bbox_to_anchor=(
            1.02,
            0.5,
        ),
        loc="center left",
        frameon=False,
        fontsize=16,
        alignment="left",
    )

    plt.tight_layout()

    # 7. Save Figure
    charts_dir = os.path.join(
        output_dir,
        "charts",
    )

    os.makedirs(
        charts_dir,
        exist_ok=True,
    )

    out_fig = os.path.join(
        charts_dir,
        "chart_pixel_per_class_net_change.png",
    )

    plt.savefig(
        out_fig,
        format="png",
        bbox_inches="tight",
        dpi=300,
    )
    print(f"Chart successfully saved at: {out_fig}")
    plt.show()

###############################################################################
#                                                                             #
#                  5. NUMBER OF CHANGES FUNCTIONS                             #
#                                                                             #
###############################################################################
def export_global_change_frequency_tasks(
    year_list: list[int],
    drive_folder: str,
    scale: int = 300,
    max_pixels: float = 1e13,
) -> list:
    """
    Triggers GEE tasks to count pixels changing in each interval,
    categorized by their total number of changes across the series.

    Parameters
    ----------
    year_list : list[int]
        List of years for indexing.
    drive_folder : str
        Directory to save the CSV files in Drive.
    scale : int, optional
        Spatial resolution in meters. Defaults to 300.
    max_pixels : float, optional
        Maximum pixels for reduceRegion. Defaults to 1e13.

    Returns
    -------
    list
        List of triggered Earth Engine Task objects.
    """
    # 1. Define global bounding box geometry
    global_geom = ee.Geometry.Rectangle(
        [
            -180,
            -90,
            180,
            90,
        ],
        'EPSG:4326',
        False,
    )

    # 2. Extract images for all years
    images = []
    for year in year_list:
        img = ee.ImageCollection(
            utils.GLANCE_COLLECTION_ID,
        ).filterDate(
            f"{year}-01-01",
            f"{year}-12-31",
        ).mosaic().select(
            utils.GLANCE_CLASS_BAND,
        )
        images.append(
            img,
        )

    # 3. Compute change images per interval
    change_images = []
    n_intervals = len(
        year_list,
    ) - 1

    for i in range(
        n_intervals,
    ):
        img_curr = images[
            i
        ]
        img_next = images[
            i + 1
        ]

        # 4. Binary change mapping (1 if changed, 0 otherwise)
        change = img_curr.neq(
            img_next,
        ).rename(
            "change",
        )
        change_images.append(
            change,
        )

    # 5. Compute total changes over the entire time series
    # Summing all boolean change images
    total_changes = ee.ImageCollection(
        change_images,
    ).sum().rename(
        "total_changes",
    )

    tasks_list = []

    # 6. Generate an export task for each interval
    for i in range(
        n_intervals,
    ):
        y_start = year_list[
            i
        ]
        y_end = year_list[
            i + 1
        ]
        interval_label = f"{y_start}_{y_end}"

        # 7. Mask total_changes to only pixels that changed in THIS interval
        interval_change_mask = change_images[
            i
        ]
        masked_total = total_changes.updateMask(
            interval_change_mask,
        )

        # 8. Reduce region to get the frequency histogram
        histogram = masked_total.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=global_geom,
            scale=scale,
            maxPixels=max_pixels,
            tileScale=16,
        )

        # 9. Extract the dictionary and convert it to a FeatureCollection
        counts_dict = ee.Dictionary(
            histogram.get(
                "total_changes",
            ),
        )
        feature = ee.Feature(
            None,
            counts_dict,
        )
        feature_collection = ee.FeatureCollection(
            [
                feature,
            ]
        )

        # 10. Define the export task parameters
        export_name = f"Number_Change_{interval_label}"
        task = ee.batch.Export.table.toDrive(
            collection=feature_collection,
            description=export_name,
            folder=drive_folder,
            fileFormat="CSV",
        )

        # 11. Start the task and append it to the list
        task.start()
        print(
            f"Task started: {export_name} (Scale: {scale}m)",
        )
        tasks_list.append(
            task,
        )

    return tasks_list

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_global_change_frequency_bar_chart(
    input_dir: str,
    output_dir: str,
) -> None:
    """
    Create a stacked bar chart showing the sequence of changes per interval
    by compiling multiple GEE-exported CSV files.

    Parameters
    ----------
    input_dir : str
        Directory containing the 'Number_Change_*.csv' files.
    output_dir : str
        Directory to save the resulting figure.

    Returns
    -------
    None
    """
    # 1. Read and compile the GEE CSV files
    search_pattern = os.path.join(
        input_dir,
        "Number_Change_*.csv",
    )
    csv_files = glob.glob(
        search_pattern,
    )

    if not csv_files:
        print(
            f"No Number_Change CSVs found in {input_dir}",
        )
        return

    records = {}
    for file_path in csv_files:
        basename = os.path.basename(
            file_path,
        )
        # Extract interval label from filename (e.g. "Number_Change_2001_2010.csv" -> "2001-2010")
        interval_str = basename.replace(
            "Number_Change_",
            "",
        ).replace(
            ".csv",
            "",
        )

        parts = interval_str.split(
            "_",
        )
        if len(
            parts,
        ) == 2:
            label = f"{parts[0]}-{parts[1]}"
        else:
            label = interval_str

        df_temp = pd.read_csv(
            file_path,
        )

        # 2. Filter numeric columns and remove GEE system columns
        num_cols = df_temp.select_dtypes(
            include=[
                'number',
            ],
        ).columns
        num_cols = [
            c for c in num_cols
            if 'system' not in c
        ]

        if num_cols:
            # The histogram dictionary values are flattened into these columns
            row_data = df_temp[
                num_cols
            ].sum()
            records[
                label
            ] = row_data

    if not records:
        print(
            "No valid data found in CSVs.",
        )
        return

    # 3. Build a consolidated DataFrame
    df = pd.DataFrame.from_dict(
        records,
        orient='index',
    )
    df.fillna(
        0,
        inplace=True,
    )

    # 4. Fix column names (GEE often exports numeric keys as floats like "1.0")
    new_cols = {}
    for c in df.columns:
        try:
            int_c = int(
                float(
                    c,
                ),
            )
            new_cols[
                c
            ] = str(
                int_c,
            )
        except ValueError:
            new_cols[
                c
            ] = str(
                c,
            )

    df.rename(
        columns=new_cols,
        inplace=True,
    )

    # 5. Sort columns numerically
    sorted_cols = sorted(
        df.columns,
        key=lambda x: int(x) if x.isdigit() else float('inf')
    )
    df = df[
        sorted_cols
    ]

    # 6. Sort index chronologically
    df.sort_index(
        inplace=True,
    )

    # 7. Determine Unit Scaling
    max_val = df.sum(
        axis=1,
    ).max()

    if max_val >= 1_000_000_000_000:
        factor = 1_000_000_000_000.0
        suffix = " (trillion pixels)"
    elif max_val >= 1_000_000_000:
        factor = 1_000_000_000.0
        suffix = " (billion pixels)"
    elif max_val >= 1_000_000:
        factor = 1_000_000.0
        suffix = " (million pixels)"
    elif max_val >= 1_000:
        factor = 1_000.0
        suffix = " (thousand pixels)"
    else:
        factor = 1.0
        suffix = ""

    df_scaled = df / factor

    # 8. Setup Figure and Colors
    fig, ax = plt.subplots(
        figsize=(
            14,
            6,
        ),
    )

    n_cols = len(
        df.columns,
    )

    cmap = plt.cm.viridis_r

    if n_cols > 1:
        colors = [
            cmap(
                i / (
                    n_cols - 1
                ),
            )
            for i in range(
                n_cols,
            )
        ]
    else:
        colors = [
            cmap(
                0.5,
            ),
        ]

    # 9. Plot Stacked Bars
    bottom = pd.Series(
        0.0,
        index=df_scaled.index,
    )

    for i, col in reversed(
        list(
            enumerate(
                df.columns,
            )
        )
    ):
        vals = df_scaled[
            col
        ]

        if vals.sum() == 0:
            label_txt = "_nolegend_"
        else:
            label_txt = f"{col}"

        ax.bar(
            df_scaled.index,
            vals,
            bottom=bottom,
            label=label_txt,
            color=colors[
                i
            ],
            edgecolor="none",
            linewidth=0.5,
            width=0.9,
        )
        bottom += vals

    # 10. Formatting Axes and Labels
    y_label_text = f"Change{suffix}"

    ax.set_ylabel(
        y_label_text,
        fontsize=18,
    )

    ax.yaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=5,
            integer=True,
        ),
    )

    ax.set_title(
        "Number of Changes during Time Intervals",
        fontsize=20,
        pad=15,
    )

    # 11. X-Axis labels
    labels = ax.get_xticklabels()
    n_labels = len(
        labels,
    )

    if n_labels <= 6:
        rotation = 0
        ha = "center"
    elif n_labels <= 12:
        rotation = 45
        ha = "right"
    else:
        rotation = 90
        ha = "center"

    plt.setp(
        labels,
        rotation=rotation,
        ha=ha,
        fontsize=18,
    )

    ax.tick_params(
        axis="y",
        rotation=0,
        labelsize=18,
    )

    # 12. Legend
    handles, labels = ax.get_legend_handles_labels()

    leg = ax.legend(
        handles[::-1],
        labels[::-1],
        title="Changes",
        title_fontsize=16,
        bbox_to_anchor=(
            1.02,
            0.5,
        ),
        loc="center left",
        frameon=False,
        fontsize=16,
    )

    for patch in leg.get_patches():
        patch.set_linewidth(
            0,
        )

    plt.tight_layout()

    # 13. Save Figure
    charts_dir = os.path.join(
        output_dir,
        "charts",
    )
    os.makedirs(
        charts_dir,
        exist_ok=True,
    )

    output_fig = os.path.join(
        charts_dir,
        "chart_number_change_time_interval.png",
    )

    plt.savefig(
        output_fig,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print(
        f"Chart saved to: {output_fig}",
    )

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

def plot_number_of_changes_distribution(
    input_dir: str,
    output_dir: str,
) -> None:
    """
    Generate a single stacked bar chart of the number of changes distribution.

    This function calculates the percentage of unique pixels that underwent
    1, 2, 3, or N total changes relative to the ENTIRE valid study area,
    using GEE-exported CSVs.

    Parameters
    ----------
    input_dir : str
        Directory containing both 'Pixel_Counts_LULC_*.csv' (for total area)
        and 'Number_Change_*.csv' files.
    output_dir : str
        Directory to save the resulting figure.

    Returns
    -------
    None
    """
    # 1. Read the pixel counts to find the TRUE total valid study area
    pixel_count_files = glob.glob(
        os.path.join(
            input_dir,
            "Pixel_Counts_LULC_*.csv",
        )
    )
    if not pixel_count_files:
        print(
            "No Pixel_Counts_LULC CSVs found to calculate total study area.",
        )
        return

    # Read the first pixel count file to get total valid pixels
    df_pixels = pd.read_csv(
        pixel_count_files[0],
    )
    num_cols_pixels = [
        c for c in df_pixels.select_dtypes(
            include=['number'],
        ).columns
        if 'system' not in c
    ]
    total_study_area_pixels = df_pixels[
        num_cols_pixels
    ].sum().sum()

    # 2. Read and compile the Number_Change CSVs
    change_files = glob.glob(
        os.path.join(
            input_dir,
            "Number_Change_*.csv",
        )
    )
    if not change_files:
        print(
            "No Number_Change CSVs found.",
        )
        return

    records = {}
    for file_path in change_files:
        df_temp = pd.read_csv(
            file_path,
        )
        num_cols = [
            c for c in df_temp.select_dtypes(
                include=['number'],
            ).columns
            if 'system' not in c
        ]
        if num_cols:
            records[
                os.path.basename(file_path)
            ] = df_temp[
                num_cols
            ].sum()

    if not records:
        print(
            "No valid data found in Number_Change CSVs.",
        )
        return

    df_compiled = pd.DataFrame.from_dict(
        records,
        orient='index',
    ).fillna(
        0,
    )

    # Fix column names to string integers
    new_cols = {}
    for c in df_compiled.columns:
        try:
            new_cols[
                c
            ] = str(
                int(float(c))
            )
        except ValueError:
            new_cols[
                c
            ] = str(
                c,
            )
    df_compiled.rename(
        columns=new_cols,
        inplace=True,
    )

    # 3. Calculate the true number of unique pixels per change category
    unique_pixels_per_change = {}

    for col_name in df_compiled.columns:
        if not col_name.isdigit():
            continue

        n_changes = int(
            col_name,
        )
        total_transitions = df_compiled[
            col_name
        ].sum()

        if n_changes > 0:
            unique_pixels = total_transitions / n_changes
        else:
            unique_pixels = 0

        unique_pixels_per_change[
            n_changes
        ] = unique_pixels

    # 4. Calculate percentages relative to the entire study area
    percentages = {}
    for n_changes, count in unique_pixels_per_change.items():
        if total_study_area_pixels > 0:
            pct = (count / total_study_area_pixels) * 100.0
        else:
            pct = 0.0

        percentages[
            n_changes
        ] = pct

    # 5. Setup Colors
    active_changes = [
        k for k, v in percentages.items()
        if v > 0
    ]

    if not active_changes:
        active_changes = list(
            percentages.keys(),
        )

    n_colors = len(
        active_changes,
    )
    cmap = plt.cm.viridis_r

    sorted_changes_desc = sorted(
        active_changes,
        reverse=True,
    )

    colors = {
        n: cmap(
            i / (n_colors - 1)
        )
        if n_colors > 1
        else cmap(
            0.5,
        )
        for i, n in enumerate(
            sorted(
                active_changes,
            )
        )
    }

    # 6. Create the Figure
    fig, ax = plt.subplots(
        figsize=(
            6,
            6,
        ),
    )

    bottom = 0.0
    for n_change in sorted_changes_desc:
        val = percentages[
            n_change
        ]
        if val > 0:
            ax.bar(
                0,
                val,
                bottom=bottom,
                color=colors[
                    n_change
                ],
                width=0.4,
                edgecolor="none",
            )
            bottom += val

    # 7. Formatting the Axes
    ax.set_ylabel(
        "Change (% of study area)",
        fontsize=16,
    )

    ax.set_title(
        "Number of Changes Overall",
        fontsize=18,
        pad=15,
    )

    for spine in [
        "top",
        "right",
        "bottom",
        "left",
    ]:
        ax.spines[
            spine
        ].set_visible(
            True,
        )
        ax.spines[
            spine
        ].set_color(
            "black",
        )
        ax.spines[
            spine
        ].set_linewidth(
            0.5,
        )

    ax.tick_params(
        axis="y",
        which="major",
        labelsize=18,
    )

    ax.set_xticks(
        [],
    )

    # Note: Keep Y-axis limit at 105 so the scale is always absolute
    max_y = bottom * 1.05 if bottom > 0 else 1.0

    ax.set_ylim(
        0,
        max_y,
    )

    # Define the number of bins
    ax.yaxis.set_major_locator(
        mticker.MaxNLocator(
            integer=True,
            nbins=10,
        ),
    )

    # 8. Legend
    legend_elements = []

    for n in sorted(active_changes):
        legend_elements.append(
            Patch(
                facecolor=colors[n],
                label=str(n),
            )
        )

    ax.legend(
        handles=legend_elements,
        title="Changes",
        title_fontsize=16,
        loc="center left",
        bbox_to_anchor=(
            1.05,
            0.5,
        ),
        fontsize=14,
        frameon=False,
    )

    # Force the main plotting box to always occupy the exact same spatial coordinates in the figure
    fig.subplots_adjust(
        left=0.15,
        right=0.75,
        bottom=0.1,
        top=0.9,
    )

    # 9. Save and show the figure
    charts_dir = os.path.join(
        output_dir,
        "charts",
    )
    os.makedirs(
        charts_dir,
        exist_ok=True,
    )

    out_fig_path = os.path.join(
        charts_dir,
        "chart_number_changes_percentage_overall.png",
    )

    plt.savefig(
        out_fig_path,
        dpi=300,
        bbox_inches="tight",
        format="png",
    )
    plt.show()
    print(
        f"Chart saved to: {out_fig_path}",
    )

###############################################################################
#                                                                             #
#                  5. TRANSITION MATRIX                                       #
#                                                                             #
###############################################################################
def calculate_global_transition_matrix(year_list, scale=300):
    """
    Calculates global transition matrices for all consecutive years in a list,
    plus the long-term transition between the first and last years.

    This function performs the calculation entirely on the Earth Engine 
    servers. It creates transition images where values represent pairs 
    of classes (e.g., class 5 to 1 becomes 501) using a 100x multiplier.

    Parameters
    ----------
    year_list : list of int
        List of years to process (e.g., [2001, 2010, 2019]).
    scale : int, optional
        The pixel scale in meters for the global reduction. 
        Note: scale=30 is native but may time out with getInfo(). 
        Defaults to 1000.

    Returns
    -------
    dict of pd.DataFrame
        A dictionary where keys are strings "YYYY_YYYY" and values are 
        pivot tables representing the transition matrix with class names.
    """
    # 1. Initialize core Earth Engine components
    collection = ee.ImageCollection(GLANCE_COLLECTION_ID)
    global_geom = ee.Geometry.Rectangle([-180, -90, 180, 90], 'EPSG:4326', False)
    class_names = {k: v['name'] for k, v in GLANCE_METADATA.items()}
    
    # 2. Define transition pairs (consecutive intervals)
    pairs = [(year_list[i], year_list[i+1]) for i in range(len(year_list) - 1)]
    
    # 3. Add long-term transition (First Year to Last Year)
    if len(year_list) > 2:
        pairs.append((year_list[0], year_list[-1]))
    
    all_results = {}

    # 4. Iterate through each year pair for processing
    for y1, y2 in pairs:
        label = f"{y1}_{y2}"
        
        # 5. Filter images and create mosaics for the pair
        img_start = collection.filterDate(f"{y1}-01-01", f"{y1}-12-31") \
                              .mosaic() \
                              .select(GLANCE_CLASS_BAND)
                              
        img_end = collection.filterDate(f"{y2}-01-01", f"{y2}-12-31") \
                            .mosaic() \
                            .select(GLANCE_CLASS_BAND)

        # 6. Compute transitions: (StartYear * 100) + EndYear
        transition_image = img_start.multiply(100).add(img_end)

        # 7. Perform global reduction to get frequency histogram
        stats = transition_image.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=global_geom,
            scale=scale,
            maxPixels=1e13
        ).get(GLANCE_CLASS_BAND).getInfo()

        # 8. Process histogram data into list of records
        records = []
        for code, count in stats.items():
            code_int = int(float(code))
            id_from = code_int // 100
            id_to = code_int % 100
            
            if id_from in class_names and id_to in class_names:
                records.append({
                    "From": class_names[id_from],
                    "To": class_names[id_to],
                    "Pixels": int(count)
                })

        # 9. Convert records to DataFrame and pivot into Matrix
        if records:
            df_temp = pd.DataFrame(records)
            matrix = df_temp.pivot(index="From", columns="To", values="Pixels").fillna(0)
            all_results[label] = matrix
        
        print(f"Successfully computed: {label}")

    return all_results

###############################################################################
#                                                                             #
#                  5.1 TRANSITION MATRIX                                      #
#                                                                             #
###############################################################################
def export_global_transition_tasks(
    year_list,
    drive_folder="GLANCE_Transitions",
    scale=30
):
    """
    Triggers asynchronous GEE tasks to export global transition matrices.

    Each task computes a frequency histogram for a year pair and saves 
    the result as a CSV file in a specific Google Drive folder.

    Parameters
    ----------
    year_list : list of int
        List of years to process (e.g., [2001, 2010, 2019]).
    drive_folder : str, optional
        Name of the folder in Google Drive to save the CSVs. 
        Defaults to "GLANCE_Transitions".
    scale : int, optional
        Spatial resolution in meters. Use 30 for native resolution. 
        Defaults to 30.

    Returns
    -------
    list of ee.batch.Task
        A list of triggered Earth Engine Task objects for monitoring.
    """
    # 1. Initialize Earth Engine resources using constants from utils.py
    collection = ee.ImageCollection(GLANCE_COLLECTION_ID)
    global_geom = ee.Geometry.Rectangle(
        [-180, -90, 180, 90], 
        'EPSG:4326', 
        False
    )
    
    # 2. Define transition pairs (consecutive intervals)
    pairs = [
        (year_list[i], year_list[i+1]) 
        for i in range(len(year_list) - 1)
    ]
    
    # 3. Add long-term transition pair (First Year to Last Year)
    if len(year_list) > 2:
        pairs.append(
            (year_list[0], year_list[-1])
        )
    
    triggered_tasks = []

    # 4. Iterate through each pair to define and start export tasks
    for y1, y2 in pairs:
        label = f"transition_{y1}_{y2}"
        
        # 5. Filter and mosaic images for the start and end years
        img_start = collection.filterDate(
            f"{y1}-01-01", 
            f"{y1}-12-31"
        ).mosaic().select(GLANCE_CLASS_BAND)
        
        img_end = collection.filterDate(
            f"{y2}-01-01", 
            f"{y2}-12-31"
        ).mosaic().select(GLANCE_CLASS_BAND)

        # 6. Create transition image: (Start * 100) + End
        transition_image = img_start.multiply(100).add(img_end)

        # 7. Reduce the image to a frequency histogram (Table format)
        # Using a Feature to wrap the result for CSV export
        transition_stats = transition_image.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=global_geom,
            scale=scale,
            maxPixels=1e13
        )

        # 8. Create a feature collection with the statistics for export
        feature = ee.Feature(None, transition_stats)
        fc = ee.FeatureCollection([feature])

        # 9. Configure the Batch Export Task to Google Drive
        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description=label,
            folder=drive_folder,
            fileNamePrefix=label,
            fileFormat='CSV'
        )

        # 10. Start the task on the server and store the object
        task.start()
        triggered_tasks.append(task)
        print(f"Task started for {label} (Scale: {scale}m)")

    return triggered_tasks

###############################################################################
#                                                                             #
#                  6. Convert CSV to Transition Matrix                        #
#                                                                             #
###############################################################################

def load_global_transition_matrices(
    drive_path
):
    """
    Loads exported GEE CSVs from Drive and converts them to transition matrices.

    Parameters
    ----------
    drive_path : str
        The full path to the Google Drive folder containing the CSV files.

    Returns
    -------
    dict of pd.DataFrame
        A dictionary where keys are 'YYYY_YYYY' and values are pivot matrices.
    """

    # 1. Identify all CSV files in the specified directory
    search_pattern = os.path.join(drive_path, "*.csv")
    file_list = glob.glob(search_pattern)
    
    # 2. Create a mapping of class IDs to names from metadata
    class_names = {
        k: v['name'] 
        for k, v in GLANCE_METADATA.items()
    }
    
    all_matrices = {}

    # 3. Iterate through each file to reconstruct the matrix
    for filepath in file_list:
        df_raw = pd.read_csv(filepath)
        
        # 4. Extract the transition label from the filename
        filename = os.path.basename(filepath)
        label = filename.replace(".csv", "").replace("transition_", "")
        
        # 5. Parse the histogram string from the 'LC' column
        # GEE exports the dictionary as a string in the first row
        if 'LC' not in df_raw.columns:
            continue
            
        # Clean the string and parse the {key=value} format
        hist_str = df_raw['LC'].iloc[0]
        # Remove braces and split by comma-space
        clean_str = hist_str.strip('{}')
        pairs = clean_str.split(', ')
        hist_data = {}
        for pair in pairs:
            if '=' in pair:
                k, v = pair.split('=')
                hist_data[k] = float(v) # Fixes the parsing for GEE CSV format

        records = []
        
        # 6. Decode transition codes into 'From' and 'To' classes
        for code, count in hist_data.items():
            code_int = int(float(code))
            id_from = code_int // 100
            id_to = code_int % 100

            # 7. Filter records using valid metadata classes
            if id_from in class_names and id_to in class_names:
                records.append({
                    "From": class_names[id_from],
                    "To": class_names[id_to],
                    "Pixels": int(count)
                })

        # 8. Pivot the records into a formal transition matrix
        if records:
            df_temp = pd.DataFrame(records)
            matrix = df_temp.pivot(
                index="From", 
                columns="To", 
                values="Pixels"
            ).fillna(0)

            # 9. Store the resulting DataFrame in the output dictionary
            all_matrices[label] = matrix
            
    return all_matrices

def convert_matrices_to_area(
    matrices_dict,
    pixel_size=30
):
    """
    Converts pixel counts in transition matrices to area in square kilometers.

    Parameters
    ----------
    matrices_dict : dict of pd.DataFrame
        A dictionary where values are transition matrices in pixel counts.
    pixel_size : int, optional
        The edge length of a single pixel in meters. Defaults to 30.

    Returns
    -------
    dict of pd.DataFrame
        A dictionary of transition matrices with values in km^2.
    """
    # 1. Calculate the conversion factor from pixels to km^2
    # Area of one pixel in m^2 = pixel_size * pixel_size
    # Conversion to km^2 = m^2 / 1,000,000
    conversion_factor = (pixel_size ** 2) / 1000000
    
    area_matrices = {}

    # 2. Iterate through the dictionary of matrices
    for label, matrix in matrices_dict.items():
        # 3. Multiply the entire DataFrame by the conversion factor
        # Pandas handles the element-wise multiplication automatically
        area_matrix = matrix * conversion_factor
        
        # 4. Round the results to two decimal places for readability
        area_matrices[label] = area_matrix.round(2)
        
    return area_matrices

def save_area_matrices_to_csv(
    area_matrices,
    output_dir
):
    """
    Saves a dictionary of area matrices to individual CSV files.

    Parameters
    ----------
    area_matrices : dict of pd.DataFrame
        Dictionary containing the transition matrices in km^2.
    output_dir : str
        The directory path where the CSV files will be saved.

    Returns
    -------
    list of str
        A list of file paths to the saved CSV files.
    """
    # 1. Check if the output directory exists and create it if necessary
    if not os.path.exists(output_dir):
        os.makedirs(
            output_dir, 
            exist_ok=True
        )
    
    saved_files = []

    # 2. Iterate through the dictionary to process each transition matrix
    for label, matrix in area_matrices.items():
        # 3. Construct the specific filename for the km2 results
        filename = f"transition_matrix_km2_{label}.csv"
        filepath = os.path.join(
            output_dir, 
            filename
        )
        
        # 4. Save the DataFrame to CSV including the class names in the index
        matrix.to_csv(
            filepath,
            index=True
        )
        
        saved_files.append(filepath)
        print(f"Successfully saved: {filename}")
        
    return saved_files

###############################################################################
#                                                                             #
#                  7. Compute Sum matrix                                      #
#                                                                             #
###############################################################################

def compute_sum_matrix(
    input_dir: str,
    output_path: str,
) -> pd.DataFrame:
    """
    Compute the SUM transition matrix by aggregating all annual intervals.

    Parameters
    ----------
    input_dir : str
        Path to the directory containing annual km2 CSV files.
    output_path : str
        Full path (including filename) to save the resulting SUM matrix.

    Returns
    -------
    pd.DataFrame
        The aggregated SUM transition matrix.
    """
    # 1. List all annual transition files (e.g., 2001_2002, 2002_2003...)
    # Matches the pattern: transition_matrix_km2_YYYY_YYYY.csv
    file_pattern = os.path.join(
        input_dir,
        "transition_matrix_km2_????_????.csv",
    )
    all_files = glob.glob(file_pattern)

    if not all_files:
        raise FileNotFoundError(
            f"No annual km2 matrices found in {input_dir}",
        )

    # 2. Sort files to ensure chronological order (optional, but good practice)
    all_files.sort()

    df_sum = None

    # 3. Iterate and aggregate
    for file_path in all_files:
        # Load current annual matrix
        df_annual = pd.read_csv(
            file_path,
            index_col=0,
        )

        if df_sum is None:
            # Initialize with the first matrix
            df_sum = df_annual.copy()
        else:
            # Sum values cell by cell
            df_sum = df_sum.add(
                df_annual,
                fill_value=0.0,
            )

    # 4. Save the consolidated SUM matrix
    if df_sum is not None:
        df_sum.to_csv(output_path)
        print(f"SUM matrix successfully saved to: {output_path}")

    return df_sum

###############################################################################
#                                                                             #
#                  8. Compute Exchange and Shift                              #
#                                                                             #
###############################################################################
def compute_and_save_components(
    df_sum: pd.DataFrame,
    df_ext: pd.DataFrame,
    output_dir: str,
    period_label: str = "2001-2019",
) -> None:
    """
    Decompose Sum and Extent matrices into change components.

    Logic:
    1. Allocation: Derived from Extent matrix (Aggregate level).
    2. Alternation: Derived from (Sum - Extent) (Trajectory level).
    Note: Shift components can be negative based on algebraic remainder.

    Parameters
    ----------
    df_sum : pd.DataFrame
        Aggregated transition matrix (Sum of annual intervals).
    df_ext : pd.DataFrame
        Direct transition matrix (Start year vs End year).
    output_dir : str
        Directory path to save the resulting CSV files.
    period_label : str, optional
        Year range label for filename, by default "2001-2019".

    Returns
    -------
    None
    """
    def _get_exchange_and_shift(
        matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decompose matrix into positive Exchange and signed Shift."""
        m_calc = matrix.copy()
        np.fill_diagonal(m_calc, 0.0)
        
        # Exchange captures symmetrical swaps (always positive magnitude)
        exchange = np.maximum(0, np.minimum(m_calc, m_calc.T))
        
        # Shift is the signed algebraic remainder
        shift = m_calc - exchange
        
        return exchange, shift

    # 1. Align and Sort Matrices based on GLANCE_METADATA order
    name_to_id = {v['name']: k for k, v in GLANCE_METADATA.items()}
    
    def _sort_key(label):
        if label in name_to_id:
            return (0, name_to_id[label])
        try:
            return (0, int(label))
        except (ValueError, TypeError):
            return (1, str(label))

    all_labels = sorted(
        list(set(df_sum.index).union(df_sum.columns)),
        key=_sort_key
    )
    
    df_s = df_sum.reindex(index=all_labels, columns=all_labels).fillna(0.0)
    df_e = df_ext.reindex(index=all_labels, columns=all_labels).fillna(0.0)

    # 2. Calculate Components
    # Allocation: Based on direct Extent matrix
    alloc_exc, alloc_shift = _get_exchange_and_shift(df_e.values)

    # Alternation: Sum - Extent (Removes np.maximum to allow negative shifts)
    alternation_raw = df_s.values - df_e.values
    alt_exc, alt_shift = _get_exchange_and_shift(alternation_raw)

    # 3. Export to CSV
    components = {
        "sum": df_s.values,
        "extent": df_e.values,
        "allocation_exchange": alloc_exc,
        "allocation_shift": alloc_shift,
        "alternation_exchange": alt_exc,
        "alternation_shift": alt_shift,
    }

    for name, data in components.items():
        df_out = pd.DataFrame(
            data,
            index=all_labels,
            columns=all_labels
        )
        
        fname = f"transition_matrix_{name}_{period_label}.csv"
        path = os.path.join(output_dir, fname)
        df_out.to_csv(path)
        
        print(f"Component saved: {fname}")

###############################################################################
#                                                                             #
#                  9. Reorder Matrices by net change                          #
#                                                                             #
###############################################################################
def reorder_matrices_by_net_change(
    df_sum: pd.DataFrame,
    df_ext: pd.DataFrame,
    df_ext_exc: pd.DataFrame,
    df_ext_shift: pd.DataFrame,
    df_alt_exc: pd.DataFrame,
    df_alt_shift: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Reorder matrices from largest losers to largest gainers using net change.

    Parameters
    ----------
    df_sum : pd.DataFrame
        The aggregated SUM matrix used to calculate the sorting order.
    df_ext, df_ext_exc, df_ext_shift, df_alt_exc, df_alt_shift : pd.DataFrame
        The other component matrices to be reordered.

    Returns
    -------
    tuple[pd.DataFrame, ...]
        All input dataframes reindexed with the same optimized order.
    """
    # 1. Calculate Net Change (Gains - Losses)
    # Diagonal is ignored to focus only on transitions
    m_values = df_sum.values.copy()
    np.fill_diagonal(m_values, 0.0)
    
    gains = m_values.sum(axis=0)
    losses = m_values.sum(axis=1)
    net_change = gains - losses
    
    # 2. Define the sorting order (ascending: losers first)
    net_series = pd.Series(net_change, index=df_sum.index)
    order_labels = net_series.sort_values(ascending=True).index.tolist()

    # 3. Helper to apply the same order to any dataframe
    def _apply_order(df: pd.DataFrame) -> pd.DataFrame:
        return df.reindex(index=order_labels, columns=order_labels).fillna(0.0)

    # 4. Return all matrices reordered
    return (
        _apply_order(df_sum),
        _apply_order(df_ext),
        _apply_order(df_ext_exc),
        _apply_order(df_ext_shift),
        _apply_order(df_alt_exc),
        _apply_order(df_alt_shift),
    )
###############################################################################
#                                                                             #
#                  10. Plot heat maps function                                #
#                                                                             #
###############################################################################
import os
from typing import Iterable, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

def annotate_heatmap(
    ax: plt.Axes,
    M: np.ndarray,
    fontsize: int = 8,
) -> None:
    """
    Annotate a heatmap with integer cell values and adaptive text color.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object where the heatmap is plotted.
    M : np.ndarray
        The matrix containing the values to display.
    fontsize : int, optional
        The font size of the annotations, by default 8.
    """
    if M.size == 0:
        return

    M_off = M.copy()
    np.fill_diagonal(M_off, np.nan)
    data_off = M_off[np.isfinite(M_off)]

    has_pos = np.any(data_off > 0)
    has_neg = np.any(data_off < 0)

    max_pos = float(np.nanmax(data_off[data_off > 0])) if has_pos else 0.0
    min_neg = float(np.nanmin(data_off[data_off < 0])) if has_neg else 0.0

    thresh_pos = 0.5 * max_pos if has_pos else np.inf
    thresh_neg = 0.5 * min_neg if has_neg else -np.inf

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            # Skip diagonal annotation as per reference
            if i == j:
                continue

            v = float(M[i, j])
            txt = f"{int(round(v))}"

            # Adaptive color logic
            if (has_pos and v >= thresh_pos) or (has_neg and v <= thresh_neg):
                color = "white"
            else:
                color = "black"

            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=color,
                clip_on=True,
            )

def _unit_label(
    suffix: str,
    base_label: str = "number of pixels",
) -> str:
    """
    Build a descriptive label for the colorbar based on a unit suffix.

    Parameters
    ----------
    suffix : str
        The suffix for the unit (e.g., 'k', 'M').
    base_label : str, optional
        The base label text, by default "number of pixels".

    Returns
    -------
    str
        The formatted unit label.
    """
    mapping = {
        "": base_label,
        "k": "thousand pixels",
        "M": "million pixels",
        "B": "billion pixels",
        "T": "trillions pixels",
    }
    return mapping.get(
        suffix,
        f"{base_label} ({suffix})",
    )


def _unit_formatter(
    factor: float,
    suffix: str,
    decimals: int = 1,
) -> mticker.FuncFormatter:
    """
    Build a tick formatter that scales values and appends a suffix.

    Parameters
    ----------
    factor : float
        The factor to divide the values by.
    suffix : str
        The string suffix to append.
    decimals : int, optional
        Number of decimal places, by default 1.

    Returns
    -------
    mticker.FuncFormatter
        A formatter function for the plot ticks.
    """
    fmt = f"{{:.{decimals}f}}{suffix}"

    def _fmt(x: float, pos: int) -> str:
        return fmt.format(x / factor)

    return mticker.FuncFormatter(_fmt)

def label_id_to_name(
    labels: Iterable[str],
    class_labels_dict: dict,
) -> list[str]:
    """
    Map class ID strings to human-readable names using class_labels_dict.

    Parameters
    ----------
    labels : Iterable[str]
        Class IDs as strings (e.g. ["1", "2", "3"]).
    class_labels_dict : dict
        Dictionary containing metadata for each class ID.

    Returns
    -------
    list[str]
        List of class names mapped from IDs.
    """
    id_to_name = {
        int(k): v.get(
            "rename",
            v.get(
                "name",
                str(k),
            ),
        )
        for k, v in class_labels_dict.items()
    }

    names: list[str] = []
    for lab in labels:
        try:
            cid = int(str(lab))
            names.append(
                id_to_name.get(
                    cid,
                    str(lab),
                ),
            )
        except Exception:
            names.append(str(lab))

    return names

def plot_heatmap(
    df: pd.DataFrame,
    title: str,
    class_labels_dict: dict,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "YlOrRd",
    vmin: float = 0.0,
    vmax: Optional[float] = None,
    rotate_xticks_deg: int = 90,
    cbar_label: str = "Number of pixels",
    annotate: bool = True,
    cell_size_inch: float = 0.8,
    ann_fontsize: int = 8,
    cbar_fraction: float = 0.025,
    cbar_pad: float = 0.02,
    tick_fontsize_x: Optional[int] = None,
    tick_fontsize_y: Optional[int] = None,
    axis_label_fontsize: Optional[int] = None,
    title_fontsize: Optional[int] = None,
) -> None:
    """
    Plot a square matrix as a heatmap with adaptive integer colorbar.

    Parameters
    ----------
    df : pd.DataFrame
        Square dataframe to plot.
    title : str
        Plot title.
    save_path : str, optional
        Path to save the PNG image.
    figsize : tuple, optional
        Figure size in inches.
    cmap : str, optional
        Colormap name, by default "YlOrRd".
    vmin : float, optional
        Minimum value for scaling, by default 0.0.
    vmax : float, optional
        Maximum value for scaling.
    rotate_xticks_deg : int, optional
        X-tick rotation, by default 90.
    cbar_label : str, optional
        Label for the colorbar.
    annotate : bool, optional
        Whether to annotate cells.
    cell_size_inch : float, optional
        Size of each cell in inches.
    ann_fontsize : int, optional
        Annotation font size.
    cbar_fraction : float, optional
        Colorbar fraction.
    cbar_pad : float, optional
        Colorbar padding.
    tick_fontsize_x : int, optional
        X-tick font size. Required.
    tick_fontsize_y : int, optional
        Y-tick font size. Required.
    axis_label_fontsize : int, optional
        Axis label font size.
    title_fontsize : int, optional
        Title font size.
    """
    if tick_fontsize_x is None or tick_fontsize_y is None:
        raise ValueError("Set `tick_fontsize_x` and `tick_fontsize_y` explicitly.")

    axis_label_fontsize = axis_label_fontsize or 12
    title_fontsize = title_fontsize or 14

    labels = list(df.index)
    matrix_values = df.values.astype(float)

    # Scale Logic
    matrix_scale = matrix_values.copy()
    np.fill_diagonal(matrix_scale, 0.0)
    finite_vals = matrix_scale[np.isfinite(matrix_scale)]

    if finite_vals.size == 0:
        has_negative, vmin_eff, vmax_eff = False, 0.0, 1.0
    else:
        has_negative = float(np.nanmin(finite_vals)) < 0.0
        vmin_eff = float(np.nanmin(finite_vals)) if has_negative else vmin
        vmax_eff = float(np.nanmax(finite_vals)) if vmax is None else float(vmax)
        if vmin_eff == vmax_eff:
            vmax_eff += 1.0

    # Layout Initialization
    nrows, ncols = df.shape
    figsize = figsize or (cell_size_inch * ncols, cell_size_inch * nrows)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    cmap_neg = mcolors.LinearSegmentedColormap.from_list(
        "CustomBlues", ["#08306b", "#b3e0ff"]
    )

    # Plotting Layers
    if has_negative:
        m_pos = np.ma.masked_less_equal(matrix_values, 0.0)
        ax.imshow(m_pos, aspect="equal", cmap=plt.cm.YlOrRd, 
                  norm=mcolors.Normalize(vmin=0.0, vmax=vmax_eff))
        m_neg = np.ma.masked_where(matrix_values >= 0.0, matrix_values)
        ax.imshow(m_neg, aspect="equal", cmap=cmap_neg, 
                  norm=mcolors.Normalize(vmin=vmin_eff, vmax=0.0))
    else:
        m_pos = np.ma.masked_equal(matrix_values, 0.0)
        ax.imshow(m_pos, aspect="equal", cmap=plt.cm.YlOrRd, 
                  norm=mcolors.Normalize(vmin=vmin_eff, vmax=vmax_eff))

    # Black Diagonal Overlay
    diag_mask = np.eye(nrows, dtype=bool)
    m_diag = np.ma.masked_where(~diag_mask, np.ones_like(matrix_values))
    ax.imshow(m_diag, aspect="equal", cmap=mcolors.ListedColormap(["black"]))

    # Labels and Titles
    tick_names = label_id_to_name(
        labels=labels,
        class_labels_dict=GLANCE_METADATA,
    )
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(tick_names, rotation=rotate_xticks_deg, fontsize=tick_fontsize_x)
    ax.set_yticklabels(tick_names, fontsize=tick_fontsize_y)
    ax.set_xlabel("To class", fontsize=axis_label_fontsize)
    ax.set_ylabel("From class", fontsize=axis_label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)

    # Legend Construction
    n_bar = 256
    vals = np.linspace(vmin_eff, vmax_eff, n_bar)
    colors_bar = [
        cmap_neg((v - vmin_eff) / (0.0 - vmin_eff)) if has_negative and v < 0 
        else (1.0, 1.0, 1.0, 1.0) if v == 0 
        else plt.cm.YlOrRd(v / vmax_eff) 
        for v in vals
    ]

    sm = plt.cm.ScalarMappable(
        cmap=mcolors.ListedColormap(colors_bar),
        norm=mcolors.Normalize(vmin=vmin_eff, vmax=vmax_eff)
    )
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, fraction=cbar_fraction, pad=cbar_pad)

    # --- CUSTOM COLORBAR SCALE LOGIC ---
    max_abs = float(np.nanmax(np.abs(finite_vals))) if finite_vals.size > 0 else 0.0
    if max_abs >= 1e6:
        factor, suffix = 1e6, "M"
    elif max_abs >= 1e3:
        factor, suffix = 1e3, "k"
    else:
        factor, suffix = 1.0, ""

    cbar.locator = mticker.MaxNLocator(nbins=5, integer=True, steps=[1, 2, 5, 10])
    cbar.formatter = _unit_formatter(factor=factor, suffix="", decimals=0)
    cbar.set_label(
        _unit_label(suffix, base_label=cbar_label),
        rotation=270,
        labelpad=15,
        fontsize=12
    )
    cbar.update_ticks()

    if annotate:
        annotate_heatmap(ax=ax, M=matrix_values, fontsize=ann_fontsize)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
