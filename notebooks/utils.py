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
            GLANCE_COLLECTION_ID,
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

import ee

def export_global_number_of_changes_raster_task(
    year_list: list[int],
    drive_folder: str,
    scale: int = 300,
) -> ee.batch.Task:
    """
    Compute and save a raster representing the total number of class changes per pixel
    using Google Earth Engine, and export it to Google Drive.

    Parameters
    ----------
    year_list : list[int]
        List of years for indexing the GLanCE collection.
    drive_folder : str
        Directory in Google Drive to save the raster.
    scale : int, optional
        Spatial resolution for the export in meters, by default 300.

    Returns
    -------
    ee.batch.Task
        The submitted Earth Engine task.
    """
    # Load GLanCE image collection using the existing global constant
    collection = ee.ImageCollection(GLANCE_COLLECTION_ID)

    # Generate sequential year pairs (e.g., 2001-2010, 2010-2019...)
    pairs = []
    for i in range(len(year_list) - 1):
        pairs.append((year_list[i], year_list[i + 1]))

    # Initialize an empty image to accumulate changes (starting at 0) with the correct band name
    change_count_img = ee.Image(0).toInt32().rename(GLANCE_CLASS_BAND)

    for y1, y2 in pairs:
        # Robust date filtering using calendarRange (works on system:time_start)
        img1 = collection.filter(ee.Filter.calendarRange(y1, y1, 'year')) \
                         .select(GLANCE_CLASS_BAND).mosaic()
        
        img2 = collection.filter(ee.Filter.calendarRange(y2, y2, 'year')) \
                         .select(GLANCE_CLASS_BAND).mosaic()
        
        # Identify pixels where the class changed (img1 != img2)
        has_changed = img1.neq(img2).rename(GLANCE_CLASS_BAND)
        
        # Accumulate the changes
        change_count_img = change_count_img.add(has_changed)

    # Retrieve global boundaries
    world_geometry = ee.Geometry.BBox(-180, -90, 180, 90)

    # Mask the change_count_img using the extent of the original dataset to exclude oceans/voids
    valid_mask = collection.filter(ee.Filter.calendarRange(year_list[0], year_list[0], 'year')) \
                           .select(GLANCE_CLASS_BAND).mosaic().mask()
                           
    masked_change_count = change_count_img.updateMask(valid_mask)

    # Unmask void/nodata pixels to NODATA_VALUE before export to differentiate from 0 (no change)
    final_export_image = masked_change_count.unmask(NODATA_VALUE).toByte()

    start_year = year_list[0]
    end_year = year_list[-1]
    task_name = f"Number_of_Changes_Raster_{start_year}_{end_year}"

    # Create and submit the export task
    task = ee.batch.Export.image.toDrive(
        image=final_export_image,
        description=task_name,
        folder=drive_folder,
        fileNamePrefix=task_name,
        region=world_geometry,
        scale=scale,
        maxPixels=1e13,
        crs="EPSG:4326",
        fileFormat="GeoTIFF"
    )
    task.start()
    
    print(f"Task started: {task_name} (Scale: {scale}m, NoData: {NODATA_VALUE})")
    return task

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

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the arrow on.
    location : str, optional
        Location of the arrow.
    shadow : bool, optional
        Whether to draw a shadow.
    rotation : dict, optional
        Rotation parameters.
    scale : float, optional
        Scale of the arrow.
    """
    pass

from pyproj import Geod

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pyproj import Transformer, Geod
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
from matplotlib_scalebar.scalebar import ScaleBar
# Make sure north_arrow is imported/defined elsewhere in your utils.py

def compute_display_pixel_size_km(
    raster_path: str,
    downsample_factor: float,
) -> float:
    """
    Compute horizontal resolution in kilometers per displayed pixel.

    Parameters
    ----------
    raster_path : str
        Path to a raster file (or VRT) used to derive spatial extent and CRS.
    downsample_factor : float
        Float factor used to downsample the raster width for display (e.g., 0.05).

    Returns
    -------
    float
        Pixel size in kilometers for the downsampled display grid.
    """

    with rasterio.open(raster_path) as src:
        left, bottom, right, top = src.bounds
        lat_mid_src = (top + bottom) / 2.0

        to_ll = Transformer.from_crs(
            src.crs,
            "EPSG:4326",
            always_xy=True,
        )
        lon_l, lat_mid = to_ll.transform(left, lat_mid_src)
        lon_r, _ = to_ll.transform(right, lat_mid_src)
        lon_c, _ = to_ll.transform((left + right) / 2.0, lat_mid_src)

        geod = Geod(ellps="WGS84")
        
        # Handle global rasters by splitting the distance calculation 
        # to avoid the distance between -180 and 180 resulting in 0.
        if abs(lon_r - lon_l) > 350:
            _, _, dist1 = geod.inv(lon_l, lat_mid, lon_c, lat_mid)
            _, _, dist2 = geod.inv(lon_c, lat_mid, lon_r, lat_mid)
            width_m = dist1 + dist2
        else:
            _, _, width_m = geod.inv(lon_l, lat_mid, lon_r, lat_mid)

        cols_disp = max(1, int(src.width * downsample_factor))

        return (width_m / cols_disp) / 1_000


def plot_number_of_changes_map(
    output_dir: str,
    nodata_val: int,
    raster_filename: str = "Number_of_Changes_Raster",
    scale_factor: float = 0.05,
) -> None:
    """
    Plot the Number of Changes raster map with cartographic elements.

    Parameters
    ----------
    output_dir : str
        Directory containing the exported GEE tiles and where the map will be saved.
    nodata_val : int
        Value representing NoData in the raster to be masked out.
    raster_filename : str, optional
        Prefix of the raster tiles to plot.
    scale_factor : float, optional
        Scale factor to downsample the massive global raster to fit into memory.

    Returns
    -------
    None
    """

    # 1. Locate all raster tiles exported by GEE
    raster_files = glob.glob(os.path.join(output_dir, f"{raster_filename}*.tif"))
    if not raster_files:
        raise FileNotFoundError(
            f"Raster tiles not found for prefix: {raster_filename}. Make sure the GEE export finished."
        )

    # 2. Create a temporary Virtual Raster (VRT) to merge tiles dynamically
    vrt_path = os.path.join(output_dir, "merged_changes.vrt")
    files_str = " ".join([f'"{f}"' for f in raster_files])
    os.system(f"gdalbuildvrt {vrt_path} {files_str}")

    # 3. Calculate pixel size for scale bar
    pixel_size_km = compute_display_pixel_size_km(
        raster_path=vrt_path,
        downsample_factor=scale_factor
    )

    # 4. Read raster and basic metadata with downsampling
    with rasterio.open(vrt_path) as src:
        out_shape = (
            int(src.height * scale_factor),
            int(src.width * scale_factor),
        )
        data = src.read(
            1,
            out_shape=out_shape,
            resampling=rasterio.enums.Resampling.nearest,
        )

        # Force masking using the provided nodata value
        data = np.ma.masked_equal(data, nodata_val)

        src_crs = src.crs
        # Adjust the affine transform for the new downsampled resolution
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[1]),
            (src.height / data.shape[0]),
        )
        height, width = data.shape

    # 5. Figure
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)

    # Data range
    min_val = int(np.ma.min(data))
    max_val = int(np.ma.max(data))

    # 6. Colormap (gray for 0 + jet/viridis for 1..max)
    original_cmap = plt.get_cmap("viridis_r")
    color_list = ["#c0c0c0"] + [
        original_cmap(i) for i in np.linspace(0, 1, max_val)
    ]
    cmap = ListedColormap(color_list)

    # Discrete normalization
    bounds = np.arange(min_val, max_val + 2) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    # 7. Plot raster
    ax.imshow(
        data,
        cmap=cmap,
        interpolation="nearest",
        norm=norm
    )

    # 8. Discrete box legend
    legend_elements = []
    present_values = np.unique(data.compressed())

    for i in range(min_val, max_val + 1):
        if i in present_values:
            legend_elements.append(
                Patch(
                    facecolor=cmap(norm(i)),
                    edgecolor="none",
                    linewidth=0,
                    label=str(i),
                )
            )

    ax.legend(
        handles=legend_elements,
        title="Changes",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=10,
        title_fontsize=12,
        alignment="left",
        handlelength=2.0,
        handleheight=1.5,
    )

    # 9. Cartographic elements
    # If the X-axis is in degrees (max value <= 180), use 1 degree in meters.
    # Otherwise, use the pixel size in meters.
    degree_in_meters = 111320.0
    dx_meters = degree_in_meters if ax.get_xlim()[1] <= 180.5 else (pixel_size_km * 1000)

    # Custom formatter to prevent 'Mm' (Megameters) and force 'km'
    def km_formatter(value, unit):
        if unit == "Mm":
            return f"{int(value * 1000)} km"
        return f"{int(value)} {unit}"

    scalebar = ScaleBar(
        dx=dx_meters,
        units="m",
        length_fraction=0.15,
        location="lower left",
        box_alpha=0.6,
        scale_formatter=km_formatter
    )
    ax.add_artist(scalebar)

    # Note: ensure north_arrow function exists in your utils.py
    try:
        north_arrow(
            ax,
            location="upper right",
            shadow=False,
            rotation={"degrees": 0},
            scale=0.5,
        )
    except NameError:
        print("north_arrow function not found. Skipping north arrow.")

    # 10. Axes styling
    ax.set_title("Number of Changes", fontsize=18, pad=10)
    ax.set_aspect("equal")

    to_latlon = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    def format_lon(x, pos):
        x = np.clip(x, 0, width - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, height // 2, x)
        lon, _ = to_latlon.transform(x_proj, y_proj)
        return f"{lon:.1f}°"

    def format_lat(y, pos):
        y = np.clip(y, 0, height - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, y, width // 2)
        _, lat = to_latlon.transform(x_proj, y_proj)
        return f"{lat:.1f}°"

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

    ax.tick_params(axis="both", which="major", labelsize=10, pad=4)
    plt.setp(ax.get_yticklabels(), rotation=90, va="center")

    # 11. Save and Show
    maps_dir = os.path.join(output_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)
    output_figure_path = os.path.join(maps_dir, "map_number_of_changes.png")

    plt.savefig(
        output_figure_path,
        dpi=300,
        bbox_inches="tight",
        format="png",
        pad_inches=0.5,
    )
    plt.show()
    print(f"Map figure saved successfully to: {output_figure_path}")


###############################################################################
#                                                                             #
#                  5. TRAJECTORY ANALYSIS                                     #
#                                                                             #
###############################################################################
def plot_trajectory_contributions(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Create stacked bar chart for trajectory contributions per interval.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with intervals as index and trajectory IDs (2, 3, 4, 5) as columns.
    output_path : str
        Path to output directory for saving figure.
    """
    # 0. Ensure columns are integers to match logic
    df = df.copy()
    df.columns = df.columns.astype(int)

    # 1. Calculate the maximum value to determine scale factor
    max_val = df.sum(axis=1).max()

    if max_val >= 1_000_000_000_000:
        scale_factor = 1_000_000_000_000
        y_label = "Change (trillion pixels)"
    elif max_val >= 1_000_000_000:
        scale_factor = 1_000_000_000
        y_label = "Change (billion pixels)"
    elif max_val >= 1_000_000:
        scale_factor = 1_000_000
        y_label = "Change (million pixels)"
    elif max_val >= 1_000:
        scale_factor = 1_000
        y_label = "Change (thousand pixels)"
    else:
        scale_factor = 1
        y_label = "Change (pixels)"

    # Apply scaling
    df_scaled = df / scale_factor

    # 2. Define colors and stacking order
    colors = {
        2: "#990033",
        3: "#FDE725",
        4: "#ff9900",
        5: "#000066"
    }

    # Stacking order: 5 (bottom), 4, 3, 2 (top)
    stack_order = [5, 4, 3, 2]

    # 3. Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # 4. Plot stacked bars
    bottom = pd.Series(0.0, index=df_scaled.index)

    for traj_id in stack_order:
        if traj_id in df_scaled.columns:
            values = df_scaled[traj_id]
            ax.bar(
                df_scaled.index,
                values,
                label=f"{traj_id}",
                bottom=bottom,
                color=colors[traj_id],
                edgecolor="none",
                width=0.9,
            )
            bottom += values

    # 5. Customize axes and labels
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_title("Trajectories during Time Intervals", fontsize=20, pad=15)

    # X-Axis formatting: Horizontal labels
    ax.tick_params(axis="x", labelsize=18, rotation=90)

    # Y-Axis formatting (mticker)
    ax.tick_params(axis="y", labelsize=18)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))

    # Spines visible, NO GRID
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)

    # 6. Legend
    handles, labels = ax.get_legend_handles_labels()

    if handles:
        # Reorder handles to match 2, 3, 4, 5
        legend_order_map = {"2": 0, "3": 1, "4": 2, "5": 3}

        # Sort handles based on labels
        sorted_pairs = sorted(
            zip(handles, labels),
            key=lambda x: legend_order_map.get(x[1], 99),
        )
        sorted_handles, sorted_labels = zip(*sorted_pairs)

        ax.legend(
            sorted_handles,
            sorted_labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            title="Trajectory",
            title_fontsize=14,
            alignment="left",
            fontsize=14,
            frameon=False,
        )

    plt.tight_layout()

    # 7. Save figure
    charts_dir = os.path.join(output_path, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    output_fig = os.path.join(charts_dir, "graphic_trajectory_time_interval.png")
    plt.savefig(output_fig, dpi=300, bbox_inches="tight", format="png")
    plt.show()

    print(f"Figure saved to: {output_fig}")

def export_trajectory_intervals_csv_gee(
    year_list: list,
    drive_folder: str,
    scale: int = 300,
) -> ee.batch.Task:
    """
    Compute trajectory interval contributions using GEE and export to CSV.
    Returns pixel counts instead of area.

    Parameters
    ----------
    year_list : list
        List of integer years to process.
    drive_folder : str
        The destination folder in Google Drive.
    scale : int, optional
        The spatial resolution for the export in meters. Default is 300.
        
    Returns
    -------
    ee.batch.Task
        The submitted Earth Engine task object.
    """
    # 1. Build the stack (returns the image stack and band names)
    image_stack, band_names = build_glance_stack(
        year_list=year_list,
        collection_id=GLANCE_COLLECTION_ID,
        band_name=GLANCE_CLASS_BAND,
        nodata_val=NODATA_VALUE
    )
    
    # 2. Calculate trajectory
    trajectory_image = calculate_trajectory_gee(image_stack, band_names)

    # 3. Filter valid trajectories (we only care about 2, 3, 4, 5)
    valid_traj_mask = trajectory_image.gte(2).And(trajectory_image.lte(5))
    trajectory_image = trajectory_image.updateMask(valid_traj_mask)

    # 4. Define a global bounding box for the export
    global_region = ee.Geometry.Polygon(
        [[[-180.0, -90.0], [180.0, -90.0], [180.0, 90.0], [-180.0, 90.0], [-180.0, -90.0]]],
        None, 
        False,
    )

    # 5. Process each interval using GEE server-side mapping
    length = len(year_list)
    indices = ee.List.sequence(0, length - 2)

    def process_interval(idx):
        idx = ee.Number(idx)
        b_names = ee.List(band_names)

        # Get current and next band names
        b1_name = ee.String(b_names.get(idx))
        b2_name = ee.String(b_names.get(idx.add(1)))

        # Select the images for the interval
        img1 = image_stack.select(b1_name)
        img2 = image_stack.select(b2_name)

        # Identify changes between t and t+1
        change_mask = img1.neq(img2)

        # Mask the trajectory image with the changes in this specific interval
        traj_for_interval = trajectory_image.updateMask(change_mask)

        # Compute frequency histogram of trajectory classes (which gives PIXEL COUNTS)
        hist = traj_for_interval.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=global_region,
            scale=scale,
            maxPixels=1e13,
            tileScale=16,
        ).get('trajectory')

        # Handle potential null returns if there are no changes
        hist_dict = ee.Dictionary(ee.Algorithms.If(hist, hist, {}))

        # Format interval label (e.g., "2001-2010")
        y_list = ee.List(year_list)
        y_start = ee.Number(y_list.get(idx)).format('%d')
        y_end = ee.Number(y_list.get(idx.add(1))).format('%d')
        interval_label = y_start.cat('-').cat(y_end)

        # Return as Feature (row for the CSV)
        return ee.Feature(None, {
            'Interval': interval_label,
            '2': ee.Number(hist_dict.get('2', 0)),
            '3': ee.Number(hist_dict.get('3', 0)),
            '4': ee.Number(hist_dict.get('4', 0)),
            '5': ee.Number(hist_dict.get('5', 0)),
        })

    # 6. Map over the intervals
    features = ee.FeatureCollection(indices.map(process_interval))

    # 7. Prepare the CSV Export task
    y_start = str(year_list[0])
    y_end = str(year_list[-1])
    task_desc = f"Trajectory_Contributions_{y_start}_{y_end}"
    
    task = ee.batch.Export.table.toDrive(
        collection=features,
        description=task_desc,
        folder=drive_folder,
        fileNamePrefix=task_desc,
        fileFormat="CSV"
    )

    # 8. Start the task
    task.start()
    print(f"Task '{task_desc}' submitted to Google Earth Engine.")
    
    return task

def export_trajectory_overall_csv_gee(
    year_list: list,
    drive_folder: str,
    scale: int = 300,
) -> ee.batch.Task:
    """
    Compute overall trajectory contributions using GEE and export to CSV.

    Parameters
    ----------
    year_list : list
        List of integer years to process.
    drive_folder : str
        The destination folder in Google Drive.
    scale : int, optional
        The spatial resolution for the export in meters. Default is 300.
        
    Returns
    -------
    ee.batch.Task
        The submitted Earth Engine task object.
    """
    # 1. Build the stack (returns the image stack and band names)
    image_stack, band_names = build_glance_stack(
        year_list=year_list,
        collection_id=GLANCE_COLLECTION_ID,
        band_name=GLANCE_CLASS_BAND,
        nodata_val=NODATA_VALUE
    )
    
    # 2. Calculate overall trajectory
    trajectory_image = calculate_trajectory_gee(image_stack, band_names)

    # 3. Filter valid trajectories (we only care about 2, 3, 4, 5)
    valid_traj_mask = trajectory_image.gte(2).And(trajectory_image.lte(5))
    trajectory_image = trajectory_image.updateMask(valid_traj_mask)

    # 4. Define a global bounding box for the export
    global_region = ee.Geometry.Polygon(
        [[[-180.0, -90.0], [180.0, -90.0], [180.0, 90.0], [-180.0, 90.0], [-180.0, -90.0]]],
        None, 
        False,
    )

    # 5. Compute frequency histogram of trajectory classes
    hist = trajectory_image.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=global_region,
        scale=scale,
        maxPixels=1e13,
        tileScale=16,
    ).get('trajectory')

    # Handle potential null returns if there are no changes
    hist_dict = ee.Dictionary(ee.Algorithms.If(hist, hist, {}))

    # 6. Format into a Feature
    y_start = str(year_list[0])
    y_end = str(year_list[-1])
    period_label = f"{y_start}-{y_end}"

    feature = ee.Feature(None, {
        'Period': period_label,
        '2': ee.Number(hist_dict.get('2', 0)),
        '3': ee.Number(hist_dict.get('3', 0)),
        '4': ee.Number(hist_dict.get('4', 0)),
        '5': ee.Number(hist_dict.get('5', 0)),
    })

    fc = ee.FeatureCollection([feature])

    # 7. Prepare the CSV Export task
    task_desc = f"Trajectory_Overall_{y_start}_{y_end}"
    
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=task_desc,
        folder=drive_folder,
        fileNamePrefix=task_desc,
        fileFormat="CSV"
    )

    # 8. Start the task
    task.start()
    print(f"Task '{task_desc}' submitted to Google Earth Engine.")
    
    return task

def plot_trajectory_map(
    output_dir: str,
    vrt_filename: str = "merged_trajectory.vrt",
    scale_factor: float = 0.05,
    nodata_val: int = 255,
) -> None:
    """
    Plot the Trajectory raster map with cartographic elements,
    using an optimized downsampling approach.

    Parameters
    ----------
    output_dir : str
        Directory containing the exported GEE tiles and where the map will be saved.
    vrt_filename : str, optional
        Name of the VRT file to plot.
    scale_factor : float, optional
        Scale factor to downsample the massive global raster to fit into memory.
    nodata_val : int, optional
        Value representing NoData in the raster to be masked out.
    """

    # 1. Locate VRT
    raster_path = os.path.join(output_dir, vrt_filename)
    if not os.path.exists(raster_path):
        raise FileNotFoundError(f"Raster (VRT) not found: {raster_path}")

    # 2. Calculate pixel size for scale bar
    # Make sure compute_display_pixel_size_km is available in utils.py
    pixel_size_km = compute_display_pixel_size_km(
        raster_path=raster_path,
        downsample_factor=scale_factor
    )

    # 3. Read raster and basic metadata with downsampling
    with rasterio.open(raster_path) as src:
        out_shape = (
            max(1, int(src.height * scale_factor)),
            max(1, int(src.width * scale_factor)),
        )
        data = src.read(
            1,
            out_shape=out_shape,
            resampling=rasterio.enums.Resampling.nearest,
        )

        # Force masking using the provided nodata value
        data = np.ma.masked_equal(data, nodata_val)

        src_crs = src.crs
        # Adjust the affine transform for the new downsampled resolution
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[1]),
            (src.height / data.shape[0]),
        )
        height, width = data.shape

    # 4. Figure
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)

    # 5. Colormap
    cmap = ListedColormap(
        [
            "#d9d9d9",  # Trajectory 1
            "#990033",  # Trajectory 2
            "#FDE725",  # Trajectory 3
            "#ff9900",  # Trajectory 4
            "#000066",  # Trajectory 5
        ]
    )
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = BoundaryNorm(bounds, cmap.N)

    # 6. Plot raster
    ax.imshow(
        data,
        cmap=cmap,
        interpolation="nearest",
        norm=norm
    )

    # 7. Legend
    legend_elements = [
        Patch(
            facecolor="#d9d9d9",
            label="1 All stable",
            edgecolor="black",
            linewidth=0,
        ),
        Patch(
            facecolor="#990033",
            label="2 Extent stable with alternation",
            edgecolor="black",
            linewidth=0,
        ),
        Patch(
            facecolor="#FDE725",
            label="3 Extent change without alternation",
            edgecolor="black",
            linewidth=0,
        ),
        Patch(
            facecolor="#ff9900",
            label=(
                "4 Extent change with alternation\n"
                "   where extent transition matches\n"
                "   a time interval transition"
            ),
            edgecolor="black",
            linewidth=0,
        ),
        Patch(
            facecolor="#000066",
            label=(
                "5 Extent change with alternation shift\n"
                "   where extent transition differs\n"
                "   from all time interval transitions"
            ),
            edgecolor="black",
            linewidth=0,
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=14,
        borderpad=1.2,
        title="Trajectory",
        title_fontsize=14,
        alignment="left",
        handletextpad=0.8,
        columnspacing=2,
        labelspacing=0.8,
        handlelength=2.0,
        handleheight=1.5,
    )

    # 8. Cartographic elements
    scalebar = ScaleBar(
        dx=pixel_size_km,
        units="km",
        length_fraction=0.15,
        location="lower left",
        box_alpha=0.6,
        scale_formatter=lambda value, unit: f"{int(value * 1000)} km" if "M" in unit else f"{int(value)} km"
    )
    ax.add_artist(scalebar)

    try:
        # Ensure north_arrow is also in utils
        north_arrow(
            ax,
            location="upper right",
            shadow=False,
            rotation={"degrees": 0},
            scale=0.5,
        )
    except NameError:
        print("north_arrow function not found. Skipping north arrow.")

    # 9. Axes styling
    ax.set_title("Trajectories", fontsize=18, pad=10)
    ax.set_aspect("equal")

    to_latlon = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    def format_lon(x, pos):
        x = np.clip(x, 0, width - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, height // 2, x)
        lon, _ = to_latlon.transform(x_proj, y_proj)
        return f"{lon:.1f}°"

    def format_lat(y, pos):
        y = np.clip(y, 0, height - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, y, width // 2)
        _, lat = to_latlon.transform(x_proj, y_proj)
        return f"{lat:.1f}°"

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

    ax.tick_params(axis="both", which="major", labelsize=10, pad=4)
    plt.setp(ax.get_yticklabels(), rotation=90, va="center")

    # 10. Save and Show
    maps_dir = os.path.join(output_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)
    output_figure_path = os.path.join(maps_dir, "map_trajectories.png")

    plt.savefig(
        output_figure_path,
        dpi=300,
        bbox_inches="tight",
        format="png",
        pad_inches=0.5,
    )
    plt.show()
    print(f"Map figure saved successfully to: {output_figure_path}")

def export_alternation_shift_task_gee(
    year_list: list,
    drive_folder: str,
    scale: int = 300,
    nodata_val: int = NODATA_VALUE,
) -> ee.batch.Task:
    """
    Compute and export a raster representing the Alternation Shift Component using GEE.
    Calculated as: Total Changes - Quantity (Extension) - Total Exchange.
    """
    print(f"Preparing Alternation Shift GEE Task for {year_list[0]}-{year_list[-1]}...")

    # 1. Fetch all images in the time series
    imgs = []
    for y in year_list:
        img = ee.ImageCollection(GLANCE_COLLECTION_ID).filter(
            ee.Filter.calendarRange(y, y, 'year')
        ).select(GLANCE_CLASS_BAND).mosaic()
        img = img.updateMask(img.neq(nodata_val))
        imgs.append(img)

    # 2. Calculate Total Changes across all intervals
    total_changes = ee.Image(0).toUint8()
    for t in range(len(imgs) - 1):
        change = imgs[t].neq(imgs[t+1])
        total_changes = total_changes.add(change)

    # 3. Calculate Quantity Component (start != end)
    quantity = imgs[0].neq(imgs[-1]).toUint8()

    # 4. Calculate Total Exchange
    classes = list(GLANCE_METADATA.keys())
    total_exchange = ee.Image(0).toUint8()

    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            class_a = classes[i]
            class_b = classes[j]

            count_a_b = ee.Image(0)
            count_b_a = ee.Image(0)

            for t in range(len(imgs) - 1):
                img_t = imgs[t]
                img_t1 = imgs[t+1]

                trans_a_b = img_t.eq(class_a).And(img_t1.eq(class_b))
                count_a_b = count_a_b.add(trans_a_b)

                trans_b_a = img_t.eq(class_b).And(img_t1.eq(class_a))
                count_b_a = count_b_a.add(trans_b_a)

            pair_exchange = count_a_b.min(count_b_a).multiply(2)
            total_exchange = total_exchange.add(pair_exchange)

    # 5. Calculate Alternation Shift (Changes - Quantity - Exchange)
    # .max(0) ensures no negative values if discrepancies arise
    shift = total_changes.subtract(quantity).subtract(total_exchange).max(0).toUint8()

    # 6. Apply NoData and set properties
    shift = shift.unmask(nodata_val).set('system:no_data_value', nodata_val).toUint8()

    # 7. Define global bounding box for the export
    global_region = ee.Geometry.Polygon(
        [[[-180.0, -90.0], [180.0, -90.0], [180.0, 90.0], [-180.0, 90.0], [-180.0, -90.0]]],
        None, False
    )

    # 8. Define and start the Earth Engine export task
    task_desc = f"Alternation_Shift_{year_list[0]}_{year_list[-1]}"
    task = ee.batch.Export.image.toDrive(
        image=shift,
        description=task_desc,
        folder=drive_folder,
        scale=scale,
        region=global_region,
        maxPixels=1e13,
    )

    task.start()
    print(f"Task '{task_desc}' submitted to Google Earth Engine with NoData: {nodata_val}")
    return task
###############################################################################
#                                                                             #
#                  5. TRANSITION MATRIX                                       #
#                                                                             #
###############################################################################
def calculate_trajectory_gee(
    image_stack: ee.Image,
    band_names: list,
) -> ee.Image:
    """
    Classify a single pixel trajectory into five categories based on mathematical logic using GEE.

    Parameters
    ----------
    image_stack : ee.Image
        An ee.Image where each band represents a chronological time step.
    band_names : list
        A list of strings representing the ordered band names in the stack.

    Returns
    -------
    ee.Image
        An ee.Image containing the classified trajectory codes (1 to 5).
    """
    # 1. Extract the start and end images from the stack
    start_img = image_stack.select(band_names[0])
    end_img = image_stack.select(band_names[-1])

    # 2. Check if the start class equals the end class
    start_equals_end = start_img.eq(end_img)

    # 3. Verify if all intermediate values match the start value
    all_match_start = image_stack.eq(start_img).reduce(ee.Reducer.min())

    # 4. Assign Trajectory 1 for completely stable pixels
    traj_1 = start_equals_end.And(all_match_start).multiply(1)

    # 5. Assign Trajectory 2 for stable extent with alternation
    traj_2 = start_equals_end.And(all_match_start.Not()).multiply(2)

    # 6. Identify pixels with extent change
    extent_change = start_equals_end.Not()

    # 7. Initialize images to track direct transitions and path changes
    has_direct_transition = ee.Image(0)
    path_changes = ee.Image(0)

    length = len(band_names)

    # 8. Iterate over the time steps to evaluate transitions
    for i in range(length - 1):
        current_band = image_stack.select(band_names[i])
        next_band = image_stack.select(band_names[i + 1])

        # 9. Check for a direct transition from start to end class
        is_direct = current_band.eq(start_img).And(next_band.eq(end_img))
        has_direct_transition = has_direct_transition.Or(is_direct)

        # 10. Increment path changes when the class changes between steps
        is_change = current_band.neq(next_band)
        path_changes = path_changes.add(is_change)

    # 11. Assign Trajectory 5 for extent change without direct transition
    traj_5 = extent_change.And(has_direct_transition.Not()).multiply(5)

    # 12. Assign Trajectory 3 for extent change without alternation
    traj_3 = extent_change.And(has_direct_transition).And(path_changes.eq(1)).multiply(3)

    # 13. Assign Trajectory 4 for extent change with alternation
    traj_4 = extent_change.And(has_direct_transition).And(path_changes.gt(1)).multiply(4)

    # 14. Combine all trajectory maps into a single output image
    trajectory_image = traj_1.add(traj_2).add(traj_3).add(traj_4).add(traj_5)

    return trajectory_image.rename('trajectory')


def build_glance_stack(
    year_list: list,
    collection_id: str,
    band_name: str,
    nodata_val: int,
) -> tuple:
    """
    Build an Earth Engine image stack from the specified collection,
    masking out NoData values.

    Parameters
    ----------
    year_list : list
        List of integer years to process.
    collection_id : str
        The GEE ImageCollection ID.
    band_name : str
        The band name to select.
    nodata_val : int
        The NoData value to mask out.

    Returns
    -------
    tuple
        A tuple containing the ee.Image stack and the list of band names.
    """
    # 1. Initialize lists for image bands and names
    images = []
    b_names = []

    # 2. Loop through the requested years
    for year in year_list:
        b_name = f"y{year}"
        b_names.append(b_name)

        # 3. Retrieve the image for the specific year
        img = ee.ImageCollection(collection_id).filter(
            ee.Filter.calendarRange(year, year, 'year')
        ).select(band_name).mosaic()

        # 4. Mask out the NoData value
        img = img.updateMask(img.neq(nodata_val)).rename(b_name)

        images.append(img)

    # 5. Combine into a single multi-band image
    stack = ee.Image(images)

    return stack, b_names


def export_trajectory_task_gee(
    year_list: list,
    drive_folder: str,
    scale: int,
    collection_id: str,
    band_name: str,
    nodata_val: int,
) -> ee.batch.Task:
    """
    Generate the trajectory raster and submit an export task to Google Earth Engine.

    Parameters
    ----------
    year_list : list
        List of integer years to process.
    drive_folder : str
        The destination folder in Google Drive.
    scale : int
        The spatial resolution for the export in meters.
    collection_id : str
        The GEE ImageCollection ID.
    band_name : str
        The band name to select.
    nodata_val : int
        The NoData value to mask out.

    Returns
    -------
    ee.batch.Task
        The submitted Earth Engine task object.
    """
    # 1. Build the image stack
    image_stack, band_names = build_glance_stack(
        year_list=year_list,
        collection_id=collection_id,
        band_name=band_name,
        nodata_val=nodata_val,
    )

    # 2. Generate the trajectory classification image
    trajectory_image = calculate_trajectory_gee(
        image_stack,
        band_names,
    )

    # 3. Apply NoData unmasking to match the project's standard
    trajectory_image = trajectory_image.unmask(nodata_val).toUint8()

    # 4. Define a global bounding box for the export
    global_region = ee.Geometry.Polygon(
        [[[-180.0, -90.0], [180.0, -90.0], [180.0, 90.0], [-180.0, 90.0], [-180.0, -90.0]]],
        None,
        False,
    )

    # 5. Define the Earth Engine export task
    task_desc = f"Trajectory_Analysis_{year_list[0]}_{year_list[-1]}"
    task = ee.batch.Export.image.toDrive(
        image=trajectory_image,
        description=task_desc,
        folder=drive_folder,
        scale=scale,
        region=global_region,
        maxPixels=1e13,
    )

    # 6. Start the export task
    task.start()
    print(f"Task '{task_desc}' submitted to Google Earth Engine.")

    return task

def plot_trajectory_contributions(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Create a stacked bar chart for trajectory contributions per interval.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with intervals as index and trajectory IDs (2, 3, 4, 5) as columns.
    output_path : str
        Path to output directory for saving figure.
    """
    # 0. Ensure columns are integers to match logic
    df = df.copy()
    df.columns = df.columns.astype(int)

    # 1. Calculate the maximum value to determine scale factor
    max_val = df.sum(axis=1).max()

    if max_val >= 1_000_000_000_000:
        scale_factor = 1_000_000_000_000
        y_label = "Change (trillion pixels)"
    elif max_val >= 1_000_000_000:
        scale_factor = 1_000_000_000
        y_label = "Change (billion pixels)"
    elif max_val >= 1_000_000:
        scale_factor = 1_000_000
        y_label = "Change (million pixels)"
    elif max_val >= 1_000:
        scale_factor = 1_000
        y_label = "Change (thousand pixels)"
    else:
        scale_factor = 1
        y_label = "Change (pixels)"

    # Apply scaling
    df_scaled = df / scale_factor

    # 2. Define colors and stacking order
    colors = {
        2: "#990033",
        3: "#FDE725",
        4: "#ff9900",
        5: "#000066",
    }

    # Stacking order: 5 (bottom), 4, 3, 2 (top)
    stack_order = [5, 4, 3, 2]

    # 3. Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # 4. Plot stacked bars
    bottom = pd.Series(0.0, index=df_scaled.index)

    for traj_id in stack_order:
        if traj_id in df_scaled.columns:
            values = df_scaled[traj_id]
            ax.bar(
                df_scaled.index,
                values,
                label=f"{traj_id}",
                bottom=bottom,
                color=colors[traj_id],
                edgecolor="none",
                width=0.9,
            )
            bottom += values

    # 5. Customize axes and labels
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_title("Trajectories during Time Intervals", fontsize=20, pad=15)

    # X-Axis formatting: Horizontal labels
    ax.tick_params(axis="x", labelsize=18, rotation=90)

    # Y-Axis formatting (mticker)
    ax.tick_params(axis="y", labelsize=18)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))

    # Spines visible, NO GRID
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)

    # 6. Legend
    handles, labels = ax.get_legend_handles_labels()

    if handles:
        # Reorder handles to match 2, 3, 4, 5
        legend_order_map = {"2": 0, "3": 1, "4": 2, "5": 3}

        # Sort handles based on labels
        sorted_pairs = sorted(
            zip(handles, labels),
            key=lambda x: legend_order_map.get(x[1], 99),
        )
        sorted_handles, sorted_labels = zip(*sorted_pairs)

        ax.legend(
            sorted_handles,
            sorted_labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            title="Trajectory",
            title_fontsize=14,
            alignment="left",
            fontsize=14,
            frameon=False,
        )

    plt.tight_layout()

    # 7. Save figure
    charts_dir = os.path.join(output_path, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    output_fig = os.path.join(charts_dir, "graphic_trajectory_time_interval.png")
    plt.savefig(output_fig, dpi=300, bbox_inches="tight", format="png")
    plt.show()

    print(f"Figure saved to: {output_fig}")

def plot_trajectory_distribution(
    output_path: str,
    raster_prefix: str = "Trajectory_Analysis",
) -> None:
    """
    Generate and save a stacked bar chart of trajectory class distributions.

    Reads multiple tiled rasters using a Virtual Raster (VRT) and processes
    them in chunks to avoid memory issues, calculating percentages.

    Parameters
    ----------
    output_path : str
        Directory path containing the input rasters and for output charts.
    raster_prefix : str, optional
        Prefix of the raster files to be merged into VRT. Default is
        "Trajectory_Analysis".

    Returns
    -------
    None
        The function saves the plot to disk and displays it.

    Raises
    ------
    FileNotFoundError
        If no TIFF files matching the prefix are found.
    """
    # 1. Build Virtual Raster (VRT) from GEE tiles
    tif_pattern = os.path.join(output_path, f"{raster_prefix}*.tif")
    tif_files = glob.glob(tif_pattern)

    if not tif_files:
        raise FileNotFoundError(f"No TIFF files found matching {tif_pattern}")

    vrt_path = os.path.join(output_path, "merged_trajectory.vrt")

    print(f"Building VRT from {len(tif_files)} tiles...")
    vrt_cmd = ["gdalbuildvrt", vrt_path] + tif_files
    subprocess.run(vrt_cmd, check=True)
    print(f"VRT created successfully at: {vrt_path}")

    # 2. Process Raster Data in chunks (block by block)
    data_counts = {}
    print("Processing raster data (this may take a few minutes for global scale)...")
    with rasterio.open(vrt_path) as src:
        nodata = src.nodata
        for ji, window in src.block_windows(1):
            traj_data = src.read(1, window=window)
            valid_data = traj_data[traj_data != nodata]
            if valid_data.size > 0:
                unique_vals, counts = np.unique(valid_data, return_counts=True)
                for val, count in zip(unique_vals, counts):
                    data_counts[val] = data_counts.get(val, 0) + count

    total_pixels = sum(data_counts.values())

    # Calculate percentages for classes 2, 3, 4, and 5
    percentages = {
        i: float((data_counts.get(i, 0) / total_pixels) * 100.0) if total_pixels > 0 else 0.0
        for i in [2, 3, 4, 5]
    }

    ordered_trajs = [5, 4, 3, 2]
    colors = {5: "#000066", 4: "#ff9900", 3: "#FDE724", 2: "#990033"}

    fig, ax = plt.subplots(figsize=(6, 6))

    bottom = 0.0
    for traj in ordered_trajs:
        val = percentages[traj]
        ax.bar(0, val, bottom=bottom, color=colors[traj], width=0.4, edgecolor="none")
        bottom += val

    ax.set_ylabel("Change (% of study area)", fontsize=16)
    ax.set_title("Trajectories Overall", fontsize=18, pad=15)

    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(0.5)

    ax.tick_params(axis="y", which="major", labelsize=18)
    ax.set_xticks([])
    ax.set_ylim(0, bottom * 1.05)

    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    legend_elements = [
        Patch(facecolor=colors[2], label="2"),
        Patch(facecolor=colors[3], label="3"),
        Patch(facecolor=colors[4], label="4"),
        Patch(facecolor=colors[5], label="5"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        title="Trajectory",
        title_fontsize=14,
        alignment="left",
        fontsize=14,
        frameon=False,
    )

    fig.subplots_adjust(left=0.15, right=0.75, bottom=0.1, top=0.9)

    charts_dir = os.path.join(output_path, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    out_fig_path = os.path.join(charts_dir, "graphic_trajectory_percentage_overall.png")
    plt.savefig(out_fig_path, dpi=300, bbox_inches="tight", format="png")
    plt.show()
    print(f"Figure saved to: {out_fig_path}")

###############################################################################
#                                                                             #
#                  5.1 CHANGE COMPONENTS                                      #
#                                                                             #
###############################################################################

def export_quantity_component_task_gee(
    year_list: list,
    drive_folder: str,
    scale: int = 300,
    nodata_val: int = 255,
) -> ee.batch.Task:
    """
    Compute and export a raster representing the Quantity Component of change using GEE.

    A pixel's quantity component is 1 when the begin class differs from
    the finish class; otherwise, it is 0.

    Parameters
    ----------
    year_list : list
        List of years representing the timeline.
    drive_folder : str
        Google Drive folder name for exports.
    scale : int, optional
        Spatial resolution in meters, by default 300.
    nodata_val : int, optional
        NoData value to be used for masking, by default 255.

    Returns
    -------
    ee.batch.Task
        The submitted Earth Engine export task.
    """
    # 1. Fetch start and end years
    start_year = year_list[0]
    end_year = year_list[-1]

    # 2. Fetch start and end images directly from the collection
    # Note: GLANCE_COLLECTION_ID and GLANCE_CLASS_BAND must be defined in utils.py
    start_img = ee.ImageCollection(GLANCE_COLLECTION_ID).filter(
        ee.Filter.calendarRange(start_year, start_year, 'year')
    ).select(GLANCE_CLASS_BAND).mosaic()
    start_img = start_img.updateMask(start_img.neq(nodata_val))

    end_img = ee.ImageCollection(GLANCE_COLLECTION_ID).filter(
        ee.Filter.calendarRange(end_year, end_year, 'year')
    ).select(GLANCE_CLASS_BAND).mosaic()
    end_img = end_img.updateMask(end_img.neq(nodata_val))

    # 3. Compute Quantity Component: 1 if start != end, else 0
    quantity_image = start_img.neq(end_img).multiply(1).toUint8()

    # 4. Apply NoData unmasking and set the system:no_data_value property
    quantity_image = quantity_image.unmask(nodata_val)
    quantity_image = quantity_image.set('system:no_data_value', nodata_val)

    # 5. Define a global bounding box for the export
    global_region = ee.Geometry.Polygon(
        [[[-180.0, -90.0], [180.0, -90.0], [180.0, 90.0], [-180.0, 90.0], [-180.0, -90.0]]],
        None, False
    )

    # 6. Define the Earth Engine export task
    task_desc = f"Quantity_Component_{start_year}_{end_year}"
    task = ee.batch.Export.image.toDrive(
        image=quantity_image,
        description=task_desc,
        folder=drive_folder,
        scale=scale,
        region=global_region,
        maxPixels=1e13,
    )

    # 7. Start the export task
    task.start()
    print(f"Task '{task_desc}' submitted to Google Earth Engine with NoData: {nodata_val}")

    return task

def plot_quantity_component_map(
    output_dir: str,
    nodata_val: int,
    raster_filename: str,
    scale_factor: float = 0.05,
) -> None:
    """
    Plot the Quantity Component (Extent Change) raster map with cartographic elements.

    Parameters
    ----------
    output_dir : str
        Directory containing the exported GEE tiles and where the map will be saved.
    nodata_val : int
        Value representing NoData in the raster to be masked out.
    raster_filename : str
        Prefix of the raster tiles to plot.
    scale_factor : float, optional
        Scale factor to downsample the massive global raster to fit into memory.

    Returns
    -------
    None
    """
    # 1. Locate all raster tiles exported by GEE
    raster_files = glob.glob(os.path.join(output_dir, f"{raster_filename}*.tif"))
    if not raster_files:
        raise FileNotFoundError(
            f"Raster tiles not found for prefix: {raster_filename}. Make sure the GEE export finished."
        )

    # 2. Create a temporary Virtual Raster (VRT) to merge tiles dynamically
    vrt_path = os.path.join(output_dir, "merged_quantity.vrt")
    files_str = " ".join([f'"{f}"' for f in raster_files])
    os.system(f"gdalbuildvrt {vrt_path} {files_str}")

    # 3. Calculate pixel size for scale bar
    # Assumes compute_display_pixel_size_km is defined in the same module
    pixel_size_km = compute_display_pixel_size_km(
        raster_path=vrt_path,
        downsample_factor=scale_factor,
    )

    # 4. Read raster and basic metadata with downsampling
    with rasterio.open(vrt_path) as src:
        out_shape = (
            max(1, int(src.height * scale_factor)),
            max(1, int(src.width * scale_factor)),
        )
        data = src.read(
            1,
            out_shape=out_shape,
            resampling=rasterio.enums.Resampling.nearest,
        )

        # Force masking using the provided nodata value
        data = np.ma.masked_equal(data, nodata_val)

        src_crs = src.crs
        # Adjust the affine transform for the new downsampled resolution
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[1]),
            (src.height / data.shape[0]),
        )
        height, width = data.shape

    # 5. Figure
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)

    # 6. Colormap for 0 and 1
    cmap = ListedColormap(["#c0c0c0", "#fde725"])
    bounds = [-0.5, 0.5, 1.5]
    norm = BoundaryNorm(bounds, cmap.N)

    # 7. Plot raster
    ax.imshow(
        data,
        cmap=cmap,
        interpolation="nearest",
        norm=norm,
    )

    # 8. Legend (adapted for 0 and 1 only)
    legend_elements = [
        Patch(
            facecolor="#c0c0c0",
            label="0",
            edgecolor="black",
            linewidth=0,
        ),
        Patch(
            facecolor="#fde725",
            label="1",
            edgecolor="black",
            linewidth=0,
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=14,
        borderpad=1.2,
        title="Change",
        title_fontsize=14,
        alignment="left",
        handletextpad=0.8,
        columnspacing=2,
        labelspacing=0.8,
        handlelength=2.0,
        handleheight=1.5,
    )

    # 9. Cartographic elements
    degree_in_meters = 111320.0
    dx_meters = degree_in_meters if ax.get_xlim()[1] <= 180.5 else (pixel_size_km * 1000)

    def km_formatter(value, unit):
        if unit == "Mm":
            return f"{int(value * 1000)} km"
        return f"{int(value)} {unit}"

    scalebar = ScaleBar(
        dx=dx_meters,
        units="m",
        length_fraction=0.15,
        location="lower left",
        box_alpha=0.6,
        scale_formatter=km_formatter,
    )
    ax.add_artist(scalebar)

    try:
        # Assumes north_arrow is defined in the same module
        north_arrow(
            ax,
            location="upper right",
            shadow=False,
            rotation={"degrees": 0},
            scale=0.5,
        )
    except NameError:
        print("north_arrow function not found. Skipping north arrow.")

    # 10. Axes styling
    ax.set_title("Extent Change", fontsize=18, pad=10)
    ax.set_aspect("equal")

    to_latlon = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    def format_lon(x, pos):
        x = np.clip(x, 0, width - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, height // 2, x)
        lon, _ = to_latlon.transform(x_proj, y_proj)
        return f"{lon:.1f}°"

    def format_lat(y, pos):
        y = np.clip(y, 0, height - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, y, width // 2)
        _, lat = to_latlon.transform(x_proj, y_proj)
        return f"{lat:.1f}°"

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

    ax.tick_params(axis="both", which="major", labelsize=10, pad=4)
    plt.setp(ax.get_yticklabels(), rotation=90, va="center")

    # 11. Save and Show
    maps_dir = os.path.join(output_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)
    output_figure_path = os.path.join(maps_dir, "map_extent_change.png")

    plt.savefig(
        output_figure_path,
        dpi=300,
        bbox_inches="tight",
        format="png",
        pad_inches=0.5,
    )
    plt.show()
    print(f"Map figure saved successfully to: {output_figure_path}")

def export_alternation_exchange_task_gee(
    year_list: list,
    drive_folder: str,
    scale: int = 300,
    nodata_val: int = 255,
) -> ee.batch.Task:
    """
    Compute and export a raster representing the Alternation Exchange Component using GEE.
    This replaces the local block-by-block Numba matrix calculation.

    Parameters
    ----------
    year_list : list
        List of years to process.
    drive_folder : str
        Google Drive folder name for exports.
    scale : int, optional
        Spatial resolution in meters, by default 300.
    nodata_val : int, optional
        NoData value to be used for masking, by default 255.

    Returns
    -------
    ee.batch.Task
        The submitted Earth Engine export task.
    """
    print(f"Preparing Alternation Exchange GEE Task for {year_list[0]}-{year_list[-1]}...")

    # 1. Fetch all images in the time series
    # Note: GLANCE_COLLECTION_ID and GLANCE_CLASS_BAND must be defined in utils.py
    imgs = []
    for y in year_list:
        img = ee.ImageCollection(GLANCE_COLLECTION_ID).filter(
            ee.Filter.calendarRange(y, y, 'year')
        ).select(GLANCE_CLASS_BAND).mosaic()
        img = img.updateMask(img.neq(nodata_val))
        imgs.append(img)

    # 2. Get unique classes from metadata
    # Note: GLANCE_METADATA must be defined in utils.py
    classes = list(GLANCE_METADATA.keys())

    # 3. Accumulate total exchange
    total_exchange = ee.Image(0).toUint8()

    # Loop through all unique pairs of classes to find A->B and B->A exchanges
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            class_a = classes[i]
            class_b = classes[j]

            count_a_b = ee.Image(0)
            count_b_a = ee.Image(0)

            # Sum transitions over time
            for t in range(len(imgs) - 1):
                img_t = imgs[t]
                img_t1 = imgs[t+1]

                # Transition A -> B
                trans_a_b = img_t.eq(class_a).And(img_t1.eq(class_b))
                count_a_b = count_a_b.add(trans_a_b)

                # Transition B -> A
                trans_b_a = img_t.eq(class_b).And(img_t1.eq(class_a))
                count_b_a = count_b_a.add(trans_b_a)

            # Exchange for this pair is min(A->B, B->A)
            # Multiplied by 2 because both directions contribute to the total exchange
            # (matching the original Numba matrix addition logic)
            pair_exchange = count_a_b.min(count_b_a).multiply(2)

            total_exchange = total_exchange.add(pair_exchange)

    # 4. Apply NoData and set properties
    total_exchange = total_exchange.unmask(nodata_val).set('system:no_data_value', nodata_val).toUint8()

    # 5. Define a global bounding box for the export
    global_region = ee.Geometry.Polygon(
        [[[-180.0, -90.0], [180.0, -90.0], [180.0, 90.0], [-180.0, 90.0], [-180.0, -90.0]]],
        None, False
    )

    # 6. Define and start the Earth Engine export task
    task_desc = f"Alternation_Exchange_{year_list[0]}_{year_list[-1]}"
    task = ee.batch.Export.image.toDrive(
        image=total_exchange,
        description=task_desc,
        folder=drive_folder,
        scale=scale,
        region=global_region,
        maxPixels=1e13,
    )

    task.start()
    print(f"Task '{task_desc}' submitted to Google Earth Engine with NoData: {nodata_val}")
    return task

def plot_alternation_exchange_map(
    output_dir: str,
    nodata_val: int,
    raster_filename: str,
    scale_factor: float = 0.05,
) -> None:
    """
    Plot the Alternation Exchange raster map with cartographic elements.

    Parameters
    ----------
    output_dir : str
        Directory containing the exported GEE tiles and where the map will be saved.
    nodata_val : int
        Value representing NoData in the raster to be masked out.
    raster_filename : str
        Prefix of the raster tiles to plot.
    scale_factor : float, optional
        Scale factor to downsample the massive global raster to fit into memory.

    Returns
    -------
    None
    """
    # 1. Locate all raster tiles exported by GEE
    raster_files = glob.glob(os.path.join(output_dir, f"{raster_filename}*.tif"))
    if not raster_files:
        raise FileNotFoundError(
            f"Raster tiles not found for prefix: {raster_filename}. Make sure the GEE export finished."
        )

    # 2. Create a temporary Virtual Raster (VRT) to merge tiles dynamically
    vrt_path = os.path.join(output_dir, "merged_exchange.vrt")
    files_str = " ".join([f'"{f}"' for f in raster_files])
    os.system(f"gdalbuildvrt {vrt_path} {files_str}")

    # 3. Calculate pixel size for scale bar
    pixel_size_km = compute_display_pixel_size_km(
        raster_path=vrt_path,
        downsample_factor=scale_factor,
    )

    # 4. Read raster and basic metadata with downsampling
    with rasterio.open(vrt_path) as src:
        out_shape = (
            max(1, int(src.height * scale_factor)),
            max(1, int(src.width * scale_factor)),
        )
        data = src.read(
            1,
            out_shape=out_shape,
            resampling=rasterio.enums.Resampling.nearest,
        )

        # Force masking using the provided nodata value
        data_masked = np.ma.masked_equal(data, nodata_val)

        src_crs = src.crs
        # Adjust the affine transform for the new downsampled resolution
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[1]),
            (src.height / data.shape[0]),
        )
        height, width = data.shape

    # 5. Figure
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)

    # Determine max value for colormap
    try:
        data_max = int(np.ma.max(data_masked))
    except:
        data_max = 1

    if data_max <= 0:
        data_max = 1

    # 6. Discrete Colormap Configuration
    original_cmap = plt.get_cmap("viridis_r")
    # Define the color for value 0 (Background/Gray)
    colors_list = ["#c0c0c0"] + [
        original_cmap(i) for i in np.linspace(0, 1, data_max)
    ]
    cmap = ListedColormap(colors_list)
    bounds = np.arange(-0.5, data_max + 1.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    # 7. Plot raster
    ax.imshow(
        data_masked,
        cmap=cmap,
        interpolation="nearest",
        norm=norm,
    )

    # 8. Legend Configuration
    legend_elements = []

    # Extract unique values actually present in the masked raster data
    present_values = np.unique(data_masked.compressed())

    for i in range(0, data_max + 1):
        # Append to legend ONLY if the value is present in the map
        if i in present_values:
            legend_elements.append(
                Patch(
                    facecolor=cmap(norm(i)),
                    edgecolor="none",
                    linewidth=0,
                    label=str(i),
                ),
            )

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=12,
        borderpad=1.2,
        title="Exchange",
        title_fontsize=14,
        alignment="left",
        handletextpad=0.8,
        columnspacing=2,
        labelspacing=0.8,
        handlelength=2.0,
        handleheight=1.5,
    )

    # 9. Cartographic elements
    degree_in_meters = 111320.0
    dx_meters = degree_in_meters if ax.get_xlim()[1] <= 180.5 else (pixel_size_km * 1000)

    def km_formatter(value, unit):
        if unit == "Mm":
            return f"{int(value * 1000)} km"
        return f"{int(value)} {unit}"

    scalebar = ScaleBar(
        dx=dx_meters,
        units="m",
        length_fraction=0.15,
        location="lower left",
        box_alpha=0.6,
        scale_formatter=km_formatter,
    )
    ax.add_artist(scalebar)

    try:
        north_arrow(
            ax,
            location="upper right",
            shadow=False,
            rotation={"degrees": 0},
            scale=0.5,
        )
    except NameError:
        print("north_arrow function not found. Skipping north arrow.")

    # 10. Axes styling
    ax.set_title("Alternation Exchange", fontsize=18, pad=10)
    ax.set_aspect("equal")

    to_latlon = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    def format_lon(x, pos):
        x = np.clip(x, 0, width - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, height // 2, x)
        lon, _ = to_latlon.transform(x_proj, y_proj)
        return f"{lon:.1f}°"

    def format_lat(y, pos):
        y = np.clip(y, 0, height - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, y, width // 2)
        _, lat = to_latlon.transform(x_proj, y_proj)
        return f"{lat:.1f}°"

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

    ax.tick_params(axis="both", which="major", labelsize=10, pad=4)
    plt.setp(ax.get_yticklabels(), rotation=90, va="center")

    # 11. Save and Show
    maps_dir = os.path.join(output_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)
    output_figure_path = os.path.join(maps_dir, "map_alternation_exchange.png")

    plt.savefig(
        output_figure_path,
        dpi=300,
        bbox_inches="tight",
        format="png",
        pad_inches=0.5,
    )
    plt.show()
    print(f"Map figure saved successfully to: {output_figure_path}")

def plot_alternation_shift_map(
    output_dir: str,
    nodata_val: int,
    raster_filename: str,
    scale_factor: float = 0.05,
) -> None:
    """
    Plot the Alternation Shift raster map with cartographic elements.

    Parameters
    ----------
    output_dir : str
        Directory containing the exported GEE tiles and where the map will be saved.
    nodata_val : int
        Value representing NoData in the raster to be masked out.
    raster_filename : str
        Prefix of the raster tiles to plot.
    scale_factor : float, optional
        Scale factor to downsample the massive global raster to fit into memory.

    Returns
    -------
    None
    """
    # 1. Locate all raster tiles exported by GEE
    raster_files = glob.glob(os.path.join(output_dir, f"{raster_filename}*.tif"))
    if not raster_files:
        raise FileNotFoundError(
            f"Raster tiles not found for prefix: {raster_filename}. Make sure the GEE export finished."
        )

    # 2. Create a temporary Virtual Raster (VRT) to merge tiles dynamically
    vrt_path = os.path.join(output_dir, "merged_shift.vrt")
    files_str = " ".join([f'"{f}"' for f in raster_files])
    os.system(f"gdalbuildvrt {vrt_path} {files_str}")

    # 3. Calculate pixel size for scale bar
    pixel_size_km = compute_display_pixel_size_km(
        raster_path=vrt_path,
        downsample_factor=scale_factor,
    )

    # 4. Read raster and basic metadata with downsampling
    with rasterio.open(vrt_path) as src:
        out_shape = (
            max(1, int(src.height * scale_factor)),
            max(1, int(src.width * scale_factor)),
        )
        data = src.read(
            1,
            out_shape=out_shape,
            resampling=rasterio.enums.Resampling.nearest,
        )

        # Force masking using the provided nodata value
        data_masked = np.ma.masked_equal(data, nodata_val)

        src_crs = src.crs
        # Adjust the affine transform for the new downsampled resolution
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[1]),
            (src.height / data.shape[0]),
        )
        height, width = data.shape

    # 5. Figure
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)

    # Determine max value for colormap
    try:
        data_max = int(np.ma.max(data_masked))
    except:
        data_max = 1

    if data_max <= 0:
        data_max = 1

    # 6. Discrete Colormap Configuration
    original_cmap = plt.get_cmap("viridis_r")
    # Define the color for value 0 (Background/Gray)
    colors_list = ["#c0c0c0"] + [
        original_cmap(i) for i in np.linspace(0, 1, data_max)
    ]
    cmap = ListedColormap(colors_list)
    bounds = np.arange(-0.5, data_max + 1.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    # 7. Plot raster
    ax.imshow(
        data_masked,
        cmap=cmap,
        interpolation="nearest",
        norm=norm,
    )

    # 8. Legend Configuration
    legend_elements = []

    # Extract unique values actually present in the masked raster data
    present_values = np.unique(data_masked.compressed())

    for i in range(0, data_max + 1):
        # Append to legend ONLY if the value is present in the map
        if i in present_values:
            legend_elements.append(
                Patch(
                    facecolor=cmap(norm(i)),
                    edgecolor="none",
                    linewidth=0,
                    label=str(i),
                ),
            )

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=12,
        borderpad=1.2,
        title="Shift\nFrequency",
        title_fontsize=14,
        alignment="left",
        handletextpad=0.8,
        columnspacing=2,
        labelspacing=0.8,
        handlelength=2.0,
        handleheight=1.5,
    )

    # 9. Cartographic elements
    degree_in_meters = 111320.0
    dx_meters = degree_in_meters if ax.get_xlim()[1] <= 180.5 else (pixel_size_km * 1000)

    def km_formatter(value, unit):
        if unit == "Mm":
            return f"{int(value * 1000)} km"
        return f"{int(value)} {unit}"

    scalebar = ScaleBar(
        dx=dx_meters,
        units="m",
        length_fraction=0.15,
        location="lower left",
        box_alpha=0.6,
        scale_formatter=km_formatter,
    )
    ax.add_artist(scalebar)

    try:
        north_arrow(
            ax,
            location="upper right",
            shadow=False,
            rotation={"degrees": 0},
            scale=0.5,
        )
    except NameError:
        print("north_arrow function not found. Skipping north arrow.")

    # 10. Axes styling
    ax.set_title("Alternation Shift", fontsize=18, pad=10)
    ax.set_aspect("equal")

    to_latlon = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    def format_lon(x, pos):
        x = np.clip(x, 0, width - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, height // 2, x)
        lon, _ = to_latlon.transform(x_proj, y_proj)
        return f"{lon:.1f}°"

    def format_lat(y, pos):
        y = np.clip(y, 0, height - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, y, width // 2)
        _, lat = to_latlon.transform(x_proj, y_proj)
        return f"{lat:.1f}°"

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

    ax.tick_params(axis="both", which="major", labelsize=10, pad=4)
    plt.setp(ax.get_yticklabels(), rotation=90, va="center")

    # 11. Save and Show
    maps_dir = os.path.join(output_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)
    output_figure_path = os.path.join(maps_dir, "map_alternation_shift.png")

    plt.savefig(
        output_figure_path,
        dpi=300,
        bbox_inches="tight",
        format="png",
        pad_inches=0.5,
    )
    plt.show()
    print(f"Map figure saved successfully to: {output_figure_path}")
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
    # Extract the unit from the parentheses, if any. E.g., 'Area (km²)' -> '(km²)'
    unit = f"({base_label.split('(')[-1]}" if '(' in base_label else f"of {base_label}"

    if "pixels" in base_label.lower():
        mapping = {
            "": base_label,
            "k": "thousand pixels",
            "M": "million pixels",
            "B": "billion pixels",
            "T": "trillions pixels",
        }
    else:
        mapping = {
            "": base_label,
            "k": f"Thousands {unit}",
            "M": f"Millions {unit}",
            "B": f"Billions {unit}",
            "T": f"Trillions {unit}",
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
