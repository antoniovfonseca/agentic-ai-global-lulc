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

def process_and_plot_pixel_counts(
    input_dir=DEFAULT_INPUT_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    no_data_value=NODATA_VALUE,
):
    """
    Processes raster images to count pixels per class, plots a stacked bar chart,
    and exports the results to a CSV file.

    Parameters
    ----------
    input_dir : str, optional
        Directory path where the masked .tif files are located. 
        Defaults to DEFAULT_INPUT_DIR.
    output_dir : str, optional
        Directory path where the output plot and CSV will be saved.
        Defaults to DEFAULT_OUTPUT_DIR.
    no_data_value : int, optional
        Pixel value to be treated as NoData. Defaults to NODATA_VALUE.

    Returns
    -------
    pd.DataFrame
        The pivot table containing pixel counts per year and class.
    """

    # 1. File Discovery and Sorting
    # Find all .tif files in the input directory
    search_pattern = os.path.join(input_dir, "*.tif")
    raw_paths = glob.glob(search_pattern)

    # Sort files chronologically using the helper function
    # Note: Ensure 'get_year_from_filename' is defined in utils.py
    image_paths = sorted(raw_paths, key=get_year_from_filename)
    years = [get_year_from_filename(p) for p in image_paths]

    # Stop if no files are found
    if not image_paths:
        print(f"No .tif files found in {input_dir}")
        return None

    records: list[dict] = []

    # 2. Iterate through each year and corresponding image path
    for year, path in zip(
        years,
        image_paths,
    ):
        # 3. Read the raster data
        with rasterio.open(
            path,
        ) as src:
            data = src.read(
                1,
            )

        # 4. Count unique pixel values
        values, counts = np.unique(
            data,
            return_counts=True,
        )

        # 5. Process counts and map to class names
        for value, count in zip(
            values,
            counts,
        ):
            value = int(value)

            # Filter out NoData values
            if value == no_data_value:
                continue

            # Skip classes not defined in the global metadata
            if value not in GLANCE_METADATA:
                continue

            records.append(
                {
                    "Year": year,
                    "ClassID": value,
                    "ClassName": GLANCE_METADATA[value]["name"],
                    "Pixels": int(count),
                },
            )

    # 6. Create DataFrame and Pivot Table
    df_pixels = pd.DataFrame(
        records,
    )

    pivot_pixels = (
        df_pixels.pivot_table(
            index="Year",
            columns="ClassName",
            values="Pixels",
            aggfunc="sum",
        )
        .fillna(
            0,
        )
        .astype(
            int,
        )
    )

    years_array = pivot_pixels.index.values

    # 7. Determine Y-axis scaling factor and label
    max_val = pivot_pixels.to_numpy().max()

    if max_val >= 1_000_000:
        scale_factor = 1_000_000
        y_label = "Area (million pixels)"
    elif max_val >= 1_000:
        scale_factor = 1_000
        y_label = "Area (thousand pixels)"
    elif max_val >= 100:
        scale_factor = 100
        y_label = "Area (hundred pixels)"
    else:
        scale_factor = 1
        y_label = "Area (pixels)"

    pivot_scaled = pivot_pixels / scale_factor

    # 8. Prepare color map and sorting logic
    class_ids_plot = sorted(GLANCE_METADATA.keys())

    color_map = {
        GLANCE_METADATA[class_id]["name"]: GLANCE_METADATA[class_id]["color"]
        for class_id in class_ids_plot
    }

    # Calculate Net Change to determine stack order
    first_year = years_array[0]
    last_year = years_array[-1]

    # Handle cases where a class might be missing in first or last year
    try:
        net_change_per_class = (
            pivot_scaled.loc[last_year]
            - pivot_scaled.loc[first_year]
        )
    except KeyError:
        # Fallback if specific years are missing, use first/last available
        net_change_per_class = (
            pivot_scaled.iloc[-1]
            - pivot_scaled.iloc[0]
        )

    # Map names back to IDs for tie-breaking
    name_to_id_map = {
        v["name"]: k
        for k, v in GLANCE_METADATA.items()
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

    # 9. Generate the Stacked Bar Chart
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

    # 10. Configure Axes
    ax.set_xticks(
        x,
    )
    ax.set_xticklabels(
        years_array,
    )

    # Adaptive rotation for X-axis labels
    n_labels = len(years_array)
    if n_labels <= 7:
        rotation = 0
        ha = "center"
    elif n_labels <= 12:
        rotation = 45
        ha = "right"
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
        labelsize=14,
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
        "Number of pixels per class",
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
            nbins=5,
            integer=True,
        ),
    )
    ax.yaxis.set_major_formatter(
        ticker.FormatStrFormatter(
            "%d",
        ),
    )

    # 11. Add Legend
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

    ax.legend(
        handles,
        labels,
        bbox_to_anchor=(
            1.05,
            1.0,
        ),
        loc="upper left",
        frameon=False,
        fontsize=12,
    )

    plt.tight_layout()

    # 12. Save Figure
    out_fig = os.path.join(
        output_dir,
        "graph_pixel_per_class.png",
    )
    plt.savefig(
        out_fig,
        format="png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()

    # 13. Save CSV
    csv_output_path = os.path.join(
        output_dir,
        "pixels_per_class_per_year.csv",
    )
    pivot_pixels.to_csv(
        csv_output_path,
        index_label="Year",
    )
    print(
        f"Files saved to: {output_dir}",
    )

    return pivot_pixels

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