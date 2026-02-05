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
    Decompose Sum and Extent matrices into 4 change components and save them.

    Parameters
    ----------
    df_sum : pd.DataFrame
        Aggregated transition matrix (Sum of all annual intervals).
    df_ext : pd.DataFrame
        Direct transition matrix (Extent: start year vs end year).
    output_dir : str
        Directory path to save the resulting CSV files.
    period_label : str, optional
        Year range label for filename (e.g., "2001-2019"), by default "2001-2019".

    Returns
    -------
    None
    """
    def _get_exchange_and_shift(
        matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Internal helper to decompose a matrix into Exchange and Shift."""
        m_calc = matrix.copy()
        np.fill_diagonal(
            m_calc,
            0.0,
        )
        exchange = np.minimum(
            m_calc,
            m_calc.T,
        )
        shift = m_calc - exchange
        return exchange, shift

    # 1. Flexible sorting to handle names like 'Developed'
    # It tries to match the name to the metadata order, otherwise uses alphabetical
    name_to_id = {v['name']: k for k, v in GLANCE_METADATA.items()}
    
    def _custom_sort_key(label):
        # Try to get ID from metadata name, then try to parse as int, else string
        if label in name_to_id:
            return (0, name_to_id[label])
        try:
            return (0, int(label))
        except (ValueError, TypeError):
            return (1, str(label))

    all_labels = sorted(
        list(
            set(df_sum.index).union(df_sum.columns),
        ),
        key=_custom_sort_key,
    )
    
    # 2. Reindex and align matrices
    df_sum = df_sum.reindex(
        index=all_labels,
        columns=all_labels,
    ).fillna(0.0)
    
    df_ext = df_ext.reindex(
        index=all_labels,
        columns=all_labels,
    ).fillna(0.0)

    # 3. Calculate Allocation Components (from Extent)
    alloc_exc, alloc_shift = _get_exchange_and_shift(
        df_ext.values,
    )

    # 4. Calculate Alternation Components (from Sum - Extent)
    alternation_raw = np.maximum(
        df_sum.values - df_ext.values,
        0.0,
    )
    alt_exc, alt_shift = _get_exchange_and_shift(
        alternation_raw,
    )

    # 5. Map and Save all 6 required files
    components = {
        "allocation_exchange": alloc_exc,
        "allocation_shift": alloc_shift,
        "alternation_exchange": alt_exc,
        "alternation_shift": alt_shift,
        "sum": df_sum.values,
        "extent": df_ext.values,
    }

    for name, data in components.items():
        df_out = pd.DataFrame(
            data,
            index=all_labels,
            columns=all_labels,
        )
        
        file_name = f"transition_matrix_{name}_{period_label}.csv"
        save_path = os.path.join(
            output_dir,
            file_name,
        )
        
        df_out.to_csv(
            save_path,
        )
        
        print(
            f"Component matrix saved to: {save_path}",
        )

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(
    df: pd.DataFrame,
    title: str,
    save_path: str,
    tick_fontsize_x: int = 12,
    tick_fontsize_y: int = 12,
    axis_label_fontsize: int = 14,
    title_fontsize: int = 16,
    ann_fontsize: int = 10,
    cbar_label: str = "Area (km²)",
    cbar_fraction: float = 0.046,
    cbar_pad: float = 0.04,
) -> None:
    """
    Generate a high-fidelity heatmap matching the reference notebook structure.

    Parameters
    ----------
    df : pd.DataFrame
        Square matrix containing transition data or change components.
    title : str
        The descriptive title to be displayed on the plot.
    save_path : str
        Full system path where the PNG file will be exported.
    tick_fontsize_x : int, optional
        Font size for the X-axis labels, by default 12.
    tick_fontsize_y : int, optional
        Font size for the Y-axis labels, by default 12.
    axis_label_fontsize : int, optional
        Font size for the axis titles, by default 14.
    title_fontsize : int, optional
        Font size for the main plot title, by default 16.
    ann_fontsize : int, optional
        Font size for the numerical annotations inside cells, by default 10.
    cbar_label : str, optional
        Label for the colorbar scale, by default "Area (km²)".
    cbar_fraction : float, optional
        Fraction of original axes to use for colorbar, by default 0.046.
    cbar_pad : float, optional
        Padding between the heatmap and the colorbar, by default 0.04.

    Returns
    -------
    None
    """
    # 1. Formatting Setup
    # Masking diagonal to highlight transitions only (Persistence is excluded)
    plot_data = df.copy()
    mask = np.eye(
        len(plot_data), 
        dtype=bool,
    )

    # 2. Figure and Axes Initialization
    sns.set_theme(style="white")
    fig, ax = plt.subplots(
        figsize=(14, 11),
    )

    # 3. Heatmap Core Logic
    # Using 'YlOrRd' as the standardized color palette for transitions
    heatmap = sns.heatmap(
        plot_data,
        mask=mask,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.8,
        linecolor="white",
        square=True,
        cbar_kws={
            "label": cbar_label,
            "fraction": cbar_fraction,
            "pad": cbar_pad,
            "shrink": 0.8,
        },
        annot_kws={
            "size": ann_fontsize,
            "weight": "normal",
        },
        ax=ax,
    )

    # 4. Axis and Label Styling
    ax.set_title(
        title, 
        fontsize=title_fontsize, 
        fontweight="bold", 
        pad=25,
    )
    ax.set_xlabel(
        "To (End Year)", 
        fontsize=axis_label_fontsize, 
        labelpad=15,
    )
    ax.set_ylabel(
        "From (Start Year)", 
        fontsize=axis_label_fontsize, 
        labelpad=15,
    )

    # Label rotation for categorical clarity (e.g., 'Herbaceous')
    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=tick_fontsize_x,
    )
    plt.yticks(
        rotation=0,
        fontsize=tick_fontsize_y,
    )

    # 5. Output Management
    plt.tight_layout()
    
    # Ensure directory exists
    output_dir = os.path.dirname(save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    
    plt.show()
    plt.close(fig)

    print(f"Successfully generated and saved: {os.path.basename(save_path)}")
