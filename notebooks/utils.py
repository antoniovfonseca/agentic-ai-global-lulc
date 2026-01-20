import ee
import geemap

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

    # 1. Define the GLANCE Collection and Classification Band
    collection_id = "projects/GLANCE/DATASETS/V001"
    class_band = "LC"

    # 2. Define Visualization Parameters according to the Official GLANCE Legend
    vis_params = {
        'min': 1,
        'max': 7,
        'palette': [
            '0000FF', # 1. Water
            'AAAAFF', # 2. Ice/Snow
            'FF0000', # 3. Developed
            '964B00', # 4. Barren
            '006400', # 5. Trees
            'FFBB22', # 6. Shrub
            'FFFF4C'  # 7. Herbaceous
        ]
    }

    # 3. Create the Map Object
    m = geemap.Map()
    m.setCenter(0, 20, 2)

    # 4. Filter and Select the Data
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    try:
        collection = ee.ImageCollection(collection_id)
        image = collection.filterDate(start_date, end_date).mosaic()
        glance_layer = image.select(class_band)

        # 5. Add the Layer and Legend
        m.addLayer(glance_layer, vis_params, f"GLANCE LC {year}")
        
        legend_dict = {
            'Water': '0000FF', 'Ice/Snow': 'AAAAFF', 'Developed': 'FF0000',
            'Barren': '964B00', 'Trees': '006400', 'Shrub': 'FFBB22',
            'Herbaceous': 'FFFF4C'
        }
        m.add_legend(title="GLANCE Classes", legend_dict=legend_dict)

        print(f"GLANCE map for year {year} generated successfully.")
        return m

    except Exception as e:
        print(f"Error loading data for year {year}: {e}")
        return m