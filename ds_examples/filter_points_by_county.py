"""
Filter DataFrame points by UK County boundaries using GeoPandas

This script demonstrates how to:
1. Load UK Counties shapefile from online source
2. Load CSV data with lat/lon coordinates
3. Filter points that fall within a selected county
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def load_uk_counties_shapefile(url=None):
    """
    Load UK Counties and Unitary Authorities shapefile
    
    Parameters:
    -----------
    url : str, optional
        Direct URL to the shapefile. If None, uses the default CKAN resource URL
    
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame containing UK county boundaries
    """
    if url is None:
        # Direct download URL for the shapefile from CKAN
        url = "https://ckan.publishing.service.gov.uk/dataset/7d535263-3fcc-4a14-ba37-d34f36381fa5/resource/7d535263-3fcc-4a14-ba37-d34f36381fa5/download/ctyua_dec_2024_boundaries_uk_buc.zip"
    
    # Read shapefile directly from URL (geopandas can handle zipped shapefiles)
    counties_gdf = gpd.read_file(url)
    
    return counties_gdf


def load_data_from_csv(csv_path):
    """
    Load data from CSV file with time, id, latitude, longitude, temperature columns
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with the data
    """
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_columns = ['time', 'id', 'latitude', 'longitude', 'temperature']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV missing required columns: {missing_columns}")
    
    return df


def create_geodataframe_from_points(df, lat_col='latitude', lon_col='longitude', crs='EPSG:4326'):
    """
    Convert DataFrame with lat/lon columns to GeoDataFrame with Point geometries
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing latitude and longitude columns
    lat_col : str
        Name of latitude column
    lon_col : str
        Name of longitude column
    crs : str
        Coordinate Reference System (default: EPSG:4326 for WGS84)
    
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame with Point geometries
    """
    # Create Point geometries from lat/lon
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    
    return gdf


def filter_points_by_county(points_gdf, counties_gdf, county_name):
    """
    Filter points that fall within a specific county
    
    Parameters:
    -----------
    points_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing point data
    counties_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing county boundaries
    county_name : str
        Name of the county to filter by (e.g., 'Kent', 'Lancashire')
    
    Returns:
    --------
    geopandas.GeoDataFrame
        Filtered GeoDataFrame containing only points within the selected county
    """
    # Ensure both GeoDataFrames use the same CRS
    if points_gdf.crs != counties_gdf.crs:
        points_gdf = points_gdf.to_crs(counties_gdf.crs)
    
    # Select the specific county
    # Note: You may need to check the actual column name in the shapefile
    # Common names: 'CTYUA24NM', 'NAME', 'County'
    county_name_col = None
    for possible_col in ['CTYUA24NM', 'CTYUA_NAME', 'NAME', 'name', 'County']:
        if possible_col in counties_gdf.columns:
            county_name_col = possible_col
            break
    
    if county_name_col is None:
        raise ValueError(f"Could not find county name column. Available columns: {counties_gdf.columns.tolist()}")
    
    selected_county = counties_gdf[counties_gdf[county_name_col] == county_name]
    
    if selected_county.empty:
        available_counties = sorted(counties_gdf[county_name_col].unique())
        raise ValueError(f"County '{county_name}' not found. Available counties:\n{available_counties}")
    
    # Spatial join to filter points within the county
    # 'inner' join keeps only points within the county
    filtered_points = gpd.sjoin(points_gdf, selected_county, how='inner', predicate='within')
    
    # Remove duplicate columns from the join
    filtered_points = filtered_points.drop(columns=['index_right'], errors='ignore')
    
    return filtered_points


def main_example():
    """
    Example usage of the functions
    """
    # Example: Load UK counties shapefile
    print("Loading UK counties shapefile...")
    counties_gdf = load_uk_counties_shapefile()
    print(f"Loaded {len(counties_gdf)} counties")
    print(f"Counties CRS: {counties_gdf.crs}")
    print(f"Available columns: {counties_gdf.columns.tolist()}")
    
    # Example: Load your CSV data
    # Uncomment and modify the path to your CSV file
    # csv_path = 'your_data.csv'
    # df = load_data_from_csv(csv_path)
    
    # Create example data for demonstration
    print("\nCreating example data...")
    example_data = pd.DataFrame({
        'time': ['2024-01-01 12:00', '2024-01-01 13:00', '2024-01-01 14:00', '2024-01-01 15:00'],
        'id': [1, 2, 3, 4],
        'latitude': [51.5074, 53.4808, 51.4545, 52.4862],  # London, Manchester, Kent, Birmingham areas
        'longitude': [-0.1278, -2.2426, 0.3456, -1.8904],
        'temperature': [15.2, 12.8, 14.5, 13.9]
    })
    
    # Convert to GeoDataFrame
    print("Converting to GeoDataFrame...")
    points_gdf = create_geodataframe_from_points(example_data)
    print(f"Created {len(points_gdf)} points")
    
    # Filter by county (example: 'Kent')
    print("\nFiltering points by county...")
    try:
        # You need to specify the exact county name from the shapefile
        # Use counties_gdf['CTYUA24NM'].unique() to see all available county names
        county_name = 'Kent'  # Change this to your desired county
        filtered_data = filter_points_by_county(points_gdf, counties_gdf, county_name)
        
        print(f"\nFound {len(filtered_data)} points in {county_name}")
        print("\nFiltered data:")
        print(filtered_data[['time', 'id', 'latitude', 'longitude', 'temperature']])
        
        # Convert back to regular DataFrame if needed
        result_df = pd.DataFrame(filtered_data.drop(columns='geometry'))
        
        # Optionally save to CSV
        # result_df.to_csv(f'filtered_data_{county_name}.csv', index=False)
        
    except ValueError as e:
        print(f"Error: {e}")
    
    # List all available counties
    print("\n" + "="*50)
    print("Available UK Counties and Unitary Authorities:")
    print("="*50)
    # Determine the county name column
    for col in ['CTYUA24NM', 'CTYUA_NAME', 'NAME', 'name']:
        if col in counties_gdf.columns:
            county_list = sorted(counties_gdf[col].unique())
            for i, county in enumerate(county_list, 1):
                print(f"{i}. {county}")
            break


# Alternative approach using simpler syntax
def filter_points_by_county_simple(df, counties_gdf, county_name, lat_col='latitude', lon_col='longitude'):
    """
    Simplified one-function approach to filter points by county
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with latitude and longitude columns
    counties_gdf : geopandas.GeoDataFrame
        GeoDataFrame with county boundaries
    county_name : str
        Name of county to filter by
    lat_col : str
        Name of latitude column
    lon_col : str
        Name of longitude column
    
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame with points in the selected county
    """
    # Convert points to GeoDataFrame
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    # Ensure same CRS
    points_gdf = points_gdf.to_crs(counties_gdf.crs)
    
    # Find county name column
    county_col = next((col for col in ['CTYUA24NM', 'CTYUA_NAME', 'NAME', 'name', 'County'] 
                       if col in counties_gdf.columns), None)
    
    # Filter county
    county_boundary = counties_gdf[counties_gdf[county_col] == county_name]
    
    # Spatial join
    filtered = gpd.sjoin(points_gdf, county_boundary, how='inner', predicate='within')
    
    # Return as regular DataFrame
    return pd.DataFrame(filtered.drop(columns='geometry'))


if __name__ == "__main__":
    main_example()
