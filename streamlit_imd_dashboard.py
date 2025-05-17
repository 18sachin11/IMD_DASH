import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
import os
from tqdm import tqdm
import zipfile
import io
import tempfile
from pathlib import Path
import pymannkendall as mk
import matplotlib.path as mpltPath

# Streamlit page configuration
st.set_page_config(page_title="IMD Rainfall Analysis", layout="wide")

# Initialize session state
if 'region' not in st.session_state:
    st.session_state.region = None
if 'downloaded_files' not in st.session_state:
    st.session_state.downloaded_files = []
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'grid_points' not in st.session_state:
    st.session_state.grid_points = None
if 'step_completed' not in st.session_state:
    st.session_state.step_completed = {'upload': False, 'download': False, 'process': False}

# Default region (Kerala)
DEFAULT_REGION = {"name": "Kerala", "lat": (8.5, 12.5), "lon": (74.5, 77.5), "geometry": None}

# Function to download NetCDF file
def download_netcdf(year, download_folder="imd_netcdf_files"):
    os.makedirs(download_folder, exist_ok=True)
    url = "https://www.imdpune.gov.in/cmpg/Griddata/RF25.php"
    payload = {"RF25": str(year)}
    filename = f"RF25_{year}.nc"
    output_path = os.path.join(download_folder, filename)
    
    if os.path.exists(output_path):
        return output_path
    
    try:
        response = requests.post(url, data=payload, stream=True, timeout=10)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=filename)
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        progress_bar.close()
        
        if total_size != 0 and os.path.getsize(output_path) != total_size:
            os.remove(output_path)
            raise Exception("Download size mismatch")
        
        return output_path
    
    except requests.RequestException as e:
        st.error(f"Failed to download NetCDF file for year {year}: {e}")
        return None

# Function to inspect NetCDF file structure
def inspect_netcdf(file_path):
    try:
        ds = xr.open_dataset(file_path)
        info = {
            "Dimensions": list(ds.dims.keys()),
            "Coordinates": list(ds.coords.keys()),
            "Variables": list(ds.variables.keys()),
            "Dataset Summary": str(ds)
        }
        return info
    except Exception as e:
        return {"Error": f"Failed to inspect file: {e}"}

# Function to process spatial file
def process_spatial_file(uploaded_file):
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            if uploaded_file.name.endswith('.zip'):
                zip_path = os.path.join(tmp_dir, uploaded_file.name)
                with open(zip_path, 'wb') as f:
                    f.write(uploaded_file.read())
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmp_dir)
                
                shp_file = next((f for f in os.listdir(tmp_dir) if f.endswith('.shp')), None)
                if not shp_file:
                    st.error("No .shp file found in the uploaded zip.")
                    return None
                
                gdf = gpd.read_file(os.path.join(tmp_dir, shp_file))
            
            elif uploaded_file.name.endswith('.geojson'):
                gdf = gpd.read_file(uploaded_file)
            
            elif uploaded_file.name.endswith('.kml'):
                gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'r'
                gdf = gpd.read_file(uploaded_file, driver='KML')
            
            else:
                st.error("Unsupported file format. Please upload a .zip (shapefile), .geojson, or .kml file.")
                return None
            
            if gdf.crs is not None and gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs(epsg=4326)
            
            bounds = gdf.total_bounds
            lat_range = (bounds[1], bounds[3])
            lon_range = (bounds[0], bounds[2])
            region_name = os.path.splitext(uploaded_file.name)[0]
            
            return {"name": region_name, "lat": lat_range, "lon": lon_range, "geometry": gdf}
    
    except Exception as e:
        st.error(f"Error processing spatial file: {e}")
        return None

# Function to perform IDW interpolation
def idw_interpolation(x, y, values, xi, yi, power=2):
    values_grid = np.zeros_like(xi)
    for i in range(xi.shape[0]):
        for j in range(xi.shape[1]):
            dist = np.sqrt((x - xi[i, j])**2 + (y - yi[i, j])**2)
            dist = np.where(dist == 0, 1e-10, dist)
            weights = 1 / dist**power
            values_grid[i, j] = np.sum(values * weights) / np.sum(weights)
    return values_grid

# Function to process NetCDF files and generate data
@st.cache_data
def process_data(start_yr, end_yr, lat_range, lon_range, _geometry):
    try:
        datasets = []
        for file_path in st.session_state.downloaded_files:
            if file_path:
                ds = xr.open_dataset(file_path)
                year = int(os.path.basename(file_path).split('_')[1].split('.')[0])
                
                expected_dims = ['latitude', 'longitude']
                if not all(dim in ds.dims for dim in expected_dims):
                    if all(dim in ds.dims for dim in ['LATITUDE', 'LONGITUDE']):
                        ds = ds.rename({'LATITUDE': 'latitude', 'LONGITUDE': 'longitude'})
                    elif all(dim in ds.dims for dim in ['lat', 'lon']):
                        ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
                    else:
                        available_dims = list(ds.dims.keys())
                        st.error(f"Invalid dimensions in {file_path}. Expected {expected_dims}, ['LATITUDE', 'LONGITUDE'], or ['lat', 'lon'], found {available_dims}.")
                        return None, None, None, None
                
                if year <= 2023:
                    rain_var = 'RAINFALL'
                    if 'TIME' in ds.dims:
                        ds = ds.rename({'TIME': 'time'})
                    elif 'time' not in ds.dims:
                        st.error(f"Expected time dimension 'TIME' or 'time' not found in {file_path}.")
                        return None, None, None, None
                else:
                    rain_var = 'rf'
                    if 'time' not in ds.dims:
                        st.error(f"Expected time dimension 'time' not found in {file_path}.")
                        return None, None, None, None
                
                if rain_var not in ds.variables:
                    st.error(f"Rainfall variable '{rain_var}' not found in {file_path}.")
                    return None, None, None, None
                
                ds = ds.sel(latitude=slice(lat_range[0], lat_range[1]), longitude=slice(lon_range[0], lon_range[1]))
                datasets.append(ds)
        
        if not datasets:
            st.error("No valid data downloaded.")
            return None, None, None, None
        
        data = xr.concat(datasets, dim='time')
        
        lats = data.latitude.values
        lons = data.longitude.values
        grid_points = [Point(lon, lat) for lat in lats for lon in lons]
        grid_gdf = gpd.GeoDataFrame({'geometry': grid_points}, crs="EPSG:4326")
        clipped_grid = gpd.sjoin(grid_gdf, _geometry, how='inner', predicate='within')
        if clipped_grid.empty:
            st.error("No grid points found within the uploaded shapefile.")
            return None, None, None, None
        
        clipped_coords = [(point.y, point.x) for point in clipped_grid.geometry]
        
        daily_dfs = []
        for lat, lon in clipped_coords:
            point_data = data[rain_var].sel(latitude=lat, longitude=lon, method='nearest')
            df = point_data.to_dataframe(name='rainfall').reset_index()
            df['latitude'] = lat
            df['longitude'] = lon
            daily_dfs.append(df[['time', 'latitude', 'longitude', 'rainfall']])
        daily_df = pd.concat(daily_dfs, ignore_index=True)
        daily_df['date'] = pd.to_datetime(daily_df['time'])
        daily_csv = daily_df.pivot_table(
            values='rainfall',
            index=['latitude', 'longitude'],
            columns='date',
            fill_value=0
        ).reset_index()
        daily_csv.columns = ['latitude', 'longitude'] + [col.strftime('%d-%m-%Y') for col in daily_csv.columns[2:]]
        
        df = daily_df.copy()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['hydro_year'] = df['year'].where(df['month'] >= 6, df['year'] + 1)
        
        def get_season(month):
            if month in [1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Pre-Monsoon'
            elif month in [6, 7, 8, 9]:
                return 'Monsoon'
            elif month in [10, 11, 12]:
                return 'Post Monsoon'
        
        df['season'] = df['month'].apply(get_season)
        df['season_year'] = df.apply(lambda x: f"{x['season']} ({x['year']})", axis=1)
        season_order = []
        for year in range(start_yr, end_yr + 1):
            season_order.extend([
                f"Winter ({year})",
                f"Pre-Monsoon ({year})",
                f"Monsoon ({year})",
                f"Post Monsoon ({year})"
            ])
        available_seasons = df['season_year'].unique()
        season_order = [s for s in season_order if s in available_seasons]
        seasonal_csv = df.pivot_table(
            values='rainfall',
            index=['latitude', 'longitude'],
            columns='season_year',
            aggfunc='sum',
            fill_value=0
        ).reset_index()
        seasonal_csv = seasonal_csv[['latitude', 'longitude'] + season_order]
        
        yearly_csv = df.pivot_table(
            values='rainfall',
            index=['latitude', 'longitude'],
            columns='hydro_year',
            aggfunc='sum',
            fill_value=0
        ).reset_index()
        yearly_csv.columns = ['latitude', 'longitude'] + [str(col) for col in yearly_csv.columns[2:]]
        
        return daily_csv, seasonal_csv, yearly_csv, clipped_grid
    
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return None, None, None, None

# Function to generate region map
def generate_map(geometry, grid_points, region_name):
    fig, ax = plt.subplots(figsize=(10, 10))
    geometry.plot(ax=ax, color='blue', alpha=0.3, edgecolor='black')
    if grid_points is not None:
        grid_points.plot(ax=ax, color='red', markersize=10, label='Grid Points')
    ax.set_title(f'Map of {region_name} with IMD Grid Points')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    plt.tight_layout()
    return fig

# Function to generate yearly rainfall maps with IDW interpolation and clipping
def generate_yearly_rainfall_maps(geometry, yearly_data, region_name):
    figs = []
    years = [col for col in yearly_data.columns if col not in ['latitude', 'longitude']]
    
    bounds = geometry.total_bounds
    lon_min, lat_min, lon_max, lat_max = bounds[0], bounds[1], bounds[2], bounds[3]
    
    grid_resolution = 100
    lon_grid = np.linspace(lon_min, lon_max, grid_resolution)
    lat_grid = np.linspace(lat_min, lat_max, grid_resolution)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
    
    paths = []
    for geom in geometry.geometry:
        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            paths.append(np.array(list(zip(x, y))))
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                paths.append(np.array(list(zip(x, y))))
    
    for year in years:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        lons = yearly_data['longitude'].values
        lats = yearly_data['latitude'].values
        rainfall = yearly_data[year].values
        
        interpolated_rainfall = idw_interpolation(lons, lats, rainfall, lon_grid, lat_grid, power=2)
        
        mask = np.zeros_like(interpolated_rainfall, dtype=bool)
        grid_points = np.array([lon_grid.ravel(), lat_grid.ravel()]).T
        for path in paths:
            path = mpltPath.Path(path)
            inside = path.contains_points(grid_points)
            mask |= inside.reshape(lon_grid.shape)
        masked_rainfall = np.ma.masked_where(~mask, interpolated_rainfall)
        
        cmap = plt.get_cmap('viridis')
        mesh = ax.pcolormesh(lon_grid, lat_grid, masked_rainfall, cmap=cmap, shading='auto')
        
        plt.colorbar(mesh, ax=ax, label='Rainfall (mm)')
        
        geometry.plot(ax=ax, color='none', edgecolor='black', linewidth=1)
        
        ax.set_title(f'Yearly Rainfall for {year} - {region_name} (IDW Interpolation)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        
        plt.tight_layout()
        figs.append(fig)
    
    return figs

# Function to generate rainfall charts
def generate_rainfall_charts(daily_csv, seasonal_csv, yearly_csv, region_name):
    figs = []
    
    daily_avg = daily_csv.iloc[:, 2:].mean().reset_index()
    daily_avg.columns = ['date', 'rainfall']
    daily_avg['date'] = pd.to_datetime(daily_avg['date'], format='%d-%m-%Y')
    
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(daily_avg['date'], daily_avg['rainfall'], color='blue')
    ax1.set_title(f'Daily Average Rainfall - {region_name}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Rainfall (mm)')
    ax1.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    figs.append(fig1)
    
    seasonal_avg = seasonal_csv.iloc[:, 2:].mean().reset_index()
    seasonal_avg.columns = ['season', 'rainfall']
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.bar(seasonal_avg['season'], seasonal_avg['rainfall'], color='green')
    ax2.set_title(f'Seasonal Average Rainfall - {region_name}')
    ax2.set_xlabel('Season')
    ax2.set_ylabel('Rainfall (mm)')
    ax2.grid(True, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    figs.append(fig2)
    
    yearly_avg = yearly_csv.iloc[:, 2:].mean().reset_index()
    yearly_avg.columns = ['year', 'rainfall']
    
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.bar(yearly_avg['year'], yearly_avg['rainfall'], color='orange')
    ax3.set_title(f'Yearly Average Rainfall - {region_name}')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Rainfall (mm)')
    ax3.grid(True, axis='y')
    plt.tight_layout()
    figs.append(fig3)
    
    return figs

# Function to perform trend analysis using Mann-Kendall test
def perform_trend_analysis(daily_csv, seasonal_csv, yearly_csv, region_name):
    results = []
    figs = []
    
    daily_avg = daily_csv.iloc[:, 2:].mean().reset_index()
    daily_avg.columns = ['date', 'rainfall']
    daily_avg['date'] = pd.to_datetime(daily_avg['date'], format='%d-%m-%Y')
    daily_result = mk.original_test(daily_avg['rainfall'])
    
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(daily_avg['date'], daily_avg['rainfall'], color='blue', label='Daily Rainfall')
    ax1.set_title(f'Daily Rainfall Trend - {region_name}\nTrend: {daily_result.trend}, p-value: {daily_result.p:.4f}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Rainfall (mm)')
    ax1.grid(True)
    ax1.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    figs.append(fig1)
    results.append({
        'Type': 'Daily',
        'Trend': daily_result.trend,
        'P-value': daily_result.p,
        'Z': daily_result.z
    })
    
    seasonal_avg = seasonal_csv.iloc[:, 2:].mean().reset_index()
    seasonal_avg.columns = ['season', 'rainfall']
    seasonal_result = mk.original_test(seasonal_avg['rainfall'])
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(seasonal_avg['season'], seasonal_avg['rainfall'], marker='o', color='green', label='Seasonal Rainfall')
    ax2.set_title(f'Seasonal Rainfall Trend - {region_name}\nTrend: {seasonal_result.trend}, p-value: {seasonal_result.p:.4f}')
    ax2.set_xlabel('Season')
    ax2.set_ylabel('Rainfall (mm)')
    ax2.grid(True)
    ax2.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    figs.append(fig2)
    results.append({
        'Type': 'Seasonal',
        'Trend': seasonal_result.trend,
        'P-value': seasonal_result.p,
        'Z': seasonal_result.z
    })
    
    yearly_avg = yearly_csv.iloc[:, 2:].mean().reset_index()
    yearly_avg.columns = ['year', 'rainfall']
    yearly_result = mk.original_test(yearly_avg['rainfall'])
    
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(yearly_avg['year'], yearly_avg['rainfall'], marker='o', color='orange', label='Yearly Rainfall')
    ax3.set_title(f'Yearly Rainfall Trend - {region_name}\nTrend: {yearly_result.trend}, p-value: {yearly_result.p:.4f}')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Rainfall (mm)')
    ax3.grid(True)
    ax3.legend()
    plt.tight_layout()
    figs.append(fig3)
    results.append({
        'Type': 'Yearly',
        'Trend': yearly_result.trend,
        'P-value': yearly_result.p,
        'Z': yearly_result.z
    })
    
    return results, figs

# Function to export daily rainfall as shapefile
def export_daily_rainfall_shapefile(daily_csv, region_name):
    try:
        # Create GeoDataFrame
        geometry = [Point(xy) for xy in zip(daily_csv['longitude'], daily_csv['latitude'])]
        gdf = gpd.GeoDataFrame(daily_csv, geometry=geometry, crs="EPSG:4326")
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            shp_path = os.path.join(tmp_dir, "daily_rainfall.shp")
            gdf.to_file(shp_path)
            
            # Create zip file
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for ext in ['shp', 'shx', 'dbf', 'prj']:
                    file_path = os.path.join(tmp_dir, f"daily_rainfall.{ext}")
                    if os.path.exists(file_path):
                        zip_file.write(file_path, f"daily_rainfall.{ext}")
            
            zip_buffer.seek(0)
            return zip_buffer.getvalue()
    
    except Exception as e:
        st.error(f"Error exporting shapefile: {e}")
        return None

# Streamlit app layout
st.title("IMD Rainfall Analysis Dashboard")
st.markdown("Analyze IMD daily gridded rainfall data (0.25° x 0.25°) based on IMD seasons: Winter (Jan-Feb), Pre-Monsoon (Mar-May), Monsoon (Jun-Sep), Post Monsoon (Oct-Dec).")

# Sidebar for user inputs
st.sidebar.header("Settings")
start_yr = st.sidebar.slider("Start Year", min_value=1901, max_value=2023, value=2021)
end_yr = st.sidebar.slider("End Year", min_value=1901, max_value=2023, value=2022)

if start_yr > end_yr:
    st.error("Start year must be less than or equal to end year.")
else:
    # Step 1: Upload Spatial File
    st.header("Step 1: Upload Region File")
    uploaded_file = st.file_uploader("Upload a .zip (shapefile), .geojson, or .kml file", type=['zip', 'geojson', 'kml'])
    if st.button("Process Spatial File", disabled=not uploaded_file):
        with st.spinner("Processing spatial file..."):
            region = process_spatial_file(uploaded_file)
            if region:
                st.session_state.region = region
                st.session_state.step_completed['upload'] = True
                st.success(f"Region {region['name']} processed successfully. Lat: {region['lat']}, Lon: {region['lon']}")
            else:
                st.session_state.region = DEFAULT_REGION
                st.session_state.step_completed['upload'] = True
                st.warning(f"Using default region: {DEFAULT_REGION['name']}")

    # Step 2: Download NetCDF Files
    if st.session_state.step_completed['upload']:
        st.header("Step 2: Download Rainfall Data")
        if st.button("Download NetCDF Files"):
            with st.spinner("Downloading NetCDF files..."):
                st.session_state.downloaded_files = []
                st.info(f"Downloading data for years {start_yr} to {end_yr}")
                for year in range(start_yr, end_yr + 1):
                    if year < 1900 or year > 2024:
                        st.warning(f"Year {year} is outside valid range (2000-2024). Skipping.")
                        continue
                    file_path = download_netcdf(year)
                    if file_path:
                        st.session_state.downloaded_files.append(file_path)
                if st.session_state.downloaded_files:
                    st.session_state.step_completed['download'] = True
                    downloaded_years = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in st.session_state.downloaded_files]
                    if any(y < start_yr or y > end_yr for y in downloaded_years):
                        st.warning(f"Unexpected years detected in downloaded files: {downloaded_years}")
                    st.success(f"Downloaded {len(st.session_state.downloaded_files)} NetCDF files: {', '.join([os.path.basename(f) for f in st.session_state.downloaded_files])}")
                else:
                    st.error("No files downloaded.")
        
        if st.session_state.downloaded_files:
            if st.button("Inspect NetCDF File Structure"):
                with st.spinner("Inspecting NetCDF file..."):
                    file_path = st.session_state.downloaded_files[0]
                    info = inspect_netcdf(file_path)
                    st.write("### NetCDF File Structure")
                    for key, value in info.items():
                        st.write(f"**{key}**: {value}")

    # Step 3: Process Rainfall Data
    if st.session_state.step_completed['download']:
        st.header("Step 3: Process Rainfall Data")
        if st.button("Process Data"):
            with st.spinner("Processing rainfall data..."):
                region = st.session_state.region
                daily_csv, seasonal_csv, yearly_csv, clipped_grid = process_data(
                    start_yr, end_yr, region['lat'], region['lon'], region['geometry']
                )
                if daily_csv is not None:
                    st.session_state.processed_data = {
                        'daily': daily_csv,
                        'seasonal': seasonal_csv,
                        'yearly': yearly_csv
                    }
                    st.session_state.grid_points = clipped_grid
                    st.session_state.step_completed['process'] = True
                    st.success("Data processed successfully.")
                else:
                    st.error("Failed to process data.")

    # Step 4: Generate Outputs
    if st.session_state.step_completed['process']:
        st.header("Step 4: Generate Map and View Results")
        if st.button("Generate Map and Tables"):
            region = st.session_state.region
            geometry = region['geometry']
            grid_points = st.session_state.grid_points
            region_name = region['name']
            
            st.subheader(f"Map of {region_name} with IMD Grid Points")
            fig = generate_map(geometry, grid_points, region_name)
            st.pyplot(fig)
            
            st.subheader("Rainfall Data Tables")
            daily_csv = st.session_state.processed_data['daily']
            seasonal_csv = st.session_state.processed_data['seasonal']
            yearly_csv = st.session_state.processed_data['yearly']
            
            st.subheader("Daily Rainfall Data")
            st.dataframe(daily_csv, use_container_width=True)
            
            st.subheader("Seasonal Rainfall Data")
            st.dataframe(seasonal_csv, use_container_width=True)
            
            st.subheader("Yearly Rainfall Data")
            st.dataframe(yearly_csv, use_container_width=True)
        
        if st.button("Generate Yearly Rainfall Maps"):
            region = st.session_state.region
            geometry = region['geometry']
            yearly_csv = st.session_state.processed_data['yearly']
            region_name = st.session_state.region['name']
            
            st.subheader("Yearly Rainfall Spatial Maps (IDW Interpolation)")
            with st.spinner("Generating yearly rainfall maps..."):
                figs = generate_yearly_rainfall_maps(geometry, yearly_csv, region_name)
                for fig in figs:
                    st.pyplot(fig)
        
        if st.button("Generate Rainfall Charts"):
            daily_csv = st.session_state.processed_data['daily']
            seasonal_csv = st.session_state.processed_data['seasonal']
            yearly_csv = st.session_state.processed_data['yearly']
            region_name = st.session_state.region['name']
            
            st.subheader("Rainfall Charts")
            with st.spinner("Generating rainfall charts..."):
                figs = generate_rainfall_charts(daily_csv, seasonal_csv, yearly_csv, region_name)
                for fig in figs:
                    st.pyplot(fig)
        
        if st.button("Perform Trend Analysis"):
            daily_csv = st.session_state.processed_data['daily']
            seasonal_csv = st.session_state.processed_data['seasonal']
            yearly_csv = st.session_state.processed_data['yearly']
            region_name = st.session_state.region['name']
            
            st.subheader("Trend Analysis (Mann-Kendall Test)")
            with st.spinner("Performing trend analysis..."):
                results, figs = perform_trend_analysis(daily_csv, seasonal_csv, yearly_csv, region_name)
                st.write("### Trend Analysis Results")
                st.table(results)
                for fig in figs:
                    st.pyplot(fig)
        
        if st.button("Export Daily Rainfall as Shapefile"):
            daily_csv = st.session_state.processed_data['daily']
            region_name = st.session_state.region['name']
            
            st.subheader("Export Daily Rainfall Data")
            with st.spinner("Generating shapefile..."):
                zip_data = export_daily_rainfall_shapefile(daily_csv, region_name)
                if zip_data:
                    st.download_button(
                        label="Download Daily Rainfall Shapefile",
                        data=zip_data,
                        file_name=f"{region_name}_daily_rainfall.zip",
                        mime="application/zip"
                    )
                    st.success("Shapefile generated successfully. Click to download.")
                else:
                    st.error("Failed to generate shapefile.")
