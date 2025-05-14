import streamlit as st
import os
import tempfile
import zipfile
import requests
import xarray as xr
import rioxarray
import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import mapping
import pandas as pd
from rasterio.mask import mask

# ---------------------------
# Helper functions
# ---------------------------
def download_imd_nc(start_year, end_year, download_folder):
    os.makedirs(download_folder, exist_ok=True)
    url = "https://www.imdpune.gov.in/cmpg/Griddata/RF25.php"
    for year in range(start_year, end_year + 1):
        payload = {"RF25": str(year)}
        resp = requests.post(url, data=payload, stream=True)
        resp.raise_for_status()
        filename = os.path.join(download_folder, f"RF25_{year}.nc")
        with open(filename, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return download_folder


def nc_to_daily_tifs(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(input_folder):
        if not fname.endswith('.nc'): continue
        year = int(fname.split('_')[-1].split('.')[0])
        path = os.path.join(input_folder, fname)
        ds = xr.open_dataset(path)
        if year > 2023:
            var_name, time_var = 'rf', 'time'
        else:
            var_name, time_var = 'RAINFALL', 'TIME'
        da = ds[var_name]
        for t in da.coords[time_var]:
            arr = da.sel({time_var: t})
            arr.rio.write_crs('EPSG:4326', inplace=True)
            day_str = str(t.values).split('T')[0]
            out_tif = os.path.join(output_folder, f"imd_pcp_{day_str}.tif")
            arr.rio.to_raster(out_tif, driver='GTiff')
    return output_folder


def daily_to_monthly(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for year_month in sorted({f[8:15] for f in os.listdir(input_folder) if f.endswith('.tif')}):
        parts = year_month.split('-')
        year, month = int(parts[0]), int(parts[1])
        files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f[8:15]==year_month]
        if not files: continue
        arrays, meta, nodata = [], None, None
        for f in files:
            with rasterio.open(f) as src:
                d = src.read(1).astype(np.float32)
                if meta is None:
                    meta, nodata = src.meta.copy(), src.nodata or -9999
                    meta.update(nodata=nodata)
                d[d==nodata] = np.nan
                arrays.append(d)
        stack = np.stack(arrays)
        msum = np.nansum(stack, axis=0)
        msum[np.all(np.isnan(stack), axis=0)] = nodata
        out = os.path.join(output_folder, f"imd_pcp_m_{year}_{month:02d}.tif")
        meta.update(dtype=rasterio.float32, count=1)
        with rasterio.open(out, 'w', **meta) as dst:
            dst.write(msum, 1)
    return output_folder


def monthly_to_annual(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    years = sorted({int(f.split('_')[3]) for f in os.listdir(input_folder)})
    for year in years:
        files = []
        for m in list(range(6,13)) + list(range(1,6)):
            y = year if m>=6 else year+1
            p = os.path.join(input_folder, f"imd_pcp_m_{y}_{m:02d}.tif")
            if os.path.exists(p): files.append(p)
        arrays, meta, nodata = [], None, None
        for f in files:
            with rasterio.open(f) as src:
                d = src.read(1).astype(np.float32)
                if meta is None:
                    meta, nodata = src.meta.copy(), src.nodata or -9999
                    meta.update(nodata=nodata)
                d[d==nodata] = np.nan
                arrays.append(d)
        stack = np.stack(arrays)
        asum = np.nansum(stack, axis=0)
        asum[np.all(np.isnan(stack), axis=0)] = nodata
        out = os.path.join(output_folder, f"imd_pcp_a_{year}_{year+1}.tif")
        meta.update(dtype=rasterio.float32, count=1)
        with rasterio.open(out, 'w', **meta) as dst:
            dst.write(asum, 1)
    return output_folder


def clip_and_export(raster_folder, shapefile_path, out_tif_folder, out_csv_path):
    os.makedirs(out_tif_folder, exist_ok=True)
    gdf = gpd.read_file(shapefile_path)
    geoms = [mapping(geom) for geom in gdf.geometry]
    records = []
    for tif in sorted(os.listdir(raster_folder)):
        if not tif.endswith('.tif'): continue
        src_path = os.path.join(raster_folder, tif)
        with rasterio.open(src_path) as src:
            out_img, out_transform = mask(src, geoms, crop=True)
            meta = src.meta.copy()
            meta.update({
                'height': out_img.shape[1],
                'width': out_img.shape[2],
                'transform': out_transform
            })
            out_tif = os.path.join(out_tif_folder, tif)
            with rasterio.open(out_tif, 'w', **meta) as dst:
                dst.write(out_img)
            data = out_img[0]
            rows, cols = np.where(data != src.nodata)
            xs, ys = rasterio.transform.xy(out_transform, rows, cols)
            for x, y, v in zip(xs, ys, data[rows, cols]):
                records.append({'x': x, 'y': y, 'value': float(v), 'file': tif})
    pd.DataFrame(records).to_csv(out_csv_path, index=False)
    return out_tif_folder, out_csv_path

# ---------------------------
# Streamlit App
# ---------------------------
def main():
    st.title("📊 IMD Precipitation Processing Dashboard")
    st.sidebar.header("Configuration")
    start_year, end_year = st.sidebar.slider(
        "Select year range", min_value=1951, max_value=2024, value=(2023, 2024)
    )
    shapefile = st.sidebar.file_uploader(
        "Upload area boundary (KML, GeoJSON, or ZIP with .shp)",
        type=["kml", "geojson", "zip"]
    )

    if st.sidebar.button("Start Processing"):
        if shapefile is None:
            st.error("Please upload a valid shapefile boundary before proceeding.")
            return
        with tempfile.TemporaryDirectory() as tmp:
            if shapefile.type == "application/zip":
                zf = zipfile.ZipFile(shapefile)
                zf.extractall(tmp)
                shp_file = next((os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith('.shp')), None)
            else:
                ext = os.path.splitext(shapefile.name)[1]
                shp_file = os.path.join(tmp, f"boundary{ext}")
                with open(shp_file, 'wb') as f:
                    f.write(shapefile.getbuffer())
            if not shp_file or not os.path.exists(shp_file):
                st.error("No .shp file found in the uploaded archive.")
                return

            st.info("🔄 Downloading data...")
            nc_folder = download_imd_nc(start_year, end_year, "imd_netcdf")
            st.info("🔄 Converting to daily TIFFs...")
            daily_folder = nc_to_daily_tifs(nc_folder, "pcp_imd_daily")
            st.info("🔄 Aggregating monthly...")
            monthly_folder = daily_to_monthly(daily_folder, "pcp_imd_monthly")
            st.info("🔄 Aggregating annual...")
            annual_folder = monthly_to_annual(monthly_folder, "pcp_imd_annual")
            st.info("🔄 Clipping and exporting...")
            tif_out, csv_out = clip_and_export(annual_folder, shp_file, "clipped_tifs", "clipped_data.csv")

            st.success("✅ Processing complete!")
            with open(csv_out, 'rb') as f:
                st.download_button(
                    label="Download CSV of clipped data",
                    data=f,
                    file_name="clipped_data.csv",
                    mime="text/csv"
                )
            zip_path = os.path.join(os.getcwd(), "clipped_tifs.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for root, _, files in os.walk(tif_out):
                    for file in files:
                        zipf.write(os.path.join(root, file), arcname=file)
            with open(zip_path, 'rb') as f:
                st.download_button(
                    label="Download clipped TIFFs",
                    data=f,
                    file_name="clipped_tifs.zip",
                    mime="application/zip"
                )

if __name__ == "__main__":
    main()
