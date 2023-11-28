import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, time
import os
import netCDF4 as nc
import plotly.graph_objects as go
import plotly.express as px
import xarray as xr
import plotly.figure_factory as ff
from ipywidgets import interact, interact_manual, widgets

DATA_PATH = '/Users/padamjungthapa/Downloads/hycom_global_2016093000_t000.nc'

def load_data(depth, min_lat, max_lat, min_lon, max_lon):
    ds = xr.open_dataset(DATA_PATH, decode_times=False)
    ds_out = ds.where(ds['depth'] == depth, drop=True)

    ds_out = ds_out.where((ds_out['lat'] >= min_lat) & (ds_out['lat'] <= max_lat), drop=True)
    ds_out = ds_out.where((ds_out['lon'] >= min_lon) & (ds_out['lon'] <= max_lon), drop=True)

    data = ds_out.to_dataframe()
    return data, ds

default_min_lat = -90
default_max_lat = 90
default_min_lon = -180
default_max_lon = 180

data, _ = load_data(0, default_min_lat, default_max_lat, default_min_lon, default_max_lon)
print(data)


data, ds = load_data(0, default_min_lat, default_max_lat, default_min_lon, default_max_lon)

def plot_globe(projection, color, variable, density, contour, scatter, shuffle, resolution, depth, min_lat, max_lat, min_lon, max_lon):
    data, _ = load_data(depth, min_lat, max_lat, min_lon, max_lon)
    data = data.dropna()
    if (shuffle):
        data = data.sample(frac=1, random_state=42)



    if(variable == 'vectors'):
        
        directions = np.arctan2(data['water_v'], data['water_u'])
        magnitudes = np.sqrt(data['water_u']**2 + data['water_v']**2)

        directions_degrees = np.degrees(directions)
        
        fig = go.Figure()

        lats=np.vstack(data[::resolution].index.to_numpy())[:,2]
        lons=np.vstack(data[::resolution].index.to_numpy())[:,3]

        for i in range(len(data[::resolution])):
          
            lat_offset = magnitudes.values[i] * np.cos(np.radians(directions_degrees.values[i]))
            lon_offset = magnitudes.values[i] * np.sin(np.radians(directions_degrees.values[i]))
            
            fig.add_trace(go.Scattergeo(
                lat=[lats[i], (lats[i]+lat_offset)*1.02],#data['water_v'][::resolution].values[i]*10)],
                lon=[lons[i], (lons[i]+lon_offset)*1.02],#data['water_u'][::resolution].values[i]*10)],
                mode='lines',
                line=dict(color='black', width=2),
                hoverinfo='skip',
                showlegend=False
            ))

            fig.add_trace(go.Scattergeo(
                lat=[lats[i]],
                lon=[lons[i]],
                mode='markers',
                marker=dict(color='red', size=3),
                showlegend=False
            ))


        fig.update_geos(projection_type=projection, landcolor="white", oceancolor="LightSkyBlue", showcoastlines=True)
        fig.update_layout(scene=dict(aspectmode="auto"))

    elif(contour):
        lats=np.vstack(data[::resolution].index.to_numpy())[:,2]
        lons=np.vstack(data[::resolution].index.to_numpy())[:,3]
        fig=go.Figure()
        
        lats=np.vstack(data[::resolution].index.to_numpy())[:,2]
        lons=np.vstack(data[::resolution].index.to_numpy())[:,3]
        fig.add_trace(go.Contour(
            x=lons,
            y=lats,
            z=data[variable][::resolution],
            #line_smoothing=0.85,
            colorscale=color,
            colorbar=dict(title=f'{variable}'),
            contours=dict(showlabels=True, labelfont=dict(color='white')),
            connectgaps = True
        ))
        
    elif(density):
        lats=np.vstack(data[::resolution].index.to_numpy())[:,2]
        lons=np.vstack(data[::resolution].index.to_numpy())[:,3]
    
        fig = px.density_mapbox(data[::resolution], lat=lats, lon=lons, z=f'{variable}', radius=10,
                        center=dict(lat=0, lon=180), zoom=0,
                        color_continuous_scale = color,
                        mapbox_style='open-street-map')

    elif(scatter):

        lats=np.vstack(data[::resolution].index.to_numpy())[:,2]
        lons=np.vstack(data[::resolution].index.to_numpy())[:,3]

        fig = px.scatter_mapbox(data[::resolution], lat=lats, lon=lons,
                            color='water_temp',
                            # size='',
                            color_continuous_scale=color,
                            zoom=3,
                            center={"lat": 0, "lon": 0},
                            mapbox_style="carto-positron")

        
    else:
        fig = px.scatter_geo(data[::resolution],
                             lat=np.vstack(data[::resolution].index.to_numpy())[:,2],
                             lon=np.vstack(data[::resolution].index.to_numpy())[:,3],
                             color=variable,
                             projection=projection,
                             color_continuous_scale=color,
                             color_continuous_midpoint=data.describe()[variable]['mean'])

        fig.update_geos(projection_type=projection)

        fig.update_layout(coloraxis_colorbar=dict(title='Legend'))
  
    fig.show()
    
proj_list=['natural earth', 'equirectangular', 'mercator', 'orthographic', 'hammer',  'robinson']



# Sliders and checkboxes
variable_dropdown = widgets.Dropdown(options=['water_temp', 'salinity', 'surf_el', 'vectors'],
                                    value='water_temp', description='Select Variable:')

projection_dropdown = widgets.Dropdown(options=proj_list,
                                    value='natural earth', description='Select Projection:')

color_dropdown = widgets.Dropdown(options=['viridis', 'icefire', 'agsunset', 'purples', 'mint'],
                                    value='viridis', description='Select Color:')



shuffle_checkbox = widgets.Checkbox(value=False, description='Random Sample')
scatter_checkbox = widgets.Checkbox(value=False, description='Scatter MapBox Plot')
contour_checkbox = widgets.Checkbox(value=False, description='Contour Plot')
density_checkbox = widgets.Checkbox(value=False, description='Density Plot')
resolution_slider = widgets.IntSlider(value=5000, min=1, max=20000, step=1, description='Resolution:')

depth_slider = widgets.SelectionSlider(
    options=ds['depth'].values,
    value=ds['depth'].values[0],
    description='Depth:',
)

min_lat_slider = widgets.FloatSlider(value=-90, min=-90, max=90, step=0.1, description='Min Latitude:')
max_lat_slider = widgets.FloatSlider(value=90, min=-90, max=90, step=0.1, description='Max Latitude:')
min_lon_slider = widgets.FloatSlider(value=-180, min=-180, max=180, step=0.1, description='Min Longitude:')
max_lon_slider = widgets.FloatSlider(value=180, min=-180, max=180, step=0.1, description='Max Longitude:')


interact_manual(plot_globe, color=color_dropdown, contour=contour_checkbox,
                density=density_checkbox,
                shuffle=shuffle_checkbox,
                scatter = scatter_checkbox,
                projection=projection_dropdown,
                variable=variable_dropdown,
                resolution=resolution_slider, depth=depth_slider,
                min_lat=min_lat_slider, max_lat=max_lat_slider,
                min_lon=min_lon_slider, max_lon=max_lon_slider)


# Streamlit app layout
st.title('Ocean Data Visualization')
    
# Sidebar for user inputs
with st.sidebar:
    st.header('Settings')
    
    # Adding latitude and longitude sliders
    min_lat = st.slider('Min Latitude:', -90.0, 90.0, -90.0)
    max_lat = st.slider('Max Latitude:', -90.0, 90.0, 90.0)
    min_lon = st.slider('Min Longitude:', -180.0, 180.0, -180.0)
    max_lon = st.slider('Max Longitude:', -180.0, 180.0, 180.0)

    variable = st.selectbox('Select Variable:', ['water_temp', 'salinity', 'surf_el', 'vectors'])
    projection = st.selectbox('Select Projection:', ['natural earth', 'equirectangular', 'mercator', 'orthographic', 'hammer', 'robinson'])
    color = st.selectbox('Select Color:', ['viridis', 'icefire', 'agsunset', 'purples', 'mint'])
    density = st.checkbox('Density Plot', value=False)
    shuffle = st.checkbox('Random Sample', value=False)
    contour = st.checkbox('Contour Plot', value=False)
    scatter_mapbox = st.checkbox('Scatter MapBox Plot', value=False)  # Added checkbox for Scatter MapBox Plot
    resolution = st.slider('Resolution:', 1, 20000, 5000)

# Main area
if st.button('Generate Plot'):
    data, ds = load_data(0, min_lat, max_lat, min_lon, max_lon)  # Load data with user-defined parameters
    plot_globe(projection, color, variable, density, contour, scatter_mapbox, shuffle, resolution, 0, min_lat, max_lat, min_lon, max_lon)

