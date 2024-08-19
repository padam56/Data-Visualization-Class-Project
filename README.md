# HYCOM Data Visualization Project

## Overview:

This project visualizes oceanographic data from the HYCOM dataset, focusing on variables like temperature, salinity, surface elevation, and water velocity vectors. It includes a Jupyter notebook for data analysis and a Streamlit application for interactive visualization. The project aims to provide insights into ocean conditions by allowing users to interactively explore the data through various visualization techniques.

## Installation:

Ensure Python is installed on your system. Install the required libraries using pip:

```bash
pip install xarray pandas numpy matplotlib netCDF4 plotly streamlit
```

## Files:
CSCI6634_ClassProject_HYCOM DATA.ipynb: Demonstrates loading HYCOM data, computing vorticity, and visualizing data using matplotlib and Plotly.
Hycomviz_Streamlit.py: Streamlit application for interactive web-based visualization.
Dataset: https://myuno-my.sharepoint.com/:u:/g/personal/pthapa_uno_edu/EXXvyuqw0sRAiqSJ5Ba9cZ0Bq1zFnEWmD2juewZB9QXmZg?e=7fbJAG

## Usage:

Jupyter Notebook
Navigate to the notebook directory and launch Jupyter:

```bash
jupyter notebook CSCI6634_ClassProject_HYCOM DATA.ipynb
```

The notebook includes:

- Loading HYCOM data for specified depth and geographical coordinates.
- Computing vorticity from velocity data.
- Interactive visualization widgets for selecting variables, projection types, and visualization techniques.

## Streamlit Application

Run the Streamlit app:

```bash
streamlit run Hycomviz_Streamlit.py
```

The Streamlit app features:
- Sidebar for setting parameters like variable selection, projection type, and geographical range.
- Buttons to generate plots based on user inputs.
- Support for various plot types including vector plots, contour plots, density maps, and scatter maps.

Features:
- Data Loading: Load HYCOM data for specific depths and geographical coordinates.
- Vorticity Computation: Calculate vorticity from water velocity data.
- Interactive Visualization: Utilize Plotly and Streamlit for dynamic data exploration.
- Customizable Parameters: Select among different variables, projection types, and visualization techniques.
- Geographical Filtering: Adjust latitude and longitude ranges to focus on specific areas.


Contributors:
- Padam Jung Thapa (University of New Orleans)
