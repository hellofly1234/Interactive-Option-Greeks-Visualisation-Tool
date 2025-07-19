#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Use %pip to install necessary libraries.
get_ipython().run_line_magic('pip', 'install yfinance ipywidgets plotly')


# In[3]:


# Enable ipywidgets extension.
# Run this with ! to execute it in the shell, not Python.
get_ipython().system('jupyter nbextension enable --py widgetsnbextension --sys-prefix')


# In[1]:


# Standard libraries
import numpy as np
import pandas as pd
from scipy.stats import norm

# Widgets and display tools for interactivity
import ipywidgets as widgets
from IPython.display import display

# For 3D plotting
import plotly.graph_objects as go

# For saving files in memory
import io


# In[2]:


# Black-Scholes option pricing and Greeks
def compute_all_greeks_and_price(S, K, T, r, sigma, option='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option == 'call':
        delta = norm.cdf(d1)
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d2)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = -norm.cdf(-d1)
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return delta, gamma, theta, vega, rho, price


# In[3]:


# Generates meshgrid for strike prices and expiry times, then computes selected Greek
def generate_surface_data(S, r, sigma, param_type, option='call'):
    strike_prices = np.linspace(100, 200, 30)
    days_to_expiry = np.linspace(5, 120, 30) / 365

    K_grid, T_grid = np.meshgrid(strike_prices, days_to_expiry)

    delta, gamma, theta, vega, rho, price = compute_all_greeks_and_price(S, K_grid, T_grid, r, sigma, option)

    greek_map = {
        'Delta': delta,
        'Gamma': gamma,
        'Theta': theta,
        'Vega': vega,
        'Rho': rho,
        'Price': price
    }

    Z = greek_map[param_type]

    return K_grid, T_grid, Z


# In[4]:


# 3D plot of the surface using Plotly
def plot_surface_plotly(K_grid, T_grid, Z, param_type):
    fig = go.Figure(data=[go.Surface(
        z=Z,
        x=K_grid,
        y=T_grid * 365,
        colorscale='Viridis',
        colorbar=dict(title=param_type)
    )])

    fig.update_layout(
        title=f"{param_type} Surface",
        scene=dict(
            xaxis_title='Strike Price',
            yaxis_title='Days to Expiry',
            zaxis_title=param_type,
        ),
        autosize=True,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    fig.show()


# In[5]:


# Converts the plot data into a CSV-ready DataFrame
def export_csv(K_grid, T_grid, Z, param_type):
    df = pd.DataFrame({
        'Strike Price': K_grid.flatten(),
        'Days to Expiry': (T_grid.flatten() * 365),
        param_type: Z.flatten()
    })
    return df

# Exports interactive HTML version of the plot
def export_html(fig, filename="plot.html"):
    fig.write_html(filename)


# In[6]:


# Create interactive controls
param_dropdown = widgets.Dropdown(
    options=['Delta', 'Gamma', 'Theta', 'Vega', 'Rho', 'Price'],
    value='Delta',
    description='Parameter:',
)

option_dropdown = widgets.Dropdown(
    options=['call', 'put'],
    value='call',
    description='Option Type:',
)

S_slider = widgets.IntSlider(value=150, min=50, max=300, step=5, description='Stock Price:')
sigma_slider = widgets.FloatSlider(value=0.3, min=0.1, max=1.0, step=0.01, description='Volatility:')
r_slider = widgets.FloatSlider(value=0.05, min=0.0, max=0.1, step=0.001, description='Risk-free Rate:')

button_export_csv = widgets.Button(description="Export CSV")
button_export_html = widgets.Button(description="Export Plot HTML")

# Output areas
output_plot = widgets.Output()
output_export = widgets.Output()


# In[11]:


# Stores the last computed figure and data for export functions
last_fig = None
last_data = None

def update_surface(param_type, option, S, sigma, r):
    global last_fig, last_data
    K_grid, T_grid, Z = generate_surface_data(S, r, sigma, param_type, option)
    output_plot.clear_output(wait=True)
    with output_plot:
        fig = go.Figure(data=[go.Surface(
            z=Z,
            x=K_grid,
            y=T_grid * 365,
            colorscale='Viridis',
            colorbar=dict(title=param_type)
        )])
        fig.update_layout(
            title=f"{param_type} Surface",
            scene=dict(
                xaxis_title='Strike Price',
                yaxis_title='Days to Expiry',
                zaxis_title=param_type,
            ),
            autosize=True,
            margin=dict(l=65, r=50, b=65, t=90)
        )
        fig.show()
    last_fig = fig
    last_data = (K_grid, T_grid, Z, param_type)

def export_csv_clicked(b):
    global last_data
    if last_data is None:
        with output_export:
            output_export.clear_output()
            print("No data to export yet.")
        return
    K_grid, T_grid, Z, param_type = last_data
    df = export_csv(K_grid, T_grid, Z, param_type)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_str = csv_buffer.getvalue()
    
    # Save the CSV to file and display preview
    with output_export:
        output_export.clear_output()
        print("CSV Export Preview (first 5 rows):")
        display(df.head())
        print("\nDownload your CSV file below:")
        from IPython.display import FileLink
        filename = "greek_surface_data.csv"
        with open(filename, "w") as f:
            f.write(csv_str)
        display(FileLink(filename))

def export_html_clicked(b):
    global last_fig
    if last_fig is None:
        with output_export:
            output_export.clear_output()
            print("No plot to export yet.")
        return
    filename = "greek_surface_plot.html"
    last_fig.write_html(filename)
    with output_export:
        output_export.clear_output()
        print("The file has been saved in the current directory.")

# Attach the export button click handlers
button_export_csv.on_click(export_csv_clicked)
button_export_html.on_click(export_html_clicked)


# In[12]:


# Bundle all widgets into a layout
ui = widgets.VBox([
    param_dropdown,
    option_dropdown,
    S_slider,
    sigma_slider,
    r_slider,
    widgets.HBox([button_export_csv, button_export_html]),
    output_export,
    output_plot,
])

# Create interactive output handler
out = widgets.interactive_output(update_surface, {
    'param_type': param_dropdown,
    'option': option_dropdown,
    'S': S_slider,
    'sigma': sigma_slider,
    'r': r_slider
})

# Display UI and output side by side
display(ui, out)


# In[ ]:




