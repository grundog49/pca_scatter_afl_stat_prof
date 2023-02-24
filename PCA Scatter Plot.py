###############################################################################
#                                PCA Dashboard                                #
###############################################################################

# Import packages for use

from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import pdist, squareform
import itertools
from scipy.spatial.distance import pdist
import seaborn as sns
from scipy.optimize import Bounds, minimize
from os import path
import seaborn as sns
from pyclustertend import compute_ivat_ordered_dissimilarity_matrix, compute_ordered_dissimilarity_matrix

# Define any functions and constants

###############################################################################
#                              Simple Line Graph                              #
###############################################################################

# Define the app to be formated

app = Dash(__name__)
server = app.server

# Define the colours to be used for the graphs

colors = {
    'background': '#FFFFFF',
    'text': '#000000'
}

# Import the data to be used for the graphic

df = pd.read_excel('https://github.com/grundog49/pca_scatter_afl_stat_prof/blob/38e5a1e001b6fc702b0127a7e6a2bd51dfe14000/AFL_player_data.xlsx')

# Manipulate the data to be used for the graphic

pca = PCA(n_components=3)
df[["PCA1","PCA2","PCA3"]] = pca.fit_transform(df.loc[:,"KI":"BO"])
df["Year"] = df["year"].astype(str)
df["Club"] = df["club"]
name_df = df['Player'].str.split(", ", expand=True)
df['Label'] = name_df.iloc[:,1]+' '+name_df.iloc[:,0]+', ('+df["Year"]+')'

# Define the figure to form the component of the dashboard

fig = px.scatter_3d(df, x='PCA1',y='PCA2',z='PCA3',color='club',animation_frame='Year', opacity=0.5, template='plotly_dark',hover_name="Label", color_discrete_map={
                "ADL": "#080067",
                "BRS": "#DB3D00",
                "COL": "#FFFFFF",
                "CAR": "#040031",
                "ESS": "#CC0000",
                "FRE":"#9300FF",
                'GCS':'#FF0000',
                'GWS':'#FF7B00',
                'HAW':'#FF8E00',
                'MEL':'#080067',
                'NM':'#1400FF',
                'PORT':'#00BDC1',
                'RICH':'#FFF300',
                'STK':'#FF0000',
                'SYD':'#FF0000',
                'WB':'#0005BE',
                'WCE':'#FFCA00'}
            )
fig.update_layout(
    font_family="sans-serif",
    font_color='white',
    font_size = 14,
    title_font_family="sans-serif",
    title_font_color='white',
    title_font_size = 24,
    title_x = 0.5,
    transition = {'duration': 2500}
    )

# Define and layout the app

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='AFL players between 1999 - 2022',
        style={
            'textAlign': 'center',
            'color': colors['text'],
            'font-family':"sans-serif",
            'padding': 10, 'flex': 1
        }
    ),

    html.Div(children='Visualising players statsitical profiles!', style={
        'textAlign': 'center',
        'color': colors['text'],
        'font-family':"sans-serif",
        'padding': 10, 'flex': 1
    }),
    dcc.Graph(
            id='graph',
            style= {'display':'flex', 'flex-direction':'column', 'height':'100%', 'width':'100%'},
            figure = fig
        )],
    )

if __name__ == '__main__':
    app.run_server()
