import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn import manifold

from IPython.display import display, HTML
import IPython.display

import plotly.offline as offline
import plotly.graph_objs as go
import plotly.express as px

from scipy.io import wavfile  # install : conda install scipy
# from pygame import mixer      # pip install pygame

import dash
from dash import Dash, dcc, html, Input, Output, State, callback, callback_context

import dash_bootstrap_components as dbc

# pathの設定
path_sound = "./sound"

# グラフ用データの読み込み
df_graph = pd.read_csv("./DataFrame/df_graph.csv", index_col = 0)
df_csv_t = pd.read_csv("./DataFrame/df_csv_t.csv", index_col = 0)

# グラフ作る
l_content = ['Sample', 'OSC', 'Leads','Plucked','ONII-CHAN Lead','ONII-CHAN Pluck']

layout_scatter = go.Layout(
        title={
            "text" : "<b>Sound Map",
            "font" : {
                "size"  : 26,
                "color" : "gray"
            }
        },
        showlegend=False,
        margin=dict(l=80, r=80, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis = {
            "title":None,
            "range":[-1.2,1.2]
        },
        yaxis = {
            "title":None,
            "range":[-1.2,1.2]
        }
    )

layout_line = go.Layout(
        title={
            "text" : "<b>Spectrum",
            "font" : {
                "size"  : 26,
                "color" : "gray",
            }
        },
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)'
    )

fig_scatter = go.Figure(layout=layout_scatter)

for content in l_content:
    df = df_graph[df_graph["content"]==content]
    fig_scatter.add_trace(go.Scatter(
        x = df["embedding_x"],
        y = df["embedding_y"],
        mode = 'markers',
        text = df["sound_name"],
        hovertemplate = "%{text}",
        name = content
    ))

fig_line = go.Figure(layout = layout_line)

# ダッシュボードに表示
app = Dash(external_stylesheets=[dbc.themes.FLATLY])

sidebar = html.Div(
    [
        dbc.Row(
        [
            html.P(id='placeholder'),
            html.H5(
                'Sound Visualizer',
                className="ml-2 text-white font-italic"
            )
        ],
        className='bg-primary'
        ),

        dbc.Row(
            [
                html.P(
                    'Search by Category',
                    style={'margin-top': '16px', 'margin-bottom': '4px'},
                    className='font-weight-bold'),
                html.Div(
                    [
                        dcc.Checklist(["All"], ['All'], id="all-checklist", inline=True),
                        dcc.Checklist(l_content, l_content, id="category-checklist",inline=False),
                    ],
                    style={'margin-left': '8px'},
                )
            ],
        ),
        dbc.Row(
            [
                html.P(
                    'Search by Sound Name',
                    style={'margin-top': '16px', 'margin-bottom': '4px'},
                    className='font-weight-bold'
                ),
                html.Div([
                    dcc.Dropdown(
                        df_graph["sound_name"], multi=True, id="dropdown")
                ])
            ]
        )
    ]
)

content = html.Div(
    [
        dbc.Row(
            [
                html.Div([
                    dcc.Graph(
                        id='graph_scatter',
                        figure=fig_scatter
                    )
                ]),
            ],
        ),
        dbc.Row(
            [
                html.Div([
                    dcc.Graph(
                        id     = 'graph_line',
                        figure = fig_line
                    )
                ])
            ],
        )
    ],
    style={"height": "80vh", 'margin': '8px'}
)


app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(sidebar, width=3, className='bg-light'),
                dbc.Col(content, width=9)
            ],
            style={"height": "100vh"}
            ),
    ],
    fluid=True
    )

@app.callback(
    Output("graph_line","figure"),
    Input("graph_scatter", "clickData")
)
def sound(clickData):

    fig_line = go.Figure()

    if clickData:

        sound_name = clickData['points'][0]['text']
        content    = df_graph[df_graph["sound_name"] == sound_name]["content"].values[0]

        wav_file   = os.path.join(path_sound, content , sound_name) + ".wav"

        # # wavファイルをロードして再生
        # mixer.init()  # mixerを初期化
        # mixer.music.load(wav_file)  # wavをロード
        # mixer.music.play(1)  # wavを1回再生
    
        fig_line = go.Figure()
        fig_line.add_trace(
            go.Scatter(
                x = df_csv_t.index,
                y = df_csv_t[sound_name]
            )
        )

        fig_line.update_xaxes(type="log")

    return fig_line

@callback(
    Output("category-checklist", "value"),
    Output("all-checklist", "value"),
    Output("graph_scatter", "figure"),
    Input("category-checklist", "value"),
    Input("all-checklist", "value"),
    Input("dropdown", "value"),
)
def sync_checklists(category_selected, all_selected, dropdown):
    ctx = callback_context
    input_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if input_id == "category-checklist":
        all_selected = ["All"] if set(category_selected) == set(l_content) else []
    else:
        category_selected = l_content if all_selected else []


    fig = go.Figure(layout=layout_scatter)

    for content in l_content:
        if content in category_selected:
            df = df_graph[df_graph["content"]==content]
            fig.add_trace(go.Scatter(
                x = df["embedding_x"],
                y = df["embedding_y"],
                mode = 'markers',
                text = df["sound_name"],
                hovertemplate = "%{text}",
                name = content
            ))
    
    if dropdown is not None:
        df_graph_dropdown = df_graph[df_graph["sound_name"].isin(dropdown)]
        fig.add_trace(go.Scatter(
            x = df_graph_dropdown["embedding_x"],
            y = df_graph_dropdown["embedding_y"],
            mode = 'markers',
            marker = {
                "color" : "green",
                "size"  : 15
            },
            text = df_graph_dropdown.index.values,
            hovertemplate = "%{text}",
            name = "Search Result"
        ))

    return category_selected, all_selected, fig

port = int(os.environ.get("PORT", 5000))
# app.run_server(debug=True, port=port)

server = app.server
app.run_server(debug=True, port=port)