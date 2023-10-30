# ライブラリの読み込み
import os

import numpy as np
import pandas as pd

from scipy.io import wavfile
import wave
import base64

import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, callback, callback_context, clientside_callback
import dash_bootstrap_components as dbc

from dotenv import load_dotenv
load_dotenv(".env")

# pathの設定
path_sound = "./sound"

# 使用する音源

if os.environ.get('USE_ALL_SOUNDS', False):
    l_content       = ['Genre','Sample','KOMPLETE','OSC', 'Leads','ONII-CHAN Lead','NextLight Serum Free LD','Synth','ONII-CHAN Chord','Plucked','ONII-CHAN Pluck',"ONII-CHAN Pad",'Noise']
    l_content_color = ["","#EFCAD6","#658080","#70C5CA","#8CA231","#E2DA56","#BA2320","#924727","#68230D","#B032EB","#A8CC8C", "#CE306A","#6B2220"]
    l_content_init  = ['Genre', 'Sample', 'KOMPLETE', 'OSC']
else:
    # 本番環境で使う音源をここに書く
    l_content       = ['Genre','Sample', 'OSC', 'Leads','ONII-CHAN Lead']
    l_content_color = ["","#EFCAD6","#658080","#70C5CA","#E2DA56",]
    l_content_init  = ['Genre', 'Sample', 'KOMPLETE', 'OSC']

# グラフ用データの読み込み
df_graph = pd.read_csv("./DataFrame/df_graph.csv", index_col = 0)
df_csv_t = pd.read_csv("./DataFrame/df_csv_t.csv", index_col = 0)

# クラスタの特徴量

dict_cluster_name ={
    0: "Saw",
    1: "Square",
    2: "Cluster2",
    3: "Cluster3",
    4: "Hard Synth",
    5: "Flute",
    6: "Organ",
    7: "Cluster7",
    8: "Cluster8",
    9: "Synth",
    10: "Cluster10",
    11: "Cluster11",
    12: "Brass",
    13: "Piano",
    14: "Cluster14",
    15: "Cluster15",
    16: "Cluster16",
    17: "Cluster17",
    18: "Cluster18",
    19: "Cluster19"
}

dict_cluster_color ={
    0:"black",
    1:"black",
    2:"black",
    3:"black",
    4:"black",
    5:"black",
    6:"black",
    7:"black",
    8:"black",
    9:"black",
    10:"black",
    11:"black",
    12:"black",
    13:"black",
    14:"black",
    15:"black",
    16:"black",
    17:"black",
    18:"black",
    19:"black"
}

# グラフ作る
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
            "range":[-1.1,1.1]
        },
        yaxis = {
            "title":None,
            "range":[-1.1,1.1]
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
        # margin=dict(l=20, r=5, t=50, b=0),
        paper_bgcolor='rgba(0,0,0,0)'
    )

fig_scatter = go.Figure(layout=layout_scatter)

for i, content in enumerate(l_content):
    if content == "Genre": continue
    else:
        df = df_graph[df_graph["content"]==content]
        fig_scatter.add_trace(go.Scatter(
            x = df["embedding_x"],
            y = df["embedding_y"],
            marker_color = l_content_color[i],
            mode = 'markers',
            text = df["sound_name"],
            hovertemplate = "%{text}",
            name = content
        ))

if "Genre" in l_content:
    df_only_center = df_graph[df_graph["content"] == "Center"]
    for cluster_name, cluster_id in zip(df_only_center["sound_name"], df_only_center["cluster"]):

        # Genre領域のサイズを決定
        size = df_graph[df_graph["cluster"] == cluster_id].shape[0]
        if size < 10: continue
        size = size*1.5

        fig_scatter.add_trace(go.Scatter(
            x = df_only_center[df_only_center["cluster"] == cluster_id]["embedding_x"],
            y = df_only_center[df_only_center["cluster"] == cluster_id]["embedding_y"],
            marker_size  = size,
            marker_color = dict_cluster_color[cluster_id],
            marker_opacity = 0.2,
            mode = 'markers+text',
            text = dict_cluster_name[cluster_id],
            textposition='top center',
            hoverinfo = 'skip',
            name = "Genre"
        ))    

fig_line = go.Figure(layout = layout_line)

# ダッシュボードに表示
app = Dash(external_stylesheets=[dbc.themes.FLATLY])
server = app.server
sidebar = html.Div(
    [
        dbc.Row(
        [
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
                        dcc.Checklist(["All"], [], id="all-checklist", inline=True),
                        dcc.Checklist(l_content, l_content_init, id="category-checklist",inline=False),
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
                        df_graph[df_graph["content"] != "Center"]["sound_name"], multi=True, id="dropdown")
                ])
            ]
        ),
        dbc.Row([
            html.Div(id="placeholder", style={"display": "none"})
        ])
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
    Output("placeholder","children"),
    Input("graph_scatter", "clickData")
)
def sound(clickData):
    fig_line = go.Figure()
    audio    = ""
    if clickData:

        sound_name = clickData['points'][0]['text']
        content    = df_graph[df_graph["sound_name"] == sound_name]["content"].values[0]

        if content != "Genre":

            wav_file         = os.path.join(path_sound, content , sound_name) + ".wav"
            wav_file_encoded = base64.b64encode(open(wav_file, 'rb').read())

            # スペクトルを表示
            fig_line = go.Figure()
            fig_line.add_trace(
                go.Scatter(
                    x = df_csv_t.index,
                    y = df_csv_t[sound_name]
                )
            )

            audio = html.Audio(
                id='audio-player2',
                src='data:audio/mpeg;base64,{}'.format(wav_file_encoded.decode()),
                controls=False,
                autoPlay=True,
            )

            fig_line.update_xaxes(type="log")

    return fig_line, audio

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

    for i, content in enumerate(l_content):
        if content in category_selected:
            if content == "Genre": continue
            else:
                df = df_graph[df_graph["content"]==content]
                fig.add_trace(go.Scatter(
                    x = df["embedding_x"],
                    y = df["embedding_y"],
                    marker_color = l_content_color[i],
                    mode = 'markers',
                    text = df["sound_name"],
                    hovertemplate = "%{text}",
                    name = content
                ))

    if "Genre" in category_selected:
        df_only_center = df_graph[df_graph["content"] == "Center"]
        for cluster_name, cluster_id in zip(df_only_center["sound_name"], df_only_center["cluster"]):

            # Genre領域のサイズを決定
            size = df_graph[df_graph["cluster"] == cluster_id].shape[0]
            if size < 10: continue
            size = size*1.5

            fig.add_trace(go.Scatter(
                x = df_only_center[df_only_center["cluster"] == cluster_id]["embedding_x"],
                y = df_only_center[df_only_center["cluster"] == cluster_id]["embedding_y"],
                marker_size  = size,
                marker_color = dict_cluster_color[cluster_id],
                marker_opacity = 0.2,
                mode = 'markers+text',
                text = dict_cluster_name[cluster_id],
                textposition='top center',
                hoverinfo = 'skip',
                name = "Genre"
            ))    

    if dropdown is not None:
        df_graph_dropdown = df_graph[df_graph["sound_name"].isin(dropdown)]
        fig.add_trace(go.Scatter(
            x = df_graph_dropdown["embedding_x"],
            y = df_graph_dropdown["embedding_y"],
            mode = 'markers',
            marker = {
                "color" : "red",
                "size"  : 15
            },
            text = df_graph_dropdown["sound_name"],
            hovertemplate = "%{text}",
            name = "Search Result"
        ))

    return category_selected, all_selected, fig

if __name__=='__main__':
    app.run_server(debug=True, port = int(os.environ.get('PORT', 1238)))