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
from pygame import mixer      # pip install pygame

import dash
from dash import Dash, dcc, html, Input, Output, State, callback, callback_context

import dash_bootstrap_components as dbc

# csvファイルの読み込み
l_sound_name = [] #サウンドの名前
l_content    = [] #サウンドが所属するグループ
l_data       = [] #サウンドの周波数情報
l_columns    = [] #サウンドの周波数情報ラベル

path_csv     = "https://github.com/smaruyama0115/synthesizer/raw/master/csv/"
path_sound   = "https://github.com/smaruyama0115/synthesizer/raw/master/sound/"

for current_dir, _, files_list in os.walk(path_csv):
    for file in files_list:
        if os.path.splitext(file)[1] == ".csv":

            # サウンド名と所属グループを取得
            l_sound_name.append(os.path.splitext(file)[0])
            l_content.append(os.path.split(current_dir)[-1])

            # csvファイルの読み込み
            df_tmp = pd.read_csv(os.path.join(current_dir, file))
            l_data.append(list(df_tmp["Power (dB)"]))

l_columns = list(df_tmp["Frequency (Hz)"])


df_csv = pd.DataFrame({"sound_name":l_sound_name, "content":l_content})

l_data = np.array(l_data).T
df_csv = pd.concat([df_csv, pd.DataFrame(zip(*l_data), columns=l_columns)], axis=1)

# 初めの3つのピークの最大値を100に合わせ、0以下は削除
df_processing = df_csv
df_processing.iloc[:,2:] = df_processing.iloc[:,2:].apply(lambda x: (x - max(x.iloc[6],x.iloc[12],x.iloc[18]) + 100), axis=1)
df_processing.iloc[:,2:] = df_processing.iloc[:,2:].map(lambda x: max(x,0))

# 特徴量をまとめたデータフレームを作成作成
df_analysis = pd.DataFrame({
    "sound_name" : df_csv["sound_name"],
    "content"    : df_csv["content"]
})

# 最初の方のピークを特徴量に加える
for i in range(7):
    col_name = "peak_" + str(i)
    amp_peak = df_processing.apply(lambda x: x.iloc[6*i+6], axis=1)
    df_analysis[col_name] = amp_peak

# 各周波数帯のピークを数える
amp_threshold        = [0, 10, 20, 30, 40, 50, 60, 60, 70, 80, 90, 100, 110, 120]
freq_threshold       = [ 500, 1000, 2000, 3000, 4000, 6000, 10000]
freq_threshold_index = [  24,   49,   93,  140,  186,  279,   465]

for freq_idx in range(len(freq_threshold)):
    for amp_idx in range(len(amp_threshold)):
        col_name = "count_" + str(freq_threshold[freq_idx]) + "_" + str(int(amp_threshold[amp_idx]*100))
        if freq_idx == len(freq_threshold) - 1:
            tmp = df_csv.iloc[:,2:].apply(lambda x: np.sum(x.iloc[freq_threshold_index[freq_idx]::6] > amp_threshold[amp_idx]), axis=1)
        else:
            tmp = df_csv.iloc[:,2:].apply(lambda x: np.sum(x.iloc[freq_threshold_index[freq_idx]:freq_threshold_index[freq_idx + 1]:6] > amp_threshold[amp_idx]), axis=1)
        df_analysis[col_name] = tmp

# 各列を正規化
df_analysis.iloc[:,2:] = df_analysis.iloc[:,2:].apply(lambda x: (x - np.mean(x)) / np.std(x) if np.std(x)>0 else 0, axis=0)

# T-SNE かけてみる
mapper = manifold.TSNE(random_state=0)
embedding_tsne = mapper.fit_transform(df_analysis.iloc[:,2:])

# グラフ用にデータ生計
embedding   = embedding_tsne

df_graph = pd.DataFrame({
    "sound_name"  : df_analysis["sound_name"],
    "content"     : df_analysis["content"],
    "embedding_x" : embedding[:,0]/np.max(np.abs(embedding_tsne[:, 0])),
    "embedding_y" : embedding[:,1]/np.max(np.abs(embedding_tsne[:, 1]))
})

# 列をサウンド名、行を周波数にとったデータフレームを作成
df_csv_t = df_csv.iloc[:,2:].T
df_csv_t.columns = df_csv["sound_name"]

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

        # wavファイルをロードして再生
        mixer.init()  # mixerを初期化
        mixer.music.load(wav_file)  # wavをロード
        mixer.music.play(1)  # wavを1回再生
    
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

app.run_server(debug=False)