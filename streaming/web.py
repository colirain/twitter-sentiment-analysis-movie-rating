#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Tingyu Li (tl2861)
# ---------------------------

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Output, Input
import plotly
import plotly.graph_objs as go
from collections import deque
import flask
import csv
import numpy as np
# from django_plotly_dash import DjangoDash

MOVIE_SELECT = None
DF = None
DF_POINTER = 0
X = deque(maxlen = 6)
Y = deque(maxlen = 6)

tags = ['AvengersEndgame','CaptainMarvel','gameofthrones']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash('web', external_stylesheets=external_stylesheets)

app.config.suppress_callback_exceptions = True

app.layout = html.Div(
    [
        # html.Img(src='1.jpg'),
        html.Div([

            html.Span("REAL TIME MOVIE REVIEW - TWITTER", className='app-title')
            ],
            className="row header"
            ),
        html.Div([
            html.Div([
                # html.Div(id = 'hot-movie'),
                html.H4("Hotest HashTags", style = {"text-align" : "center"}),
                dcc.Graph(id = 'hot-movie', animate = True),
            ],
            style={
                # "verticalAlign":"left",
                "margin": "5%",
                "width": "40%",
                "height" : "400px",
                "float" : "left",
                },
            ),
            html.Div([
                html.H4("Elo Rank", style = {"text-align" : "center"}),
                html.Div(id = 'elo-rank'),
            ],
            style={
                # "verticalAlign":"right",
                "margin": "5%",
                "width": "40%",
                "height" : "400px",
                "float" : "right",
                },
            ),
        ],
        style = {
            "height" : "400px",
        }
        ),
        dcc.Interval(
                id = 'tag-update',
                interval = 1*1000,
                n_intervals=0
                ),
        dcc.Interval(
                id = 'elo-update',
                interval = 1*1000,
                n_intervals = 0
                ),
        html.Div([
            html.H4("Live Movie Score", style = {"text-align" : "center"}),
            dcc.Dropdown(
                id = 'movie-select',
                options = [{'label':i, 'value':i} for i in tags],
                value = 'None',
            ),
            dcc.Graph(id='live-graph', animate = True, )
        ],
        style = {
            "margin" : "5%",
            # "margin-top" : "10%"
            "padding-top" : "100px"
            },
        ),
        dcc.Interval(
            id = 'graph-update',
            interval = 1*1000,
            n_intervals=0
            ),
        html.Div([
            html.H4("Historical Moving Average Movie Score", style = {"text-align" : "center"}),
            dcc.Graph(id='live-history', animate = True),
            dcc.Interval(
                id = 'history-update',
                interval = 1*1000,
                n_intervals = 0
                ),
        ],
        style = {
            "margin" : "5%",
            # "margin-top" : "1%"
            },
        ),


    ],
    className = "row",
    style={"margin": "0%"},
)


@app.callback(
    dash.dependencies.Output('hot-movie', 'figure'),
                        [Input('tag-update', 'n_intervals')])
def update_tage(n):
    labels = []
    values = []
    try:
        with open ('hashtag_count.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter = ',')
            for row in csv_reader:
                labels.append(row[0])
                values.append(row[1])
    except:
        pass
    data = plotly.graph_objs.Pie(
            labels = labels, values = values,
            )
    layout = {
        'paper_bgcolor' : '#e5e8e6',
    }
    return {'data' : [data], 
                'layout' : layout,
                }


@app.callback(
    dash.dependencies.Output('live-graph', 'figure'),
                    [Input('movie-select', 'value'),
                     Input('graph-update', 'n_intervals')]
    )
def update_graph(select_move, n):
    global MOVIE_SELECT
    global DF_POINTER
    global X
    global Y
    print ('--------- {}'.format(select_move))
    if select_move ==  'None' or select_move is None:
        print ('enter here')
        X.clear()
        Y.clear()
        DF_POINTER = 0
        return {'data': []}
    if select_move != MOVIE_SELECT:
        X.clear()
        Y.clear()
        DF_POINTER = 0
    update_queue(select_move)

    data = plotly.graph_objs.Scatter(
            x = list(X),
            y = list(Y),
            name = select_move + ' Review Score',
            mode = 'lines+markers'
        )
    try:
        x_min = min(X)
        x_max = max(X)
        y_min = min(Y)
        y_max = max(Y)
    except:
        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0
    return {'data': [data], 'layout' : go.Layout(xaxis = dict(range=[x_min, x_max], title='time'),
                                                 yaxis = dict(range=[y_min, y_max], title='score'),
                                                 # paper_bgcolor = '#e5e8e6',
                                                 )}

def update_queue(select_move):
    global DF_POINTER
    if select_move != 'None':
        df = pd.read_csv('{}.csv'.format(select_move))
        while DF_POINTER < len(df.index):
            try:
                X.append(df.iloc[DF_POINTER]['time'])
                Y.append(df.iloc[DF_POINTER]['score'])
                DF_POINTER += 1
            except:
                pass


@app.callback(
    dash.dependencies.Output('live-history', 'figure'),
                    [Input('history-update', 'n_intervals')]
    )

def update_history(n):
    df = []
    for tag in tags:
        try:
            df.append(pd.read_csv('{}_avg.csv'.format(tag)))
        except:
            df.append(pd.DataFrame(columns = ['movie','time','score']))
    data = [plotly.graph_objs.Scatter(
        x = df[i]['time'].to_list(),
        y = df[i]['score'].to_list(),
        name = val,
        mode = 'lines+markers',
        ) for i, val in enumerate(tags)]
    x_min_list = []
    x_max_list = []
    for i, val in enumerate(tags):
        df[i] = pd.to_datetime(df[i]['time'])
        x_min_list.append(df[i].min())
        x_max_list.append(df[i].max())
    x_min = min(x_min_list)
    x_max = max(x_max_list)
    layout = go.Layout(
                xaxis={'title': 'time', 'range' : [x_min, x_max]},
                yaxis={'title': 'score', 'range' : [0,1]},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest')
    return {'data' : data, 'layout' : layout}

@app.callback(
    dash.dependencies.Output('elo-rank', 'children'),
                    [Input('elo-update', 'n_intervals')]
    )

def update_rank(n):
    try:
    	dataframe = pd.read_csv('elo.csv')
    except:
        #return html.Table(html.Tr(['index','hashtag','rank']))
        dataframe = pd.DataFrame(columns = ['movie','rank'])
    dataframe = dataframe.sort_values(by = ['rank'], ascending = False)
    dataframe = dataframe.reset_index(drop=True).set_index(np.arange(len(dataframe.index)))
    dataframe.reset_index(level=0, inplace=True)
    print(dataframe)
    return html.Table(
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(len(dataframe))]

        )


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port='8111')
