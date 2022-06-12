#backend imports
from modules.model import NNDefaker
from modules.matcher import Matcher
from modules.nerweighter import get_nerwindow_comparison
import sys
import json

WEIGHTS = [0.5, 0.5]

#frontend imports
import dash
from dash import html, dcc, Output, State, Input,dash_table
import uuid
import os
import pandas as pd
import numpy as np

import time


if __name__ == '__main__':
    matcher = Matcher('./resources/news.csv')
    defaker = NNDefaker('./resources/first_classificator.pck')
    with open('./resources/mos_idf.json') as fin:
        idf_dict = json.load(fin)
    app = dash.Dash(__name__, url_base_pathname='/')


def get_rating(header, data):
    global matcher
    global defaker
    matched = matcher.match([header])
    nerprobs = get_nerwindow_comparison(data, list(matched.text))
    matched['probs'] = list(nerprobs.values())
    nnprobs = list(defaker.infer_text([data] + list(matched.text)))[1:]
    matched['nnprobs'] = nnprobs
    matched['probs'] = matched.apply(lambda row: WEIGHTS[0] * row['probs'] + WEIGHTS[1]
                                     * row['nnprobs'] if row['probs'] not in [0.0, 1.0] else row['probs'], axis=1)
    if len(matched['probs'])>0:
        proba = matched['probs'].iloc[0]
    else:
        proba = 0
    #print(matched.columns)
    return "Вероятность фейка: %.5f" % (proba), matched





def serve_layout():
    session_id = str(uuid.uuid4())
    return html.Div(children=[
         html.Label('Введите информацию о новости'),html.Br(), html.Br(),
        html.Label('Заголовок:'), html.Br(), html.Br(),
        dcc.Input( id="header_input", value='', type='text', style={'width': '90%', 'height': '40px'}),
        html.Br(), html.Br(), html.Label('Текст:'), html.Br(), html.Br(),
        dcc.Textarea( id="data_input", value='', style={'width': '90%', 'height': '400px'}), html.Br(), html.Br(),
        html.Button(id='submit_button', n_clicks=0, children='Проверить', style={'height': '40px'}), html.Br(), html.Br(),
        html.Div(id = "data_output", style={'font-size': '26px'}),
        html.Br(), html.Br(),html.Br(), html.Br(),
    ])




@app.callback(Output('data_output', 'children'),
              State('data_input', 'value'),
              State('header_input', 'value'),
              Input('submit_button', 'n_clicks') )
def read_value(data, header, n_clicks):

    if len(header)>0 and len(data)>0:
        #time.sleep(3);
        rating_text, matched = get_rating(header, data)
        matched["Url"] = matched["id"].apply(lambda x: "https://mos.ru/news/item/" + str(x))

        matched["Fake"] = matched["probs"].apply(lambda x: "%.3f" % (x))
        matched["Title"] = matched["title"].apply(lambda x: x[0:min(len(x), 40)])
        matched["Date"] = matched["date"].apply(lambda x: x[0:min(len(x), 16)])
        return html.Div([
            rating_text ,  html.Br(),
            #"Первоисточник: ",
            #html.A(url, href=url),html.Br(),
            dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i}
                     for i in ["Url", "Fake", "Date", "Title"] ],   #df.columns],
            data=matched.to_dict('records'),
            style_cell=dict(textAlign='left'),
            style_header=dict(backgroundColor="grey"),
            style_data=dict(backgroundColor="white")
    ),
        ])
    else:
        return ""




if __name__ == '__main__':
    app.index_string="""<!DOCTYPE html>
<html>
<head>
<title> Проверка достоверности новостей </title>
{%metas%}

    {%favicon%}
    {%css%}
<style>
        ._dash-loading-callback {
        position: fixed;
        z-index: 100;
        }

        ._dash-loading-callback::after {
        content: 'Подождите...';
        font-family: sans-serif;
        padding-top: 250px;
        color: #000;

        -webkit-animation: fadein 0.5s ease-in 1s forwards; /* Safari, Chrome and Opera > 12.1 */
           -moz-animation: fadein 0.5s ease-in 1s forwards; /* Firefox < 16 */
            -ms-animation: fadein 0.5s ease-in 1s forwards; /* Internet Explorer */
             -o-animation: fadein 0.5s ease-in 1s forwards; /* Opera < 12.1 */
                animation: fadein 0.5s ease-in 1s forwards;
        /* prevent flickering on every callback */
        -webkit-animation-delay: 0.5s;
        animation-delay: 0.5s;

        /* The banner */
        opacity: 0;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.5);
        text-align: center;
        cursor: progress;
        z-index: 100000;

        background-image: url('assets/loading.gif');
        background-position: center center;
        background-repeat: no-repeat;
        }

        @keyframes fadein {
          from { opacity: 0; }
          to   { opacity: 1; }
        }

        /* Firefox < 16 */
        @-moz-keyframes fadein {
          from { opacity: 0; }
          to   { opacity: 1; }
        }

        /* Safari, Chrome and Opera > 12.1 */
        @-webkit-keyframes fadein {
          from { opacity: 0; }
          to   { opacity: 1; }
        }

        /* Internet Explorer */
        @-ms-keyframes fadein {
          from { opacity: 0; }
          to   { opacity: 1; }
        }

        /* Opera < 12.1 */
        @-o-keyframes fadein {
          from { opacity: 0; }
          to   { opacity: 1; }
        }
#page {
    width: 100%;
    background-color: #dddddd;
    color: #191970;
}
#header {

    margin-top: 0px;
    background-color: black; //#cc2222;
    text-align:center;
    font-size: 26px;

    font-family: Verdana, Geneva, sans-serif;
    letter-spacing: 2 px;

}
#header_text {
  color: white; #F8F32B;
  background: black;  //#cc2222;
  margin-top: 0px;

}

#logo {
    width: 5px;
    margin: 1;
    padding: 10px;
}
#content {
    background-color:#fff;

    width:70%;
    margin: 0 auto;
    text-align: left;
    float:center;
    padding:5px;
}
#footer {
    //position: absolute;
    background-color: black; //#cc2222;
    clear:both;
    text-align:center;
    padding:5px;
    width: 100%;
    color: white;
    bottom:0;
    left:0;

}
</style>
</head>

<body>
<div id="header">
<h1 id="header_text"> Проверка достоверности новостей</h1>
</div>

<div id="content">
{%app_entry%}
</div>

<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
<div id="footer">
Copyright © 2022
</div>
</body>
     """
    app.layout = serve_layout
    app.run_server(debug=False, host="0.0.0.0" )
