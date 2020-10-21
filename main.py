# Import general libraries
import pandas as pd
import numpy as np
import json
import pprint
from datetime import datetime
import random
import os
import time
import regex as re
import base64
import PyPDF2
import io

# # Torch + Huggingface
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    squad_convert_examples_to_features,
)

from transformers.data.processors.squad import (
    SquadResult,
    SquadExample,
)
from transformers.data.metrics.squad_metrics import compute_predictions_logits

# Import dash
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly
import plotly.express as px

app = dash.Dash(external_stylesheets=[dbc.themes.CERULEAN])

app.title = "DSTA Smart Search System"

app_name = "DSTA Smart Search System"

server = app.server  # for deployment

""" Config """
# Documents
# You could replace this data with your own documents
f = open("data/train-v2.0.json")
data = json.load(f)["data"]

# Config
n_best_size = 1
max_answer_length = 30
do_lower_case = True
null_score_diff_threshold = 0.0

# Setup model
use_own_model = False
if use_own_model:  # For use if you have your own pre-trained model
    model_name_or_path = "/content/model_output"
else:  # Obtain model from huggingface library https://huggingface.co/models
    model_name_or_path = "ktrapeznikov/albert-xlarge-v2-squad-v2"

config_class, model_class, tokenizer_class = (
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
)
config = config_class.from_pretrained(model_name_or_path)
tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)
model = model_class.from_pretrained(model_name_or_path, config=config)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
""" End of Config """


""" Helper Functions """


def load_cat_data():
    """
    Loads the categories of data into a list for dropdown selection

    Input: None
    Output: List of categories (string)
    """
    result = []

    for i in data:  # for all categories
        result.append({"label": i["title"], "value": i["title"]})

    f.close()
    return result


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def run_prediction(question_texts, context_text):
    """Setup function to compute predictions"""
    examples = []

    for i, question_text in enumerate(question_texts):
        example = SquadExample(
            qas_id=str(i),
            question_text=question_text,
            context_text=context_text,
            answer_text=None,
            start_position_character=None,
            title="Predict",
            is_impossible=False,
            answers=None,
        )

        examples.append(example)
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        is_training=False,
        return_dataset="pt",
        threads=1,
    )
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=10)
    all_results = []

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            example_indices = batch[3]

            outputs = model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)

    output_prediction_file = "predictions.json"
    output_nbest_file = "nbest_predictions.json"
    output_null_log_odds_file = "null_predictions.json"

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False,  # verbose_logging
        True,  # version_2_with_negative
        null_score_diff_threshold,
        tokenizer,
    )
    return predictions


# To parse custom uploaded files in tab 3
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)

    try:
        if "pdf" in filename:
            # Assume that the user uploaded a PDF file
            pdf = PyPDF2.PdfFileReader(io.BytesIO(decoded))
        # elif 'csv' in filename:
        #     # Assume that the user uploaded a CSV file
        #     df = pd.read_csv(
        #         io.StringIO(decoded.decode('utf-8')))
        # elif 'xls' in filename:
        #     # Assume that the user uploaded an excel file
        #     df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])

    result = ""

    for pages in range(pdf.getNumPages()):
        result += pdf.getPage(pages).extractText().replace("\n", "").replace("\\", "")

    return result


""" End of Helper Functions """


""" Front End"""
# Navigation bar
navbar = dbc.Navbar(
    [
        html.A(
            dbc.Row(
                [
                    dbc.Col(
                        dbc.NavbarBrand(
                            "DSTA Smart Search System",
                            className="ml-auto",
                            style={"font-size": 30},
                        )
                    ),
                ],
                align="left",
                no_gutters=True,
            ),
            href="",  # to be updated
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
    ],
    color="dark",
    dark=True,
    style={"width": "100%"},
)

### SQuAD 2.0 tab
squadTab = html.Div(
    children=[
        html.P(
            "This application is gpu enabled: "
            + ("True" if torch.cuda.is_available() else "False")
        ),
        # Category box
        html.H3("Step 1: Please select your category"),
        html.Div(
            [
                dcc.Dropdown(
                    id="category-dropdown",
                    options=load_cat_data(),
                    placeholder="Select a category",
                ),
            ]
        ),
        html.Br(),
        # Query box
        html.H3("Step 2: Please input your query"),
        dbc.Textarea(
            id="text_search",
            className="mb-3",
            placeholder="e.g. What is computational complexity principle?",
        ),
        # Search button
        html.H3("Step 3: Click the search button to generate results"),
        html.Button(
            "Search", id="button_search", className="btn btn-success btn-lg btn-block"
        ),
        html.Br(),
        # Output table
        html.H3("Step 4: Results"),
        dcc.Loading(  # add loading animation
            children=[
                html.Div(
                    id="output_search_table",
                    style={
                        "marginTop": 20,
                        "marginLeft": 10,
                        "width": "80%",
                        # "display": "inline-block",
                    },
                )
            ],
            type="graph",
            fullscreen=False,
        ),
        # Some blank rows
        html.Br(),
        html.Br(),
    ],
    style={
        "marginLeft": 40,
        "width": "80%",
        # "display": "inline-block",
    },
)

### Custom context tab
contextTab = html.Div(
    children=[
        html.P(
            "This application is gpu enabled: "
            + ("True" if torch.cuda.is_available() else "False")
        ),
        # Context box
        html.H3("Step 1: Please input your custom context"),
        dbc.Textarea(
            id="text_custom_context",
            className="mb-3",
            placeholder="e.g. Computational complexity theory focuses on classifying computational problems according to their resource usage, and relating these classes to each other. A computational problem is a task solved by a computer. A computation problem is solvable by mechanical application of mathematical steps, such as an algorithm.",
        ),
        # Query box
        html.H3("Step 2: Please input your query"),
        dbc.Textarea(
            id="text_search_context",
            className="mb-3",
            placeholder="e.g. What is computational complexity principle?",
        ),
        # Search button
        html.H3("Step 3: Click the search button to generate results"),
        html.Button(
            "Search",
            id="button_search_context",
            className="btn btn-success btn-lg btn-block",
        ),
        html.Br(),
        # Output table
        html.H3("Step 4: Results"),
        dcc.Loading(  # add loading animation
            children=[
                html.Div(
                    id="output_search_table_context",
                    style={
                        "marginTop": 20,
                        "marginLeft": 10,
                        "width": "80%",
                        # "display": "inline-block",
                    },
                )
            ],
            type="graph",
            fullscreen=False,
        ),
        # Some blank rows
        html.Br(),
        html.Br(),
    ],
    style={
        "marginLeft": 40,
        "width": "80%",
        # "display": "inline-block",
    },
)

### Custom input file tab
fileTab = html.Div(
    children=[
        html.P(
            "This application is gpu enabled: "
            + ("True" if torch.cuda.is_available() else "False")
        ),
        # File upload box
        html.H3("Step 1: Please upload your documents (only PDF files are accepted)"),
        html.Div(
            [
                dcc.Upload(
                    id="upload_document",
                    children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "10px",
                    },
                    # Allow multiple files to be uploaded
                    multiple=True,
                ),
            ]
        ),
        # Confirmation of file names
        html.H3("Step 1.5: Please wait for your file names to be listed below"),
        dbc.Spinner(children=[html.Div(id="document_confirmation")]),
        # Query box
        html.H3("Step 2: Please input your query"),
        dbc.Textarea(
            id="text_search_file",
            className="mb-3",
            placeholder="e.g. What is computational complexity principle?",
        ),
        # Search button
        html.H3("Step 3: Click the search button to generate results"),
        html.Button(
            "Search",
            id="button_search_file",
            className="btn btn-success btn-lg btn-block",
        ),
        html.Br(),
        # Output table
        html.H3("Step 4: Results"),
        dcc.Loading(  # add loading animation
            children=[
                html.Div(
                    id="output_search_table_file",
                    style={
                        "marginTop": 20,
                        "marginLeft": 10,
                        "width": "80%",
                        # "display": "inline-block",
                    },
                )
            ],
            type="graph",
            fullscreen=False,
        ),
        # Some blank rows
        html.Br(),
        html.Br(),
    ],
    style={
        "marginLeft": 40,
        "width": "80%",
        # "display": "inline-block",
    },
)

# Layout of entire app
app.layout = html.Div(
    [
        navbar,
        dbc.Tabs(
            [
                dbc.Tab(squadTab, id="label_tab1", label="SQuAD 2.0"),
                dbc.Tab(contextTab, id="label_tab2", label="Custom Context"),
                dbc.Tab(fileTab, id="label_tab3", label="Custom Input File"),
            ],
            style={"font-size": 20, "background-color": "#b9d9eb"},
        ),
    ],
)

""" End of Front End"""

""" Callbacks"""
### For squadTab
@app.callback(
    Output("output_search_table", "children"),  # output results in table
    [Input("button_search", "n_clicks")],  # upon clicking search button
    state=[
        State("category-dropdown", "value"),  # selection of category
        State("text_search", "value"),  # Input query
    ],
)
# retrieve query
def search(n_clicks, cat_val, text_val):
    startTime = datetime.now()
    # Prepare inputs for prediction
    idx = 0
    questions = [text_val]
    context = ""
    # Obtain index of the relevant category
    for i in range(len(data)):
        if data[i]["title"] == cat_val:
            idx = i

    # Loop through the category and merge all paragraphs
    for i in range(len(data[idx]["paragraphs"])):
        context += data[idx]["paragraphs"][i]["context"]

    # # Debug, to be overwritten when GPU is sourced
    # context = data[idx]["paragraphs"][50]["context"]

    # Run prediction
    predictions = run_prediction(questions, context)
    endTimePred = datetime.now()

    # Return results
    answers = []
    for key in predictions.keys():
        answers.append(predictions[key])

    if answers[0] == "":  # Invalid questions
        df = pd.DataFrame(
            {
                "Answers": ["Invalid question", ""],
                "Documents": [
                    "Invalid question",
                    "This query was completed in: "
                    + str((datetime.now() - startTime).total_seconds())
                    + "s.",
                ],
            }
        )
    else:
        # Retrieve documents and highlight answers
        documents_df, answers_df = (
            [],
            [],
        )  # Create 2 lists to house answers and documents
        for (
            ans
        ) in (
            answers
        ):  # Retrieve documents for all the answers. Usually we only have 1 answer
            for i in range(len(data[idx]["paragraphs"])):
                ctx = data[idx]["paragraphs"][i]["context"]
                if re.search(ans, ctx):
                    answers_df.append(ans)
                    documents_df.append(
                        str(re.sub(ans, "**" + ans + "**", ctx))
                    )  # Use markdown to bold the answer
        documents_df.extend(
            [
                "The answers were generated in: "
                + str((endTimePred - startTime).total_seconds())
                + "s.",
                "This documents were retrieved in: "
                + str((datetime.now() - endTimePred).total_seconds())
                + "s.",
            ]
        )
        answers_df.extend(["", ""])

        df = pd.DataFrame({"Answers": answers_df, "Documents": documents_df})

    output = dash_table.DataTable(
        id="dash_tb",
        columns=[{"name": i, "id": i, "presentation": "markdown",} for i in df.columns],
        data=df.to_dict("records"),
        style_header={
            "fontWeight": "bold",
            "backgroundColor": "rgb(30, 30, 30)",
            "color": "white",
            "font-size": 20,
        },
        style_cell={
            "whiteSpace": "normal",
            "height": "auto",
            "textAlign": "left",
            "maxWidth": 1080,
        },
    )
    return output


### For contextTab
@app.callback(
    Output("output_search_table_context", "children"),  # output results in table
    [Input("button_search_context", "n_clicks")],
    state=[  # upon clicking search button
        State("text_custom_context", "value"),  # Custom context
        State("text_search_context", "value"),
    ],  # Input query
)
# retrieve query
def search(n_clicks, context_val, question_val):
    startTime = datetime.now()

    # Prepare inputs for prediction
    question = [str(question_val)]
    context = str(context_val)

    # Run prediction
    predictions = run_prediction(question, context)
    endTimePred = datetime.now()

    # Return results
    answers = []
    for key in predictions.keys():
        answers.append(predictions[key])

    if answers[0] == "":  # Invalid questions
        df = pd.DataFrame(
            {
                "Answers": ["Invalid question", ""],
                "Documents": [
                    "Invalid question",
                    "This query was completed in: "
                    + str((endTimePred - startTime).total_seconds())
                    + "s.",
                ],
            }
        )
    else:
        # Retrieve documents and highlight answers
        documents_df, answers_df = (
            [],
            [],
        )  # Create 2 lists to house answers and documents
        for (
            ans
        ) in (
            answers
        ):  # Retrieve documents for all the answers. Usually we only have 1 answer
            ctx = context
            if re.search(ans, ctx):  # If answer is found in the context provided
                answers_df.append(ans)
                documents_df.append(
                    str(re.sub(ans, "**" + ans + "**", ctx))
                )  # Use markdown to bold the answer
        documents_df.append(
            "The answers were generated in: "
            + str((endTimePred - startTime).total_seconds())
            + "s."
        )
        answers_df.append("")

        df = pd.DataFrame({"Answers": answers_df, "Documents": documents_df})

    output = dash_table.DataTable(
        id="dash_tb",
        columns=[{"name": i, "id": i, "presentation": "markdown",} for i in df.columns],
        data=df.to_dict("records"),
        style_header={
            "fontWeight": "bold",
            "backgroundColor": "rgb(30, 30, 30)",
            "color": "white",
            "font-size": 20,
        },
        style_cell={
            "whiteSpace": "normal",
            "height": "auto",
            "textAlign": "left",
            "maxWidth": 1080,
        },
    )
    return output


### For fileTab
@app.callback(
    Output("output_search_table_file", "children"),  # output results in table
    [
        Input("button_search_file", "n_clicks"),
        Input("upload_document", "contents"),
    ],  # upon clicking search button and uploading file
    state=[
        State("upload_document", "filename"),
        State("upload_document", "last_modified"),
        State("text_search_file", "value"),
    ],
)
# retrieve query
def search(n_clicks, list_of_contents, list_of_names, list_of_dates, text_val):
    startTime = datetime.now()

    # Prepare inputs for prediction
    questions = [text_val]
    context = ""
    contexts = []

    # Retrieve all the contexts by parsing the pdf
    if list_of_contents is not None:
        contexts = [
            parse_contents(c, n, d)
            for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
        ]

    # Loop through all the documents and merge all texts
    for ctx in contexts:
        context += ctx

    # Run prediction
    predictions = run_prediction(questions, context)
    endTimePred = datetime.now()

    # Return results
    answers = []
    for key in predictions.keys():
        answers.append(predictions[key])

    if answers[0] == "":  # Invalid questions
        df = pd.DataFrame(
            {
                "Answers": ["Invalid question", ""],
                "Documents": [
                    "Invalid question",
                    "This query was completed in: "
                    + str((endTimePred - startTime).total_seconds())
                    + "s.",
                ],
            }
        )
    else:
        # Retrieve documents and highlight answers
        documents_df, answers_df = (
            [],
            [],
        )  # Create 2 lists to house answers and documents
        for (
            ans
        ) in (
            answers
        ):  # Retrieve documents for all the answers. Usually we only have 1 answer
            for i in range(len(contexts)):  # loop through all contexts
                ctx = contexts[i]
                filename = list_of_names[i]
                if re.search(ans, ctx):
                    answers_df.append(ans)
                    documents_df.append(
                        "**"
                        + filename
                        + ":** "
                        + str(re.sub(ans, "**" + ans + "**", ctx))
                    )  # Use markdown to bold the answer
        documents_df.extend(
            [
                "The answers were generated in: "
                + str((endTimePred - startTime).total_seconds())
                + "s.",
                "This documents were retrieved in: "
                + str((datetime.now() - endTimePred).total_seconds())
                + "s.",
            ]
        )
        answers_df.extend(["", ""])

        df = pd.DataFrame({"Answers": answers_df, "Documents": documents_df})

    output = dash_table.DataTable(
        id="dash_tb",
        columns=[{"name": i, "id": i, "presentation": "markdown",} for i in df.columns],
        data=df.to_dict("records"),
        style_header={
            "fontWeight": "bold",
            "backgroundColor": "rgb(30, 30, 30)",
            "color": "white",
            "font-size": 20,
        },
        style_cell={
            "whiteSpace": "normal",
            "height": "auto",
            "textAlign": "left",
            "maxWidth": 1080,
        },
    )
    return output


### For fileTab
@app.callback(
    Output("document_confirmation", "children"),  # output results in table
    [Input("upload_document", "contents"),],
    state=[
        State("upload_document", "filename"),
        State("upload_document", "last_modified"),
    ],
)
# retrieve query
def search(list_of_contents, list_of_names, list_of_dates):
    contexts = [
        parse_contents(c, n, d)
        for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
    ]
    return html.Ul([html.Li(names) for names in list_of_names])


#     return output
""" End of Callbacks"""

if __name__ == "__main__":
    # app.run_server(debug=True)  # change debug to true when developing
    app.run_server(host="0.0.0.0", port=80)

