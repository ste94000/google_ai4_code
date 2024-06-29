import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from google_ai4_code_script.params import *
from google_ai4_code_script.model import *
from google_ai4_code_script.data import *
from google_ai4_code_script.utils import *

# DISTILBERT

def main_distilbert(only_markdown: bool = True, ):

    df = get_df_distilbert(only_markdown=only_markdown)

    input_ids, attention_mask = tokenize_distilbert(df["source"])

    dataset = get_dataset_distilbert(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    model = get_model_distilbert()
    model.load_weights(WEIGHTS_PATH_DISTILBERT)

    pred = model.predict(dataset)
    df['rank_pred'] = pred
    df.sort_values('rank_pred', inplace=True)

    return df

# CODEBERT

def main_codebert():
    df = get_df_codebert()


    fts = get_features(df)
    input_ids, attention_mask, features = tokenize_codebert(df, fts)

    dataset = get_dataset_codebert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        features=features,
    )

    model = get_model_codebert()
    model.load_weights(WEIGHTS_PATH_CODEBERT)

    pred = model.predict(dataset)
    df['rank_pred'] = pred
    df.sort_values('rank_pred', inplace=True)

    return df[['cell_id', 'cell_type', 'source', 'rank_pred']]

if __name__ == '__main__':
    print(main_codebert())
