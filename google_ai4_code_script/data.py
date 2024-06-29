import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from typing import Optional, Tuple, List
import glob
from google_ai4_code_script.params import *
from google_ai4_code_script.utils import *


def read_notebook(path: str) -> pd.DataFrame:
    with open(path) as file:
        df = pd.DataFrame(json.load(file))
    df["id"] = os.path.splitext(os.path.basename(path))[0]
    return df

# DISTILBERT

def get_df_distilbert(only_markdown: bool = True) -> pd.DataFrame:
    paths = 'input/json/0009d135ece78d.json'
    df = read_notebook(paths)

    if only_markdown:
        df = df[df["cell_type"] == "markdown"]
        df = df.drop("cell_type", axis=1)

    df = df.rename_axis("cell_id").reset_index()
    print(df)
    return df

def get_dataset_distilbert(
    input_ids: np.array,
    attention_mask: np.array,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        {"input_ids": input_ids, "attention_mask": attention_mask}
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# CODEBERT

def get_df_codebert(notebook_json):
    #paths = 'input/ipynb/test.ipynb'
    df = convert_notebook(notebook_json)
    print(df)
    #df = read_notebook(paths).set_index("id", append=True).swaplevel().reset_index()
    df["source"] = df["source"].str.slice(0, MD_MAX_LEN_CODEBERT)
    df["rank"] = df.groupby(["id", "cell_type"]).cumcount()
    df["pct_rank"] = df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
    df.rename(columns={'level_1': 'cell_id'}, inplace=True)
    return df

def clean_code(cell: str) -> str:
    return str(cell).replace("\\n", "\n")

def sample_cells(cells: List[str], n: int) -> List[str]:
    cells = [clean_code(cell) for cell in cells]
    if n >= len(cells):
        return cells
    else:
        results = []
        step = len(cells) / n
        idx = 0
        while int(np.round(idx)) < len(cells):
            results.append(cells[int(np.round(idx))])
            idx += step
        if cells[-1] not in results:
            results[-1] = cells[-1]
        return results

def get_features(df: pd.DataFrame) -> dict:
    features = {}
    total_md = df[df.cell_type == "markdown"].shape[0]
    code_sub_df = df[df.cell_type == "code"]
    total_code = code_sub_df.shape[0]
    codes = sample_cells(code_sub_df.source.values, 20)
    features["total_code"] = total_code
    features["total_md"] = total_md
    features["codes"] = codes
    return features

def get_dataset_codebert(
    input_ids: np.array,
    attention_mask: np.array,
    features: np.array,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        {"input_ids": input_ids, "attention_mask": attention_mask, "feature": features}
    )
    dataset = dataset.batch(BATCH_SIZE)
    return dataset.prefetch(tf.data.AUTOTUNE)


if __name__ == '__main__':
    print(get_df_codebert())
