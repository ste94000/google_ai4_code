import glob
import json
import os
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
from tqdm.notebook import tqdm
import uuid


BATCH_SIZE = 32
SLICES = 8
MD_MAX_LEN = 64
TOTAL_MAX_LEN = 512
STRATEGY = tf.distribute.get_strategy()
BASE_MODEL = "../models/codebert-base"
TOKENIZER = transformers.AutoTokenizer.from_pretrained(BASE_MODEL)
INPUT_PATH = "../raw_data/AI4Code"


notebook_path='../notebooks/test_notebook.ipynb'

def generate_cell_id():
    return uuid.uuid4().hex[:8]

def convert_notebook(notebook_path: str) -> pd.DataFrame:

    with open(notebook_path, 'r', encoding='utf-8') as file:
        notebook_json = json.load(file)

    cells = notebook_json['cells']
    data = []
    for cell in cells:
        cell_type = cell['cell_type']
        source = ''.join(cell['source'])  # Join list of strings into a single string
        cell_id = generate_cell_id()
        data.append({'cell_id': cell_id, 'cell_type': cell_type, 'source': source})

    return(pd.DataFrame(data).assign(id=os.path.basename(notebook_path).split(".")[0]).set_index('cell_id'))

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
    for i, sub_df in tqdm(df.groupby("id"), desc="Features"):
        features[i] = {}
        total_md = sub_df[sub_df.cell_type == "markdown"].shape[0]
        code_sub_df = sub_df[sub_df.cell_type == "code"]
        total_code = code_sub_df.shape[0]
        codes = sample_cells(code_sub_df.source.values, 20)
        features[i]["total_code"] = total_code
        features[i]["total_md"] = total_md
        features[i]["codes"] = codes
    return features


def tokenize(df: pd.DataFrame, fts: dict) -> dict:
    input_ids = np.zeros((len(df), TOTAL_MAX_LEN), dtype=np.int32)
    attention_mask = np.zeros((len(df), TOTAL_MAX_LEN), dtype=np.int32)
    features = np.zeros((len(df),), dtype=np.float32)

    for i, row in tqdm(
        df.reset_index(drop=True).iterrows(), desc="Tokens", total=len(df)
    ):
        row_fts = fts[row.id]

        inputs = TOKENIZER.encode_plus(
            row.source,
            None,
            add_special_tokens=True,
            max_length=MD_MAX_LEN,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        code_inputs = TOKENIZER.batch_encode_plus(
            [str(x) for x in row_fts["codes"]] or [""],
            add_special_tokens=True,
            max_length=23,
            padding="max_length",
            truncation=True,
        )

        ids = inputs["input_ids"]
        for x in code_inputs["input_ids"]:
            ids.extend(x[:-1])
        ids = ids[:TOTAL_MAX_LEN]
        if len(ids) != TOTAL_MAX_LEN:
            ids = ids + [
                TOKENIZER.pad_token_id,
            ] * (TOTAL_MAX_LEN - len(ids))

        mask = inputs["attention_mask"]
        for x in code_inputs["attention_mask"]:
            mask.extend(x[:-1])
        mask = mask[:TOTAL_MAX_LEN]
        if len(mask) != TOTAL_MAX_LEN:
            mask = mask + [
                TOKENIZER.pad_token_id,
            ] * (TOTAL_MAX_LEN - len(mask))

        input_ids[i] = ids
        attention_mask[i] = mask
        features[i] = (
            row_fts["total_md"] / (row_fts["total_md"] + row_fts["total_code"]) or 1
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "features": features,
    }

def get_ranks(base: pd.Series, derived: List[str]) -> List[str]:
    return [base.index(d) for d in derived]


def get_dataset(
    input_ids: np.array,
    attention_mask: np.array,
    feature: np.array,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        {"input_ids": input_ids, "attention_mask": attention_mask, "feature": feature}
    )
    dataset = dataset.batch(BATCH_SIZE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def get_model() -> tf.keras.Model:
    backbone = transformers.TFAutoModel.from_pretrained(BASE_MODEL)
    input_ids = tf.keras.layers.Input(
        shape=(TOTAL_MAX_LEN,),
        dtype=tf.int32,
        name="input_ids",
    )
    attention_mask = tf.keras.layers.Input(
        shape=(TOTAL_MAX_LEN,),
        dtype=tf.int32,
        name="attention_mask",
    )
    feature = tf.keras.layers.Input(
        shape=(1,),
        dtype=tf.float32,
        name="feature",
    )
    x = backbone({"input_ids": input_ids, "attention_mask": attention_mask})[0]
    x = tf.concat([x[:, 0, :], feature], axis=1)
    outputs = tf.keras.layers.Dense(1, activation="linear", dtype="float32")(x)
    return tf.keras.Model(
        inputs=[input_ids, attention_mask, feature],
        outputs=outputs,
    )



paths = glob.glob(os.path.join(INPUT_PATH, "test", "*.json"))
df = (
    pd.concat([read_notebook(x) for x in tqdm(paths, desc="Concat")])
    .set_index("id", append=True)
    .swaplevel()
    .sort_index(level="id", sort_remaining=False)
).reset_index()
df["source"] = df["source"].str.slice(0, MD_MAX_LEN)
df["rank"] = df.groupby(["id", "cell_type"]).cumcount()
df["pct_rank"] = df.groupby(["id", "cell_type"])["rank"].rank(pct=True)

fts = get_features(df)
