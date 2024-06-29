import pandas as pd
import numpy as np
import tensorflow as tf
import transformers
from typing import Tuple
from tqdm import tqdm
from google_ai4_code_script.params import *

# DISTILBERT

def tokenize_distilbert(source: pd.Series) -> Tuple[np.array, np.array]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(DISTILBERT_BASE, do_lower_case=True)

    input_ids = np.zeros((len(source), SEQ_LEN_DISTILBERT), dtype="int32")
    attention_mask = np.zeros((len(source), SEQ_LEN_DISTILBERT), dtype="int32")

    for i, x in enumerate(tqdm(source, total=len(source))):
        encoding = tokenizer.encode_plus(
            x,
            None,
            add_special_tokens=True,
            max_length=SEQ_LEN_DISTILBERT,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        input_ids[i] = encoding["input_ids"]
        attention_mask[i] = encoding["attention_mask"]

    return input_ids, attention_mask

def get_model_distilbert() -> tf.keras.Model:
    backbone = transformers.TFDistilBertModel.from_pretrained(DISTILBERT_BASE)
    input_ids = tf.keras.layers.Input(
        shape=(SEQ_LEN_DISTILBERT,),
        dtype=tf.int32,
        name="input_ids",
    )
    attention_mask = tf.keras.layers.Input(
        shape=(SEQ_LEN_DISTILBERT,),
        dtype=tf.int32,
        name="attention_mask",
    )
    print(type(input_ids), type(attention_mask))

    x = backbone(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
    )
    outputs = tf.keras.layers.Dense(1, activation="linear", dtype="float32")(x[0][:, 0, :])

    model = tf.keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=outputs,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    return model


# CODEBERT

def tokenize_codebert(df: pd.DataFrame, fts: dict) -> dict:

    tokenizer = transformers.AutoTokenizer.from_pretrained(CODEBERT_BASE)

    input_ids = np.zeros((len(df), TOTAL_MAX_LEN_CODEBERT), dtype=np.int32)
    attention_mask = np.zeros((len(df), TOTAL_MAX_LEN_CODEBERT), dtype=np.int32)
    features = np.zeros((len(df),), dtype=np.float32)

    for i, row in tqdm(
        df.reset_index(drop=True).iterrows(), desc="Tokens", total=len(df)
    ):
        #row_fts = fts[row.cell_id]

        inputs = tokenizer.encode_plus(
            row.source,
            None,
            add_special_tokens=True,
            max_length=MD_MAX_LEN_CODEBERT,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        code_inputs = tokenizer.batch_encode_plus(
            [str(x) for x in fts["codes"]] or [""],
            add_special_tokens=True,
            max_length=23,
            padding="max_length",
            truncation=True,
        )

        ids = inputs["input_ids"]
        for x in code_inputs["input_ids"]:
            ids.extend(x[:-1])
        ids = ids[:TOTAL_MAX_LEN_CODEBERT]
        if len(ids) != TOTAL_MAX_LEN_CODEBERT:
            ids = ids + [
                tokenizer.pad_token_id,
            ] * (TOTAL_MAX_LEN_CODEBERT - len(ids))

        mask = inputs["attention_mask"]
        for x in code_inputs["attention_mask"]:
            mask.extend(x[:-1])
        mask = mask[:TOTAL_MAX_LEN_CODEBERT]
        if len(mask) != TOTAL_MAX_LEN_CODEBERT:
            mask = mask + [
                tokenizer.pad_token_id,
            ] * (TOTAL_MAX_LEN_CODEBERT - len(mask))

        input_ids[i] = ids
        attention_mask[i] = mask
        features[i] = (
            fts["total_md"] / (fts["total_md"] + fts["total_code"]) or 1
        )

    return input_ids, attention_mask, features

def get_model_codebert() -> tf.keras.Model:
    backbone = transformers.TFAutoModel.from_pretrained(CODEBERT_BASE)
    input_ids = tf.keras.layers.Input(
        shape=(TOTAL_MAX_LEN_CODEBERT,),
        dtype=tf.int32,
        name="input_ids",
    )
    attention_mask = tf.keras.layers.Input(
        shape=(TOTAL_MAX_LEN_CODEBERT,),
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
