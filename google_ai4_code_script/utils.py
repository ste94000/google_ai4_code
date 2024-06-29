import os
import json
import pandas as pd
import numpy as np
import uuid

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
    return(pd.DataFrame(data).assign(id=os.path.basename(notebook_path).split(".")[0]).set_index('cell_id')).reset_index()

def convert_df_to_ipynb(df):
    return

if __name__ == '__main__':
    pass
