import os
import json
import pandas as pd
import numpy as np
import uuid
#import nbformat as nbf


def generate_cell_id():
    return uuid.uuid4().hex[:8]

def convert_notebook(notebook_json) -> pd.DataFrame:
    #with open(notebook_file, 'r', encoding='utf-8') as file:
    #notebook_json = json.load(notebook_file)
    cells = notebook_json['cells']
    data = []
    for cell in cells:
        cell_type = cell['cell_type']
        source = ''.join(cell['source'])  # Join list of strings into a single string
        cell_id = generate_cell_id()
        data.append({'cell_id': cell_id, 'cell_type': cell_type, 'source': source})
    return(pd.DataFrame(data).assign(id=os.path.basename('notebook_id')).set_index('cell_id')).reset_index()


#def dataframe_to_notebook(df: pd.DataFrame, output_path: str):

    #nb = nbf.v4.new_notebook()

    #cells = []
    #for index, row in df.iterrows():
        #cell_type = row['cell_type']
        #source = row['source']

        #if cell_type == 'markdown':
            #cell = nbf.v4.new_markdown_cell(source)
        #elif cell_type == 'code':
            #cell = nbf.v4.new_code_cell(source)
        #else:
            #raise ValueError(f"Unsupported cell type: {cell_type}")

        #cells.append(cell)
    #nb['cells'] = cells

    #with open(output_path, 'w', encoding='utf-8') as f:
        #nbf.write(nb, f)


def convert_df_to_ipynb(df):
    return

if __name__ == '__main__':
    pass
