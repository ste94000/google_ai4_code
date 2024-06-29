import os
import json
import pandas as pd
import numpy as np

def convert_ipynb_to_json(ipynb_path, json_path):
    """
    Convertit un fichier .ipynb en fichier .json.

    :param ipynb_path: Chemin du fichier .ipynb
    :param json_path: Chemin de sortie pour le fichier .json
    """
    # Lire le fichier .ipynb
    with open(ipynb_path, 'r', encoding='utf-8') as ipynb_file:
        notebook_content = json.load(ipynb_file)

    # Écrire le contenu dans un fichier .json
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(notebook_content, json_file, indent=2)

def convert_df_to_json(df):
    #json_dict = {'cell_type':{df['cell_id'][i]:df['cell_type'][i] for i in range(len(df))}, 'source':{df['cell_id'][i]:df['source'][i] for i in range(len(df))}}
    df['cell_type'] = 'markdown'
    df = df[['cell_id', 'cell_type', 'source']].set_index('cell_id')
    print(df)
    return df.to_json('input/json/test.json', index=True)

if __name__ == '__main__':
    pass
