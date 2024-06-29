import pandas as pd
from google_ai4_code_script.main import main_codebert
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile
import json
from fastapi.responses import FileResponse


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/predict")
def predict(notebook_file: UploadFile):
    notebook_json = json.load(notebook_file.file)
    result = main_codebert(notebook_json=notebook_json)
    print(result)

    return result.to_dict()
