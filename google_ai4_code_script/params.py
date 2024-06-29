import os

# Data
DATA_PATH = os.environ.get('DATA_PATH')

# Models
DISTILBERT_BASE = os.environ.get('DISTILBERT_BASE')
CODEBERT_BASE = os.environ.get('CODEBERT_BASE')
WEIGHTS_PATH_DISTILBERT = os.environ.get('WEIGHTS_PATH_DISTILBERT')
WEIGHTS_PATH_CODEBERT = os.environ.get('WEIGHTS_PATH_CODEBERT')

BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))

# Tokenizer
SEQ_LEN_DISTILBERT = int(os.environ.get('SEQ_LEN_DISTILBERT'))
MD_MAX_LEN_CODEBERT = int(os.environ.get('MD_MAX_LEN_CODEBERT'))
TOTAL_MAX_LEN_CODEBERT = int(os.environ.get('TOTAL_MAX_LEN_CODEBERT'))
