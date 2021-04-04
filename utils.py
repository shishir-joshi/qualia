"""
Utils
"""
import re
import os
import ast
import json
import logging
import numpy as np
import time
import dill as pickle
import hnswlib
import numpy as np
# import mysql.connector
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

from embedding_model import Model



"""
PATHS COMMON TO THE TRAINING AND CLUSTER PREP SCRIPTS:
"""
# MODEL_DIR       = '/work/paptronics/model/v9/'
MODEL_DIR       = 'model/v1/'
INDEX_DIR       = MODEL_DIR+'hnsw_index/'
TRAIN_DATA      = 'data/train/'
# Default params:
EMB_DIM            = 768 #1000
MAX_NEAREST_NBRS   = 30
MAX_SEARCH_THREADS = -1
INDEX_SIZE_PATH    = INDEX_DIR+'hnsw_curr_index_size.pkl'
SAMPLE_DATASET = 'data/arxivData.json'

SVD_FNAME       = 'svd2000_v8.pkl'

# TODO:
DATA_PATH       = TRAIN_DATA + 'train_99k_full_questions_with_sub_balanced_v2.pkl'
MODEL_FILE_NAME = 'tfidf_vec_ngrams3_mindf5_full_dataset_balanced_sublin_tf_v8.pkl'
MATX_FILE_NAME  = 'tfidf_matx_ngrams3_mindf5_full_dataset_balanced_sublin_tf_v8.pkl'
UNIQ_IDS        = 'tfidf_matx_ngrams3_mindf5_ids_full_dataset_balanced_sublin_tf_v8.pkl'



# Index files

TMP_INDEX_SAVE_FILE = 'tmp_hnsw_index.bin'   #<-- New save
# INDEX_SAVE_FILE = 'hnsw_index_mindf5_svd2000_full_set_v8.bin'   #<-- Current one
INDEX_SAVE_FILE = 'hnsw_index.bin'   #<-- Current one
INDEX_SAVE_FILE_1 = 'hnsw_index.bin'   #<-- Backup n-1
INDEX_SAVE_FILE_2 = 'hnsw_index.bin'   #<-- Backup n-2

def create_logger(name, log_dir, filename, level, log_to_console=False):
    LOG_FILENAME = log_dir+filename
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # create file handler which logs even debug messages
    fh = logging.FileHandler(LOG_FILENAME)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    if log_to_console:
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(ch)
    return logger
