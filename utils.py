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
import mysql.connector
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

from transformer import Model



"""
PATHS COMMON TO THE TRAINING AND CLUSTER PREP SCRIPTS:
"""
# MODEL_DIR       = '/work/paptronics/model/v9/'
MODEL_DIR       = '/mnt/z/Project/semantic_search/qualia/model/v8/'
TRAIN_DATA      = '/mnt/z/Project/semantic_search/qualia/data/train/'

SVD_FNAME       = 'svd2000_v8.pkl'

# TODO:
DATA_PATH       = TRAIN_DATA + 'train_99k_full_questions_with_sub_balanced_v2.pkl'
MODEL_FILE_NAME = 'tfidf_vec_ngrams3_mindf5_full_dataset_balanced_sublin_tf_v8.pkl'
MATX_FILE_NAME  = 'tfidf_matx_ngrams3_mindf5_full_dataset_balanced_sublin_tf_v8.pkl'
UNIQ_IDS        = 'tfidf_matx_ngrams3_mindf5_ids_full_dataset_balanced_sublin_tf_v8.pkl'

# Default params:
EMB_DIM            = 768 #1000
MAX_NEAREST_NBRS   = 30
MAX_SEARCH_THREADS = -1
INDEX_SIZE_PATH    = 'hnsw_curr_index_size_v8.pkl'


# Index files

TMP_INDEX_SAVE_FILE = 'tmp_hnsw_index.bin'   #<-- New save
# INDEX_SAVE_FILE = 'hnsw_index_mindf5_svd2000_full_set_v8.bin'   #<-- Current one
INDEX_SAVE_FILE = 'hnsw_index.bin'   #<-- Current one
INDEX_SAVE_FILE_1 = 'hnsw_index.bin'   #<-- Backup n-1
INDEX_SAVE_FILE_2 = 'hnsw_index.bin'   #<-- Backup n-2


#TODO: temp, ideally you shouldnt have an external mapping, your index needs to only have db uniq ids
label_text_mapping_file = MODEL_DIR + 'label_text_mapping.pkl'

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

def load_model():
    MODEL = Model()
    MODEL.make()
    return MODEL

def load_svd():
    # Load SVD
    with open(MODEL_DIR+SVD_FNAME, 'rb') as f:
        svd = pickle.load(f)
    return svd

def load_index():
    p = hnswlib.Index(space='cosine', dim=EMB_DIM)
    p.load_index(MODEL_DIR+INDEX_SAVE_FILE)
    p.set_ef(200)
    return p

def load_search_engine_components():
    mod = load_model()
    # svd = load_svd()
    ind = load_index()
    return mod, ind

def load_connection_engine():
    cnx = mysql.connector.connect(user='qbsdbuser', password='f1901cd0',
                                host='94.237.31.125',
                                database='qbs')
    return cnx
