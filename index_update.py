"""
index update service 
1. load data object
2. init hnsw index
3. add data object in batches, optionally pop each element after adding into index
4. save index into file every batch, print batch id 
5. store final index
6. create (query,data,cos_sim) df, 

3 indexes stored at all times: 
n-2, n-1, n <-- this will be the active one always

when new data is being added to n:
    load n, 
    add to n
    save as n+1
    test n+1, if pass:
        phase out n-2, make n-1 == n - 2
        phase out n-1, make n == n - 1
        make n = n+1
    else:
        keep n as it is

"""
import os
import shutil
import re
import hnswlib
import spacy
import time
import scipy.spatial
import pandas as pd
import numpy as np
import sklearn
import dill as pickle
import mysql.connector
import logging
from datetime import datetime
from nltk import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from stopwords import added_stopwords
from utils import *
from tests import *

# Logging
time_stamp = time.strftime("%Y-%m-%d")
LOG_DIR = "/var/log/qbs/"
LOG_PREFIX = "index_update_service"
LOG_SUFFIX = time_stamp + ".log"

LOG_FILENAME = LOG_PREFIX + LOG_SUFFIX

print("LOG FNAME: ", LOG_FILENAME)

logger = create_logger(name='index_update_service',
                       log_dir=LOG_DIR,
                       filename=LOG_FILENAME,
                       level=logging.DEBUG,
                       log_to_console=True)

logger.info(f"Loading connection engine")
conn = load_connection_engine()


# TODO:
def fetch_new_questions_from_db():
    QUERY = """
        SELECT
            q.id,
            CONCAT(q.question, ' ', CONCAT(',', GROUP_CONCAT(qho.content))) AS question_text
        FROM
            question q
            LEFT JOIN question_has_options qho ON qho.question_id = q.id
        WHERE
            q.processed = False
        GROUP BY
            q.id,
            q.question;
    """
    cur = conn.cursor(buffered=False)
    cur.execute(QUERY)
    text = []
    ids = []
    for t in cur:
        ids.append(t[0])
        text.append(t[1])
    return text, ids


def update_processed_questions(update_list):
    update_list = tuple(update_list)
    QUERY = f"""
    UPDATE question q
    SET q.processed = True
    WHERE q.id IN {update_list}
    """
    # cur = conn.cursor(buffered=False)
    # cur.execute(QUERY)
    logger.info(f"""UPDATE QRY: 
            {QUERY}
    """)


def tmp_load_some_data():
    with open(DATA_PATH, 'rb') as f:
        train_data = pickle.load(f)
    train_data = train_data.iloc[:300]
    q_id = train_data['q_id'].values
    text = train_data['text'].values
    return text, q_id


def run_tests(new_index, vectorizer, svd):
    # 1. Does it pass the bad batch?
    matching_score = test_recall(new_index, vectorizer, svd, logger)
    logger.info(f"Recall on master eval dataset = {matching_score}")
    # 2. Is latency acceptable?
    return "passed", None, matching_score


def make_index(tfidf_vecs, unique_ids, hnsw_params, prev_index_size):
    def _batch(iterable1, iterable2, n=1):
        l1 = iterable1.shape[0]
        l2 = iterable2.shape[0]
        print(l1, l2)
        assert l1 == l2
        # print("LEN OF ITER", l)
        for ndx in range(0, l1, n):
            # print("ITER", iterable[ndx:min(ndx + n, l)])
            yield iterable1[ndx:min(ndx + n, l1)], iterable2[ndx:min(ndx + n, l1)]

    curr_idx_size = prev_index_size
    dim = hnsw_params.get("dim")
    M = hnsw_params.get("M", 200)
    ef_construction = hnsw_params.get("ef_construction", 200)
    item_batch_size = hnsw_params.get("item_batch_size", 10)
    num_threads = hnsw_params.get("num_threads", -1)

    num_elements = tfidf_vecs.shape[0]
    logger.info(f"Num Elements in this batch: {num_elements}")

    logger.info(f"Generating data \n\n {tfidf_vecs.shape}\n\n")
    data = tfidf_vecs
    data_labels = unique_ids

    logger.info("Declaring index")
    # possible options are l2, cosine or ip
    p = hnswlib.Index(space='cosine', dim=dim)

    try:
        logger.info(f"""
            Loading Previously saved index:
                Loading File:     {MODEL_DIR+INDEX_SAVE_FILE}
                New max_elements: {prev_index_size + num_elements}
                Params :          {hnsw_params}
        """)
        p.load_index(MODEL_DIR+INDEX_SAVE_FILE,
                     max_elements=prev_index_size + num_elements)
        curr_idx_size += num_elements
    except Exception as e:
        logger.info("First time?")
        logger.info(e, exc_info=False)
        # Initing index - the maximum number of elements should be known beforehand
        logger.info(f"New Index size: {num_elements}")
        curr_idx_size = num_elements
        p.init_index(max_elements=num_elements,
                     ef_construction=ef_construction,
                     M=M)

    # List of qids to update as 'processed=True'
    update_list = []
    # Element insertion (can be called several times):
    logger.info(f"Adding n={item_batch_size} batches of items from the data")
    for d, dl in _batch(data, data_labels, n=item_batch_size):
        p.add_items(data=d,
                    ids=dl,
                    num_threads=num_threads)
        logger.info(f"Adding keys to update list")
        update_list.extend(dl)

    # Save index so far
    p.save_index(MODEL_DIR+TMP_INDEX_SAVE_FILE)

    logger.info(f"""Index Created and saved, 
                    current calculated index capacity: {p.get_current_count()}
                    Update list len:                   {len(update_list)}""")
    update_processed_questions(update_list)
    logger.info(f"STOP TIME: {datetime.now()}")
    return p


def run_index_creation():
    def _batch(iterable1, iterable2, n=1):
        l1 = iterable1.shape[0]
        l2 = iterable2.shape[0]
        assert l1 == l2
        for ndx in range(0, l1, n):
            yield iterable1[ndx:min(ndx + n, l1)], iterable2[ndx:min(ndx + n, l1)]

    # Load New Data
    # TODO:
    # text, ids = fetch_new_questions_from_db()
    text, ids = tmp_load_some_data()

    # Extract text and ids
    # vec, svd, ind = load_search_engine_components()
    vectorizer = load_vectorizer()
    SVD = load_svd()

    # tfidf vectorize the texts
    logger.info(f"TFIDF")
    text_vec = vectorizer.transform(text)
    logger.info(f"Text Vectorized dims: {text_vec.shape}")

    # dimensionality reduction
    logger.info(f"SVD")
    text_vec = SVD.transform(text_vec)
    logger.info(f"Text SVD'd dims: {text_vec.shape}")

    hnsw_params = {
        "dim": SVD_DIM,  # SVD dims
        "M": 64,
        "ef_construction": 100,
        "item_batch_size": 1000
    }

    with open(MODEL_DIR+INDEX_SIZE_PATH, 'rb') as f:
        prev_index_size = pickle.load(f)

    logger.info(f"Previous index size = {prev_index_size}")

    # for tfidf_vecs, uniq_ids in _batch(tfidf, full_q_smpl_train_uniq_ids, n=chunk_size):
    logger.info(f"Add new vecs to the index")
    # logger.info(tfidf_vecs, uniq_ids)
    new_index = make_index(
        text_vec,
        ids,
        hnsw_params,
        prev_index_size
    )
    # index_file_name=MODEL_DIR + INDEX_SAVE_FILE
    logger.info(f"Current index size {new_index.get_current_count()}")

    # Test new index
    logger.info(f"Running Tests")
    status, latency, recall = run_tests(new_index, vectorizer, SVD)
    if status == 'passed':
        logger.info(f"Index Passes tests {latency}, {recall}")
        # Phase out n-2, make n-1 as n-2
        logger.info(
            f"Moving {MODEL_DIR+INDEX_SAVE_FILE_1} into {MODEL_DIR+INDEX_SAVE_FILE_2}")
        shutil.copy(MODEL_DIR+INDEX_SAVE_FILE_1, MODEL_DIR+INDEX_SAVE_FILE_2)
        # Phase out n-1, make n as n-1
        logger.info(
            f"Moving {MODEL_DIR+INDEX_SAVE_FILE} into {MODEL_DIR+INDEX_SAVE_FILE_1}")
        shutil.copy(MODEL_DIR+INDEX_SAVE_FILE, MODEL_DIR+INDEX_SAVE_FILE_1)
        # Phase out n, make n+1 as n
        logger.info(
            f"Moving {MODEL_DIR+INDEX_SAVE_FILE} into {MODEL_DIR+INDEX_SAVE_FILE_1}")
        shutil.copy(MODEL_DIR+TMP_INDEX_SAVE_FILE, MODEL_DIR+INDEX_SAVE_FILE)
        # Save Index Size:
        logger.info(f"Saving Index Size")
        index_size = new_index.get_current_count()
        with open(MODEL_DIR+INDEX_SIZE_PATH, 'wb') as f:
            pickle.dump(index_size, f)
    else:
        logger.info("New index fails tests, keeping the current index as is.")


def main():
    try:
        run_index_creation()
    except Exception as e:
        logger.info(e, exc_info=True)


if __name__ == "__main__":
    main()
