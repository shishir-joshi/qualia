"""
API
"""
import numpy as np
import pandas as pd
import dill as pickle
from utils import *
from tests import *
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel


app = FastAPI()

vectorizer, svd, index = load_search_engine_components()

conn = load_connection_engine()

# TODO: temp, remove once index labels are fixed
with open(label_text_mapping_file, 'rb') as f:
# with open('/work/paptronics/model/v7/'+'label_text_mapping.pkl', 'rb') as f:
    label_text_mapping = pickle.load(f)

# Logging
time_stamp = time.strftime("%Y-%m-%d")
LOG_DIR = "/var/log/qbs/"
LOG_PREFIX = "api"
LOG_SUFFIX = time_stamp + ".log"

LOG_FILENAME = LOG_PREFIX + LOG_SUFFIX

print("LOG FNAME: ", LOG_FILENAME)

logger = create_logger(name='api',
                       log_dir=LOG_DIR,
                       filename=LOG_FILENAME,
                       level=logging.DEBUG,
                       log_to_console=True)

logger.info("API LOADED")

class QueryQuestion(BaseModel):
    question_text: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/test_query_by_question_content")
def test_query_by_question_content(query: QueryQuestion):
    logger.info(f"API: test_query_by_question_content")
    stime = datetime.now()
    logger.info(f"REQ START: {stime}")
    # clean_question_text = re.sub(r'\\+', r'\\', query.question_text)
    logger.info(f"TOKENS: {stemm_tokenize(query.question_text)}")
    # clean_question_text = re.sub(r'\\n[a-z]\)', r',', clean_question_text)
    # logger.info(f"TOKENS: {stemm_tokenize(query.question_text)}")
    vec = vectorizer.transform([query.question_text])

    def display_scores(vectorizer, tfidf_result):
        # http://stackoverflow.com/questions/16078015/
        scores = zip(vectorizer.get_feature_names(),
                     np.asarray(tfidf_result.sum(axis=0)).ravel()
                     )
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        for item in sorted_scores:
            if item[1] > 0.0:
                logger.info("{0:50} Score: {1}".format(item[0], item[1]))

    logger.info(f"QUERY: {query.question_text}")
    display_scores(vectorizer, vec)
    logger.info("SVD")
    vec = [svd.transform(x.A)[0] for x in vec]
    logger.info(f"SVD: {len(vec)}")
    labels, distances = index.knn_query(list(vec),
                                        k=MAX_NEAREST_NBRS,
                                        num_threads=MAX_SEARCH_THREADS)
    logger.info(f"QUERY DONE")
    if labels is None:
        logger.info(f"Labels is none {labels}")
        return {"matches": []}

    resp = []
    for q_ids, dists in zip(labels, distances):
        raw_set = []
        for qid, d in zip(q_ids, dists):
            jacc_sim = jaccard_index(
                            set(stemm_tokenize(query.question_text)), 
                            set(stemm_tokenize(label_text_mapping[qid]))
                        )
            raw_set.append((int(qid), d, label_text_mapping[qid], jacc_sim))
        raw_set = sorted(raw_set, key=lambda x: x[3], reverse=True)
        resp.append([{k: (v, "%.3f"%((1-d)*100))} for k, d, v, _ in raw_set])
    
    logger.info("FEATURES OF TOP MATCH")
    logger.info("---------------------------------------------------------------------------------------")
    logger.info("---------------------------------------------------------------------------------------")
    logger.info(stemm_tokenize(list(resp[0][0].values())[0][0]))
    logger.info(list(resp[0][0].values())[0][0])
    display_scores(vectorizer, vectorizer.transform([list(resp[0][0].values())[0][0]]))
    logger.info("---------------------------------------------------------------------------------------")
    logger.info("---------------------------------------------------------------------------------------")
    logger.info(resp)
    logger.info(f"REQ STOP: {datetime.now() - stime}")
    return {"matches": resp[0]}


@app.post("/query_by_question_content")
def query_by_question_content(query: QueryQuestion):
    logger.info(f"API: query_by_question_content")
    stime = datetime.now()
    logger.info(f"REQ START: {stime}")
    logger.info(f"TOKENS: {stemm_tokenize(query.question_text)}")
    vec = vectorizer.transform([query.question_text])
    
    def display_scores(vectorizer, tfidf_result):
        # http://stackoverflow.com/questions/16078015/
        scores = zip(vectorizer.get_feature_names(),
                     np.asarray(tfidf_result.sum(axis=0)).ravel()
                     )
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        for item in sorted_scores:
            if item[1] > 0.0:
                logger.info("{0:50} Score: {1}".format(item[0], item[1]))

    logger.info(f"QUERY: {query.question_text}")
    display_scores(vectorizer, vec)

    vec = [svd.transform(x.A)[0] for x in vec]
    # print("SVD: ", list(vec))
    labels, distances = index.knn_query(list(vec),
                                        k=MAX_NEAREST_NBRS,
                                        num_threads=MAX_SEARCH_THREADS)
    if labels is None:
        logger.info(f"Labels is none {labels}")
        return {"matches": []}
    
    resp = []
    for q_ids, dists in zip(labels, distances):
        raw_set = []
        for qid, d in zip(q_ids, dists):
            jacc_sim = jaccard_index(
                            set(stemm_tokenize(query.question_text)), 
                            set(stemm_tokenize(label_text_mapping[qid]))
                        )
            raw_set.append((int(qid), d, label_text_mapping[qid], jacc_sim))
        raw_set = sorted(raw_set, key=lambda x: x[3], reverse=True)
        resp.append([[k, "%.3f"%((1-d)*100)] for k, d, v, _ in raw_set])
    logger.info(resp)
    logger.info(f"REQ STOP: {datetime.now() - stime}")
    return {"matches": resp[0]}
