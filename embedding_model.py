"""
The Model
"""
import logging
import pandas as pd
import numpy as np
# Check that PyTorch sees it
import torch
import transformers
import sentence_transformers

from sentence_transformers import SentenceTransformer, models
import scipy.spatial

import pandas as pd

logger = logging.getLogger("sentence_process_logger")

logger.info("Loading model")
logger.info(torch.cuda.is_available())

BASE_DIR = '/mnt/z/Project/'
MODEL_DIR = 'model/'
DBERT_MODEL_DIR = MODEL_DIR+'distilbert_no_tags_full/'

BATCH_SIZE = 16


class Model():
    def __init__(self):
        self.DBERT_MODEL_DIR = DBERT_MODEL_DIR
        self.word_embedding_model = None
        self.pooling_model = None
        self.model = None
        self.BATCH_SIZE = 16
        
    def make(self):
        # Use Fine tuned distilBert for mapping tokens to embeddings
        self.word_embedding_model = models.Transformer(self.DBERT_MODEL_DIR)
        # Apply mean pooling to get one fixed sized sentence vector
        self.pooling_model = models.Pooling(self.word_embedding_model.get_word_embedding_dimension())
        self.model = SentenceTransformer(modules=[self.word_embedding_model, self.pooling_model])
        print(f"Model Compiled: {self.DBERT_MODEL_DIR}")
        logger.info("Model Compiled")
    
    def make_pretrained(self, model_name):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Model Compiled: {model_name}")
        print(f"Model Compiled: {model_name}")

    def encode_sentences(self, sentences):
        sentence_embeddings = self.model.encode(sentences)
        return sentence_embeddings
