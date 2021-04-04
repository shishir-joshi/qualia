"""
HNSW Utility Class
Needs:
    1. Index Creation
    2. Index Management(Adding, deleting)
    3. Handling labels(maintain mappings, )
"""
import os
import shutil
import hnswlib
import pandas as pd
import numpy as np
import pickle
from utils import *

"""
making new index -- needs new index size, 
"""
class Index():
    """
    Index Classes
    """
    def __init__(self, hnsw_params:dict):
        self.SAVE_DIR = hnsw_params.get("save_dir", INDEX_DIR)
        self.SAVE_FILE = hnsw_params.get("save_file", "hnsw_index.bin")
        self.CURR_IDX_SIZE = self.find_current_index_size()
        self.M = hnsw_params.get("M", 200)
        self.ef_construction = hnsw_params.get("ef_construction", 200)
        self.item_batch_size = hnsw_params.get("item_batch_size", 10)
        self.num_threads = hnsw_params.get("num_threads", MAX_SEARCH_THREADS)
        self.num_elements = hnsw_params.get("num_elements", None)
        self.label_mapping = hnsw_params.get("label_mapping", None)
        self.index = hnswlib.Index(space='cosine', dim=EMB_DIM)
        self.index_loaded = False

    def init_search(self):
        """
        load any presaved index if not, return empty index def
        """
        print("Loading saved index")
        self.index.load_index(self.SAVE_DIR+self.SAVE_FILE,
                              max_elements=self.CURR_IDX_SIZE)
        self.index_loaded = True

    def find_current_index_size(self):
        """
            Find saved index size 
        """
        try:
            return pickle.load(open(INDEX_SIZE_PATH, 'rb'))
        except Exception as e:
            print("No Index size pre saved")
            return None

    def define_index(self, idx_size):
        """
        define fresh new index with params given when init
        save in given path
        """
        print("First time?")
        # Initing index - the maximum number of elements should be known beforehand
        self.CURR_IDX_SIZE = idx_size
        self.index.init_index(max_elements=self.CURR_IDX_SIZE,
                              ef_construction=self.ef_construction,
                              M=self.M)
        print(f"Saving Fresh New index: {self.SAVE_DIR+self.SAVE_FILE}")
        self.index.save_index(self.SAVE_DIR+self.SAVE_FILE)
        print(f"Save index size: {self.CURR_IDX_SIZE}")
        pickle.dump(self.CURR_IDX_SIZE, open(INDEX_SIZE_PATH, 'wb'))

    def _batch(self, iterable1, iterable2, n=1):
        l1 = iterable1.shape[0]
        l2 = iterable2.shape[0]
        assert l1 == l2
        # print("LEN OF ITER", l)
        for ndx in range(0, l1, n):
            # print("ITER", iterable[ndx:min(ndx + n, l)])
            yield iterable1[ndx:min(ndx + n, l1)], iterable2[ndx:min(ndx + n, l1)]
    
    def get_current_count(self):
        if not self.index_loaded:
            print("Index Not Loaded, please run init_search() first")
            return False
        return self.index.get_current_count()


    def update_index(self, data, data_labels):
        """
        batch add things to index and save to save path
        data should be:
            [(vector, label)]
        """
        
        if not self.index_loaded:
            print("Index Not Loaded, please run init_search() first")
            return False
                
        print(f"""
            Loading Previously saved index:
                Loading File:     {self.SAVE_FILE}
                New max_elements: {self.CURR_IDX_SIZE + len(data)}
        """)
        try:
            num_elements = len(data)
            self.index.load_index(self.SAVE_DIR+self.SAVE_FILE,
                                  max_elements=self.CURR_IDX_SIZE+num_elements)
            self.CURR_IDX_SIZE += num_elements
        except Exception as e:
            print(e)
            print("Probably not saved earlier")
            return None
        
        # List of qids to update as 'processed=True'
        update_list = [] 
        data = np.array(data)  
        data_labels = np.array(data_labels) 
        # Element insertion (can be called several times):
        print(f"Adding n={self.item_batch_size} batches of items from the data")
        for d, dl in self._batch(data, data_labels, n=self.item_batch_size):
            self.index.add_items(data=d,
                                 ids=dl,
                                 num_threads=self.num_threads)
            update_list.extend(dl)

        print(f"Saving new index of size {self.index.get_current_count()}")
        
        # Save index so far
        self.index.save_index(self.SAVE_DIR+self.SAVE_FILE)
        
        self.CURR_IDX_SIZE = self.index.get_current_count()
        print(f"Save index size: {self.CURR_IDX_SIZE}")
        pickle.dump(self.CURR_IDX_SIZE, open(INDEX_SIZE_PATH, 'wb'))

    def search(self, query_vector, max_nearest_nbrs):
        if not self.index_loaded:
            print("Index Not Loaded, please run init_search() first")
            return False

        labels, distances = self.index.knn_query(list(query_vector),
                                                 k=max_nearest_nbrs,
                                                 num_threads=self.num_threads)
        if labels is None:
            print(f"No Matches found")
            return {"matches": []}

        if self.label_mapping is not None:
            resp = []
            for q_ids, dists in zip(labels, distances):
                raw_set = []
                for qid, d in zip(q_ids, dists):
                    raw_set.append((int(qid), d, self.label_mapping[qid]))
                raw_set = sorted(raw_set, key=lambda x: x[1], reverse=False)
                resp.append([{k: (v, "%.3f"%((1-d)*100))} for k, d, v in raw_set])
        
        return {"matches": resp[0]}

    def __str__(self):
        return f"""
        HNSW Index Params: 
            SAVE_DIR: {self.SAVE_DIR},
            SAVE_FILE: {self.SAVE_FILE},
            CURR_IDX_SIZE: {self.CURR_IDX_SIZE},
            M: {self.M},
            ef_construction: {self.ef_construction},
            item_batch_size: {self.item_batch_size},
            num_threads: {self.num_threads},
            index_loaded: {self.index_loaded}
        """  