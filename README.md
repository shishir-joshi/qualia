# qualia
Online Real Time Semantic Search using Transformers and HNSW nearest neighbour search.

Uses Sentence Transformers (https://github.com/UKPLab/sentence-transformers) as embedding model
and HNSWlib (https://github.com/nmslib/hnswlib) as approximate nearest neighbour search index based on cosine similarity.

The aim is to make it "Online" - to be able to add new documents in parallel with querying. 

TODO
====

1. Make Data Ingestor class to handle any document type
2. Make arrangements for fine tuning 
3. Make it "Online" - Add Index management(cycling backups, switching to fresh index)
4. Make better demo notebook
5. Make arXiv specific ranking scheme - embed documents at the sentence level
6. MAKE BETTER README
