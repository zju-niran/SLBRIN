# SBRIN
Please understand that the ReadMe instructions will be added after the paper is submitted.

# Folder Description
* data
  * table: the origin datasets, including NYCT, UNIFORM and NORMAL.
  * index: the index entries of the origin datasets sorted by geohash.
  * query: the query conditions, including point query, range query and knn query.
  * create_data.py: the python scripts to create tables, indexes and queries.
* result
  * create_result.py: the python scripts to create results, such as figures in the paper.
* src: the source code
  * spatial_index: the source code of our SBRIN and other competitors. 
  * experiment: the experiments in the paper.
  * learned_model*.py: the RMI version for learned model.
* requirement.txt: a list python dependencies, which can install by 'pip install -r requirement.txt'