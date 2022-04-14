import os
import sys
import time

import numpy as np

sys.setrecursionlimit(5000)
sys.path.append('/home/zju/wlj/st-learned-index')
from src.spatial_index.common_utils import Region, quick_sort
from src.spatial_index.geohash_utils import Geohash

os.chdir(os.path.dirname(os.path.realpath(__file__)))
# path = '../../data/trip_data_1_10w.npy'
path = '../../data/trip_data_1_filter.npy'
data = np.load(path).tolist()
geohash = Geohash.init_by_precision(data_precision=6, region=Region(40, 42, -75, -73))
start_time = time.time()
data = [[t[0], t[1], geohash.encode(t[0], t[1])] for t in data]
quick_sort(data, 2, 0, len(data) - 1)
end_time = time.time()
search_time = end_time - start_time
print("Load data time: %s" % search_time)
