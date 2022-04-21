import os
import sys
import time

import numpy as np

sys.path.append('/home/zju/wlj/st-learned-index')
from src.spatial_index.common_utils import Region
from src.spatial_index.geohash_utils import Geohash

os.chdir(os.path.dirname(os.path.realpath(__file__)))
# path = '../../data/table/trip_data_1_filter_10w.npy'
path = '../../data/table/trip_data_1_filter.npy'
data = np.load(path, allow_pickle=True)[:, [10, 11, -1]]
geohash = Geohash.init_by_precision(data_precision=6, region=Region(40, 42, -75, -73))
start_time = time.time()
data = [(data[i][0], data[i][1], geohash.encode(data[i][0], data[i][1]), i) for i in range(len(data))]
data = np.array(sorted(data, key=lambda x: x[2]), dtype=[("0", 'f8'), ("1", 'f8'), ("2", 'i8'), ("3", 'i4')])
end_time = time.time()
search_time = end_time - start_time
print("Load data time: %s" % search_time)
