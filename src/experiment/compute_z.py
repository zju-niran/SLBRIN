import os
import sys
import time

import pandas as pd

sys.path.append('/home/zju/wlj/st-learned-index')
from src.spatial_index.common_utils import Region
from src.spatial_index.geohash_utils import Geohash

os.chdir(os.path.dirname(os.path.realpath(__file__)))
path = '../../data/trip_data_1_filter.csv'
data = pd.read_csv(path)
print("Start")
start_time = time.time()
geohash = Geohash.init_by_precision(data_precision=6, region=Region(40, 42, -75, -73))
data["z"] = data.apply(lambda t: geohash.point_to_z(t.x, t.y), 1)
data.sort_values(by=["z"], ascending=True, inplace=True)
data.reset_index(drop=True, inplace=True)
data_list = data.values.tolist()
end_time = time.time()
search_time = end_time - start_time
print("Load data time: %s" % search_time)
