import os

from src.spatial_index.common_utils import read_data_and_search, create_data_z
from src.spatial_index.zm_index import ZMIndex

if __name__ == '__main__':
    os.chdir('D:\\Code\\Paper\\st-learned-index')
    path = 'C:\\Users\\123\\Desktop\\trip_data_2.csv'
    # create_data_z(path_100000, path_100000_z, 1, 2)
    path_100000 = 'C:\\Users\\123\\Desktop\\trip_data_2_100000_random.csv'
    path_100000_z = 'C:\\Users\\123\\Desktop\\trip_data_2_100000_random_z.csv'
    # path_test = 'data/test_x_y_index.csv'
    # create_data(path_test)
    # read_data_and_search(path_100000, RTree(), 1, 2, None, 0)
    # read_data_and_search(path_100000, QuadTree(), 1, 2, None, 0)
    read_data_and_search(path_100000_z, ZMIndex(), None, None, 5, 0)
