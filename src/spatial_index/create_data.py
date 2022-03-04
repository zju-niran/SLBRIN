import csv
import os

import pandas

from src.spatial_index.common_utils import Point, Region


def count_csv(path):
    with open(path) as f1:
        reader = csv.reader(f1)
        count = 0
        for line in reader:
            count += 1
        return count


def filter_row_from_csv(input_path, output_path, lines_limit, range_limit=None):
    with open(input_path) as f1, open(output_path, 'w', newline='') as f2:
        count = 0
        reader = csv.reader(f1)
        writer = csv.writer(f2)
        for line in reader:
            if lines_limit and count > lines_limit:
                break

            if count == 0:
                writer.writerow(["id", "x", "y"])
            if count != 0:
                pickup_lng = line[10]
                pickup_lat = line[11]
                if range_limit and range_limit.contain(Point(float(pickup_lng), float(pickup_lat))) is False:
                    continue
                else:
                    target_row = [count, pickup_lng, pickup_lat]
                    writer.writerow(target_row)
            count += 1


def sample_from_csv(input_path, output_path, lines_limit, range_limit=None):
    df = pandas.read_csv(input_path, usecols=[10, 11])
    df = df.rename(columns={"pickup_longitude": "x", "pickup_latitude": "y"})
    df["contain"] = df.apply(
        lambda x: range_limit.contain(Point(float(x["x"]), float(x["y"]))), axis=1)
    df = df[df["contain"]]
    df = df.sample(n=lines_limit, random_state=1)
    df = df.reset_index()
    df = df.drop(columns={"index", "contain"})
    df.to_csv(output_path, index_label="index")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    headers = ['medallion',
               'hack_license',
               'vendor_id',
               'rate_code',
               'store_and_fwd_flag',
               'pickup_datetime',
               'dropoff_datetime',
               'passenger_count',
               'trip_time_in_secs',
               'trip_distance',
               'pickup_longitude',
               'pickup_latitude',
               'dropoff_longitude',
               'dropoff_latitude']
    # 数据来源：从http://www.andresmh.com/nyctaxitrips/下载trip_data.7z，拿到其中的一月份数据csv
    # 数据总记录数：14776616，region内14507253
    # csv文件size：2459600863字节=2.29GB
    input_path = "../../data/trip_data_1.csv"
    output_path = "../../data/trip_data_1_filter.csv"
    # filter_row_from_csv(input_path, output_path, None, Region(40, 42, -75, -73))
    output_path_100000_sample = '../../data/trip_data_1_100000.csv'
    sample_from_csv(input_path, output_path_100000_sample, 100000, Region(40, 42, -75, -73))
