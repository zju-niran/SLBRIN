class SpatialIndex:
    def __init__(self, name):
        self.name = name

    def point_query_single(self, point):
        return None

    def point_query(self, points):
        return [self.point_query_single(point) for point in points]

    def test_point_query(self, points):
        for point in points:
            result = self.point_query_single(point)

    def range_query_single(self, window):
        return None

    def range_query(self, windows):
        return [self.range_query_single(window) for window in windows]

    def test_range_query(self, windows):
        for window in windows:
            result = self.range_query_single(window)

    def knn_query_single(self, knn):
        return None

    def knn_query(self, knns):
        return [self.knn_query_single(knn) for knn in knns]

    def test_knn_query(self, knns):
        for knn in knns:
            result = self.knn_query_single(knn)
