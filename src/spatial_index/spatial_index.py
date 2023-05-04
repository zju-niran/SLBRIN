class SpatialIndex:
    """
    空间索引基础类
    """
    def __init__(self, name):
        self.name = name

    def insert_single(self, point):
        return

    def insert(self, points):
        for point in points:
            self.insert_single(point)

    def point_query_single(self, point):
        """
        query key by x/y point
        """
        return None

    def point_query(self, points):
        return [self.point_query_single(point) for point in points]

    def test_point_query(self, points):
        for point in points:
            self.point_query_single(point)

    def range_query_single(self, window):
        """
        query key by x1/y1/x2/y2 window
        """
        return None

    def range_query(self, windows):
        return [self.range_query_single(window) for window in windows]

    def test_range_query(self, windows):
        for window in windows:
            self.range_query_single(window)

    def knn_query_single(self, knn):
        """
        query key by x1/y1/n knn
        """
        return None

    def knn_query(self, knns):
        return [self.knn_query_single(knn) for knn in knns]

    def test_knn_query(self, knns):
        for knn in knns:
            self.knn_query_single(knn)

    def save(self):
        """
        save index into json file
        """
        return

    def load(self):
        """
        load index from json file
        """
        return

    def size(self):
        """
        get index file size
        """
        return
