class Index:
    def __init__(self, name):
        self.name = name

    def build(self, points):
        for point in points:
            self.insert(point)

    def insert(self, point):
        return

    def predict(self, point):
        return self.search(point)

    def search(self, point):
        return []

    def delete(self, point):
        return
