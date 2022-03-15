from src.spatial_index.common_utils import ZOrder, Region

"""
对brin进行改进，来适应z的索引
1. regular_pages不分page：由于非满四叉树分区数量有限，对应block的数量也有限，因此所有block存储在一个regular_page里
2. regular_pages.values里存储的不再是两个值block，而是一个值：非满四叉树和morton编码适配后，morton排序后的前后分区的morton值是连续的
因此block可以用min_z表示，next_block.min_z = cur_block.max_z + 1
3. regular_page作为brin唯一的结构：由于regular_pages不分page，而且block形成的数据量由数据的空间分布决定，而非数据本身决定，
因此revmap没有存在的意义， meta_page自然也不需要了
"""


class ZBRIN:
    def __init__(self, version=None, size=None, minreg=None, blknums=None, values=None):
        self.version = version
        self.size = size
        self.minreg = minreg
        self.blknums = blknums
        self.values = values

    @staticmethod
    def init_by_dict(d: dict):
        return ZBRIN(version=d['version'],
                     size=d['size'],
                     minreg=d['minreg'],
                     blknums=d['blknums'],
                     values=d['values'])

    def save_to_dict(self):
        return {
            'version': self.version,
            'size': self.size,
            'minreg': self.minreg,
            'blknums': self.blknums,
            'values': self.values
        }

    def build(self, quad_tree):
        """
        通过四叉树构建block range
        :param quad_tree:
        :return:
        """
        split_data = quad_tree.leaf_nodes
        self.size = len(split_data)
        self.minreg = min([item["region_width"] for item in split_data])
        self.blknums = [len(item["items"]) for item in split_data]
        self.values = [item["first_z"] for item in split_data]

    def point_query(self, point):
        """
        query index by z point
        :param point: z
        :return: index
        """
        for i in range(self.size):
            if point < self.values[i]:
                return i - 1
        return None

    def range_query(self, point1, point2):
        """
        range index by z1/z2 point
        :param point1: z
        :param point2: z
        :return: index1, index2
        """
        for i in range(self.size):
            if point1 < self.values[i]:
                break
        for j in range(i - 1, self.size):
            if point2 < self.values[j]:
                return i - 1, j - 1
        return None, None

    def range_query_old(self, point1, point2):
        """
        range index by z1/z2 point
        1. get the value1/value2 in regular_page.values which contains z1/z2
        2. get the geohash of leaf model from blknums by value1/value2
        :param point1: z
        :param point2: z
        :return: geohash1, geohash2
        """
        result = []
        z_order = ZOrder(data_precision=6, region=Region(40, 42, -75, -73))
        for i in range(self.size):
            if point1 < self.values[i]:
                z1 = self.values[i - 1]
                break
        for j in range(i - 1, self.size):
            if point2 < self.values[j]:
                z2_next = self.values[j]
                break
        # TODO: 由于z和point是多对一的关系，所以z的point不一定位于左下角，因此z-1的point也不一定在右上角
        window_left, window_bottom = z_order.z_to_min_point(z1)
        window_right, window_top = z_order.z_to_min_point(z2_next - 1)
        child_z_block_list = []
        while window_bottom < window_top:
            tmp_window_top = window_bottom + self.minreg
            while window_left < window_right:
                tmp_window_right = window_left + self.minreg
                child_z_block_list.append([z_order.point_to_z(window_left, window_bottom),
                                           z_order.point_to_z(tmp_window_right, tmp_window_top)])
                window_left = tmp_window_right
            window_bottom = tmp_window_top

        tmp_result = []
        for child_z_block in child_z_block_list:
            for k in range(i - 1, j):
                if child_z_block[1] < self.values[k]:
                    tmp_result.append([k - 1, child_z_block])
        tmp_dict = dict(tmp_result)
        return tmp_dict
