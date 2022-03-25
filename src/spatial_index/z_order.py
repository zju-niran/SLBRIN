class ZOrder:
    def __init__(self, data_precision, region):
        self.name = "Z Order"
        self.dimensions = 2
        self.bits = region.get_bits_by_region_and_precision(data_precision)
        self.data_precision = data_precision
        self.region = region
        self.region_width = region.right - region.left
        self.region_height = region.up - region.bottom

        def flp2(x):
            '''Greatest power of 2 less than or equal to x, branch-free.'''
            x |= x >> 1
            x |= x >> 2
            x |= x >> 4
            x |= x >> 8
            x |= x >> 16
            x |= x >> 32
            x -= x >> 1
            return x

        shift = flp2(self.dimensions * (self.bits - 1))
        masks = []
        lshifts = []
        max_value = (1 << (shift * self.bits)) - 1
        while shift > 0:
            mask = 0
            shifted = 0
            for bit in range(self.bits):
                distance = (self.dimensions * bit) - bit
                shifted |= shift & distance
                mask |= 1 << bit << (((shift - 1) ^ max_value) & distance)

            if shifted != 0:
                masks.append(mask)
                lshifts.append(shift)

            shift >>= 1
        self.lshifts = [0] + lshifts
        self.rshifts = lshifts + [0]
        self.max_num = 1 << self.bits
        self.masks = [self.max_num - 1] + masks

    def point_to_z(self, lng, lat):
        """
        计算point的z order
        1. 经纬度都先根据region归一化到0-1，然后缩放到0-2^self.bits
        2. 使用morton-py.pack(int, int): int计算z order，顺序是左下、右下、左上、右上
        :param lng:
        :param lat:
        :return:
        """
        lng_zoom = round((lng - self.region.left) * self.max_num / self.region_width)
        lat_zoom = round((lat - self.region.bottom) * self.max_num / self.region_height)
        return self.pack(lng_zoom, lat_zoom)

    def z_to_point(self, z):
        """
        计算z order的point
        1. 使用morton-py.unpack(int)
        2. 反归一化
        注意：使用round后，z转化的point不一定=计算z的原始point，因为保留有效位数的point和z是多对一的
        如果要一对一，则point_to_z的入口point和z_to_point的出口point都不要用round
        :param z:
        :return:
        """
        lng_zoom, lat_zoom = self.unpack(z)
        lng = lng_zoom * self.region_width / self.max_num + self.region.left
        lat = lat_zoom * self.region_height / self.max_num + self.region.bottom
        return round(lng, self.data_precision), round(lat, self.data_precision)

    def save_to_dict(self):
        return {
            'name': self.name,
            'data_precision': self.data_precision,
            'region': self.region
        }

    @staticmethod
    def init_by_dict(d: dict):
        return ZOrder(data_precision=d['data_precision'],
                      region=d['region'])

    def split(self, value):
        for o in range(len(self.masks)):
            value = (value | (value << self.lshifts[o])) & self.masks[o]
        return value

    def pack(self, *args):
        code = 0
        for i in range(self.dimensions):
            code |= self.split(args[i]) << i
        return code

    def compact(self, code):
        for o in range(len(self.masks) - 1, -1, -1):
            code = (code | (code >> self.rshifts[o])) & self.masks[o]
        return code

    def unpack(self, code):
        values = []
        for i in range(self.dimensions):
            values.append(self.compact(code >> i))
        return values
