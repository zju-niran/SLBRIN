import numpy as np
import pandas as pd

from src.spatial_index.common_utils import Region, ZOrder
from src.spatial_index.quad_tree import QuadTree


class BRIN:
    def __init__(self, version, pages_per_range, revmap_page_maxitems, regular_page_maxitems,
                 meta_page=None, revmap_pages=None, regular_pages=None):
        self.version = version
        self.pages_per_range = pages_per_range
        self.revmap_page_maxitems = revmap_page_maxitems
        self.regular_page_maxitems = regular_page_maxitems
        self.meta_page = meta_page
        self.revmap_pages = revmap_pages if revmap_pages is not None else []
        self.regular_pages = regular_pages if regular_pages is not None else []

    @staticmethod
    def init_by_dict(d: dict):
        return BRIN(version=d['version'],
                    pages_per_range=d['pages_per_range'],
                    revmap_page_maxitems=d['revmap_page_maxitems'],
                    regular_page_maxitems=d['regular_page_maxitems'],
                    meta_page=d['meta_page'],
                    revmap_pages=d['revmap_pages'],
                    regular_pages=d['regular_pages'])

    def build(self):
        return None

    def point_query(self, point):
        """
        query index by z point
        1. get the value in regular_pages.values which contains x
        2. get the geohash of leaf model from blknums by value
        :param data: z
        :return: geohash
        """
        for regular_page in self.regular_pages:
            for i in range(len(regular_page.itemoffsets)):
                value = regular_page.values[i]
                if point >= value[0]:
                    if point <= value[1]:
                        return regular_page.blknums[i]
                else:
                    return None

    def build_by_quad_tree(self, quad_tree):
        """
        不限制pages_per_range，per range的pages大小=四叉树该节点的数据数量
        :param quad_tree:
        :return:
        """
        split_data = quad_tree.geohash_items_map
        z_border_list = []
        blknum_list = []
        for geohash_key in split_data:
            z_border = split_data[geohash_key]["z_border"]
            blknum = geohash_key
            z_border_list.append(z_border)
            blknum_list.append(blknum)
        index_len = len(split_data)
        page_len = int(index_len / self.regular_page_maxitems)
        revmap_page_list = []
        for i in range(page_len + 1):
            left_index = i * self.regular_page_maxitems
            right_index = (i + 1) * self.regular_page_maxitems
            if right_index > index_len:
                right_index = index_len
            if left_index > index_len:
                break
            revmap_page_list.extend(
                [{"regular_page_id": i, "regular_page_item": j} for j in range(left_index, right_index)])
            self.regular_pages.append(RegularPage(id=i,
                                                  itemoffsets=list(range(left_index, right_index)),
                                                  blknums=blknum_list[left_index: right_index],
                                                  values=z_border_list[left_index: right_index]))
        revmap_page_len = len(revmap_page_list)
        page_len = int(revmap_page_len / self.revmap_page_maxitems)
        for i in range(page_len + 1):
            left_index = i * self.revmap_page_maxitems
            right_index = (i + 1) * self.revmap_page_maxitems
            if right_index > index_len:
                right_index = index_len
            if left_index > index_len:
                break
            self.revmap_pages.append(RevMapPage(id=i,
                                                pages=revmap_page_list[left_index: right_index]))
        self.meta_page = MetaPage(version=self.version,
                                  pages_per_range=self.pages_per_range,
                                  last_revmap_page=len(self.revmap_pages))


class MetaPage:
    def __init__(self, version, pages_per_range, last_revmap_page):
        self.version = version
        self.pages_per_range = pages_per_range
        self.last_revmap_page = last_revmap_page

    @staticmethod
    def init_by_dict(d: dict):
        return MetaPage(version=d['version'],
                        pages_per_range=d['pages_per_range'],
                        last_revmap_page=d['last_revmap_page'])


class RevMapPage:
    def __init__(self, id, pages):
        self.id = id
        self.pages = pages

    @staticmethod
    def init_by_dict(d: dict):
        return RevMapPage(id=d['id'],
                          pages=d['pages'])


class RegularPage:
    def __init__(self, id, itemoffsets, blknums, values,
                 attnums=None, allnulls=None, hasnulls=None, placeholders=None):
        self.id = id
        self.itemoffsets = itemoffsets
        self.blknums = blknums
        self.attnums = attnums
        self.allnulls = allnulls
        self.hasnulls = hasnulls
        self.placeholders = placeholders
        self.values = values

    @staticmethod
    def init_by_dict(d: dict):
        return RegularPage(id=d['id'],
                           itemoffsets=d['itemoffsets'],
                           blknums=d['blknums'],
                           attnums=d['attnums'],
                           allnulls=d['allnulls'],
                           hasnulls=d['hasnulls'],
                           placeholders=d['placeholders'],
                           values=d['values'])
