class BRIN:
    def __init__(self, version, pages_per_range, revmap_page_maxitems, regular_page_maxitems,
                 meta=None, revmaps=None, block_ranges=None):
        self.version = version
        self.pages_per_range = pages_per_range
        self.revmap_page_maxitems = revmap_page_maxitems
        self.regular_page_maxitems = regular_page_maxitems
        self.meta = meta
        self.revmaps = revmaps if revmaps is not None else []
        self.block_ranges = block_ranges if block_ranges is not None else []

    @staticmethod
    def init_by_dict(d: dict):
        return BRIN(version=d['version'],
                    pages_per_range=d['pages_per_range'],
                    revmap_page_maxitems=d['revmap_page_maxitems'],
                    regular_page_maxitems=d['regular_page_maxitems'],
                    meta=d['meta'],
                    revmaps=d['revmaps'],
                    block_ranges=d['block_ranges'])

    def build(self):
        return None


class Meta:
    def __init__(self, version, pages_per_range, last_revmap_page):
        self.version = version
        self.pages_per_range = pages_per_range
        self.last_revmap_page = last_revmap_page

    @staticmethod
    def init_by_dict(d: dict):
        return Meta(version=d['version'],
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


class BlockRange:
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
        return BlockRange(id=d['id'],
                           itemoffsets=d['itemoffsets'],
                           blknums=d['blknums'],
                           attnums=d['attnums'],
                           allnulls=d['allnulls'],
                           hasnulls=d['hasnulls'],
                           placeholders=d['placeholders'],
                           values=d['values'])
