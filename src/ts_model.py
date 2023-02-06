import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.vector_ar.var_model import VAR

from src.spatial_index.common_utils import binary_search_less_max_duplicate


class TimeSeriesModel:
    def __init__(self,
                 old_cdfs, cur_cdf,
                 old_max_keys, cur_max_key,
                 key_list):
        # common
        self.name = "Time Series Model"
        # for ts of cdf
        self.old_cdfs = old_cdfs
        self.cur_cdf = cur_cdf
        # for ts of max key
        self.old_max_keys = old_max_keys
        self.cur_max_key = cur_max_key
        # for compute
        self.key_list = key_list

    def build(self, cdf_width, cdf_lag):
        old_cdfs_num = len(self.old_cdfs)
        if old_cdfs_num == 0:  # if old_cdfs is []
            self.cur_cdf = [0.0] * cdf_width
            self.cur_max_key = 0
        elif old_cdfs_num <= cdf_lag:  # if old_cdfs are not enough for ts_model in quantity
            self.cur_cdf = self.old_cdfs[-1]
            self.cur_max_key = self.old_max_keys[-1]
        else:
            self.cur_cdf = train_var(self.old_cdfs, cdf_width, cdf_lag)
            self.cur_max_key = round(train_ar(self.old_max_keys, cdf_lag))

    def update(self, data, cdf_width, cdf_lag):
        self.old_cdfs.append(build_cdf(data, cdf_width, self.key_list))
        self.old_max_keys.append(len(data) - 1)
        self.build(cdf_width, cdf_lag)


def train_var(data, width, lag):
    """
    y_t = const_trend + y_t-1 @ w1 + y_t-2 @ w2 + ...
    model consists of {const_trend, w1, w2, ...}, with shape of (1 + lag * width, width)
    """
    try:
        model = VAR(data).fit(maxlags=lag, verbose=True).params
    except ValueError:
        model = VAR(data).fit(maxlags=lag, verbose=True, trend="n").params
    # similar to VAR.forecast(np.array(data), steps=1)
    new_data = np.zeros(width)
    start = 0
    if model.shape[0] > lag * width:
        new_data = model[0]
        start = 1
    for i in range(0, lag):
        new_data += data[-i - 1] @ model[start: start + width]
        start += width
    return list(correct_cdf(new_data))


def train_ar(data, lag):
    """
    y_t = const_trend + y_t-1 @ w1 + y_t-2 @ w2 + ...
    model consists of {const_trend, w1, w2, ...}, with shape of 1 + lag
    """
    model = AutoReg(data, lags=lag).fit().params
    # similar to AutoReg.forecast(steps=1)
    new_data = 0
    start = 0
    if model.shape[0] > lag:
        new_data = model[0]
        start = 1
    for i in range(0, lag):
        new_data += data[-i - 1] * model[start + i]
    return correct_max_key(new_data)


def build_cdf(data, cdf_width, key_list):
    x_len = len(data)
    x_max_key = x_len - 1
    cdf = []
    p = 0
    for l in range(cdf_width):
        p = binary_search_less_max_duplicate(data, key_list[l], p, x_max_key)
        cdf.append(p / x_len)
    return cdf


def correct_cdf(cdf):
    """
    correct the predicted cdf:
    1. normalize into [0.0, 1.0]
    2. transfer into monotonically increasing
    e.g. [0.1, 0.0, 0.1, 0.7, 0.6, 1.0, 0.9] => [0.0, 0.0, 0.1, 0.7, 0.7, 1.0, 1.0]
    """
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())
    i = 0
    cdf_width = len(cdf)
    cur_v = 0.0
    while i < cdf_width and cdf[i] != 0.0:
        cdf[i] = 0.0
        i += 1
    while i < cdf_width and cdf[i] < 1.0:
        if cdf[i] < cur_v:
            cdf[i] = cur_v
        else:
            cur_v = cdf[i]
        i += 1
    while i < cdf_width:
        cdf[i] = 1.0
        i += 1
    return cdf


def correct_max_key(max_key):
    """
    correct the max_key:
    1. max_key >= 0
    """
    return max(0, max_key)
