import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.vector_ar.var_model import VAR


class TimeSeriesModel:
    def __init__(self,
                 cdf_model, old_cdfs, cur_cdf, cdf_width,
                 num_model, old_nums, cur_num,
                 lag):
        # common
        self.name = "Time Series Model"
        self.lag = lag
        # for ts of cdf
        self.cdf_model = cdf_model
        self.old_cdfs = old_cdfs
        self.cur_cdf = cur_cdf
        self.cdf_width = cdf_width
        # for ts of date num
        self.num_model = num_model
        self.old_nums = old_nums
        self.cur_num = cur_num

    def build(self):
        # self.model = VARMAX(self.old_cdfs, order=(self.cdf_lag, self.cdf_lag))
        # self.model = self.model.fit(disp=False)
        # cur_cdf = model.forecast(model.model.y, steps=1)[0] # 和self.cur_cdf一致
        self.cdf_model, self.cur_cdf = self.train_var(self.old_cdfs, self.cdf_width)
        self.num_model, self.cur_num = self.train_ar(self.old_nums)

    def update(self):
        self.old_cdfs.append(self.cur_cdf)
        self.old_nums.append(self.cur_num)
        self.build()

    def train_var(self, data, width):
        """
        y_t = const_trend + y_t-1 @ w1 + y_t-2 @ w2 + ...
        model consists of {const_trend, w1, w2, ...}, with shape of (1 + lag * width, width)
        """
        try:
            model = VAR(data).fit(maxlags=self.lag, verbose=True).params
        except ValueError:
            model = VAR(data).fit(maxlags=self.lag, verbose=True, trend="n").params
        # similar to VAR.forecast(np.array(data), steps=1)
        new_data = np.zeros(width)
        start = 0
        if model.shape[0] > self.lag * width:
            new_data = model[0]
            start = 1
        for i in range(0, self.lag):
            new_data += self.old_cdfs[-i - 1] @ model[start: start + width]
            start += width
        return model, correct_cdf(list(new_data))

    def train_ar(self, data):
        """
        y_t = const_trend + y_t-1 @ w1 + y_t-2 @ w2 + ...
        model consists of {const_trend, w1, w2, ...}, with shape of 1 + lag
        """
        model = AutoReg(data, lags=self.lag).fit().params
        # similar to AutoReg.forecast(steps=1)
        new_data = 0
        start = 0
        if model.shape[0] > self.lag:
            new_data = model[0]
            start = 1
        for i in range(0, self.lag):
            new_data += self.old_nums[-i - 1] * model[start + i]
        return model, new_data


def correct_cdf(cdf):
    """
    correct the predicted cdf:
    1. monotonically increase
    2. normalize into [0.0, 1.0]
    """
    cur_v = 0.0
    cdf_width = len(cdf)
    i = 0
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
