# VAR
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from random import random
# contrived dataset with dependency
data = list()
for i in range(100):
  v1 = i + random()
  v2 = v1 + random()
  row = [v1, v2]
  data.append(row)
data = np.array(data)
# fit model
model = VAR(data)
model_fit = model.fit(maxlags=3, verbose=True, trend="n")
# make prediction
yhat = model_fit.forecast(data, steps=1)
print(yhat)
params = model_fit.params
yhat1 = np.zeros(2)
start = 0
if params.shape[0] > 6:
    yhat1 = params[0]
    start = 1
for i in range(0, 3):
    yhat1 += data[-i - 1] @ params[start: start + 2]
    start += 2
print(yhat1)