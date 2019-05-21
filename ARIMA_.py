import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams.update({'font.size': 12})
plt.style.use('ggplot')

data = pd.read_excel(io='data.xlsx')
dat = data['location']

dat.plot()
plt.ylabel('Location')
plt.xlabel('Time')
plt.show()

# Define the d and q parameters to take any value between 0 and 1
q = d = range(0, 2)
# Define the p parameters to take any value between 0 and 3
p = range(0, 4)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 14) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))


train_data = dat[0:1122]
test_data = dat[1123:1398]

warnings.filterwarnings("ignore") # specify to ignore warning messages
#下面代码用于选择最佳的p，q,d的取值
AIC = []
SARIMAX_model = []
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(train_data,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal, results.aic), end='\r')
            AIC.append(results.aic)
            SARIMAX_model.append([param, param_seasonal])
        except:
            continue

print('The smallest AIC is {} for model SARIMAX{}x{}'.format(min(AIC), SARIMAX_model[AIC.index(min(AIC))][0],SARIMAX_model[AIC.index(min(AIC))][1]))


#Let's fit this model
mod = sm.tsa.statespace.SARIMAX(train_data,
                                order=SARIMAX_model[AIC.index(min(AIC))][0],
                                seasonal_order=SARIMAX_model[AIC.index(min(AIC))][1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)

# # 通过上述代码得到最好的配置参数
# mod = sm.tsa.statespace.SARIMAX(train_data,
#                                 order=(3, 0, 1),
#                                 seasonal_order=(1, 0, 1, 14),
#                                 enforce_stationarity=False,
#                                 enforce_invertibility=False)
results = mod.fit()

results.plot_diagnostics(figsize=(20, 14))
plt.show()

# 以下为三种评估方式，对于测试集上的预测可以通过type3
# Type 1: prediction with 1-step (using training data), each forecasted point will be used to predict the following one.
pred0 = results.get_prediction(start=64, dynamic=False)
pred0_ci = pred0.conf_int()
forecast = pd.DataFrame(pred0.predicted_mean.to_frame('location').rename_axis('number').reset_index())
forecast.to_csv('data_pre_1000.csv')

# Type 2: prediction with 1-step(using training data)
pred1 = results.get_prediction(start=64, dynamic=True)
pred1_ci = pred1.conf_int()


# Type3 : forecasting of out of training data 对测试集数据进行预测（该部分数据未参与训练过程），由于Arima算法是回归算法，因此取四舍五入后结果
pred2 = results.get_forecast(steps=275)
pred2_ci = pred2.conf_int()
print(pred2.predicted_mean)
forecast = pd.DataFrame(pred2.predicted_mean.to_frame('location').rename_axis('number').reset_index())
forecast.to_csv('data_pre.csv')

#计算测试集上的结果与预测结果的差异（由于不知道具体评价指标，没有继续完成）