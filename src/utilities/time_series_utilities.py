# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""
import pandas as pd
import statsmodels.api as sm


def test_stationarity(x):
    adf_test = sm.tsa.stattools.adfuller(x=x,
                                         regression='ct',
                                         autolag='AIC')
    adf_output = pd.Series(adf_test[0:4],
                           index=['Test Statistic', 'p-value', '# Lags Used', '# Observations Used'])
    for key, value in adf_test[4].items():
        adf_output['Critical Value (%s)' % key] = value
    print(adf_output)
