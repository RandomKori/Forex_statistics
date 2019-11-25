import statsmodels.api as sm
from datetime import datetime
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
from numpy.random import seed
from numpy.random import randn
from scipy.stats import shapiro

def statc(x):
    test = sm.tsa.adfuller(x)
    print ('adf: ', test[0]) 
    print ('p-value: ', test[1])
    print('Critical values: ', test[4])
    if test[0]> test[4]['5%']: 
        print ('есть единичные корни, ряд не стационарен')
    else:
        print ('единичных корней нет, ряд стационарен')

def hurst(ts):

    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    # Here it calculates the variances, but why it uses 
    # standard deviation and then make a root of it?
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0

def isnorm(x):
    seed(1)
    data = 5 * randn(100) + 50
    stat, p = shapiro(x)
    print('Statistics=', stat)
    print ('p-value:=', p)
    alpha = 0.05
    if p > alpha:
        print('Нормальное распределение')
    else:
        print('Распределение не соответствует нормальному')