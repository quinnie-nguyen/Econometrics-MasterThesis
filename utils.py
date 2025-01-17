import pandas
import numpy
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()

rootdir = r'D:\TU_DORTMUND\Thesis\Data\price'

def get_price_data(no_month_ahead=1, start_date = datetime.datetime(2016, 2, 1), end_date = datetime.datetime(2025, 1, 1)):
    month_to_symbol = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    yr_list = [str(yr)[-2:] for yr in range(start_date.year, end_date.year)]
    df_list = dict()
    df_list['cnt_name'] = []
    df_list['data'] = []
    for y in yr_list:
        for m in list(month_to_symbol):
            cnt_name = f"{m}-{y}"
            print(cnt_name)
            df_temp = pandas.read_csv(rf"{rootdir}\{cnt_name}.csv")
            df_list['cnt_name'] = cnt_name
            df_list['data'].append(df_temp)
    df_total = pandas.concat([df for df in df_list['data']], axis=0).drop_duplicates(keep='first')
    return df_total

def get_data_by_cnt(month, year, length = 252):
    cnt_name = f"{month}-{year}"
    price = pandas.read_csv(rf"{rootdir}\{cnt_name}.csv")
    price['LOG_RETURN'] = numpy.log(price['CLOSE']).diff()
    price['Date'] = pandas.to_datetime(price['Date'])
    price = price[['Date', 'CLOSE', 'LOG_RETURN']].iloc[-length:, :]
    return price

def compute_pacf(data, nlags):
    """
    Calculate Partial Autocorrelation Function (PACF) using Durbin-Levinson recursion.

    Parameters:
        data (array-like): Time series data.
        nlags (int): Number of lags for which to calculate PACF.

    Returns:
        pacf (array): PACF values for lags 0 to nlags.
    """
    n = len(data)
    mean = numpy.mean(data)
    data = data - mean  # Demean the data

    # Compute autocorrelations
    acf = numpy.correlate(data, data, mode='full') / (n * numpy.var(data))
    acf = acf[n-1:]  # Keep only non-negative lags

    pacf = [1]  # PACF(0) is always 1
    phi_prev = []

    for k in range(1, nlags + 1):
        # Solve Yule-Walker equations for lag k
        toeplitz_matrix = numpy.array([acf[abs(i - j)] for i in range(k) for j in range(k)]).reshape(k, k)
        rhs = acf[1:k+1]
        phi_k = numpy.linalg.solve(toeplitz_matrix, rhs)
        pacf.append(phi_k[-1])  # Append PACF value for lag k

    return numpy.array(pacf)

if __name__ == '__main__':
    data = get_price_data()
