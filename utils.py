import pandas
import numpy
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
from EGARCH import quasiML_utils as EGARCH
import os

rootdir = r'D:\TU_DORTMUND\Thesis\Data\price'
params_root_dir = r'D:\TU_DORTMUND\Thesis\Data\params'
crps_root_dir = r'D:\TU_DORTMUND\Thesis\Data\crps'

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
def get_business_day(dd):
    cal = USFederalHolidayCalendar()
    while dd.isoweekday() > 5 or dd in cal.holidays():
        dd += datetime.timedelta(days=1)
    return dd
class calibration_CNT():
    def __init__(self, contract, #'mm-yy'
                 t2m, #'months/12'
                 model, # 'EGARCH' for EGARCH skewed-t or HESTON
                 delta_hegde = False,
                 estimation_length = 3):
        self.cnt = contract
        self.start_delivery = get_business_day(datetime.datetime.strptime(self.cnt, "%b-%y").replace(day=1))
        self.t2m = t2m
        self.valdate = get_business_day(self.start_delivery - datetime.timedelta(days=round(self.t2m*365,0)))
        self.model = model
        self.delta_hegde = delta_hegde
        self.estimation_length = estimation_length
        self.params_calibration()
        self.outsample_simulation()
        attemp = 0
        while self.model_obj.outsample_crps is numpy.nan and attemp <=4:
            attemp = attemp +1
            self.valdate = get_business_day(self.valdate + datetime.timedelta(days = 1))
            self.params_calibration()
            self.outsample_simulation()


    def params_calibration(self):
        if self.model == 'EGARCH':
            self.model_obj = EGARCH.EGARCH_SST(contract=self.cnt, valdate=self.valdate, estimation_length=self.estimation_length)
            self.S0 = self.model_obj.gas_data[-1]
            self._params_dict = self.model_obj._params_dict
        else:
            self._params_dict = None

    def archive_params(self):
        df = pandas.DataFrame(self._params_dict, index=[self.valdate])
        file_path = fr"{params_root_dir}\{self.model}\{self.cnt}.csv"
        if os.path.exists(file_path):
            df_save = pandas.read_csv(file_path, index_col=0, parse_dates=True)
            df_save = pandas.concat([df_save, df], axis = 0)
            df_save = df_save.loc[~df_save.index.duplicated(keep='last')]
            df_save.to_csv(fr"{params_root_dir}\{self.model}\{self.cnt}.csv")
        else:
            df.to_csv(fr"{params_root_dir}\{self.model}\{self.cnt}.csv")

    def outsample_simulation(self):
        self.model_obj.outsample_simulation(self.model_obj.steps_needed)
        self.crps = pandas.DataFrame([self.model_obj.outsample_crps], index = [self.valdate], columns = ['CRPS'])

    def archive_crps(self):
        df = pandas.DataFrame(self.crps, index=[self.valdate])
        file_path = fr"{crps_root_dir}\{self.model}\{self.cnt}.csv"
        if os.path.exists(file_path):
            df_save = pandas.read_csv(file_path, index_col=0, parse_dates=True)
            df_save = pandas.concat([df_save, df], axis = 0)
            df_save = df_save.loc[~df_save.index.duplicated(keep='last')]
            df_save.to_csv(fr"{crps_root_dir}\{self.model}\{self.cnt}.csv")
        else:
            df.to_csv(fr"{crps_root_dir}\{self.model}\{self.cnt}.csv")


if __name__ == '__main__':
    contract = 'Jul-24'
    t2m = 3/12
    model = 'EGARCH'
    obj = calibration_CNT(contract=contract, t2m=t2m, model=model)
    obj.archive_params()
    obj.archive_crps()




