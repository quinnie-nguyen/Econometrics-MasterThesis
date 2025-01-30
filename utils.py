import pandas
import numpy
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
from EGARCH import quasiML_utils as EGARCH
from jumpdiffcalibrator import HESTON_utils as HESTON
import os

rootdir = r'D:\TU_DORTMUND\Thesis\Data\price'
params_root_dir = r'D:\TU_DORTMUND\Thesis\Data\params'
crps_root_dir = r'D:\TU_DORTMUND\Thesis\Data\crps'
pnl_root_dir = r'D:\TU_DORTMUND\Thesis\Data\pnl'

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
                 t2m,
                 valdate, #'months/12'
                 model, # 'EGARCH' for EGARCH skewed-t or HESTON
                 delta_hegde = False,
                 estimation_length = 3,
                 fixed_result = None):
        self.cnt = contract
        self.start_delivery = get_business_day(datetime.datetime.strptime(self.cnt, "%b-%y").replace(day=1))
        self.t2m = t2m
        if t2m is None:
            self.valdate = valdate
        else:
            self.valdate = get_business_day(self.start_delivery - datetime.timedelta(days=round(self.t2m*365,0)))
        self.model = model
        self.delta_hegde = delta_hegde
        self.estimation_length = estimation_length
        self.fixed_result = fixed_result
        self.params_calibration()
        if self.model == 'EGARCH' and self.model_obj.result.success == True:
            self.outsample_simulation()
        elif self.model == 'EGARCH' and self.model_obj.result.success == False:
            self.crps = pandas.DataFrame([numpy.nan], index = [self.valdate], columns = ['CRPS'])
        elif self.model == 'HESTON':
            self.outsample_simulation()

        # attemp = 0
        # while self.model_obj.outsample_crps is numpy.nan and attemp <=4:
        #     attemp = attemp +1
        #     self.valdate = get_business_day(self.valdate + datetime.timedelta(days = 1))
        #     self.params_calibration()
        #     self.outsample_simulation()


    def params_calibration(self):
        if self.model == 'EGARCH':
            self.model_obj = EGARCH.EGARCH_SST(contract=self.cnt, valdate=self.valdate, estimation_length=self.estimation_length, fixed_result=self.fixed_result)
            self.S0 = self.model_obj.gas_data[-1]
            self._params_dict = self.model_obj._params_dict
        else:
            self.model_obj = HESTON.HESTON_CNT(contract=self.cnt, valdate=self.valdate, estimation_length=self.estimation_length, fixed_param=self.fixed_result)
            self.S0 = self.model_obj.gas_data[-1]
            self._params_dict = self.model_obj._params_dict

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




class Delta_Hedge():

    def __init__(self, egarch_obj, rate = 0, step = 60,
                 s0_1 = 0.999,
                 s0_2 = 1.001,
                 moneyness = 0.9,
                 model = 'HESTON'):
        self.obj = egarch_obj
        self.t2e = (self.obj.expiry_date - self.obj.valdate).days//30
        self.rate = rate
        self.step = step
        self.s0_1 = s0_1
        self.s0_2 = s0_2
        self.moneyness = moneyness
        self.model = model
        self.delta_compute(self.moneyness)


    def outsample_simulation_egarch(self, step, s_rate):
        s0 = self.obj.gas_data[-1] * s_rate
        simulated_paths, simulated_variances = self.obj.get_paths(s0=s0, nsteps=step, nsim=self.obj.nsim, v0=self.obj.vpast)
        price_simu_outsample = pandas.DataFrame(simulated_paths).T
        #self.terminal_price = self.obj.spot[self.obj.spot['Date'] == self.obj.expiry_date]['Close'].values[0]
        #terminal_price = self.df['Price'].iloc[-1]
        #outsample_crps, fcrps1, acrps1 = pscore(price_simu_outsample.iloc[-1], self.terminal_price).compute()
        #print(f"CRPS Out-Sample: {outsample_crps}")
        return price_simu_outsample#, outsample_crps

    def outsample_simulation_heston(self, step, s_rate):
        s0 = self.obj.gas_data[-1] * s_rate
        simulated_paths, simulated_variances = self.obj.heston_cal.get_paths(s0=s0, nsteps=step, nsim=self.obj.nsim, v0=self.obj.vpast)
        price_simu_outsample = pandas.DataFrame(simulated_paths).T
        #self.terminal_price = self.obj.spot[self.obj.spot['Date'] == self.obj.expiry_date]['Close'].values[0]
        #terminal_price = self.df['Price'].iloc[-1]
        #outsample_crps, fcrps1, acrps1 = pscore(price_simu_outsample.iloc[-1], self.terminal_price).compute()
        #print(f"CRPS Out-Sample: {outsample_crps}")
        return price_simu_outsample#, outsample_crps

    def call_option_price(self, step, s0_rate, moneyness):
        if self.model == 'EGARCH':
            price_simu = self.outsample_simulation_egarch(step, s0_rate)
        else:
            price_simu = self.outsample_simulation_heston(step, s0_rate)
        self.strike = self.obj.gas_data[-1]*moneyness
        payoff = numpy.mean([i if i > 0 else 0 for i in list(price_simu.iloc[-1,:] - self.strike)])
        discount = (1+self.rate)**(self.t2e/12)
        call_price = payoff/discount
        return call_price

    def delta_compute(self, moneyness):
        call_1 = self.call_option_price(self.step, self.s0_1, moneyness)
        call_2 = self.call_option_price(self.step, self.s0_2, moneyness)

        s0_1 = self.obj.gas_data[-1] * self.s0_1
        s0_2 = self.obj.gas_data[-1] * self.s0_2

        self.delta = (call_1-call_2)/(s0_1-s0_2)

        return self.delta



class hegding_pnl():
    def __init__(self, origin_obj, moneyness = 0.95):

        self.origin_obj = origin_obj
        self.moneyness = moneyness

        self.call_price = self.origin_obj.model_obj.call_option_price(step=self.origin_obj.model_obj.steps_needed, moneyness=self.moneyness)
        self.compute_pnl()

    def compute_pnl(self):
        if self.origin_obj.model == 'EGARCH':
            fixed_rslt = self.origin_obj.model_obj.result
        else:
            fixed_rslt = self.origin_obj.model_obj.all_params
        rslt_dict = dict()
        start_date = self.origin_obj.model_obj.valdate + datetime.timedelta(days=1)
        rslt_dict['valdate'] = [start_date - datetime.timedelta(days=1)]
        rslt_dict['S'] = [0]
        rslt_dict['delta'] = [0]
        while start_date < self.origin_obj.model_obj.expiry_date:
            while start_date.isoweekday() > 5 or start_date in cal.holidays():
                start_date = start_date + datetime.timedelta(days=1)
            try:
                obj = calibration_CNT(contract=self.origin_obj.cnt, t2m=None, valdate=start_date, model=self.origin_obj.model,
                                            delta_hegde=False, estimation_length=3, fixed_result=fixed_rslt)
                delta_obj = Delta_Hedge(egarch_obj=obj.model_obj, step=obj.model_obj.steps_needed,
                                              moneyness=self.moneyness, model = obj.model)
                if delta_obj.delta > 0:
                    rslt_dict['valdate'].append(start_date)
                    rslt_dict['S'].append(obj.model_obj.gas_data[-1])
                    rslt_dict['delta'].append(delta_obj.delta)
            except:
                pass
            start_date = start_date + datetime.timedelta(days=1)

        hedge_df = pandas.DataFrame(rslt_dict)
        delta_hedge = [
            (hedge_df['delta'][i] - hedge_df['delta'][i - 1]) * hedge_df['S'][i] * (1 + self.origin_obj.model_obj.rate) ** (
                        1 / 12) for i in range(1, hedge_df.shape[0])]
        delta_hedge.insert(0, 0)
        hedge_df['delta_hedge'] = delta_hedge
        strike = self.origin_obj.model_obj.gas_data[-1] * self.moneyness
        execute_price = self.origin_obj.model_obj.terminal_price
        portfolio = -numpy.sum(hedge_df['delta_hedge']) + self.call_price * (1 + self.origin_obj.model_obj.rate) ** (1 / 12) + \
                    hedge_df['delta'].iloc[-1] * execute_price
        self.payoff = numpy.max([execute_price - strike, 0])
        pnl = portfolio - self.payoff
        self.hedge_df = hedge_df
        self.pnl = pnl

    def archive_pnl(self):
        df = pandas.DataFrame([self.pnl], index=[self.origin_obj.model_obj.valdate])
        file_path = fr"{pnl_root_dir}\{self.origin_obj.model}\{self.origin_obj.cnt}.csv"
        if os.path.exists(file_path):
            df_save = pandas.read_csv(file_path, index_col=0, parse_dates=True)
            df_save = pandas.concat([df_save, df], axis = 0)
            df_save = df_save.loc[~df_save.index.duplicated(keep='last')]
            df_save.to_csv(fr"{pnl_root_dir}\{self.origin_obj.model}\{self.origin_obj.cnt}.csv")
        else:
            df.to_csv(fr"{pnl_root_dir}\{self.origin_obj.model}\{self.origin_obj.cnt}.csv")


if __name__ == '__main__':
    contract = 'Jul-24'
    t2m = 3/12
    model = 'EGARCH'
    obj = calibration_CNT(contract=contract, t2m=t2m, model=model)
    obj.archive_params()
    obj.archive_crps()




