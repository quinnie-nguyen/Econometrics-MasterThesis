import pandas
import numpy
import datetime
import dateutil.relativedelta as relativedelta
import sys
sys.path.append(r'D:\TU_DORTMUND\ThesisCode')
import jumpdiffcalibrator as jdcal
import CRPS.CRPS as pscore
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()


class Heston():
    def __init__(self,
                 nsim = 10000,
                 r=0,
                 q=0,
                 n_mcmc_steps = 10000,
                 burn_in=5000,
                 estimation_length = 60,
                 si=0,
                 plot = True):
        self.nsim = nsim
        self.r = r
        self.q = q
        self.n_mcmc_steps = n_mcmc_steps
        self.burn_in = burn_in
        self.estimation_length = estimation_length
        self.si = si
        self.plot = plot
        df = pandas.read_csv(r"C:\temp\gas_price.csv")
        df['Unnamed: 0'] = pandas.to_datetime(df['Unnamed: 0'])
        df = df[df['Unnamed: 0'].dt.weekday < 5]
        df.columns = ['Date', 'Price']
        self.df = df
        self.gas_data = self.df.iloc[self.si:self.si + self.estimation_length + 1]['Price'].to_numpy()
        self.HestonClibration()
        self.insample_simulation()
        #self.outsample_simulation()


    def HestonClibration(self):
        self.heston_cal = jdcal.HestonCalibrator(price_series=self.gas_data, cost_of_carry=self.r - self.q,
                                                 vol_prior_mean = 0.09, vol_prior_std = 0.1)
        self.heston_cal.calibrate(n_mcmc_steps=self.n_mcmc_steps, burn_in=self.burn_in)
        all_params = self.heston_cal.params_dict
        self.mu = all_params.get("mu_final")
        self.kappa = all_params.get("kappa_final")
        self.theta = all_params.get("theta_final")
        self.sigma = all_params.get("volvol_final")
        self.rho = all_params.get("rho_final")

    def insample_simulation(self):

        s0 = self.gas_data[0]
        simulated_paths, simulated_variances = self.heston_cal.get_paths(s0=s0, nsteps=len(self.gas_data)-1, nsim=self.nsim, risk_neutral=False,
                                                                    dt=1 / 252)
        self.price_simu_insample = pandas.DataFrame(simulated_paths).T
        self.vpast = pandas.DataFrame(simulated_variances).iloc[:,-1].mean()
        self.insample_crps, fcrps, acrps = pscore(self.price_simu_insample.iloc[-1], self.gas_data[:len(self.gas_data)][-1]).compute()
        print(f"CRPS In-Sample: {self.insample_crps}")

        if self.plot==True:
            pandas.concat([self.price_simu_insample.iloc[:, :10], pandas.DataFrame(self.gas_data[:len(self.gas_data)]).rename(columns={0: "Price"})],
                          axis=1).plot(color = ['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black',
                                                 'red'])

    def outsample_simulation(self, step):

        s0 = self.gas_data[-1]
        simulated_paths, simulated_variances = self.heston_cal.get_paths(s0=s0, nsteps=step, nsim=self.nsim,
                                                                        risk_neutral=False,
                                                                        v0 = self.vpast)
        price_simu_outsample = pandas.DataFrame(simulated_paths).T
        real_gas = self.df.iloc[self.si + self.estimation_length: self.si + self.estimation_length + step + 1]['Price'].to_numpy()
        outsample_crps, fcrps1, acrps1 = pscore(price_simu_outsample.iloc[-1], real_gas[-1]).compute()
        print(f"CRPS Out-Sample: {outsample_crps}")
        if self.plot == True:
            pandas.concat([price_simu_outsample.iloc[:, :10], pandas.DataFrame(real_gas).rename(columns={0: "Price"})],
                          axis=1).plot(color= ['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black',
                                                 'red'])
        return price_simu_outsample, real_gas, outsample_crps

class HESTON_CNT():
    def __init__(self, contract = 'Jan-24', valdate = datetime.datetime(2023, 10, 1),
                 nsim = 10000,
                 r=0,
                 q=0,
                 n_mcmc_steps = 10000,
                 burn_in=5000,
                 estimation_length = 3, # month
                 plot = False, rate = 0.05,
                 fixed_param = None):
        price_df = pandas.read_csv(fr"D:\TU_DORTMUND\Thesis\Data\price\{contract}.csv")
        self.cnt = contract
        self.expiry_date = datetime.datetime.strptime(contract, '%b-%y')
        price_df['Date'] = pandas.to_datetime(price_df['Date'])
        price_df = price_df[price_df['Date'].dt.weekday < 5]
        price_df = price_df[['Date', 'CLOSE']]
        price_df.columns = ['Date', 'Price']
        self.nsim = nsim
        self.r = r
        self.q = q
        self.n_mcmc_steps = n_mcmc_steps
        self.burn_in = burn_in
        self.estimation_length = estimation_length
        self.plot = plot
        self.rate = rate
        self.df = price_df.iloc[-252:, :]
        self.valdate = valdate
        self.fixed_param = fixed_param
        self.gas_data = self.df[(self.df['Date'] >= self.valdate - relativedelta.relativedelta(months=self.estimation_length)) & (self.df['Date'] < self.valdate)]['Price'].to_numpy()
        self.HestonClibration()
        self.insample_simulation()
        spot = pandas.read_csv(r'D:\TU_DORTMUND\Thesis\Data\price\Spot_Price.csv')
        spot['Date'] = pandas.to_datetime(spot['Date'])
        self.spot = spot
        self.get_step_till_expiry()
        self.get_step_till_end()
        #self.outsample_simulation()


    def HestonClibration(self):
        self.heston_cal = jdcal.HestonCalibrator(price_series=self.gas_data, cost_of_carry=self.r - self.q,
                                                 vol_prior_mean = 0.09, vol_prior_std = 0.1)
        self.heston_cal.calibrate(n_mcmc_steps=self.n_mcmc_steps, burn_in=self.burn_in, fixed_param=self.fixed_param)
        self.all_params = self.heston_cal.params_dict
        self.mu = self.all_params.get("mu_final")
        self.kappa = self.all_params.get("kappa_final")
        self.theta = self.all_params.get("theta_final")
        self.sigma = self.all_params.get("volvol_final")
        self.rho = self.all_params.get("rho_final")



    def insample_simulation(self):

        s0 = self.gas_data[0]
        simulated_paths, simulated_variances = self.heston_cal.get_paths(s0=s0, nsteps=len(self.gas_data)-1, nsim=self.nsim, risk_neutral=False,
                                                                    dt=1 / 252)
        self.price_simu_insample = pandas.DataFrame(simulated_paths).T
        self.vpast = pandas.DataFrame(simulated_variances).iloc[:,-1].mean()
        self.insample_crps, fcrps, acrps = pscore(self.price_simu_insample.iloc[-1], self.gas_data[:len(self.gas_data)][-1]).compute()
        print(f"CRPS In-Sample: {self.insample_crps}")
    def get_business_day(self, dd):
        cal = USFederalHolidayCalendar()
        while dd.isoweekday() > 5 or dd in cal.holidays():
            dd += datetime.timedelta(days=1)
        return dd

    def get_step_till_expiry(self):
        self.first_busday_after_expiry = self.get_business_day(self.expiry_date)
        self.steps_needed = numpy.busday_count(self.valdate.date(), self.first_busday_after_expiry.date())

    def get_step_till_end(self):
        self.steps_end = numpy.busday_count(self.valdate.date(), self.df['Date'].iloc[-1].date())

    def outsample_simulation(self, step):

        s0 = self.gas_data[-1]
        simulated_paths, simulated_variances = self.heston_cal.get_paths(s0=s0, nsteps=step, nsim=self.nsim,
                                                                        risk_neutral=True,
                                                                        v0 = self.vpast)
        price_simu_outsample = pandas.DataFrame(simulated_paths).T
        #real_gas = self.df.iloc[self.si + self.estimation_length: self.si + self.estimation_length + step + 1]['Price'].to_numpy()
        self.terminal_price = self.spot[self.spot['Date'] == self.first_busday_after_expiry]['Close'].values[0]
        outsample_crps, fcrps1, acrps1 = pscore(price_simu_outsample.iloc[-1], self.terminal_price).compute()
        print(f"CRPS Out-Sample: {outsample_crps}")
        return price_simu_outsample, outsample_crps

    def call_option_price(self, step, moneyness=0.9):
        price_simu, _ = self.outsample_simulation(step)
        self.strike = self.gas_data[-1]*moneyness
        self.t2e = (self.first_busday_after_expiry - self.valdate).days // 30
        payoff = numpy.mean([i if i > 0 else 0 for i in list(price_simu.iloc[-1,:] - self.strike)])
        discount = (1+self.rate)**(self.t2e/12)
        call_price = payoff/discount
        return call_price

class Delta_Hedge():

    def __init__(self, heston_obj, rate = 0, step = 60,
                 s0_1 = 0.999,
                 s0_2 = 1.001,
                 moneyness = 1.1):
        self.obj = heston_obj
        self.t2e = (self.obj.first_busday_after_expiry - self.obj.valdate).days//30
        self.rate = rate
        self.step = step
        self.s0_1 = s0_1
        self.s0_2 = s0_2
        self.moneyness = moneyness
        self.delta_compute(self.moneyness)

    def outsample_simulation(self, step, s0_rate):
        s0 = self.obj.gas_data[-1] * s0_rate
        simulated_paths, simulated_variances = self.obj.heston_cal.get_paths(s0=s0, nsteps=step, nsim=self.obj.nsim, v0=self.obj.vpast)
        price_simu_outsample = pandas.DataFrame(simulated_paths).T
        self.terminal_price = self.obj.spot[self.obj.spot['Date'] == self.obj.first_busday_after_expiry]['Close'].values[0]
        #terminal_price = self.df['Price'].iloc[-1]
        outsample_crps, fcrps1, acrps1 = pscore(price_simu_outsample.iloc[-1], self.terminal_price).compute()
        print(f"CRPS Out-Sample: {outsample_crps}")
        return price_simu_outsample, outsample_crps

    def call_option_price(self, step, s0_rate, moneyness):
        price_simu, _ = self.outsample_simulation(step, s0_rate)
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

if __name__ == '__main__':
    ### RUN EGARCH for contract
    contract = 'Jan-24'
    start_date = datetime.datetime(2023, 10, 1)-datetime.timedelta(days=1)
    moneyness = 1.2
    call_price = 0
    while call_price <=0:
        start_date = start_date + datetime.timedelta(days=1)
        while start_date.isoweekday() > 5 or start_date in cal.holidays():
            start_date = start_date + datetime.timedelta(days=1)
        obj_0 = HESTON_CNT(contract=contract, valdate=start_date, estimation_length=3)
        call_price = obj_0.call_option_price(step=obj_0.steps_needed, moneyness=moneyness)
    param_dict = obj_0.all_params
    valdate = start_date
    rslt_dict = dict()
    rslt_dict['valdate'] = [start_date-datetime.timedelta(days=1)]
    rslt_dict['S'] = [0]
    rslt_dict['delta'] = [0]
    while valdate < datetime.datetime(2024, 1, 2):
        while valdate.isoweekday() > 5 or valdate in cal.holidays():
            valdate = valdate + datetime.timedelta(days=1)
        try:
            obj = HESTON_CNT(contract=contract, valdate=valdate, estimation_length=3, fixed_param=param_dict)
            delta_obj = Delta_Hedge(heston_obj=obj, step=obj.steps_needed, moneyness=moneyness)
            if delta_obj.delta > 0:
                rslt_dict['valdate'].append(valdate)
                rslt_dict['S'].append(obj.gas_data[-1])
                rslt_dict['delta'].append(delta_obj.delta)
        except:
            pass

        valdate = valdate + datetime.timedelta(days=1)


    hedge_df = pandas.DataFrame(rslt_dict)
    delta_hedge = [(hedge_df['delta'][i] - hedge_df['delta'][i-1])*hedge_df['S'][i]*(1+obj_0.rate)**(obj_0.t2e/12) for i in range(1, hedge_df.shape[0])]
    delta_hedge.insert(0, 0)
    hedge_df['delta_hedge'] = delta_hedge
    strike = obj_0.gas_data[-1]*moneyness
    execute_price = obj_0.terminal_price
    portfolio = -numpy.sum(hedge_df['delta_hedge']) + call_price*(1+obj_0.rate)**(obj_0.t2e/12) + hedge_df['delta'].iloc[-1]*execute_price
    payoff = numpy.max([execute_price - strike, 0])

    #### Short a Call Option and long underlying to hedge the position
    pnl = portfolio-payoff




    ### RUN HESTON for contract
    contract = 'Jan-18'
    obj = HESTON_CNT(contract=contract, valdate=datetime.datetime(2017, 10, 1))
    delta_obj = Delta_Hedge(obj, step=obj.steps_needed)
    price_path, outsample_crps_5 = obj.outsample_simulation(step=obj.steps_needed)



    insample_crps_series = []
    outsample_crps_series_5 = []
    outsample_crps_series_20 = []
    outsample_crps_series_60 = []
    estimation_length = 60
    for si in range(100):

        obj = Heston(nsim=10000,
                     r=0,
                     q=0,
                     n_mcmc_steps=10000,
                     burn_in=5000,
                     estimation_length=estimation_length,
                     si=si,
                     plot=False)
        insample_crps_series.append(obj.insample_crps)
        price_path, real_path, outsample_crps_5 = obj.outsample_simulation(step=5)
        outsample_crps_series_5.append(outsample_crps_5)

        price_path, real_path, outsample_crps_20 = obj.outsample_simulation(step=20)
        outsample_crps_series_20.append(outsample_crps_20)

        price_path, real_path, outsample_crps_60 = obj.outsample_simulation(step=60)
        outsample_crps_series_60.append(outsample_crps_60)

    df = pandas.read_csv(r"C:\temp\gas_price.csv")
    df['Unnamed: 0'] = pandas.to_datetime(df['Unnamed: 0'])
    df = df[df['Unnamed: 0'].dt.weekday < 5]
    df.columns = ['Date', 'Price']
    #valdate = df['Date'][estimation_length+1:estimation_length+100+1]
    crps_df = pandas.DataFrame([outsample_crps_series_5, outsample_crps_series_20, outsample_crps_series_60]).T
    crps_df.columns = ['Step5', 'Step20', 'Step60']

    crps_df.to_csv(rf'C:\temp\crps_Heston.csv', index=False)

    #
    # o
    # # ----- set parameters
    # #s0 = 100
    # nsteps = 50
    # nsim = 10000
    # r = 0 #0.05
    # q = 0 #0.02
    # # ----- calibrate parameters
    # n_mcmc_steps = 10000
    # burn_in = 5000
    #
    # df['Unnamed: 0'] = pandas.to_datetime(df['Unnamed: 0'])
    # df = df[df['Unnamed: 0'].dt.weekday < 5]
    # df.columns = ['Date', 'Price']
    # si = 1
    # estimation_length = 60
    # gas_data = df.iloc[si:si+estimation_length+1]['Price'].to_numpy()
    # s0=gas_data[0]
    #
    # heston_cal = jdcal.HestonCalibrator(price_series=gas_data, cost_of_carry=r - q)
    # heston_cal.calibrate(n_mcmc_steps=n_mcmc_steps, burn_in=burn_in)
    # all_params = heston_cal.params_dict
    # mu = all_params.get("mu_final")
    # kappa = all_params.get("kappa_final")
    # theta = all_params.get("theta_final")
    # sigma = all_params.get("volvol_final")
    # rho = all_params.get("rho_final")
    #
    # simulated_paths, simulated_variances = heston_cal.get_paths(s0=s0, nsteps=nsteps, nsim=nsim, risk_neutral=False, dt = 1/252)
    # price_simu = pandas.DataFrame(simulated_paths).T
    # crps, fcrps, acrps = pscore(price_simu.iloc[-1], gas_data[-1]).compute()
    #
    # simulated_paths_1, simulated_variances_1 = heston_cal.get_paths(s0=gas_data[-1], nsteps=nsteps, nsim=nsim, risk_neutral=False, v0 = pandas.DataFrame(simulated_variances).iloc[:, -1].mean())
    # price_simu_forecast = pandas.DataFrame(simulated_paths_1).T
    # real_gas = df.iloc[si+estimation_length: si+estimation_length+nsteps+1]['Price'].to_numpy()
    # crps1, fcrps1, acrps1 = pscore(price_simu_forecast.iloc[-1], real_gas[-1]).compute()
    #
    # pandas.concat([price_simu_forecast.iloc[:, :10], pandas.DataFrame(real_gas).rename(columns={0: "Price"})],
    #               axis=1).plot()
    #
